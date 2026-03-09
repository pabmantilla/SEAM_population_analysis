#!/usr/bin/env python
"""
Post-hoc script: compute mechanistic causality from existing injection results
and generate all variant-injection final plots.

Reads cluster_results/variant_inject/{source}/{locus}/k{k}/ outputs
(cluster_labels, csm_matrix, inject_mapping) and the original one-hot sequences
to compute mech_causality per variant, then generates:
  - GnomAD: mech_diversity vs AF, mech_causality vs AF
  - caQTL:  pred vs actual coefficient

Usage:
  python make_inject_plots.py
  python make_inject_plots.py --k 100 --sources gnomad caqtl_eur caqtl_afr
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# ── Paths ──
BASE_DIR = '/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis'
LCL_DIR = os.path.join(BASE_DIR, 'SEAM_ChromBPNet', 'LCL_population_variants')
LOCI_TSV = os.path.join(BASE_DIR, 'variant_data', 'GnomAD_data', 'loci_backup_all34.tsv')
MUTLIB_DIR = os.path.join(LCL_DIR, 'Mutagenisis_lib')
DEEPSHAP_DIR = os.path.join(LCL_DIR, 'DeepSHAP_lib')
VARIANT_LIBS_DIR = os.path.join(LCL_DIR, 'variant_libs')
INJECT_CLUSTER_DIR = os.path.join(LCL_DIR, 'cluster_results', 'variant_inject')
INJECT_RESULTS_DIR = os.path.join(LCL_DIR, 'results', 'results_final', 'variant_inject')

CAQTL_COEFF_DIR = os.path.join(BASE_DIR, 'variant_data', 'Alphagenome_data', 'chromatin_accessibility_qtl')
CAQTL_COEFF_FILES = {
    'caqtl_eur': os.path.join(CAQTL_COEFF_DIR, 'caqtl_european_variant_coefficient_human_predictions.feather'),
    'caqtl_afr': os.path.join(CAQTL_COEFF_DIR, 'caqtl_african_variant_coefficient_human_predictions.feather'),
}

SEQ_LENGTH = 2114
NUM_SEQS = 25000
K_DEFAULT = 100


def load_loci():
    loci = pd.read_csv(LOCI_TSV, sep='\t')
    loci['start'] = loci['tss'] - SEQ_LENGTH // 2
    loci['end'] = loci['tss'] + SEQ_LENGTH // 2
    return loci


def compute_causality_for_source(source, k):
    """Compute mech_causality for all loci of a given source using existing results."""
    inject_base = os.path.join(INJECT_CLUSTER_DIR, source)
    src_dir = os.path.join(VARIANT_LIBS_DIR, source)
    all_loci = load_loci()
    all_rows = []

    for _, row in all_loci.iterrows():
        name = row['name']
        locus_dir = os.path.join(inject_base, name, f'k{k}')
        labels_path = os.path.join(locus_dir, 'cluster_labels.npy')
        csm_path = os.path.join(locus_dir, 'csm_matrix.npy')
        mapping_path = os.path.join(locus_dir, 'inject_mapping.csv')
        preds_path = os.path.join(DEEPSHAP_DIR, f'preds_{name}_{NUM_SEQS}.npy')
        var_preds_path = os.path.join(src_dir, f'preds_{name}.npy')

        # Also need one-hot sequences to build SNP masks
        mut_path = os.path.join(MUTLIB_DIR, f'x_mut_{name}_{NUM_SEQS}.npy')
        var_xmut_path = os.path.join(src_dir, f'x_var_{name}.npy')

        if not all(os.path.exists(p) for p in [labels_path, csm_path, mapping_path,
                                                 mut_path, var_xmut_path]):
            continue

        print(f'  {name} ({source}): computing causality...')

        labels = np.load(labels_path)
        csm = np.load(csm_path)  # (n_clusters, 2114) in percent 0-100
        mapping = pd.read_csv(mapping_path)

        # Load WT one-hot (index 0 of mutagenesis lib)
        x_mut = np.load(mut_path, mmap_mode='r')
        x_wt = np.array(x_mut[0])  # (2114, 4)

        # Load variant one-hot sequences (skip index 0 = WT)
        var_xmut = np.load(var_xmut_path, mmap_mode='r')

        # Load predictions
        preds_25k = np.load(preds_path) if os.path.exists(preds_path) else None
        var_preds = np.load(var_preds_path) if os.path.exists(var_preds_path) else None

        cluster_ids = sorted(np.unique(labels))
        cluster_to_ci = {c: ci for ci, c in enumerate(cluster_ids)}

        # Mechanistic diversity per cluster
        wt_cluster = labels[0]
        maps_path = os.path.join(DEEPSHAP_DIR, f'maps_{name}_{NUM_SEQS}.npy')
        var_maps_path = os.path.join(src_dir, f'maps_{name}.npy')

        # We need injected maps to compute diversity — rebuild from injection
        # But we can read diversity from cluster_info.csv if available
        info_path = os.path.join(locus_dir, 'cluster_info.csv')
        cluster_mech_div = {}
        if os.path.exists(info_path):
            info = pd.read_csv(info_path)
            for _, ir in info.iterrows():
                cluster_mech_div[int(ir['cluster'])] = float(ir['mech_diversity'])

        wt_pred = float(preds_25k[0]) if preds_25k is not None else 0.0

        for i, (_, mrow) in enumerate(mapping.iterrows()):
            injected_idx = int(mrow['injected_idx'])
            c = int(labels[injected_idx])
            ci = cluster_to_ci[c]

            # SNP mask from variant one-hot (i+1 because index 0 is WT in var lib)
            var_onehot = np.array(var_xmut[i + 1])  # (2114, 4)
            snp_mask = np.any(var_onehot != x_wt, axis=1).astype(float)  # (2114,)

            # Mechanistic causality: CSM fraction (0-1) dot SNP mask
            mech_causality = float(np.sum((csm[ci] / 100.0) * snp_mask))

            # Variant prediction
            var_pred = float(var_preds[i + 1]) if var_preds is not None else float(preds_25k[injected_idx]) if preds_25k is not None else 0.0

            # log2 fold change: convert from natural log to log2 (matches notebook)
            log2fc = (var_pred - wt_pred) / np.log(2)

            r = {
                'locus': name,
                'variant_id': mrow.get('variant_id', ''),
                'cluster': c,
                'mech_diversity': cluster_mech_div.get(c, np.nan),
                'mech_causality': mech_causality,
                'pred': var_pred,
                'wt_pred': wt_pred,
                'log2fc': log2fc,
            }
            # Copy metadata columns
            for col in ['AF', 'rsids', 'consequence', 'offset', 'ref', 'alt', 'pos']:
                if col in mrow.index:
                    r[col] = mrow[col]
            all_rows.append(r)

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


def plot_gnomad(df, k):
    """Generate GnomAD plots: mech_diversity vs AF and mech_causality vs AF."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    df = df[df['AF'].notna() & (df['AF'] > 0)].copy()
    if df.empty:
        print('GnomAD: no variants with valid AF > 0')
        return

    locus_names = df['locus'].unique()
    cmap = plt.cm.get_cmap('tab20', max(len(locus_names), 1))
    locus_colors = {n: cmap(i) for i, n in enumerate(locus_names)}

    af_ticks = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    af_tick_labels = ['1', '1/10', '1/100', '1/1K', '1/10K', '1/100K', '1/1M']

    for y_col, y_label, fname_tag in [
        ('mech_diversity', 'Mechanistic Diversity (1 - cos sim to WT)', 'mech_diversity'),
        ('mech_causality', 'Mechanistic Causality (CSM · SNP mask)', 'mech_causality'),
    ]:
        if y_col not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 7))
        for locus_name in locus_names:
            ld = df[df['locus'] == locus_name]
            ax.scatter(ld['AF'], ld[y_col],
                       s=12, alpha=0.6, c=[locus_colors[locus_name]],
                       edgecolors='k', linewidth=0.2, label=locus_name)

        ax.set_xscale('log')
        ax.set_xticks(af_ticks)
        ax.set_xticklabels(af_tick_labels)
        ax.set_xlabel('Allele Frequency', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f'{y_label.split("(")[0].strip()} vs Allele Frequency — GnomAD (k={k})', fontsize=13)
        ax.legend(fontsize=7, ncol=3, loc='best', markerscale=1.5)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(INJECT_RESULTS_DIR, f'gnomad_{fname_tag}_vs_AF_k{k}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved {out_path}')

    df.to_csv(os.path.join(INJECT_RESULTS_DIR, f'gnomad_variant_results_k{k}.csv'), index=False)


def plot_caqtl(df, k, source):
    """Generate caQTL plot: predicted effect vs measured coefficient."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr

    coeff_path = CAQTL_COEFF_FILES.get(source)
    if not coeff_path or not os.path.exists(coeff_path):
        print(f'{source}: coefficient file not found')
        return

    coeff_df = pd.read_feather(coeff_path)
    coeff_target = coeff_df.set_index('variant_id')['target'].to_dict()
    coeff_ag_pred = coeff_df.set_index('variant_id')['prediction'].to_dict()

    df['target'] = df['variant_id'].map(coeff_target)
    df['ag_prediction'] = df['variant_id'].map(coeff_ag_pred)
    df_valid = df[df['target'].notna()].copy()
    if df_valid.empty:
        print(f'{source}: no variants matched coefficient data')
        return

    locus_names = df_valid['locus'].unique()
    cmap = plt.cm.get_cmap('tab20', max(len(locus_names), 1))
    locus_colors = {n: cmap(i) for i, n in enumerate(locus_names)}
    label = 'EUR' if source == 'caqtl_eur' else 'AFR'

    # --- Three-panel figure ---
    # 1) AlphaGenome vs Actual  2) ChromBPNet vs Actual  3) AlphaGenome vs ChromBPNet
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    plot_specs = [
        (axes[0], 'target', 'ag_prediction', f'Measured caQTL Coefficient ({label})', 'AlphaGenome Prediction', f'AlphaGenome vs Actual — {label}'),
        (axes[1], 'target', 'log2fc', f'Measured caQTL Coefficient ({label})', 'ChromBPNet log2FC', f'ChromBPNet vs Actual — {label}'),
        (axes[2], 'ag_prediction', 'log2fc', 'AlphaGenome Prediction', 'ChromBPNet log2FC', f'ChromBPNet vs AlphaGenome — {label}'),
    ]

    for ax, x_col, y_col, x_label, y_label, title in plot_specs:
        for locus_name in locus_names:
            ld = df_valid[df_valid['locus'] == locus_name]
            ax.scatter(ld[x_col], ld[y_col],
                       s=15, alpha=0.6, c=[locus_colors[locus_name]],
                       edgecolors='k', linewidth=0.2, label=locus_name)

        all_vals = np.concatenate([df_valid[x_col].values, df_valid[y_col].values])
        vmin, vmax = all_vals.min(), all_vals.max()
        margin = (vmax - vmin) * 0.05
        ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
                'r--', linewidth=1, alpha=0.5, label='y=x')

        r_pearson, p_pearson = pearsonr(df_valid[x_col], df_valid[y_col])
        r_spearman, p_spearman = spearmanr(df_valid[x_col], df_valid[y_col])
        ax.text(0.05, 0.95,
                f'Pearson r={r_pearson:.3f} (p={p_pearson:.2e})\nSpearman \u03c1={r_spearman:.3f} (p={p_spearman:.2e})',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f'{title} (k={k})', fontsize=12)
        ax.legend(fontsize=6, ncol=2, loc='best', markerscale=1.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(INJECT_RESULTS_DIR, f'{source}_pred_vs_actual_k{k}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {out_path}')

    df_valid.to_csv(os.path.join(INJECT_RESULTS_DIR, f'{source}_variant_results_k{k}.csv'), index=False)


def plot_inject_diversity_evolvability(k, source='gnomad'):
    """Per-locus and cross-locus diversity vs evolvability for injected runs.

    For each locus: plot clusters as dots (x=mean_pred - WT, y=mech_diversity),
    highlight clusters containing injected variants. Save per-locus plots to
    seq_results/{locus}/ and a cross-locus summary to results_final/variant_inject/.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    inject_base = os.path.join(INJECT_CLUSTER_DIR, source)
    all_loci = load_loci()
    all_scatter_rows = []

    for _, row in all_loci.iterrows():
        name = row['name']
        locus_dir = os.path.join(inject_base, name, f'k{k}')
        info_path = os.path.join(locus_dir, 'cluster_info.csv')
        labels_path = os.path.join(locus_dir, 'cluster_labels.npy')
        mapping_path = os.path.join(locus_dir, 'inject_mapping.csv')

        if not all(os.path.exists(p) for p in [info_path, labels_path, mapping_path]):
            continue

        info = pd.read_csv(info_path)
        labels = np.load(labels_path)
        mapping = pd.read_csv(mapping_path)

        # Which clusters contain injected variants?
        variant_clusters = set()
        for _, mrow in mapping.iterrows():
            idx = int(mrow['injected_idx'])
            variant_clusters.add(int(labels[idx]))

        # WT pred for centering
        wt_row = info[info['has_wt'] == True]
        wt_pred = float(wt_row['mean_pred'].values[0]) if not wt_row.empty else float(info['mean_pred'].mean())

        md = info['mech_diversity'].values
        mp = info['mean_pred'].values - wt_pred
        is_wt = info['has_wt'].values
        has_variant = info['cluster'].isin(variant_clusters).values

        # Collect for cross-locus
        for _, ir in info.iterrows():
            c = int(ir['cluster'])
            all_scatter_rows.append({
                'locus': name,
                'cluster': c,
                'mech_diversity': ir['mech_diversity'],
                'pred_centered': ir['mean_pred'] - wt_pred,
                'is_wt': bool(ir['has_wt']),
                'has_variant': c in variant_clusters,
            })

        # --- Per-locus plot ---
        locus_out_dir = os.path.join(LCL_DIR, 'results', 'seq_results', name)
        os.makedirs(locus_out_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))

        # Regular clusters
        mask_regular = ~is_wt & ~has_variant
        ax.scatter(mp[mask_regular], md[mask_regular], s=8, alpha=0.4, c='steelblue',
                   edgecolors='none', label='Non-WT clusters')

        # Variant-containing clusters
        mask_var = has_variant & ~is_wt
        if mask_var.any():
            ax.scatter(mp[mask_var], md[mask_var], s=25, alpha=0.7, c='orange',
                       edgecolors='k', linewidth=0.3, zorder=4, label='Variant clusters')

        # WT cluster
        if is_wt.any():
            ax.scatter(mp[is_wt], md[is_wt], s=40, c='red', marker='*',
                       edgecolors='k', linewidth=0.3, zorder=5, label='WT cluster')

        # Symmetric x-axis around 0
        xlim = max(abs(mp.min()), abs(mp.max())) * 1.1
        ax.set_xlim(-xlim, xlim)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=0.8, alpha=0.4)
        ax.set_xlabel('Functional Evolvability (pred. activity - WT)', fontsize=12)
        ax.set_ylabel('Mechanistic Diversity (1 - cos sim to WT)', fontsize=12)
        ax.set_title(f'{name} — Inject {source} — Diversity vs Evolvability (k={k})', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(locus_out_dir, f'diversity_evolvability_inject_{source}_k{k}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  {name}: saved {out_path}')

    # --- Cross-locus summary ---
    if not all_scatter_rows:
        print(f'  No inject diversity data for {source}')
        return

    sdf = pd.DataFrame(all_scatter_rows)
    locus_names = sdf['locus'].unique()
    cmap = plt.cm.get_cmap('tab20', max(len(locus_names), 1))
    locus_colors = {n: cmap(i) for i, n in enumerate(locus_names)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for locus_name in locus_names:
        ld = sdf[sdf['locus'] == locus_name]
        color = locus_colors[locus_name]

        regular = ld[(ld['is_wt'] == False) & (ld['has_variant'] == False)]
        var_clusters = ld[(ld['has_variant'] == True) & (ld['is_wt'] == False)]
        wt = ld[ld['is_wt'] == True]

        ax.scatter(regular['pred_centered'], regular['mech_diversity'],
                   s=6, alpha=0.3, c=[color], edgecolors='none', label=locus_name)
        if not var_clusters.empty:
            ax.scatter(var_clusters['pred_centered'], var_clusters['mech_diversity'],
                       s=18, alpha=0.7, c=[color], edgecolors='k', linewidth=0.3, marker='D', zorder=4)
        if not wt.empty:
            ax.scatter(wt['pred_centered'].values, wt['mech_diversity'].values,
                       s=30, c=[color], marker='*', edgecolors='k', linewidth=0.3, zorder=5)

    xlim = max(abs(sdf['pred_centered'].min()), abs(sdf['pred_centered'].max())) * 1.1
    ax.set_xlim(-xlim, xlim)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.set_xlabel('Functional Evolvability (pred. activity - WT)', fontsize=12)
    ax.set_ylabel('Mechanistic Diversity (1 - cos sim to WT)', fontsize=12)
    ax.set_title(f'Inject {source} — Diversity vs Evolvability — All Loci (k={k})', fontsize=13)
    ax.legend(fontsize=6, ncol=3, loc='best', markerscale=1.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(INJECT_RESULTS_DIR, f'{source}_diversity_evolvability_all_loci_k{k}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Summary saved: {out_path}')

    sdf.to_csv(os.path.join(INJECT_RESULTS_DIR, f'{source}_diversity_evolvability_k{k}.csv'), index=False)


def main():
    parser = argparse.ArgumentParser(description='Generate variant injection plots from existing results')
    parser.add_argument('--k', type=int, default=K_DEFAULT)
    parser.add_argument('--sources', nargs='+', default=['gnomad', 'caqtl_eur', 'caqtl_afr'],
                        choices=['gnomad', 'caqtl_eur', 'caqtl_afr'])
    args = parser.parse_args()

    os.makedirs(INJECT_RESULTS_DIR, exist_ok=True)

    for source in args.sources:
        print(f'\n{"="*50}')
        print(f'Processing {source} (k={args.k})')
        print(f'{"="*50}')

        df = compute_causality_for_source(source, args.k)
        if df.empty:
            print(f'  No results for {source}')
            continue

        if source == 'gnomad':
            plot_gnomad(df, args.k)
        else:
            plot_caqtl(df, args.k, source)

    # Diversity vs evolvability plots for GnomAD injected runs
    if 'gnomad' in args.sources:
        print(f'\n{"="*50}')
        print(f'Diversity vs Evolvability plots (gnomad, k={args.k})')
        print(f'{"="*50}')
        plot_inject_diversity_evolvability(args.k, source='gnomad')

    print('\nDone.')


if __name__ == '__main__':
    main()
