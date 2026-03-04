#!/usr/bin/env python
"""
Results analysis for SEAM eQTL LCL pipeline.
Generates CSM mismatch heatmaps, cluster prediction plots, and cluster attribution logos.

Adapted from gnomAD pipeline — references 'eQTL_LCL' source instead of 'gnomAD'.
"""
import os
import sys
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = '/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis/SEAM_CLIPNET/LCL_variants_analysis'
REVO_DIR = '/grid/wsbs/home_norepl/pmantill/SEAM_revisions/SEAM_revisions/seam+REVO_exploration'
DATA_DIR = os.path.join(BASE_DIR, 'data')
ATTRIBUTION_DIR = os.path.join(BASE_DIR, 'DeepSHAP_maps')
RESULTS_DIR = os.path.join(BASE_DIR, 'SEAM_results')
FINALS_DIR = os.path.join(RESULTS_DIR, 'results_finals')
NUM_SEQS = 25000
K = 100

sys.path.insert(0, os.path.join(REVO_DIR, '..', 'seam_repo', 'seam-nn', 'seam'))
from logomaker_batch.batch_logo import BatchLogo

# ── Loci ──────────────────────────────────────────────────────────────
LOCI_DF = pd.DataFrame({
    'name':  ['IRF7', 'HLA-A', 'HLA-B', 'HLA-C', 'HLA-G',
              'HOXA1', 'HOXA13', 'HOXC13', 'B-ACTIN', 'TBP', 'GAPDH',
              'YAP1', 'TAZ', 'PIK3R3', 'MYC', 'TNF', 'BCL2',
              'KRAS', 'EGFR', 'ERBB2', 'PIK3CA', 'CCND1', 'BRAF', 'VEGFA', 'MDM2'],
    'tss':   [616000, 29942532, 31357179, 31272092, 29827825,
              27095025, 27209044, 53976181, 5530601, 170554302, 6534512,
              102110461, 149658025, 46132640, 127736231, 31575565, 63319769,
              25250929, 55019017, 39700064, 179148357, 69641156, 140924929, 43770211, 68808177],
})
LOCI_DF['start'] = LOCI_DF['tss'] - 500
LOCI = LOCI_DF['name'].tolist()

VARIANT_SOURCE = 'eQTL_LCL'


# ── Helper: load eQTL LCL cluster mapping ─────────────────────────────
def load_variant_clusters(labels, meta_path):
    """Return dict: cluster_id -> count of eQTL LCL variants."""
    cluster_counts = {}
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path)
        var_rows = meta[meta['source'] == VARIANT_SOURCE]
        for _, vrow in var_rows.iterrows():
            seq_idx = int(vrow['seq_idx'])
            if seq_idx < len(labels):
                c = labels[seq_idx]
                cluster_counts[c] = cluster_counts.get(c, 0) + 1
    return cluster_counts


# =========================================================================
# 1. CSM Mismatch Heatmap
# =========================================================================
def plot_csm_mismatch(name, k=K):
    locus_dir = os.path.join(RESULTS_DIR, name, f'k{k}')
    csm_path = os.path.join(locus_dir, 'csm_matrix.npy')
    labels_path = os.path.join(locus_dir, 'cluster_labels.npy')
    meta_path = os.path.join(DATA_DIR, f'x_mut_{name}_{NUM_SEQS}_metadata.csv')

    if not os.path.exists(csm_path):
        print(f'{name}: csm_matrix.npy not found, skipping')
        return

    csm = np.load(csm_path)
    labels = np.load(labels_path)
    cluster_ids = sorted(np.unique(labels))

    wt_cluster = labels[0]
    wt_idx = cluster_ids.index(wt_cluster)
    variant_cluster_counts = load_variant_clusters(labels, meta_path)
    cluster_sizes = {c: int(np.sum(labels == c)) for c in cluster_ids}

    fig_height = max(6, len(cluster_ids) * 0.18)
    fig, ax = plt.subplots(figsize=(20, fig_height))
    im = ax.pcolormesh(csm, cmap='viridis', vmin=0, vmax=100)

    ax.axhline(y=wt_idx + 0.5, color='red', linestyle=':', linewidth=2, alpha=0.9)
    for c in variant_cluster_counts:
        ci = cluster_ids.index(c)
        ax.axhline(y=ci + 0.5, color='orange', linestyle=':', linewidth=1.0, alpha=0.7)

    ytick_labels = []
    for ci, c in enumerate(cluster_ids):
        lbl = f'C{c} (n={cluster_sizes[c]})'
        if c == wt_cluster:
            lbl += '  *WT*'
        if c in variant_cluster_counts:
            lbl += f'  *eQTL variants {variant_cluster_counts[c]}*'
        ytick_labels.append(lbl)

    ax.set_yticks(np.arange(len(cluster_ids)) + 0.5)
    ax.set_yticklabels(ytick_labels, fontsize=5)
    ax.set_xlabel('Position', fontsize=11)
    ax.set_ylabel('Cluster', fontsize=11)
    ax.set_title(f'{name} — CSM % Mismatch from WT (k={k})', fontsize=13)
    plt.colorbar(im, ax=ax, label='% Mismatch', shrink=0.6)
    plt.tight_layout()
    out_path = os.path.join(FINALS_DIR, f'{name}_csm_mismatch_k{k}.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()

    n_variants = sum(variant_cluster_counts.values())
    print(f'{name}: saved {out_path}')
    print(f'  WT cluster={wt_cluster}, '
          f'{n_variants} eQTL LCL variants across {len(variant_cluster_counts)} clusters')


# =========================================================================
# 2. Cluster Predictions (ordered by mean prediction)
# =========================================================================
def plot_cluster_preds(name, k=K):
    locus_dir = os.path.join(RESULTS_DIR, name, f'k{k}')
    labels_path = os.path.join(locus_dir, 'cluster_labels.npy')
    preds_path = os.path.join(ATTRIBUTION_DIR, f'preds_quantity_{name}_{NUM_SEQS}.npy')
    meta_path = os.path.join(DATA_DIR, f'x_mut_{name}_{NUM_SEQS}_metadata.csv')

    if not os.path.exists(labels_path) or not os.path.exists(preds_path):
        return

    labels = np.load(labels_path)
    preds = np.load(preds_path)
    cluster_ids = sorted(np.unique(labels))

    wt_cluster = labels[0]
    wt_pred = float(preds[0])
    variant_cluster_counts = load_variant_clusters(labels, meta_path)
    cluster_sizes = {c: int(np.sum(labels == c)) for c in cluster_ids}

    cluster_means = {}
    for c in cluster_ids:
        cluster_means[c] = float(preds[labels == c].mean())
    sorted_clusters = sorted(cluster_ids, key=lambda c: cluster_means[c])

    box_data = [preds[labels == c] for c in sorted_clusters]

    fig_height = max(6, len(sorted_clusters) * 0.18)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    bp = ax.boxplot(box_data, patch_artist=True, showfliers=False, vert=False, widths=0.6)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.6)

    ax.axvline(x=wt_pred, color='red', linestyle='--', linewidth=1.5, alpha=0.8,
               label=f'WT pred = {wt_pred:.2f}')

    wt_y = sorted_clusters.index(wt_cluster) + 1
    ax.axhline(y=wt_y, color='red', linestyle=':', linewidth=1.5, alpha=0.7)

    for c in variant_cluster_counts:
        c_y = sorted_clusters.index(c) + 1
        ax.axhline(y=c_y, color='orange', linestyle=':', linewidth=0.8, alpha=0.6)

    ytick_labels = []
    for c in sorted_clusters:
        lbl = f'C{c} (n={cluster_sizes[c]})'
        if c == wt_cluster:
            lbl += ' *WT*'
        if c in variant_cluster_counts:
            lbl += f' *eQTL:{variant_cluster_counts[c]}*'
        ytick_labels.append(lbl)

    ax.set_yticklabels(ytick_labels, fontsize=5)
    ax.set_xlabel('DNN Quantity Prediction', fontsize=11)
    ax.set_title(f'{name} — Cluster Predictions (k={k}, sorted by mean)', fontsize=13)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(locus_dir, f'cluster_preds_sorted_k{k}.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'{name}: saved {out_path}')


# =========================================================================
# 3. Cluster Attribution Logos (WT + top-2 most different eQTL clusters)
# =========================================================================
def plot_cluster_logos(name, k=K):
    locus_dir = os.path.join(RESULTS_DIR, name, f'k{k}')
    labels_path = os.path.join(locus_dir, 'cluster_labels.npy')
    maps_path = os.path.join(ATTRIBUTION_DIR, f'maps_quantity_{name}_{NUM_SEQS}.npy')
    preds_path = os.path.join(ATTRIBUTION_DIR, f'preds_quantity_{name}_{NUM_SEQS}.npy')
    meta_path = os.path.join(DATA_DIR, f'x_mut_{name}_{NUM_SEQS}_metadata.csv')
    seqs_path = os.path.join(DATA_DIR, f'x_mut_{name}_{NUM_SEQS}.npy')

    if not os.path.exists(labels_path) or not os.path.exists(maps_path):
        print(f'{name}: missing data, skipping logos')
        return

    labels = np.load(labels_path)
    maps = np.load(maps_path)
    if maps.ndim == 3 and maps.shape[0] == 4 and maps.shape[2] != 4:
        maps = maps.transpose(1, 2, 0)

    seqs = np.load(seqs_path) if os.path.exists(seqs_path) else None
    if seqs is not None:
        if seqs.ndim == 3 and seqs.shape[0] == 4 and seqs.shape[2] != 4:
            seqs = seqs.transpose(1, 2, 0)
        maps = maps * (seqs > 0).astype(np.float32)

    preds = np.load(preds_path) if os.path.exists(preds_path) else None

    wt_cluster = labels[0]
    locus_start = int(LOCI_DF.loc[LOCI_DF['name'] == name, 'start'].values[0])

    # Load eQTL variants per cluster
    variant_by_cluster = {}
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path)
        var_rows = meta[meta['source'] == VARIANT_SOURCE]
        for _, vrow in var_rows.iterrows():
            seq_idx = int(vrow['seq_idx'])
            if seq_idx < len(labels):
                c = labels[seq_idx]
                rel_pos = int(vrow['pos']) - locus_start
                pred_score = float(vrow['prediction']) if pd.notna(vrow.get('prediction')) else 0.0
                variant_by_cluster.setdefault(c, []).append({
                    'seq_idx': seq_idx,
                    'ref': vrow['ref'], 'alt': vrow['alt'],
                    'rel_pos': rel_pos, 'prediction': pred_score,
                })

    non_wt_variants = {c: v for c, v in variant_by_cluster.items() if c != wt_cluster}
    if len(non_wt_variants) < 1:
        print(f'{name}: no non-WT eQTL clusters, skipping logos')
        del maps
        gc.collect()
        return

    wt_avg_flat = maps[labels == wt_cluster].mean(axis=0).flatten()
    most_diff_c = max(non_wt_variants.keys(),
                      key=lambda c: np.linalg.norm(wt_avg_flat - maps[labels == c].mean(axis=0).flatten()))

    if preds is not None:
        cluster_mean_pred = {c: float(preds[labels == c].mean()) for c in non_wt_variants}
        highest_act_c = max(cluster_mean_pred, key=cluster_mean_pred.get)
        lowest_act_c = min(cluster_mean_pred, key=cluster_mean_pred.get)
    else:
        highest_act_c = most_diff_c
        lowest_act_c = most_diff_c

    panel_candidates = [('1. Most Different from WT', most_diff_c),
                        ('2. Highest Activity', highest_act_c),
                        ('3. Lowest Activity', lowest_act_c)]
    seen = {}
    panels = [('0. WT Cluster', wt_cluster)]
    for label, c in panel_candidates:
        if c in seen:
            idx = seen[c]
            old_label, old_c = panels[idx]
            panels[idx] = (old_label + ' + ' + label, old_c)
        else:
            seen[c] = len(panels)
            panels.append((label, c))

    n_panels = len(panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(50, 4 * n_panels))
    if n_panels == 1:
        axes = [axes]

    for i, (panel_label, cluster_id) in enumerate(panels):
        members = labels == cluster_id
        n_members = int(np.sum(members))

        cluster_avg = maps[members].mean(axis=0)

        logo = BatchLogo(cluster_avg[np.newaxis, :, :],
                         figsize=[50, 3], show_progress=False)
        logo.process_all()
        logo.draw_single(0, ax=axes[i], fixed_ylim=False, border=True)

        n_eqtl = len(variant_by_cluster.get(cluster_id, []))
        title = f'{panel_label} C{cluster_id} (n={n_members}'
        if n_eqtl > 0:
            title += f', {n_eqtl} eQTL LCL variants'
        title += ')'

        if preds is not None:
            c_preds = preds[members]
            title += (f'  |  Activity: mean={c_preds.mean():.2f}, '
                      f'max={c_preds.max():.2f}, min={c_preds.min():.2f}')

        axes[i].set_title(title, fontsize=11, fontweight='bold')
        axes[i].set_ylabel('Attribution', fontsize=9)

        variants = variant_by_cluster.get(cluster_id, [])
        if variants:
            pos_variants = {}
            for v in variants:
                rp = v['rel_pos']
                if 0 <= rp < 1000:
                    mut_str = f"{v['ref']}>{v['alt']}"
                    pos_variants.setdefault(rp, []).append((mut_str, v['prediction']))

            for pos in pos_variants:
                axes[i].axvline(x=pos, color='darkorange', linestyle='-',
                               linewidth=0.5, alpha=0.4, ymin=0.85, ymax=1.0)

            sorted_pos = sorted(pos_variants.keys())
            mut_strings = []
            for pos in sorted_pos:
                seen_m = {}
                for m, pred_score in pos_variants[pos]:
                    if m not in seen_m or abs(pred_score) > abs(seen_m[m]):
                        seen_m[m] = pred_score
                for m in sorted(seen_m):
                    pred_score = seen_m[m]
                    mut_strings.append(f'{pos}:{m} (pred={pred_score:.3f})')

            if len(mut_strings) <= 15:
                summary = '\n'.join(mut_strings)
            else:
                summary = '\n'.join(mut_strings[:12]) + \
                          f'\n... +{len(mut_strings)-12} more'

            axes[i].text(1.005, 0.5, summary, transform=axes[i].transAxes,
                        fontsize=8, fontfamily='monospace', va='center',
                        color='darkorange', fontweight='bold')

    fig.suptitle(f'{name} — Cluster Attribution Logos (k={k})', fontsize=14,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    out_path = os.path.join(locus_dir, f'cluster_logos_eqtl_k{k}.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f'{name}: saved {out_path}')
    del maps
    gc.collect()


# =========================================================================
# 4. Summary: eQTL Alignment vs Prediction Score (all loci)
# =========================================================================
def plot_eqtl_alignment_summary(k=K):
    """For each cluster containing eQTL variants across all loci:
    - Binary mask: 1 at eQTL variant positions, 0 elsewhere
    - eQTL alignment = sum(CSM_mismatch_row * mask) / cluster_size
    - Mean AlphaGenome prediction score of eQTL variants in that cluster
    Scatter plot alignment vs mean prediction, colored by locus category.
    """
    from scipy import stats

    records = []

    cat_map = {
        'IRF7': 'Oncogene', 'HLA-A': 'HLA', 'HLA-B': 'HLA', 'HLA-C': 'HLA', 'HLA-G': 'HLA',
        'HOXA1': 'Hox', 'HOXA13': 'Hox', 'HOXC13': 'Hox',
        'B-ACTIN': 'Housekeeping', 'TBP': 'Housekeeping', 'GAPDH': 'Housekeeping',
        'YAP1': 'Oncogene', 'TAZ': 'Oncogene', 'PIK3R3': 'Oncogene',
        'MYC': 'Oncogene', 'TNF': 'Oncogene', 'BCL2': 'Oncogene',
        'KRAS': 'Oncogene', 'EGFR': 'Oncogene', 'ERBB2': 'Oncogene',
        'PIK3CA': 'Oncogene', 'CCND1': 'Oncogene', 'BRAF': 'Oncogene',
        'VEGFA': 'Oncogene', 'MDM2': 'Oncogene',
    }

    for name in LOCI:
        locus_dir = os.path.join(RESULTS_DIR, name, f'k{k}')
        csm_path = os.path.join(locus_dir, 'csm_matrix.npy')
        labels_path = os.path.join(locus_dir, 'cluster_labels.npy')
        meta_path = os.path.join(DATA_DIR, f'x_mut_{name}_{NUM_SEQS}_metadata.csv')

        if not os.path.exists(csm_path) or not os.path.exists(meta_path):
            continue

        csm = np.load(csm_path)
        labels = np.load(labels_path)
        cluster_ids = sorted(np.unique(labels))
        locus_start = int(LOCI_DF.loc[LOCI_DF['name'] == name, 'start'].values[0])
        category = cat_map.get(name, 'Other')

        meta = pd.read_csv(meta_path)
        var_rows = meta[meta['source'] == VARIANT_SOURCE]

        cluster_variants = {}
        for _, vrow in var_rows.iterrows():
            seq_idx = int(vrow['seq_idx'])
            if seq_idx >= len(labels):
                continue
            c = labels[seq_idx]
            rel_pos = int(vrow['pos']) - locus_start
            pred_score = float(vrow['prediction']) if pd.notna(vrow.get('prediction')) else 0.0
            cluster_variants.setdefault(c, []).append((rel_pos, pred_score))

        wt_cluster = labels[0]
        for c, var_list in cluster_variants.items():
            if c == wt_cluster:
                continue
            ci = cluster_ids.index(c)
            csm_row = csm[ci]

            mask = np.zeros(1000, dtype=np.float32)
            pred_scores = []
            for rel_pos, pred_score in var_list:
                if 0 <= rel_pos < 1000:
                    mask[rel_pos] = 1.0
                    pred_scores.append(pred_score)

            if mask.sum() == 0 or len(pred_scores) == 0:
                continue

            eqtl_alignment_raw = float(np.sum(csm_row * mask))
            n_seqs_in_cluster = int(np.sum(labels == c))
            eqtl_alignment = eqtl_alignment_raw / n_seqs_in_cluster
            mean_pred = float(np.mean(pred_scores))

            records.append({
                'locus': name,
                'cluster': int(c),
                'eqtl_alignment': eqtl_alignment,
                'eqtl_alignment_raw': eqtl_alignment_raw,
                'n_seqs': n_seqs_in_cluster,
                'mean_prediction': mean_pred,
                'category': category,
                'n_variants': len(var_list),
            })

    if not records:
        print('No eQTL alignment data to plot')
        return

    df = pd.DataFrame(records)

    cat_colors = {
        'Oncogene': '#e74c3c',
        'HLA': '#3498db',
        'Hox': '#2ecc71',
        'Housekeeping': '#9b59b6',
    }
    cat_order = ['Oncogene', 'HLA', 'Hox', 'Housekeeping']

    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3],
                           hspace=0.05, wspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0])
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    plt.setp(ax_histx.get_xticklabels(), visible=False)
    plt.setp(ax_histy.get_yticklabels(), visible=False)

    from scipy.stats import gaussian_kde
    x_vals = df['mean_prediction'].values
    x_grid = np.linspace(x_vals.min() - 0.1, x_vals.max() + 0.1, 200)
    y_grid = np.linspace(df['eqtl_alignment'].min() - 0.1,
                         df['eqtl_alignment'].max() * 1.1, 200)

    for cat in cat_order:
        sub = df[df['category'] == cat]
        if len(sub) < 2:
            continue
        color = cat_colors[cat]

        ax.scatter(sub['mean_prediction'], sub['eqtl_alignment'],
                   c=color, label=cat, alpha=0.5, s=20, edgecolors='none')

        kde_x = gaussian_kde(sub['mean_prediction'].values)
        ax_histx.fill_between(x_grid, kde_x(x_grid), alpha=0.25, color=color)
        ax_histx.plot(x_grid, kde_x(x_grid), color=color, linewidth=2)

        kde_y = gaussian_kde(sub['eqtl_alignment'].values)
        ax_histy.fill_betweenx(y_grid, kde_y(y_grid), alpha=0.25, color=color)
        ax_histy.plot(kde_y(y_grid), y_grid, color=color, linewidth=2)

    ax.set_xlabel('Mean AlphaGenome Prediction Score (eQTL LCL variants)', fontsize=12)
    ax.set_ylabel('eQTL Alignment Score\n(sum of CSM % mismatch at variant positions / cluster size)', fontsize=12)

    ax_histx.spines['top'].set_visible(False)
    ax_histx.spines['right'].set_visible(False)
    ax_histx.spines['left'].set_visible(False)
    ax_histx.tick_params(left=False, labelleft=False)
    ax_histy.spines['top'].set_visible(False)
    ax_histy.spines['right'].set_visible(False)
    ax_histy.spines['bottom'].set_visible(False)
    ax_histy.tick_params(bottom=False, labelbottom=False)

    valid = df[['mean_prediction', 'eqtl_alignment']].dropna()
    if len(valid) > 2:
        r_pearson, p_pearson = stats.pearsonr(valid['mean_prediction'], valid['eqtl_alignment'])
        r_spearman, p_spearman = stats.spearmanr(valid['mean_prediction'], valid['eqtl_alignment'])
        ax_histx.set_title(
            f'eQTL Alignment vs AlphaGenome Prediction (all 25 loci, k={k})\n'
            f'Pearson r={r_pearson:.3f} (p={p_pearson:.2e}), '
            f'Spearman r={r_spearman:.3f} (p={p_spearman:.2e})\n'
            f'n={len(df)} clusters with eQTL LCL variants',
            fontsize=13)
    else:
        ax_histx.set_title(f'eQTL Alignment vs AlphaGenome Prediction (all 25 loci, k={k})',
                     fontsize=13)

    ax.legend(fontsize=10, markerscale=2)
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(FINALS_DIR, f'eqtl_alignment_vs_prediction_k{k}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    csv_path = os.path.join(FINALS_DIR, f'eqtl_alignment_vs_prediction_k{k}.csv')
    df.to_csv(csv_path, index=False)

    print(f'Summary: saved {out_path}')
    print(f'  {len(df)} total cluster-points across {df["locus"].nunique()} loci')
    print(f'  Data: {csv_path}')


# =========================================================================
# Main
# =========================================================================
def main():
    os.makedirs(FINALS_DIR, exist_ok=True)

    for name in LOCI:
        plot_csm_mismatch(name)
        plot_cluster_preds(name)
        plot_cluster_logos(name)

    plot_eqtl_alignment_summary()

    print(f'\nAll plots saved to {FINALS_DIR}')


if __name__ == '__main__':
    main()
