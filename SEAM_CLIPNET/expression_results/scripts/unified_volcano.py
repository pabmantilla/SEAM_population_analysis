#!/usr/bin/env python
"""
Unified figure: Mechanistic Diversity vs Predicted Expression (Transcriptional Initiation)
Combines gnomAD variants (25 loci), eQTL variants (HLA-A, HLA-C), and causal fine-mapped variants.

Outputs to: expression_results/all_variants/
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
GNOMAD_BASE = '/grid/wsbs/home_norepl/pmantill/SEAM_revisions/SEAM_revisions/seam+REVO_exploration/genomAD_compare/variants_test'
EQTL_BASE = '/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis/SEAM_CLIPNET/LCL_variants_analysis'
OUT_DIR = '/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis/SEAM_CLIPNET/expression_results/all_variants'
CAUSAL_FEATHER = '/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis/variant_data/Alphagenome_data/eqtl_variants/eqtl_variant_catalogue_causality_gene_balanced_human_predictions.feather'
NUM_SEQS = 25000
K = 100

# ── Loci ───────────────────────────────────────────────────────────────
GNOMAD_LOCI_DF = pd.DataFrame({
    'name':  ['IRF7', 'HLA-A', 'HLA-B', 'HLA-C', 'HLA-G',
              'HOXA1', 'HOXA13', 'HOXC13', 'B-ACTIN', 'TBP', 'GAPDH',
              'YAP1', 'TAZ', 'PIK3R3', 'MYC', 'TNF', 'BCL2',
              'KRAS', 'EGFR', 'ERBB2', 'PIK3CA', 'CCND1', 'BRAF', 'VEGFA', 'MDM2'],
    'tss':   [616000, 29942532, 31357179, 31272092, 29827825,
              27095025, 27209044, 53976181, 5530601, 170554302, 6534512,
              102110461, 149658025, 46132640, 127736231, 31575565, 63319769,
              25250929, 55019017, 39700064, 179148357, 69641156, 140924929, 43770211, 68808177],
    'category': ['Oncogene', 'HLA', 'HLA', 'HLA', 'HLA',
                 'Hox', 'Hox', 'Hox', 'Housekeeping', 'Housekeeping', 'Housekeeping',
                 'Oncogene', 'Oncogene', 'Oncogene', 'Oncogene', 'Oncogene', 'Oncogene',
                 'Oncogene', 'Oncogene', 'Oncogene', 'Oncogene', 'Oncogene', 'Oncogene',
                 'Oncogene', 'Oncogene'],
})
GNOMAD_LOCI = GNOMAD_LOCI_DF['name'].tolist()

EQTL_LOCI = ['HLA-A', 'HLA-C']


# =========================================================================
# 1. Compute per-sequence cosine distances for gnomAD loci
# =========================================================================
def compute_gnomad_data(cache_path=None):
    """Compute activity + cosine distance for gnomAD variant sequences across all 25 loci.
    Returns DataFrame with: locus, seq_idx, activity, cos_dist, variant_id, AF, category, source.
    Also returns random mutagenesis background sample for context.
    """
    if cache_path and os.path.exists(cache_path):
        print(f'Loading cached gnomAD data from {cache_path}')
        return pd.read_csv(cache_path)

    all_records = []

    for _, row in GNOMAD_LOCI_DF.iterrows():
        name = row['name']
        category = row['category']
        print(f'Processing gnomAD locus: {name}')

        maps_path = os.path.join(GNOMAD_BASE, 'DeepSHAP_maps', f'maps_quantity_{name}_{NUM_SEQS}.npy')
        preds_path = os.path.join(GNOMAD_BASE, 'DeepSHAP_maps', f'preds_quantity_{name}_{NUM_SEQS}.npy')
        labels_path = os.path.join(GNOMAD_BASE, 'SEAM_results', name, f'k{K}', 'cluster_labels.npy')
        meta_path = os.path.join(GNOMAD_BASE, 'data', f'x_mut_{name}_{NUM_SEQS}_metadata.csv')
        seqs_path = os.path.join(GNOMAD_BASE, 'data', f'x_mut_{name}_{NUM_SEQS}.npy')

        if not os.path.exists(maps_path) or not os.path.exists(preds_path):
            print(f'  Skipping {name}: missing maps/preds')
            continue
        if not os.path.exists(labels_path):
            print(f'  Skipping {name}: missing cluster labels')
            continue

        # Load data
        maps = np.load(maps_path)
        if maps.ndim == 3 and maps.shape[0] == 4 and maps.shape[2] != 4:
            maps = maps.transpose(1, 2, 0)

        seqs = np.load(seqs_path) if os.path.exists(seqs_path) else None
        if seqs is not None:
            if seqs.ndim == 3 and seqs.shape[0] == 4 and seqs.shape[2] != 4:
                seqs = seqs.transpose(1, 2, 0)
            maps = maps * (seqs > 0).astype(np.float32)
            del seqs

        preds = np.load(preds_path)
        labels = np.load(labels_path)
        meta = pd.read_csv(meta_path)

        # WT cluster average
        wt_cluster = labels[0]
        wt_avg = maps[labels == wt_cluster].mean(axis=0).reshape(1, -1)
        wt_norm = np.linalg.norm(wt_avg)

        # Vectorized cosine distance for ALL sequences
        n = maps.shape[0]
        maps_flat = maps.reshape(n, -1)
        norms = np.linalg.norm(maps_flat, axis=1)
        dots = (maps_flat @ wt_avg.T).squeeze()
        denom = norms * wt_norm
        denom[denom == 0] = 1e-10
        cos_dists = 1.0 - dots / denom

        del maps, maps_flat
        gc.collect()

        # gnomAD variant records
        gnomad_rows = meta[meta['source'] == 'gnomAD']
        for _, vrow in gnomad_rows.iterrows():
            si = int(vrow['seq_idx'])
            if si >= n:
                continue
            all_records.append({
                'locus': name,
                'seq_idx': si,
                'activity': float(preds[si]),
                'cos_dist': float(cos_dists[si]),
                'variant_id': vrow.get('variant_id', ''),
                'AF': float(vrow['AF']) if pd.notna(vrow.get('AF')) else 0.0,
                'category': category,
                'source': 'gnomAD',
            })

        # WT record
        all_records.append({
            'locus': name,
            'seq_idx': 0,
            'activity': float(preds[0]),
            'cos_dist': float(cos_dists[0]),
            'variant_id': 'WT',
            'AF': np.nan,
            'category': category,
            'source': 'WT',
        })

        # Random mutagenesis subsample (500 per locus for background)
        rm_rows = meta[meta['source'] == 'random_mutagenesis']
        rm_indices = rm_rows['seq_idx'].values.astype(int)
        rm_indices = rm_indices[rm_indices < n]
        if len(rm_indices) > 500:
            rng = np.random.default_rng(42)
            rm_indices = rng.choice(rm_indices, size=500, replace=False)
        for si in rm_indices:
            all_records.append({
                'locus': name,
                'seq_idx': int(si),
                'activity': float(preds[si]),
                'cos_dist': float(cos_dists[si]),
                'variant_id': '',
                'AF': np.nan,
                'category': category,
                'source': 'random_mutagenesis',
            })

        del preds, labels, cos_dists
        gc.collect()
        print(f'  {name}: {len(gnomad_rows)} gnomAD variants processed')

    df = pd.DataFrame(all_records)
    if cache_path:
        df.to_csv(cache_path, index=False)
        print(f'Cached gnomAD data to {cache_path}')
    return df


# =========================================================================
# 2. Load eQTL data from existing volcano CSVs
# =========================================================================
def load_eqtl_data():
    """Load per-sequence eQTL volcano data for HLA-A and HLA-C.
    Returns DataFrame with: locus, seq_idx, activity, cos_dist, source, variant_id.
    """
    # Load causal variant IDs
    causal_df = pd.read_feather(CAUSAL_FEATHER)
    causal_lcl = causal_df[causal_df['tissue'] == 'Cells_EBV-transformed_lymphocytes']
    causal_ids = set(causal_lcl['variant_id'].values)
    del causal_df, causal_lcl

    all_records = []
    for name in EQTL_LOCI:
        volcano_csv = os.path.join(EQTL_BASE, 'SEAM_results', 'results_finals',
                                   f'{name}_volcano_data_k{K}.csv')
        meta_path = os.path.join(EQTL_BASE, 'data', f'x_mut_{name}_{NUM_SEQS}_metadata.csv')

        if not os.path.exists(volcano_csv):
            print(f'  {name}: volcano CSV not found, skipping')
            continue

        vdf = pd.read_csv(volcano_csv)
        meta = pd.read_csv(meta_path)

        # Build variant_id lookup
        eqtl_meta = meta[meta['source'] == 'eQTL_LCL']
        idx_to_vid = dict(zip(eqtl_meta['seq_idx'].astype(int), eqtl_meta['variant_id']))

        for _, vrow in vdf.iterrows():
            si = int(vrow['seq_idx'])
            src = vrow['source']
            vid = idx_to_vid.get(si, '')

            # Determine refined source
            if src == 'eQTL_LCL':
                refined_source = 'eQTL_causal' if vid in causal_ids else 'eQTL'
            elif src == 'WT':
                refined_source = 'WT'
            else:
                refined_source = 'random_mutagenesis'

            all_records.append({
                'locus': name,
                'seq_idx': si,
                'activity': float(vrow['activity']),
                'cos_dist': float(vrow['cos_dist_from_wt']),
                'variant_id': vid,
                'AF': np.nan,
                'category': 'HLA',
                'source': refined_source,
            })

        print(f'  {name}: {len(vdf)} sequences loaded from volcano CSV')

    return pd.DataFrame(all_records)


# =========================================================================
# 3. Unified Volcano Plot
# =========================================================================
def _af_to_size(af_series, s_min=3, s_max=40):
    """Map allele frequency to point size. NaN/0 get s_min."""
    af = af_series.fillna(0).values.astype(float)
    af = np.clip(af, 0, 1)
    # Log-scale mapping: AF spans many orders of magnitude
    log_af = np.where(af > 0, np.log10(af), -7)  # floor at 1e-7
    log_min, log_max = -7, 0
    norm = (log_af - log_min) / (log_max - log_min)
    norm = np.clip(norm, 0, 1)
    return s_min + norm * (s_max - s_min)


def _scatter_panel(ax, ax_histx, ax_histy, all_df, x_col, xlabel):
    """Draw one scatter panel with per-category coloring and marginal KDEs."""
    from scipy.stats import gaussian_kde

    CAT_COLORS = {
        'Oncogene': '#e74c3c',
        'HLA': '#3498db',
        'Hox': '#9b59b6',
        'Housekeeping': '#2ecc71',
    }
    CAT_ORDER = ['Oncogene', 'HLA', 'Hox', 'Housekeeping']

    # --- Plot each gnomAD category: background (random mut) then variants ---
    cat_counts = {}
    for cat in CAT_ORDER:
        color = CAT_COLORS[cat]
        cat_data = all_df[all_df['category'] == cat]
        if len(cat_data) == 0:
            continue

        # Random mutagenesis background for this category
        rm = cat_data[cat_data['source'] == 'random_mutagenesis']
        ax.scatter(rm[x_col], rm['cos_dist'],
                   c=color, s=2, alpha=0.05, rasterized=True, zorder=1)

        # gnomAD variants (circles), sized by AF
        variants = cat_data[cat_data['source'] == 'gnomAD']
        if len(variants) > 0:
            sizes = _af_to_size(variants['AF'])
            ax.scatter(variants[x_col], variants['cos_dist'],
                       c=color, s=sizes, alpha=0.5, edgecolors='none',
                       zorder=5)

        cat_counts[cat] = len(variants)
        ax.scatter([], [], c=color, s=12, label=f'{cat} (n={len(variants):,})')

    # --- eQTL variants: HLA-blue small diamonds on top ---
    eqtl_all = all_df[all_df['source'].isin(['eQTL', 'eQTL_causal'])]
    if len(eqtl_all) > 0:
        ax.scatter(eqtl_all[x_col], eqtl_all['cos_dist'],
                   c='#3498db', s=25, marker='D', edgecolors='black',
                   linewidths=0.4, alpha=0.8, zorder=15)
        ax.scatter([], [], c='#3498db', s=25, marker='D', edgecolors='black',
                   linewidths=0.4, label=f'eQTL variant (n={len(eqtl_all)})')

    ax.set_yscale('log')
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.4f}' if v < 0.01 else f'{v:.3f}' if v < 0.1 else f'{v:.2f}'))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))

    # WT reference line
    if x_col == 'delta_activity':
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1.2, alpha=0.5,
                   label='WT ($\Delta$=0)')


    # --- Top marginal KDE: per-category, each normalized to its own peak ---
    plt.setp(ax_histx.get_xticklabels(), visible=False)
    for cat in CAT_ORDER:
        color = CAT_COLORS[cat]
        cat_data = all_df[(all_df['category'] == cat) &
                          (~all_df['source'].isin(['WT']))]
        if len(cat_data) < 3:
            continue
        vals = cat_data[x_col].dropna().values
        try:
            kde = gaussian_kde(vals)
            x_grid = np.linspace(vals.min() - 1, vals.max() + 1, 300)
            density = kde(x_grid)
            # Normalize each KDE to peak=1 so all are visible
            peak = density.max()
            if peak > 0:
                density = density / peak
            ax_histx.plot(x_grid, density, color=color, linewidth=2.5)
            ax_histx.fill_between(x_grid, density, alpha=0.1, color=color)
        except Exception:
            pass
    for spine in ['top', 'right', 'left']:
        ax_histx.spines[spine].set_visible(False)
    ax_histx.tick_params(left=False, labelleft=False)
    ax_histx.set_ylabel('Density\n(normalized)', fontsize=8)

    # --- Right marginal KDE (log-space): per-category, peak-normalized ---
    plt.setp(ax_histy.get_yticklabels(), visible=False)
    for cat in CAT_ORDER:
        color = CAT_COLORS[cat]
        cat_data = all_df[(all_df['category'] == cat) &
                          (~all_df['source'].isin(['WT']))]
        if len(cat_data) < 3:
            continue
        vals = cat_data['cos_dist'].dropna().values
        vals = vals[vals > 0]
        if len(vals) < 3:
            continue
        try:
            log_vals = np.log10(vals)
            kde = gaussian_kde(log_vals)
            y_grid_log = np.linspace(log_vals.min() - 0.3, log_vals.max() + 0.3, 300)
            y_grid = 10 ** y_grid_log
            density = kde(y_grid_log)
            peak = density.max()
            if peak > 0:
                density = density / peak
            ax_histy.plot(density, y_grid, color=color, linewidth=2.5)
            ax_histy.fill_betweenx(y_grid, density, alpha=0.1, color=color)
        except Exception:
            pass
    for spine in ['top', 'right', 'bottom']:
        ax_histy.spines[spine].set_visible(False)
    ax_histy.tick_params(bottom=False, labelbottom=False)

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel('Mechanistic Diversity\n(1 - Cosine Similarity from WT DeepSHAP)', fontsize=13)
    ax.grid(True, alpha=0.2)

    return cat_counts


def plot_unified_volcano(gnomad_df, eqtl_df):
    """Create two side-by-side panels: predicted activity (left) and delta from WT (right)."""

    gnomad_df = gnomad_df.copy()
    eqtl_df = eqtl_df.copy()
    # Only keep actual eQTL variant sequences, color as HLA
    eqtl_df = eqtl_df[eqtl_df['source'].isin(['eQTL', 'eQTL_causal'])].copy()
    eqtl_df['category'] = 'HLA'

    # Compute delta activity from per-locus WT
    wt_activity_by_locus = {}
    for _, row in gnomad_df[gnomad_df['source'] == 'WT'].iterrows():
        wt_activity_by_locus[row['locus']] = row['activity']
    for _, row in eqtl_df[eqtl_df['source'] == 'WT'].iterrows():
        if row['locus'] not in wt_activity_by_locus:
            wt_activity_by_locus[row['locus']] = row['activity']

    gnomad_df['delta_activity'] = gnomad_df.apply(
        lambda r: r['activity'] - wt_activity_by_locus.get(r['locus'], r['activity']), axis=1)
    eqtl_df['delta_activity'] = eqtl_df.apply(
        lambda r: r['activity'] - wt_activity_by_locus.get(r['locus'], r['activity']), axis=1)

    all_df = pd.concat([gnomad_df, eqtl_df], ignore_index=True)

    # 2 panels side by side, each with top + right marginals
    fig = plt.figure(figsize=(36, 14))
    gs = gridspec.GridSpec(2, 4,
                           width_ratios=[4, 1, 4, 1],
                           height_ratios=[1, 4],
                           hspace=0.05, wspace=0.12)

    # Left panel: predicted activity
    ax_L = fig.add_subplot(gs[1, 0])
    ax_Ltop = fig.add_subplot(gs[0, 0], sharex=ax_L)
    ax_Lright = fig.add_subplot(gs[1, 1], sharey=ax_L)

    cat_counts = _scatter_panel(
        ax_L, ax_Ltop, ax_Lright, all_df,
        x_col='activity',
        xlabel='Predicted Expression (Transcriptional Initiation)')
    ax_L.legend(fontsize=8, markerscale=1.5, loc='upper left',
                framealpha=0.9, edgecolor='gray')

    # Right panel: delta from WT
    ax_R = fig.add_subplot(gs[1, 2])
    ax_Rtop = fig.add_subplot(gs[0, 2], sharex=ax_R)
    ax_Rright = fig.add_subplot(gs[1, 3], sharey=ax_R)

    _scatter_panel(
        ax_R, ax_Rtop, ax_Rright, all_df,
        x_col='delta_activity',
        xlabel='$\Delta$ Predicted Expression from WT (Transcriptional Initiation)')
    ax_R.legend(fontsize=8, markerscale=1.5, loc='upper left',
                framealpha=0.9, edgecolor='gray')

    counts_str = '  |  '.join(f'{k} (n={v:,})' for k, v in cat_counts.items())
    fig.suptitle(
        f'Mechanistic Diversity vs Predicted Expression  |  {counts_str}  |  k={K}',
        fontsize=15, fontweight='bold', y=1.01)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f'unified_volcano_k{K}.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved unified volcano: {out_path}')
    return out_path


# =========================================================================
# 4. Per-category summary stats
# =========================================================================
def print_summary(gnomad_df, eqtl_df):
    """Print summary statistics for each variant category."""
    gnomad_vars = gnomad_df[gnomad_df['source'] == 'gnomAD']
    rm = gnomad_df[gnomad_df['source'] == 'random_mutagenesis']
    eqtl_nc = eqtl_df[eqtl_df['source'] == 'eQTL']
    eqtl_c = eqtl_df[eqtl_df['source'] == 'eQTL_causal']

    print('\n' + '='*70)
    print('SUMMARY STATISTICS')
    print('='*70)
    for label, sub in [('gnomAD', gnomad_vars), ('eQTL non-causal', eqtl_nc),
                       ('eQTL causal', eqtl_c), ('Random mutagenesis', rm)]:
        if len(sub) == 0:
            continue
        print(f'\n{label} (n={len(sub):,}):')
        print(f'  Activity:  mean={sub["activity"].mean():.2f}  '
              f'median={sub["activity"].median():.2f}  '
              f'std={sub["activity"].std():.2f}')
        print(f'  Cos dist:  mean={sub["cos_dist"].mean():.4f}  '
              f'median={sub["cos_dist"].median():.4f}  '
              f'std={sub["cos_dist"].std():.4f}')

    # Per-category gnomAD
    print(f'\ngnomAD by category:')
    for cat in ['HLA', 'Oncogene', 'Hox', 'Housekeeping']:
        sub = gnomad_vars[gnomad_vars['category'] == cat]
        if len(sub) == 0:
            continue
        print(f'  {cat} (n={len(sub):,}): cos_dist mean={sub["cos_dist"].mean():.4f}  '
              f'activity mean={sub["activity"].mean():.2f}')
    print('='*70)


# =========================================================================
# Main
# =========================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cache_path = os.path.join(OUT_DIR, 'gnomad_variant_data.csv')

    print('Step 1: Computing gnomAD variant cosine distances...')
    gnomad_df = compute_gnomad_data(cache_path=cache_path)
    print(f'  gnomAD data: {len(gnomad_df)} rows')

    print('\nStep 2: Loading eQTL data...')
    eqtl_df = load_eqtl_data()
    print(f'  eQTL data: {len(eqtl_df)} rows')

    print('\nStep 3: Plotting unified volcano...')
    plot_unified_volcano(gnomad_df, eqtl_df)

    print_summary(gnomad_df, eqtl_df)

    # Save combined data
    combined = pd.concat([
        gnomad_df[gnomad_df['source'] != 'random_mutagenesis'],
        eqtl_df[eqtl_df['source'].isin(['eQTL', 'eQTL_causal'])],
    ], ignore_index=True)
    combined_path = os.path.join(OUT_DIR, 'all_variants_data.csv')
    combined.to_csv(combined_path, index=False)
    print(f'\nSaved combined variant data: {combined_path}')


if __name__ == '__main__':
    main()
