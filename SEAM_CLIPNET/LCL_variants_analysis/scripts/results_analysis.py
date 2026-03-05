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
CAUSAL_FEATHER = '/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis/variant_data/Alphagenome_data/eqtl_variants/eqtl_variant_catalogue_causality_gene_balanced_human_predictions.feather'
NUM_SEQS = 25000
K = 100

sys.path.insert(0, os.path.join(REVO_DIR, '..', 'seam_repo', 'seam-nn', 'seam'))
from logomaker_batch.batch_logo import BatchLogo

# ── Loci ──────────────────────────────────────────────────────────────
LOCI_DF = pd.DataFrame({
    'name':  ['HLA-A', 'HLA-C'],
    'tss':   [29942532, 31272092],
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
# 3. Sequence-specific Attribution Logos (WT avg, each eQTL variant, most distant seq)
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

    # Load eQTL variant metadata
    variant_info = []
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path)
        var_rows = meta[meta['source'] == VARIANT_SOURCE]
        for _, vrow in var_rows.iterrows():
            seq_idx = int(vrow['seq_idx'])
            if seq_idx < len(maps):
                rel_pos = int(vrow['pos']) - locus_start
                pred_score = float(vrow['prediction']) if pd.notna(vrow.get('prediction')) else 0.0
                variant_info.append({
                    'seq_idx': seq_idx,
                    'variant_id': vrow.get('variant_id', ''),
                    'ref': vrow['ref'], 'alt': vrow['alt'],
                    'rel_pos': rel_pos, 'prediction': pred_score,
                    'cluster': labels[seq_idx],
                })

    if not variant_info:
        print(f'{name}: no eQTL variants in maps, skipping logos')
        del maps
        gc.collect()
        return

    # Compute cosine distance + activity change for all non-WT, non-eQTL seqs
    wt_avg_flat = maps[labels == wt_cluster].mean(axis=0).flatten()
    wt_avg_norm = np.linalg.norm(wt_avg_flat)
    wt_pred = float(preds[0]) if preds is not None else 0.0
    eqtl_set = {v['seq_idx'] for v in variant_info}

    n_seqs = maps.shape[0]
    cos_dists = np.zeros(n_seqs, dtype=np.float64)
    for i in range(n_seqs):
        flat = maps[i].flatten()
        norm_prod = np.linalg.norm(flat) * wt_avg_norm
        if norm_prod > 0:
            cos_dists[i] = 1.0 - np.dot(flat, wt_avg_flat) / norm_prod

    # Find most distant seq with negative activity change
    best_neg_dist, most_diff_neg_idx = -1.0, 1
    best_pos_dist, most_diff_pos_idx = -1.0, 1
    for i in range(1, n_seqs):
        if i in eqtl_set:
            continue
        d = cos_dists[i]
        act_change = float(preds[i]) - wt_pred if preds is not None else 0.0
        if act_change < 0 and d > best_neg_dist:
            best_neg_dist = d
            most_diff_neg_idx = i
        if act_change > 0 and d > best_pos_dist:
            best_pos_dist = d
            most_diff_pos_idx = i

    def _normalize(m):
        """L2 normalize a (1000, 4) map for visualization."""
        l2 = np.linalg.norm(m.flatten())
        return m / l2 if l2 > 0 else m

    # Two-hot decoding: argmax of [A,C,G,T] channel
    _IDX2BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

    def _find_snps(wt_seq, mut_seq):
        """Compare two (1000, 4) two-hot seqs, return list of (pos, ref, alt)."""
        snps = []
        for pos in range(wt_seq.shape[0]):
            if not np.array_equal(wt_seq[pos], mut_seq[pos]):
                ref_base = _IDX2BASE.get(int(np.argmax(wt_seq[pos])), '?')
                alt_base = _IDX2BASE.get(int(np.argmax(mut_seq[pos])), '?')
                snps.append((pos, ref_base, alt_base))
        return snps

    def _annotate_snps(ax, snps):
        """Draw orange vlines and build annotation text for SNPs."""
        mut_strings = []
        for pos, ref, alt in snps:
            ax.axvline(x=pos, color='darkorange', linestyle='-',
                      linewidth=1.5, alpha=0.6, ymin=0.85, ymax=1.0)
            mut_strings.append(f'{pos}: {ref}>{alt}')
        if mut_strings:
            if len(mut_strings) <= 20:
                summary = '\n'.join(mut_strings)
            else:
                summary = '\n'.join(mut_strings[:17]) + f'\n... +{len(mut_strings)-17} more'
            ax.text(1.005, 0.5, summary, transform=ax.transAxes,
                   fontsize=8, fontfamily='monospace', va='center',
                   color='darkorange', fontweight='bold')

    # Collect all normalized maps first to determine global ylim
    all_norm_maps = []

    wt_members = labels == wt_cluster
    wt_avg_normed = _normalize(maps[wt_members].mean(axis=0))
    all_norm_maps.append(wt_avg_normed)

    var_norm_maps = []
    for vinfo in variant_info:
        vm = _normalize(maps[vinfo['seq_idx']])
        var_norm_maps.append(vm)
        all_norm_maps.append(vm)

    neg_norm = _normalize(maps[most_diff_neg_idx])
    pos_norm = _normalize(maps[most_diff_pos_idx])
    all_norm_maps.append(neg_norm)
    all_norm_maps.append(pos_norm)

    # Global ylim across all panels
    global_min = min(m.min() for m in all_norm_maps)
    global_max = max(m.max() for m in all_norm_maps)
    pad = (global_max - global_min) * 0.05
    ylim = (global_min - pad, global_max + pad)

    # Panels: WT avg, each eQTL variant, most distant neg, most distant pos
    n_panels = 1 + len(variant_info) + 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(50, 4 * n_panels))

    # --- Panel 0: WT cluster average ---
    logo = BatchLogo(wt_avg_normed[np.newaxis, :, :],
                     figsize=[50, 3], show_progress=False)
    logo.process_all()
    logo.draw_single(0, ax=axes[0], fixed_ylim=False, border=True)
    axes[0].set_ylim(ylim)

    wt_title = f'WT Cluster C{wt_cluster} avg (n={int(wt_members.sum())})'
    if preds is not None:
        wt_preds = preds[wt_members]
        wt_title += (f'  |  Activity: mean={wt_preds.mean():.2f}, '
                     f'max={wt_preds.max():.2f}, min={wt_preds.min():.2f}')
    axes[0].set_title(wt_title, fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Attribution', fontsize=9)

    # --- Panels 1..N: individual eQTL variant DeepSHAP maps ---
    for vi, vinfo in enumerate(variant_info):
        ax_i = axes[1 + vi]
        logo = BatchLogo(var_norm_maps[vi][np.newaxis, :, :],
                         figsize=[50, 3], show_progress=False)
        logo.process_all()
        logo.draw_single(0, ax=ax_i, fixed_ylim=False, border=True)
        ax_i.set_ylim(ylim)

        rp = vinfo['rel_pos']
        if 0 <= rp < 1000:
            ax_i.axvline(x=rp, color='darkorange', linestyle='-',
                        linewidth=1.5, alpha=0.6, ymin=0.85, ymax=1.0)

        title = (f"eQTL: {vinfo['variant_id']}  ({vinfo['ref']}>{vinfo['alt']} @ pos {rp})"
                 f"  |  Cluster C{vinfo['cluster']}")
        if preds is not None:
            seq_idx = vinfo['seq_idx']
            title += f"  |  Activity={float(preds[seq_idx]):.3f}"
        title += f"  |  AlphaGenome pred={vinfo['prediction']:.4f}"

        ax_i.set_title(title, fontsize=11, fontweight='bold')
        ax_i.set_ylabel('Attribution', fontsize=9)

        annot = (f"pos={rp}: {vinfo['ref']}>{vinfo['alt']}\n"
                 f"pred={vinfo['prediction']:.4f}")
        ax_i.text(1.005, 0.5, annot, transform=ax_i.transAxes,
                 fontsize=9, fontfamily='monospace', va='center',
                 color='darkorange', fontweight='bold')

    # Get WT sequence for SNP comparison
    wt_seq = seqs[0] if seqs is not None else None

    # --- Most distant seq with negative activity change ---
    ax_neg = axes[-2]
    logo = BatchLogo(neg_norm[np.newaxis, :, :],
                     figsize=[50, 3], show_progress=False)
    logo.process_all()
    logo.draw_single(0, ax=ax_neg, fixed_ylim=False, border=True)
    ax_neg.set_ylim(ylim)

    neg_act = float(preds[most_diff_neg_idx]) if preds is not None else 0.0
    neg_snps = _find_snps(wt_seq, seqs[most_diff_neg_idx]) if seqs is not None else []
    neg_title = (f'Most Distant Seq (neg activity change)  idx={most_diff_neg_idx}'
                 f'  |  cos_dist={best_neg_dist:.4f}  |  Activity={neg_act:.3f}'
                 f'  |  delta={neg_act - wt_pred:.3f}  |  {len(neg_snps)} SNPs')
    ax_neg.set_title(neg_title, fontsize=11, fontweight='bold', color='purple')
    ax_neg.set_ylabel('Attribution', fontsize=9)
    _annotate_snps(ax_neg, neg_snps)

    # --- Most distant seq with positive activity change ---
    ax_pos = axes[-1]
    logo = BatchLogo(pos_norm[np.newaxis, :, :],
                     figsize=[50, 3], show_progress=False)
    logo.process_all()
    logo.draw_single(0, ax=ax_pos, fixed_ylim=False, border=True)
    ax_pos.set_ylim(ylim)

    pos_act = float(preds[most_diff_pos_idx]) if preds is not None else 0.0
    pos_snps = _find_snps(wt_seq, seqs[most_diff_pos_idx]) if seqs is not None else []
    pos_title = (f'Most Distant Seq (pos activity change)  idx={most_diff_pos_idx}'
                 f'  |  cos_dist={best_pos_dist:.4f}  |  Activity={pos_act:.3f}'
                 f'  |  delta={pos_act - wt_pred:+.3f}  |  {len(pos_snps)} SNPs')
    ax_pos.set_title(pos_title, fontsize=11, fontweight='bold', color='darkgreen')
    ax_pos.set_ylabel('Attribution', fontsize=9)
    _annotate_snps(ax_pos, pos_snps)

    fig.suptitle(f'{name} — Normalized DeepSHAP: WT Avg, eQTL Variants, Most Distant Seqs (k={k})',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    out_path = os.path.join(locus_dir, f'cluster_logos_eqtl_k{k}.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f'{name}: saved {out_path} ({len(variant_info)} variant panels + 2 distant seqs)')
    del maps
    gc.collect()


# =========================================================================
# 4. Volcano: Activity vs Mechanistic Diversity (per-sequence)
# =========================================================================
def plot_volcano(name, k=K):
    """For every sequence in a locus mutagenesis library:
    - x-axis: CLIPNET activity prediction
    - y-axis: cosine distance from WT cluster average DeepSHAP map
    Highlights eQTL variants.
    """
    maps_path = os.path.join(ATTRIBUTION_DIR, f'maps_quantity_{name}_{NUM_SEQS}.npy')
    preds_path = os.path.join(ATTRIBUTION_DIR, f'preds_quantity_{name}_{NUM_SEQS}.npy')
    labels_path = os.path.join(RESULTS_DIR, name, f'k{k}', 'cluster_labels.npy')
    meta_path = os.path.join(DATA_DIR, f'x_mut_{name}_{NUM_SEQS}_metadata.csv')
    seqs_path = os.path.join(DATA_DIR, f'x_mut_{name}_{NUM_SEQS}.npy')

    if not os.path.exists(maps_path) or not os.path.exists(preds_path):
        print(f'{name}: missing maps/preds for volcano, skipping')
        return

    maps = np.load(maps_path)
    if maps.ndim == 3 and maps.shape[0] == 4 and maps.shape[2] != 4:
        maps = maps.transpose(1, 2, 0)

    seqs = np.load(seqs_path) if os.path.exists(seqs_path) else None
    if seqs is not None:
        if seqs.ndim == 3 and seqs.shape[0] == 4 and seqs.shape[2] != 4:
            seqs = seqs.transpose(1, 2, 0)
        maps = maps * (seqs > 0).astype(np.float32)

    preds = np.load(preds_path)
    labels = np.load(labels_path)

    # WT cluster average (flattened)
    wt_cluster = labels[0]
    wt_avg = maps[labels == wt_cluster].mean(axis=0).flatten()

    # Load metadata to identify eQTL variants
    meta = pd.read_csv(meta_path)
    eqtl_idx = set()
    eqtl_labels = {}
    for _, vrow in meta[meta['source'] == VARIANT_SOURCE].iterrows():
        si = int(vrow['seq_idx'])
        eqtl_idx.add(si)
        eqtl_labels[si] = vrow.get('variant_id', '')

    # Load causal fine-mapped variant IDs
    causal_df = pd.read_feather(CAUSAL_FEATHER)
    causal_lcl = causal_df[causal_df['tissue'] == 'Cells_EBV-transformed_lymphocytes']
    causal_ids = set(causal_lcl['variant_id'].values)
    del causal_df, causal_lcl

    # Split eQTL into causal vs non-causal
    causal_eqtl = set()
    noncausal_eqtl = set()
    for si in eqtl_idx:
        vid = eqtl_labels.get(si, '')
        if vid in causal_ids:
            causal_eqtl.add(si)
        else:
            noncausal_eqtl.add(si)

    # Compute cosine distance for each sequence vs WT cluster avg
    n_seqs = maps.shape[0]
    cos_dists = np.zeros(n_seqs, dtype=np.float64)
    for i in range(n_seqs):
        flat = maps[i].flatten()
        norm_prod = np.linalg.norm(flat) * np.linalg.norm(wt_avg)
        if norm_prod > 0:
            cos_dists[i] = 1.0 - np.dot(flat, wt_avg) / norm_prod
        else:
            cos_dists[i] = 0.0

    # Separate into background vs eQTL
    bg_mask = np.ones(n_seqs, dtype=bool)
    bg_mask[0] = False  # exclude WT itself
    eqtl_mask = np.zeros(n_seqs, dtype=bool)
    for si in eqtl_idx:
        if si < n_seqs:
            bg_mask[si] = False
            eqtl_mask[si] = True

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(24, 8))

    # Background (random mutagenesis)
    ax.scatter(preds[bg_mask], cos_dists[bg_mask],
               c='#cccccc', s=3, alpha=0.15, label='Random mutagenesis', rasterized=True)

    # WT mean activity vertical line
    wt_mean_activity = float(preds[labels == wt_cluster].mean())
    ax.axvline(x=wt_mean_activity, color='red', linestyle='--', linewidth=1.5,
               alpha=0.8, label=f'WT cluster mean = {wt_mean_activity:.2f}')

    # WT
    ax.scatter(preds[0], cos_dists[0],
               c='red', s=120, marker='*', zorder=10, label='WT sequence')

    # eQTL variants: causal (green) vs non-causal (orange)
    causal_mask = np.zeros(n_seqs, dtype=bool)
    noncausal_mask = np.zeros(n_seqs, dtype=bool)
    for si in causal_eqtl:
        if si < n_seqs:
            causal_mask[si] = True
    for si in noncausal_eqtl:
        if si < n_seqs:
            noncausal_mask[si] = True

    if noncausal_mask.any():
        ax.scatter(preds[noncausal_mask], cos_dists[noncausal_mask],
                   c='darkorange', s=80, edgecolors='black', linewidths=0.8,
                   zorder=10, label=f'eQTL non-causal (n={int(noncausal_mask.sum())})')
    if causal_mask.any():
        ax.scatter(preds[causal_mask], cos_dists[causal_mask],
                   c='#2ecc71', s=80, edgecolors='black', linewidths=0.8,
                   zorder=10, label=f'eQTL causal/fine-mapped (n={int(causal_mask.sum())})')

    # Label each eQTL variant with staggered offsets to avoid overlap
    eqtl_indices = sorted(eqtl_idx)
    for vi, si in enumerate(eqtl_indices):
        if si < n_seqs:
            vid = eqtl_labels.get(si, '')
            short = vid.replace('chr6_', '').replace('_', ' ') if vid else f'idx{si}'
            color = '#2ecc71' if si in causal_eqtl else 'darkorange'
            y_offset = 8 + (vi % 3) * 14
            x_offset = 10 if vi % 2 == 0 else -60
            ax.annotate(short, (preds[si], cos_dists[si]),
                       textcoords='offset points', xytext=(x_offset, y_offset),
                       fontsize=8, fontweight='bold', color=color,
                       arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

    ax.set_xlabel('CLIPNET Activity Prediction', fontsize=13)
    ax.set_ylabel('Mechanistic Diversity\n(Cosine Distance from WT Cluster Avg DeepSHAP)', fontsize=13)
    ax.set_title(f'{name} — Activity vs Mechanistic Diversity (k={k})\n'
                 f'eQTL variants are mechanistically robust (low diversity) despite functional effects',
                 fontsize=14)
    ax.legend(fontsize=10, markerscale=1.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(FINALS_DIR, f'{name}_volcano_activity_vs_mech_diversity_k{k}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Save data
    df_out = pd.DataFrame({
        'seq_idx': np.arange(n_seqs),
        'activity': preds,
        'cos_dist_from_wt': cos_dists,
        'source': meta['source'].values[:n_seqs],
    })
    csv_path = os.path.join(FINALS_DIR, f'{name}_volcano_data_k{k}.csv')
    df_out.to_csv(csv_path, index=False)

    print(f'{name}: saved volcano {out_path}')
    print(f'  eQTL variants cosine dist: {cos_dists[eqtl_mask]}')
    print(f'  eQTL variants activity: {preds[eqtl_mask]}')
    print(f'  Background median cosine dist: {np.median(cos_dists[bg_mask]):.4f}')
    print(f'  Data: {csv_path}')

    del maps
    gc.collect()


# =========================================================================
# Main
# =========================================================================
def main():
    os.makedirs(FINALS_DIR, exist_ok=True)

    for name in LOCI:
        plot_csm_mismatch(name)
        plot_cluster_preds(name)
        plot_cluster_logos(name)

    for name in LOCI:
        plot_volcano(name)

    print(f'\nAll plots saved to {FINALS_DIR}')


if __name__ == '__main__':
    main()
