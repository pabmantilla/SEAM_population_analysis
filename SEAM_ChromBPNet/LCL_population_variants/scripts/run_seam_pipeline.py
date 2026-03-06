#!/usr/bin/env python
"""
Full SEAM pipeline for ChromBPNet LCL population variant analysis.

Steps per locus:
  1. DeepSHAP attribution maps (ChromBPNet via bpnet-lite, tangermeme)
  2. ChromBPNet predictions on mutagenesis library
  3. K-means clustering on attribution maps
  4. CSM percent-mismatch from WT
  5. Results analysis: CSM heatmaps, cluster logos, summary plots

Usage:
  python run_seam_pipeline.py --step all
  python run_seam_pipeline.py --step attribute --locus IRF7
  python run_seam_pipeline.py --step cluster --k 100
  python run_seam_pipeline.py --step results

Run with chrombpnet_torch_env:
  source .../SEAM_ChromBPNet/chrombpnet_torch_env/bin/activate
"""

import os
import sys
import gc
import argparse
import logging

import numpy as np
import pandas as pd
import torch

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = '/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis'
LCL_DIR = os.path.join(BASE_DIR, 'SEAM_ChromBPNet', 'LCL_population_variants')
LOCI_TSV = os.path.join(BASE_DIR, 'variant_data', 'GnomAD_data', 'loci_backup_all34.tsv')

MUTLIB_DIR = os.path.join(LCL_DIR, 'Mutagenisis_lib')
DEEPSHAP_DIR = os.path.join(LCL_DIR, 'DeepSHAP_lib')
CLUSTER_DIR = os.path.join(LCL_DIR, 'cluster_results')
SEQ_RESULTS_DIR = os.path.join(LCL_DIR, 'results', 'seq_results')
FINAL_RESULTS_DIR = os.path.join(LCL_DIR, 'results', 'results_final')

# ChromBPNet model
MODEL_DIR = os.path.join(BASE_DIR, 'SEAM_ChromBPNet', 'models')
MODEL_ACCESSION = 'ENCFF673TIN'  # GM12878 LCL DNase-seq

# ── Constants ──────────────────────────────────────────────────────────────
SEQ_LENGTH = 2114
NUM_SEQS = 25000
NUM_SHUFS = 50
ATTR_BATCH_SIZE = 8
PRED_BATCH_SIZE = 32
K_DEFAULT = 100

# ── Load loci ──────────────────────────────────────────────────────────────
def load_loci(loci_tsv=LOCI_TSV):
    loci = pd.read_csv(loci_tsv, sep='\t')
    loci['start'] = loci['tss'] - SEQ_LENGTH // 2
    loci['end'] = loci['tss'] + SEQ_LENGTH // 2
    return loci

LOCI = load_loci()


# =========================================================================
# Step 1: DeepSHAP Attributions (ChromBPNet via bpnet-lite + tangermeme)
# =========================================================================
def load_chrombpnet_model(device='cuda'):
    """Load ChromBPNet GM12878 LCL DNase-seq model (fold_0)."""
    import torch
    from io import BytesIO
    from bpnetlite import BPNet
    import tarfile
    import h5py

    tar_path = os.path.join(MODEL_DIR, f'{MODEL_ACCESSION}.tar.gz')
    if not os.path.exists(tar_path):
        raise FileNotFoundError(
            f'Model not found at {tar_path}. Download with:\n'
            f'  wget -P {MODEL_DIR} https://www.encodeproject.org/files/{MODEL_ACCESSION}/@@download/{MODEL_ACCESSION}.tar.gz'
        )

    print(f'Loading ChromBPNet model from {tar_path} (fold_0)...')
    with tarfile.open(tar_path, 'r:gz') as tar:
        # Find fold_0 nobias model
        h5_name = None
        for member in tar.getnames():
            if 'fold_0' in member and 'nobias' in member and member.endswith('.h5'):
                h5_name = member
                break
        if h5_name is None:
            raise ValueError(f'Could not find fold_0 nobias .h5 in {tar_path}')

        h5_data = tar.extractfile(h5_name).read()

    model = BPNet.from_chrombpnet(BytesIO(h5_data))
    model = model.to(device).eval()
    print(f'Model loaded: {h5_name}')
    return model


class CountWrapper(torch.nn.Module):
    """Wrap BPNet to return only the scalar log-count prediction."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        profile, counts = self.model(x)
        return counts


def step_attribute(loci, device='cuda'):
    """Compute DeepLIFT/SHAP attributions for all sequences in mutagenesis library."""
    import torch
    from tangermeme.deep_lift_shap import deep_lift_shap

    os.makedirs(DEEPSHAP_DIR, exist_ok=True)

    model = load_chrombpnet_model(device)
    count_model = CountWrapper(model).to(device)

    ref_data = np.load(os.path.join(MUTLIB_DIR, 'x_onehot_ref_2114bp.npz'))

    for _, row in loci.iterrows():
        name = row['name']
        maps_path = os.path.join(DEEPSHAP_DIR, f'maps_{name}_{NUM_SEQS}.npy')
        preds_path = os.path.join(DEEPSHAP_DIR, f'preds_{name}_{NUM_SEQS}.npy')

        mut_path = os.path.join(MUTLIB_DIR, f'x_mut_{name}_{NUM_SEQS}.npy')
        if not os.path.exists(mut_path):
            print(f'{name}: mutagenesis library not found at {mut_path}, skipping')
            continue

        if os.path.exists(maps_path) and os.path.exists(preds_path):
            print(f'{name}: attribution maps + predictions exist, skipping')
            continue

        print(f'\n{"="*50}')
        print(f'{name}: DeepLIFT/SHAP attributions')
        print(f'{"="*50}')

        # Load mutagenesis library: (N, L, 4) -> need (N, 4, L) for bpnet-lite
        x_mut = np.load(mut_path)  # (N, 2114, 4)
        x_mut_t = np.transpose(x_mut, (0, 2, 1))  # (N, 4, 2114)

        # --- Predictions ---
        if not os.path.exists(preds_path):
            print(f'  Computing predictions for {x_mut_t.shape[0]} sequences...')
            from tangermeme.predict import predict
            X = torch.tensor(x_mut_t, dtype=torch.float32)
            profile_preds, count_preds = predict(model, X, batch_size=PRED_BATCH_SIZE, device=device)
            counts = count_preds.numpy().squeeze()  # (N,)
            np.save(preds_path, counts)
            print(f'  Saved predictions {counts.shape} to {preds_path}')
            del X, profile_preds, count_preds
            gc.collect()
            torch.cuda.empty_cache()

        # --- Attribution maps ---
        if not os.path.exists(maps_path):
            from tangermeme.ersatz import dinucleotide_shuffle

            X = torch.tensor(x_mut_t, dtype=torch.float32)

            # Pre-compute all dinucleotide shuffles in parallel (avoids
            # the sequential per-sequence loop inside deep_lift_shap)
            CHUNK_SIZE = 256
            all_attrs = []
            for i in range(0, X.shape[0], CHUNK_SIZE):
                chunk = X[i:i + CHUNK_SIZE]

                print(f'  Chunk [{i}:{min(i+CHUNK_SIZE, X.shape[0])}] '
                      f'pre-computing {NUM_SHUFS} dinuc shuffles...')
                refs = dinucleotide_shuffle(chunk, n=NUM_SHUFS, random_state=42)
                # refs shape: (chunk_size, n_shuffles, 4, 2114)

                print(f'  Chunk [{i}:{min(i+CHUNK_SIZE, X.shape[0])}] '
                      f'running DeepLIFT/SHAP (batch_size={ATTR_BATCH_SIZE})...')
                attrs = deep_lift_shap(
                    count_model, chunk,
                    references=refs,  # pre-computed tensor, no sequential loop
                    batch_size=ATTR_BATCH_SIZE,
                    device=device,
                    hypothetical=True,
                )
                # attrs: (chunk, 4, 2114) -> project by input
                projected = (attrs * chunk.numpy()).numpy()  # (chunk, 4, 2114)
                all_attrs.append(projected)

                done = min(i + CHUNK_SIZE, X.shape[0])
                print(f'    {done}/{X.shape[0]} sequences attributed')
                del refs
                gc.collect()
                torch.cuda.empty_cache()

            maps = np.concatenate(all_attrs, axis=0)  # (N, 4, 2114)
            # Transpose to (N, 2114, 4) for consistency with CLIPNET pipeline
            maps = np.transpose(maps, (0, 2, 1))
            np.save(maps_path, maps)
            print(f'  Saved attribution maps {maps.shape} to {maps_path}')
            del X, all_attrs, maps
            gc.collect()
            torch.cuda.empty_cache()

        gc.collect()


# =========================================================================
# Step 2: K-means Clustering + CSM
# =========================================================================
def step_cluster(loci, k):
    """Cluster attribution maps and compute CSM (% mismatch from WT)."""
    from sklearn.cluster import KMeans

    os.makedirs(CLUSTER_DIR, exist_ok=True)

    for _, row in loci.iterrows():
        name = row['name']
        maps_path = os.path.join(DEEPSHAP_DIR, f'maps_{name}_{NUM_SEQS}.npy')
        mut_path = os.path.join(MUTLIB_DIR, f'x_mut_{name}_{NUM_SEQS}.npy')
        preds_path = os.path.join(DEEPSHAP_DIR, f'preds_{name}_{NUM_SEQS}.npy')

        if not os.path.exists(maps_path):
            print(f'{name}: attribution maps not found, skipping')
            continue

        locus_dir = os.path.join(CLUSTER_DIR, name, f'k{k}')
        labels_path = os.path.join(locus_dir, 'cluster_labels.npy')
        if os.path.exists(labels_path):
            print(f'{name}: cluster results exist (k={k}), skipping')
            continue

        print(f'\n{"="*50}')
        print(f'{name}: clustering (k={k})')
        print(f'{"="*50}')

        os.makedirs(locus_dir, exist_ok=True)

        maps = np.load(maps_path)  # (N, 2114, 4)
        x_mut = np.load(mut_path)  # (N, 2114, 4)

        # Flatten for clustering
        maps_flat = maps.reshape(maps.shape[0], -1)
        print(f'  Clustering {maps_flat.shape[0]} maps into {k} clusters...')
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
        labels = km.fit_predict(maps_flat)

        # CSM: percent mismatch from WT
        x_ref = x_mut[0]  # WT is index 0
        mismatch_mask = np.any(x_mut != x_ref[np.newaxis, :, :], axis=2)  # (N, 2114)
        cluster_ids = sorted(np.unique(labels))
        pct_mismatch = np.zeros((len(cluster_ids), SEQ_LENGTH))
        for ci, c in enumerate(cluster_ids):
            members = labels == c
            if np.sum(members) > 0:
                pct_mismatch[ci] = mismatch_mask[members].mean(axis=0) * 100

        # Save
        np.save(labels_path, labels)
        np.save(os.path.join(locus_dir, 'csm_matrix.npy'), pct_mismatch)

        # Cluster info CSV
        wt_cluster = labels[0]
        info_rows = []
        preds = np.load(preds_path) if os.path.exists(preds_path) else None
        for ci, c in enumerate(cluster_ids):
            members = labels == c
            n = int(np.sum(members))
            r = {
                'cluster': int(c),
                'n_seqs': n,
                'pct_of_total': round(100 * n / len(labels), 2),
                'has_wt': c == wt_cluster,
                'mean_mismatch_pct': round(float(pct_mismatch[ci].mean()), 2),
            }
            if preds is not None:
                r['mean_pred'] = round(float(preds[members].mean()), 6)
                r['std_pred'] = round(float(preds[members].std()), 6)
            info_rows.append(r)

        pd.DataFrame(info_rows).to_csv(os.path.join(locus_dir, 'cluster_info.csv'), index=False)
        print(f'  Saved cluster results to {locus_dir}')
        print(f'  WT cluster={wt_cluster}, sizes: min={min(r["n_seqs"] for r in info_rows)}, '
              f'max={max(r["n_seqs"] for r in info_rows)}')

        del maps, maps_flat, x_mut, mismatch_mask
        gc.collect()


# =========================================================================
# Step 3: Per-locus results (CSM heatmap, cluster preds, WT logo)
# =========================================================================
def step_seq_results(loci, k):
    """Generate per-locus result plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    for _, row in loci.iterrows():
        name = row['name']
        locus_cluster_dir = os.path.join(CLUSTER_DIR, name, f'k{k}')
        locus_out_dir = os.path.join(SEQ_RESULTS_DIR, name)
        os.makedirs(locus_out_dir, exist_ok=True)

        csm_path = os.path.join(locus_cluster_dir, 'csm_matrix.npy')
        labels_path = os.path.join(locus_cluster_dir, 'cluster_labels.npy')
        preds_path = os.path.join(DEEPSHAP_DIR, f'preds_{name}_{NUM_SEQS}.npy')
        maps_path = os.path.join(DEEPSHAP_DIR, f'maps_{name}_{NUM_SEQS}.npy')

        if not os.path.exists(csm_path):
            print(f'{name}: cluster results not found, skipping')
            continue

        print(f'{name}: generating seq results...')

        csm = np.load(csm_path)
        labels = np.load(labels_path)
        cluster_ids = sorted(np.unique(labels))
        wt_cluster = labels[0]
        cluster_sizes = {c: int(np.sum(labels == c)) for c in cluster_ids}

        # ── 1. CSM Heatmap ──
        fig_height = max(6, len(cluster_ids) * 0.18)
        fig, ax = plt.subplots(figsize=(20, fig_height))
        im = ax.pcolormesh(csm, cmap='viridis', vmin=0, vmax=100)

        wt_idx = cluster_ids.index(wt_cluster)
        ax.axhline(y=wt_idx + 0.5, color='red', linestyle=':', linewidth=2, alpha=0.9)

        ytick_labels = []
        for ci, c in enumerate(cluster_ids):
            lbl = f'C{c} (n={cluster_sizes[c]})'
            if c == wt_cluster:
                lbl += '  *WT*'
            ytick_labels.append(lbl)

        ax.set_yticks(np.arange(len(cluster_ids)) + 0.5)
        ax.set_yticklabels(ytick_labels, fontsize=5)
        ax.set_xlabel('Position', fontsize=11)
        ax.set_ylabel('Cluster', fontsize=11)
        ax.set_title(f'{name} — CSM % Mismatch from WT (k={k})', fontsize=13)
        plt.colorbar(im, ax=ax, label='% Mismatch', shrink=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(locus_out_dir, f'csm_mismatch_k{k}.png'), dpi=100, bbox_inches='tight')
        plt.close()

        # ── 2. Cluster Predictions Boxplot (sorted by mean pred) ──
        if os.path.exists(preds_path):
            preds = np.load(preds_path)
            wt_pred = float(preds[0])
            cluster_means = {c: float(preds[labels == c].mean()) for c in cluster_ids}
            sorted_clusters = sorted(cluster_ids, key=lambda c: cluster_means[c])

            box_data = [preds[labels == c] for c in sorted_clusters]
            fig_height = max(6, len(sorted_clusters) * 0.18)
            fig, ax = plt.subplots(figsize=(10, fig_height))
            bp = ax.boxplot(box_data, patch_artist=True, showfliers=False, vert=False, widths=0.6)
            for patch in bp['boxes']:
                patch.set_facecolor('steelblue')
                patch.set_alpha(0.6)

            ax.axvline(x=wt_pred, color='red', linestyle='--', linewidth=1.5, alpha=0.8,
                       label=f'WT pred = {wt_pred:.4f}')
            wt_y = sorted_clusters.index(wt_cluster) + 1
            ax.axhline(y=wt_y, color='red', linestyle=':', linewidth=1.5, alpha=0.7)

            ytick_labels_sorted = []
            for c in sorted_clusters:
                lbl = f'C{c} (n={cluster_sizes[c]})'
                if c == wt_cluster:
                    lbl += ' *WT*'
                ytick_labels_sorted.append(lbl)

            ax.set_yticklabels(ytick_labels_sorted, fontsize=5)
            ax.set_xlabel('ChromBPNet Log Counts Prediction', fontsize=11)
            ax.set_title(f'{name} — Cluster Predictions (k={k}, sorted)', fontsize=13)
            ax.legend(fontsize=9, loc='lower right')
            ax.grid(True, axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(locus_out_dir, f'cluster_preds_sorted_k{k}.png'),
                        dpi=100, bbox_inches='tight')
            plt.close()

        # ── 3. WT Attribution Logo + CSM ──
        if os.path.exists(maps_path):
            maps = np.load(maps_path, mmap_mode='r')
            wt_map = maps[0]  # (2114, 4)

            fig = plt.figure(figsize=(20, max(5, 2.5 + k * 0.005)))
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.5], hspace=0.35)

            # WT logo
            ax_wt = fig.add_subplot(gs[0])
            # Plot as heatmap-style for 2114bp (too wide for sequence logo)
            contrib = wt_map  # (2114, 4)
            contrib_max = np.max(np.abs(contrib)) + 1e-8
            ax_wt.plot(np.arange(SEQ_LENGTH), contrib.sum(axis=1), color='steelblue', linewidth=0.5)
            ax_wt.fill_between(np.arange(SEQ_LENGTH), contrib.sum(axis=1), alpha=0.3, color='steelblue')
            ax_wt.axvline(x=SEQ_LENGTH // 2, color='red', linestyle='--', linewidth=1, alpha=0.5, label='TSS')
            ax_wt.set_ylabel('Attribution')
            ax_wt.set_title(f'WT DeepSHAP - {name} (log counts)')
            ax_wt.legend(fontsize=8)
            ax_wt.set_xlim(0, SEQ_LENGTH)

            # CSM heatmap (compact)
            ax_csm = fig.add_subplot(gs[1])
            im = ax_csm.pcolormesh(csm, cmap='viridis', vmin=0, vmax=100)
            ax_csm.set_xlabel('Position')
            ax_csm.set_title(f'CSM: % Mismatch from WT (k={k})')
            plt.colorbar(im, ax=ax_csm, label='% Mismatch', shrink=0.8)

            plt.savefig(os.path.join(locus_out_dir, f'wt_csm_k{k}.png'), dpi=200, bbox_inches='tight')
            plt.close()
            del maps

        print(f'  Saved to {locus_out_dir}')
        gc.collect()


# =========================================================================
# Step 4: Final cross-locus results
# =========================================================================
def step_final_results(k):
    """Generate cross-locus summary plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(FINAL_RESULTS_DIR, exist_ok=True)
    all_loci = load_loci()

    # ── Collect per-locus summaries ──
    summary_rows = []
    for _, row in all_loci.iterrows():
        name = row['name']
        info_path = os.path.join(CLUSTER_DIR, name, f'k{k}', 'cluster_info.csv')
        if not os.path.exists(info_path):
            continue
        info = pd.read_csv(info_path)
        info['locus'] = name
        summary_rows.append(info)

    if not summary_rows:
        print('No cluster results found for any locus')
        return

    summary = pd.concat(summary_rows, ignore_index=True)
    summary.to_csv(os.path.join(FINAL_RESULTS_DIR, f'all_loci_cluster_info_k{k}.csv'), index=False)

    # ── WT cluster prediction vs other clusters across loci ──
    if 'mean_pred' in summary.columns:
        wt_data = summary[summary['has_wt'] == True]
        non_wt = summary[summary['has_wt'] == False]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left: distribution of mean predictions, WT vs non-WT
        axes[0].hist(non_wt['mean_pred'], bins=50, alpha=0.6, color='steelblue', label='Non-WT clusters')
        for _, wt_row in wt_data.iterrows():
            axes[0].axvline(x=wt_row['mean_pred'], color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        axes[0].set_xlabel('Mean ChromBPNet Log Counts Prediction')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'Cluster Predictions Across All Loci (k={k})')
        axes[0].legend()

        # Right: WT prediction per locus
        wt_sorted = wt_data.sort_values('mean_pred')
        axes[1].barh(range(len(wt_sorted)), wt_sorted['mean_pred'], color='steelblue', alpha=0.7)
        axes[1].set_yticks(range(len(wt_sorted)))
        axes[1].set_yticklabels(wt_sorted['locus'], fontsize=8)
        axes[1].set_xlabel('WT Cluster Mean Prediction')
        axes[1].set_title(f'WT Prediction by Locus (k={k})')

        plt.tight_layout()
        plt.savefig(os.path.join(FINAL_RESULTS_DIR, f'cross_locus_predictions_k{k}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ── Mean mismatch distribution ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in all_loci.iterrows():
        name = row['name']
        csm_path = os.path.join(CLUSTER_DIR, name, f'k{k}', 'csm_matrix.npy')
        if not os.path.exists(csm_path):
            continue
        csm = np.load(csm_path)
        mean_per_cluster = csm.mean(axis=1)
        ax.hist(mean_per_cluster, bins=30, alpha=0.3, label=name)

    ax.set_xlabel('Mean % Mismatch from WT')
    ax.set_ylabel('Count (clusters)')
    ax.set_title(f'CSM Mean Mismatch Distribution Across Loci (k={k})')
    if len(all_loci) <= 10:
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(FINAL_RESULTS_DIR, f'csm_mismatch_distribution_k{k}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f'Final results saved to {FINAL_RESULTS_DIR}')


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description='SEAM pipeline for ChromBPNet LCL population analysis')
    parser.add_argument('--step', required=True,
                        choices=['all', 'attribute', 'cluster', 'results', 'final'],
                        help='Pipeline step to run')
    parser.add_argument('--locus', default=None,
                        help='Comma-separated locus names (default: all 34)')
    parser.add_argument('--k', type=int, default=K_DEFAULT,
                        help=f'Number of clusters for k-means (default: {K_DEFAULT})')
    parser.add_argument('--device', default='cuda',
                        help='Device for PyTorch (default: cuda)')
    args = parser.parse_args()

    loci = LOCI.copy()
    if args.locus:
        names = set(args.locus.split(','))
        loci = LOCI[LOCI['name'].isin(names)]
        if loci.empty:
            raise ValueError(f'No matching loci. Available: {LOCI["name"].tolist()}')
        print(f'Processing {len(loci)} loci: {loci["name"].tolist()}')

    if args.step in ('all', 'attribute'):
        print('\n>>> Step 1: DeepSHAP Attribution Maps <<<')
        step_attribute(loci, device=args.device)

    if args.step in ('all', 'cluster'):
        print(f'\n>>> Step 2: K-means Clustering + CSM (k={args.k}) <<<')
        step_cluster(loci, args.k)

    if args.step in ('all', 'results'):
        print(f'\n>>> Step 3: Per-locus Results <<<')
        step_seq_results(loci, args.k)

    if args.step in ('all', 'final'):
        print(f'\n>>> Step 4: Cross-locus Final Results <<<')
        step_final_results(args.k)

    print('\nPipeline complete.')


if __name__ == '__main__':
    main()
