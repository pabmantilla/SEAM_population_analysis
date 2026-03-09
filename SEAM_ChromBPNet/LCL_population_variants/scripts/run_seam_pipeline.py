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
  python run_seam_pipeline.py --step inject --inject gnomad caqtl_eur caqtl_afr

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
VARIANT_LIBS_DIR = os.path.join(LCL_DIR, 'variant_libs')
CLUSTER_DIR = os.path.join(LCL_DIR, 'cluster_results')
INJECT_CLUSTER_DIR = os.path.join(LCL_DIR, 'cluster_results', 'variant_inject')
SEQ_RESULTS_DIR = os.path.join(LCL_DIR, 'results', 'seq_results')
FINAL_RESULTS_DIR = os.path.join(LCL_DIR, 'results', 'results_final')
INJECT_RESULTS_DIR = os.path.join(LCL_DIR, 'results', 'results_final', 'variant_inject')

# caQTL coefficient feather files (AlphaGenome predictions vs measured)
CAQTL_COEFF_DIR = os.path.join(BASE_DIR, 'variant_data', 'Alphagenome_data', 'chromatin_accessibility_qtl')
CAQTL_COEFF_FILES = {
    'caqtl_eur': os.path.join(CAQTL_COEFF_DIR, 'caqtl_european_variant_coefficient_human_predictions.feather'),
    'caqtl_afr': os.path.join(CAQTL_COEFF_DIR, 'caqtl_african_variant_coefficient_human_predictions.feather'),
}

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


def _attribute_one_library(mut_path, maps_path, preds_path, model, count_model, device):
    """Run predictions + DeepLIFT/SHAP on a single mutagenesis library file."""
    import torch
    from tangermeme.deep_lift_shap import deep_lift_shap

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

        CHUNK_SIZE = 256
        all_attrs = []
        for i in range(0, X.shape[0], CHUNK_SIZE):
            chunk = X[i:i + CHUNK_SIZE]

            print(f'  Chunk [{i}:{min(i+CHUNK_SIZE, X.shape[0])}] '
                  f'pre-computing {NUM_SHUFS} dinuc shuffles...')
            refs = dinucleotide_shuffle(chunk, n=NUM_SHUFS, random_state=42)

            print(f'  Chunk [{i}:{min(i+CHUNK_SIZE, X.shape[0])}] '
                  f'running DeepLIFT/SHAP (batch_size={ATTR_BATCH_SIZE})...')
            attrs = deep_lift_shap(
                count_model, chunk,
                references=refs,
                batch_size=ATTR_BATCH_SIZE,
                device=device,
                hypothetical=True,
            )
            projected = (attrs * chunk.numpy()).numpy()  # (chunk, 4, 2114)
            all_attrs.append(projected)

            done = min(i + CHUNK_SIZE, X.shape[0])
            print(f'    {done}/{X.shape[0]} sequences attributed')
            del refs
            gc.collect()
            torch.cuda.empty_cache()

        maps = np.concatenate(all_attrs, axis=0)  # (N, 4, 2114)
        maps = np.transpose(maps, (0, 2, 1))  # (N, 2114, 4)
        np.save(maps_path, maps)
        print(f'  Saved attribution maps {maps.shape} to {maps_path}')
        del X, all_attrs, maps
        gc.collect()
        torch.cuda.empty_cache()

    gc.collect()


def step_attribute(loci, device='cuda', source=None):
    """Compute DeepLIFT/SHAP attributions.

    Args:
        source: If None, uses default Mutagenisis_lib/DeepSHAP_lib paths (original pipeline).
                If 'gnomad', 'caqtl_eur', or 'caqtl_afr', uses variant_libs/{source}/
                with x_var_{name}.npy files (WT + variant one-hot sequences).
    """
    model = load_chrombpnet_model(device)
    count_model = CountWrapper(model).to(device)

    for _, row in loci.iterrows():
        name = row['name']

        if source:
            src_dir = os.path.join(VARIANT_LIBS_DIR, source)
            mut_path = os.path.join(src_dir, f'x_var_{name}.npy')
            maps_path = os.path.join(src_dir, f'maps_{name}.npy')
            preds_path = os.path.join(src_dir, f'preds_{name}.npy')
        else:
            mut_path = os.path.join(MUTLIB_DIR, f'x_mut_{name}_{NUM_SEQS}.npy')
            maps_path = os.path.join(DEEPSHAP_DIR, f'maps_{name}_{NUM_SEQS}.npy')
            preds_path = os.path.join(DEEPSHAP_DIR, f'preds_{name}_{NUM_SEQS}.npy')
            os.makedirs(DEEPSHAP_DIR, exist_ok=True)

        if not os.path.exists(mut_path):
            print(f'{name}: library not found at {mut_path}, skipping')
            continue

        if os.path.exists(maps_path) and os.path.exists(preds_path):
            print(f'{name}: attribution maps + predictions exist, skipping')
            continue

        print(f'\n{"="*50}')
        print(f'{name}: DeepLIFT/SHAP attributions{" (" + source + ")" if source else ""}')
        print(f'{"="*50}')

        _attribute_one_library(mut_path, maps_path, preds_path, model, count_model, device)


# =========================================================================
# Step 2: K-means Clustering + CSM
# =========================================================================
def _cosine_similarity(a, b):
    """Cosine similarity between two flat vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-12:
        return 0.0
    return float(dot / norm)


def step_cluster(loci, k):
    """Cluster attribution maps, compute CSM, mechanistic diversity, and functional evolvability."""
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

        # Mechanistic diversity: 1 - cos_sim(cluster_avg_map, wt_cluster_avg_map)
        wt_cluster = labels[0]
        wt_members = labels == wt_cluster
        wt_avg_map = maps_flat[wt_members].mean(axis=0)  # flat avg

        cluster_cos_sim = {}
        cluster_mech_div = {}
        for c in cluster_ids:
            members = labels == c
            cluster_avg = maps_flat[members].mean(axis=0)
            cs = _cosine_similarity(cluster_avg, wt_avg_map)
            cluster_cos_sim[c] = cs
            cluster_mech_div[c] = 1.0 - cs

        # Save
        np.save(labels_path, labels)
        np.save(os.path.join(locus_dir, 'csm_matrix.npy'), pct_mismatch)

        # Cluster info CSV with mechanistic diversity + functional evolvability
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
                'cos_sim_to_wt': round(cluster_cos_sim[c], 6),
                'mech_diversity': round(cluster_mech_div[c], 6),
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

        # ── 1. Mechanistic Diversity vs Functional Evolvability Scatter ──
        if os.path.exists(maps_path) and os.path.exists(preds_path):
            maps = np.load(maps_path)  # (N, 2114, 4)
            maps_flat = maps.reshape(maps.shape[0], -1)
            preds_all = np.load(preds_path)

            wt_members = labels == wt_cluster
            wt_avg = maps_flat[wt_members].mean(axis=0)

            mech_div = []
            mean_pred = []
            n_seqs = []
            is_wt = []
            for c in cluster_ids:
                members = labels == c
                cluster_avg = maps_flat[members].mean(axis=0)
                cs = _cosine_similarity(cluster_avg, wt_avg)
                mech_div.append(1.0 - cs)
                mean_pred.append(float(preds_all[members].mean()))
                n_seqs.append(int(members.sum()))
                is_wt.append(c == wt_cluster)

            del maps, maps_flat

            fig, ax = plt.subplots(figsize=(8, 6))
            md = np.array(mech_div)
            mp = np.array(mean_pred)
            ns = np.array(n_seqs)
            iw = np.array(is_wt)

            # Center x-axis on WT activity
            wt_pred_val = float(mp[iw][0]) if iw.any() else float(mp.mean())
            mp_centered = mp - wt_pred_val

            ax.scatter(mp_centered[~iw], md[~iw], s=8, alpha=0.5, c='steelblue',
                      edgecolors='k', linewidth=0.2, label='Non-WT clusters')
            if iw.any():
                ax.scatter(mp_centered[iw], md[iw], s=40, c='red', marker='*',
                          edgecolors='k', linewidth=0.3, zorder=5, label='WT cluster')

            xlim = max(abs(mp_centered.min()), abs(mp_centered.max())) * 1.1
            ax.set_xlim(-xlim, xlim)
            ax.axvline(x=0, color='red', linestyle='--', linewidth=0.8, alpha=0.4)
            ax.set_xlabel('Functional Evolvability (pred. activity - WT)', fontsize=12)
            ax.set_ylabel('Mechanistic Diversity (1 - cos sim to WT)', fontsize=12)
            ax.set_title(f'{name} — Func. Evolvability vs Mech. Diversity (k={k})', fontsize=13)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(locus_out_dir, f'diversity_evolvability_k{k}.png'),
                        dpi=150, bbox_inches='tight')
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
            import logomaker

            maps = np.load(maps_path, mmap_mode='r')
            wt_map = maps[0]  # (2114, 4)

            fig = plt.figure(figsize=(40, max(5, 2.5 + k * 0.005)))
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.5], hspace=0.35)

            # WT logo (sequence logo from attribution scores)
            ax_wt = fig.add_subplot(gs[0])
            logo_df = pd.DataFrame(wt_map, columns=['A', 'C', 'G', 'T'])
            logomaker.Logo(logo_df, ax=ax_wt, color_scheme='classic')
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

    # ── Mechanistic Diversity vs Functional Evolvability (all loci) ──
    scatter_rows = []
    for _, row in all_loci.iterrows():
        name = row['name']
        maps_path = os.path.join(DEEPSHAP_DIR, f'maps_{name}_{NUM_SEQS}.npy')
        preds_path = os.path.join(DEEPSHAP_DIR, f'preds_{name}_{NUM_SEQS}.npy')
        labels_path = os.path.join(CLUSTER_DIR, name, f'k{k}', 'cluster_labels.npy')
        if not all(os.path.exists(p) for p in [maps_path, preds_path, labels_path]):
            continue

        maps = np.load(maps_path)
        maps_flat = maps.reshape(maps.shape[0], -1)
        preds = np.load(preds_path)
        labels = np.load(labels_path)
        cluster_ids = sorted(np.unique(labels))
        wt_cluster = labels[0]
        wt_avg = maps_flat[labels == wt_cluster].mean(axis=0)

        for c in cluster_ids:
            members = labels == c
            cluster_avg = maps_flat[members].mean(axis=0)
            cs = _cosine_similarity(cluster_avg, wt_avg)
            scatter_rows.append({
                'locus': name,
                'cluster': c,
                'mech_diversity': 1.0 - cs,
                'mean_pred': float(preds[members].mean()),
                'n_seqs': int(members.sum()),
                'is_wt': c == wt_cluster,
            })
        del maps, maps_flat
        gc.collect()

    if scatter_rows:
        sdf = pd.DataFrame(scatter_rows)
        sdf.to_csv(os.path.join(FINAL_RESULTS_DIR, f'diversity_evolvability_k{k}.csv'), index=False)

        locus_names = sdf['locus'].unique()
        cmap = plt.cm.get_cmap('tab20', len(locus_names))
        locus_colors = {n: cmap(i) for i, n in enumerate(locus_names)}

        # Center each locus's predictions on its WT activity
        sdf['pred_centered'] = 0.0
        for locus_name in locus_names:
            mask = sdf['locus'] == locus_name
            wt_mask = mask & (sdf['is_wt'] == True)
            wt_val = sdf.loc[wt_mask, 'mean_pred'].values[0] if wt_mask.any() else sdf.loc[mask, 'mean_pred'].mean()
            sdf.loc[mask, 'pred_centered'] = sdf.loc[mask, 'mean_pred'] - wt_val

        fig, ax = plt.subplots(figsize=(10, 8))
        for locus_name in locus_names:
            ld = sdf[sdf['locus'] == locus_name]
            color = locus_colors[locus_name]
            non_wt = ld[ld['is_wt'] == False]
            wt = ld[ld['is_wt'] == True]

            ax.scatter(non_wt['pred_centered'], non_wt['mech_diversity'],
                      s=6, alpha=0.4, c=[color],
                      edgecolors='none', label=locus_name)
            if not wt.empty:
                ax.scatter(wt['pred_centered'].values, wt['mech_diversity'].values,
                          s=30, c=[color], marker='*', edgecolors='k', linewidth=0.3, zorder=5)

        xlim = max(abs(sdf['pred_centered'].min()), abs(sdf['pred_centered'].max())) * 1.1
        ax.set_xlim(-xlim, xlim)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=0.8, alpha=0.4)
        ax.set_xlabel('Functional Evolvability (pred. activity - WT)', fontsize=12)
        ax.set_ylabel('Mechanistic Diversity (1 - cos sim to WT)', fontsize=12)
        ax.set_title(f'Func. Evolvability vs Mech. Diversity — All Loci (k={k})', fontsize=13)
        ax.legend(fontsize=6, ncol=3, loc='best', markerscale=0.5)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FINAL_RESULTS_DIR, f'diversity_evolvability_all_loci_k{k}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f'Final results saved to {FINAL_RESULTS_DIR}')


# =========================================================================
# Step 5: Variant Injection — merge variant maps/preds into the 25k SEAM
#         library, replacing random non-WT sequences, then cluster + plot
# =========================================================================
def step_inject(loci, k, source, seed=42):
    """Inject variant attribution maps into the 25k mutagenesis library.

    For each locus with both a mutagenesis library and variant_libs/{source}/ data:
      - Load the 25k maps, preds, and x_mut arrays
      - Load the variant maps, preds from variant_libs/{source}/
      - Replace randomly-chosen non-WT sequences with variant sequences
      - Save injected arrays and a mapping CSV to cluster_results/variant_inject/{source}/{name}/k{k}/
      - Run k-means clustering + CSM on the injected library
    """
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(seed)
    src_dir = os.path.join(VARIANT_LIBS_DIR, source)
    inject_base = os.path.join(INJECT_CLUSTER_DIR, source)
    os.makedirs(inject_base, exist_ok=True)

    for _, row in loci.iterrows():
        name = row['name']

        # Paths for the 25k mutagenesis library
        maps_path = os.path.join(DEEPSHAP_DIR, f'maps_{name}_{NUM_SEQS}.npy')
        preds_path = os.path.join(DEEPSHAP_DIR, f'preds_{name}_{NUM_SEQS}.npy')
        mut_path = os.path.join(MUTLIB_DIR, f'x_mut_{name}_{NUM_SEQS}.npy')

        # Paths for variant library
        var_maps_path = os.path.join(src_dir, f'maps_{name}.npy')
        var_preds_path = os.path.join(src_dir, f'preds_{name}.npy')
        var_meta_path = os.path.join(src_dir, f'x_var_{name}_metadata.csv')

        # Check all inputs exist
        missing = []
        for p, desc in [(maps_path, '25k maps'), (preds_path, '25k preds'),
                        (mut_path, '25k x_mut'), (var_maps_path, 'variant maps'),
                        (var_preds_path, 'variant preds'), (var_meta_path, 'variant metadata')]:
            if not os.path.exists(p):
                missing.append(desc)
        if missing:
            print(f'{name}: missing {", ".join(missing)}, skipping')
            continue

        locus_dir = os.path.join(inject_base, name, f'k{k}')
        labels_path = os.path.join(locus_dir, 'cluster_labels.npy')
        if os.path.exists(labels_path):
            print(f'{name}: injected cluster results exist ({source}, k={k}), skipping')
            continue

        print(f'\n{"="*50}')
        print(f'{name}: variant injection ({source}, k={k})')
        print(f'{"="*50}')

        os.makedirs(locus_dir, exist_ok=True)

        # Load data
        maps_25k = np.load(maps_path)      # (25000, 2114, 4)
        preds_25k = np.load(preds_path)     # (25000,)
        x_mut_25k = np.load(mut_path)       # (25000, 2114, 4)

        var_maps = np.load(var_maps_path)   # (V, 2114, 4)
        var_preds = np.load(var_preds_path) # (V,)
        var_meta = pd.read_csv(var_meta_path)

        # Variant sequences start at index 1 (index 0 is WT in variant lib)
        n_variants = var_maps.shape[0] - 1  # exclude WT row
        if n_variants <= 0:
            print(f'{name}: no variant sequences found, skipping')
            continue

        var_maps_only = var_maps[1:]        # (n_variants, 2114, 4)
        var_preds_only = var_preds[1:]      # (n_variants,)

        # Choose random non-WT indices to replace (indices 1..24999)
        available_indices = np.arange(1, maps_25k.shape[0])
        replace_indices = rng.choice(available_indices, size=n_variants, replace=False)
        replace_indices.sort()

        # Create injected copies
        inj_maps = maps_25k.copy()
        inj_preds = preds_25k.copy()
        inj_xmut = x_mut_25k.copy()

        inj_maps[replace_indices] = var_maps_only
        inj_preds[replace_indices] = var_preds_only
        # For x_mut, use the variant one-hot sequences (from x_var file)
        var_xmut = np.load(os.path.join(src_dir, f'x_var_{name}.npy'))
        inj_xmut[replace_indices] = var_xmut[1:]  # skip WT

        print(f'  Injected {n_variants} variants at indices (first 5): {replace_indices[:5]}')

        # Save injection mapping
        inject_meta = var_meta[var_meta['source'] == 'variant'].copy()
        inject_meta['injected_idx'] = replace_indices
        inject_meta.to_csv(os.path.join(locus_dir, 'inject_mapping.csv'), index=False)

        # --- K-means clustering on injected library ---
        maps_flat = inj_maps.reshape(inj_maps.shape[0], -1)
        print(f'  Clustering {maps_flat.shape[0]} maps into {k} clusters...')
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
        labels = km.fit_predict(maps_flat)

        # CSM: percent mismatch from WT
        x_ref = inj_xmut[0]  # WT is index 0
        mismatch_mask = np.any(inj_xmut != x_ref[np.newaxis, :, :], axis=2)  # (N, 2114)
        cluster_ids = sorted(np.unique(labels))
        pct_mismatch = np.zeros((len(cluster_ids), SEQ_LENGTH))
        for ci, c in enumerate(cluster_ids):
            members = labels == c
            if np.sum(members) > 0:
                pct_mismatch[ci] = mismatch_mask[members].mean(axis=0) * 100

        # Mechanistic diversity
        wt_cluster = labels[0]
        wt_members = labels == wt_cluster
        wt_avg_map = maps_flat[wt_members].mean(axis=0)

        cluster_cos_sim = {}
        cluster_mech_div = {}
        for c in cluster_ids:
            members = labels == c
            cluster_avg = maps_flat[members].mean(axis=0)
            cs = _cosine_similarity(cluster_avg, wt_avg_map)
            cluster_cos_sim[c] = cs
            cluster_mech_div[c] = 1.0 - cs

        # Save cluster results
        np.save(labels_path, labels)
        np.save(os.path.join(locus_dir, 'csm_matrix.npy'), pct_mismatch)

        # Cluster info CSV
        info_rows = []
        for ci, c in enumerate(cluster_ids):
            members = labels == c
            n = int(np.sum(members))
            r = {
                'cluster': int(c),
                'n_seqs': n,
                'pct_of_total': round(100 * n / len(labels), 2),
                'has_wt': c == wt_cluster,
                'mean_mismatch_pct': round(float(pct_mismatch[ci].mean()), 2),
                'cos_sim_to_wt': round(cluster_cos_sim[c], 6),
                'mech_diversity': round(cluster_mech_div[c], 6),
                'mean_pred': round(float(inj_preds[members].mean()), 6),
                'std_pred': round(float(inj_preds[members].std()), 6),
            }
            info_rows.append(r)

        pd.DataFrame(info_rows).to_csv(os.path.join(locus_dir, 'cluster_info.csv'), index=False)

        # Build a cluster-index lookup for CSM rows
        cluster_to_ci = {c: ci for ci, c in enumerate(cluster_ids)}

        # Per-variant results: which cluster each variant landed in
        # Compute SNP masks for all variants (where variant differs from WT)
        x_wt = inj_xmut[0]  # (2114, 4)
        variant_rows = []
        for i, idx in enumerate(replace_indices):
            c = int(labels[idx])
            # SNP mask: positions where this variant differs from WT
            snp_mask = np.any(inj_xmut[idx] != x_wt, axis=1).astype(float)  # (2114,)
            # Mechanistic causality: CSM (as fraction 0-1) dotted with SNP mask
            ci = cluster_to_ci[c]
            mech_causality = float(np.sum((pct_mismatch[ci] / 100.0) * snp_mask))
            variant_rows.append({
                'variant_idx': int(idx),
                'cluster': c,
                'mech_diversity': cluster_mech_div[c],
                'mech_causality': mech_causality,
                'pred': float(inj_preds[idx]),
                'wt_pred': float(inj_preds[0]),
                'log2fc': float((inj_preds[idx] - inj_preds[0]) / np.log(2)),
            })
        var_results = pd.DataFrame(variant_rows)
        # Merge with variant metadata
        inject_meta_reset = inject_meta.reset_index(drop=True)
        var_results = pd.concat([inject_meta_reset, var_results], axis=1)
        var_results.to_csv(os.path.join(locus_dir, 'variant_results.csv'), index=False)

        print(f'  WT cluster={wt_cluster}, variants span {len(set(labels[replace_indices]))} clusters')
        print(f'  Saved to {locus_dir}')

        del inj_maps, inj_preds, inj_xmut, maps_flat, maps_25k, preds_25k, x_mut_25k
        gc.collect()


def step_inject_final_plots(k, sources=None):
    """Generate cross-locus plots for variant injection results.

    - GnomAD: Mechanistic Diversity vs Allele Frequency
    - caQTL: ChromBPNet predicted effect vs measured caQTL coefficient (target)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(INJECT_RESULTS_DIR, exist_ok=True)

    if sources is None:
        sources = ['gnomad', 'caqtl_eur', 'caqtl_afr']

    # ── GnomAD: Mechanistic Diversity vs Allele Frequency ──
    if 'gnomad' in sources:
        _plot_gnomad_mech_vs_af(k)

    # ── caQTL: Predicted vs Actual ──
    for src in ['caqtl_eur', 'caqtl_afr']:
        if src in sources:
            _plot_caqtl_pred_vs_actual(k, src)


def _plot_gnomad_mech_vs_af(k):
    """Mechanistic Diversity vs Allele Frequency for GnomAD variants."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    inject_base = os.path.join(INJECT_CLUSTER_DIR, 'gnomad')
    all_loci = load_loci()
    rows = []

    for _, row in all_loci.iterrows():
        name = row['name']
        vr_path = os.path.join(inject_base, name, f'k{k}', 'variant_results.csv')
        if not os.path.exists(vr_path):
            continue
        vr = pd.read_csv(vr_path)
        vr['locus'] = name
        rows.append(vr)

    if not rows:
        print('GnomAD: no variant injection results found')
        return

    df = pd.concat(rows, ignore_index=True)
    # Filter to variants with valid AF
    df = df[df['AF'].notna() & (df['AF'] > 0)].copy()
    if df.empty:
        print('GnomAD: no variants with valid AF > 0')
        return

    locus_names = df['locus'].unique()
    cmap = plt.cm.get_cmap('tab20', max(len(locus_names), 1))
    locus_colors = {n: cmap(i) for i, n in enumerate(locus_names)}

    fig, ax = plt.subplots(figsize=(10, 7))
    for locus_name in locus_names:
        ld = df[df['locus'] == locus_name]
        ax.scatter(ld['AF'], ld['mech_diversity'],
                   s=12, alpha=0.6, c=[locus_colors[locus_name]],
                   edgecolors='k', linewidth=0.2, label=locus_name)

    # Use linear scale but show 1/X style tick labels
    ax.set_xscale('log')
    af_ticks = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    af_tick_labels = ['1', '1/10', '1/100', '1/1K', '1/10K', '1/100K', '1/1M']
    ax.set_xticks(af_ticks)
    ax.set_xticklabels(af_tick_labels)
    ax.set_xlabel('Allele Frequency', fontsize=12)
    ax.set_ylabel('Mechanistic Diversity (1 - cos sim to WT)', fontsize=12)
    ax.set_title(f'Mechanistic Diversity vs Allele Frequency — GnomAD (k={k})', fontsize=13)
    ax.legend(fontsize=7, ncol=3, loc='best', markerscale=1.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(INJECT_RESULTS_DIR, f'gnomad_mech_diversity_vs_AF_k{k}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved GnomAD plot to {out_path}')

    # ── Mechanistic Causality vs Allele Frequency ──
    if 'mech_causality' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 7))
        for locus_name in locus_names:
            ld = df[df['locus'] == locus_name]
            ax.scatter(ld['AF'], ld['mech_causality'],
                       s=12, alpha=0.6, c=[locus_colors[locus_name]],
                       edgecolors='k', linewidth=0.2, label=locus_name)

        ax.set_xscale('log')
        ax.set_xticks(af_ticks)
        ax.set_xticklabels(af_tick_labels)
        ax.set_xlabel('Allele Frequency', fontsize=12)
        ax.set_ylabel('Mechanistic Causality (CSM · SNP mask)', fontsize=12)
        ax.set_title(f'Mechanistic Causality vs Allele Frequency — GnomAD (k={k})', fontsize=13)
        ax.legend(fontsize=7, ncol=3, loc='best', markerscale=1.5)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path2 = os.path.join(INJECT_RESULTS_DIR, f'gnomad_mech_causality_vs_AF_k{k}.png')
        plt.savefig(out_path2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved GnomAD causality plot to {out_path2}')

    # Save data
    df.to_csv(os.path.join(INJECT_RESULTS_DIR, f'gnomad_variant_results_k{k}.csv'), index=False)


def _plot_caqtl_pred_vs_actual(k, source):
    """ChromBPNet predicted effect vs measured caQTL coefficient."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    inject_base = os.path.join(INJECT_CLUSTER_DIR, source)
    all_loci = load_loci()

    # Load coefficient data
    coeff_path = CAQTL_COEFF_FILES.get(source)
    if not coeff_path or not os.path.exists(coeff_path):
        print(f'{source}: coefficient file not found at {coeff_path}')
        return
    coeff_df = pd.read_feather(coeff_path)
    coeff_lookup = coeff_df.set_index('variant_id')['target'].to_dict()

    rows = []
    for _, row in all_loci.iterrows():
        name = row['name']
        vr_path = os.path.join(inject_base, name, f'k{k}', 'variant_results.csv')
        if not os.path.exists(vr_path):
            continue
        vr = pd.read_csv(vr_path)
        vr['locus'] = name
        # Match variant_id to coefficient target
        vr['target'] = vr['variant_id'].map(coeff_lookup)
        rows.append(vr)

    if not rows:
        print(f'{source}: no variant injection results found')
        return

    df = pd.concat(rows, ignore_index=True)
    df_valid = df[df['target'].notna()].copy()
    if df_valid.empty:
        print(f'{source}: no variants matched coefficient data')
        return

    locus_names = df_valid['locus'].unique()
    cmap = plt.cm.get_cmap('tab20', max(len(locus_names), 1))
    locus_colors = {n: cmap(i) for i, n in enumerate(locus_names)}

    # log2fc = ChromBPNet predicted effect (variant - WT)
    # target = measured caQTL coefficient
    fig, ax = plt.subplots(figsize=(8, 8))
    for locus_name in locus_names:
        ld = df_valid[df_valid['locus'] == locus_name]
        ax.scatter(ld['target'], ld['log2fc'],
                   s=15, alpha=0.6, c=[locus_colors[locus_name]],
                   edgecolors='k', linewidth=0.2, label=locus_name)

    # Add identity line
    all_vals = np.concatenate([df_valid['target'].values, df_valid['log2fc'].values])
    vmin, vmax = all_vals.min(), all_vals.max()
    margin = (vmax - vmin) * 0.05
    ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
            'r--', linewidth=1, alpha=0.5, label='y=x')

    # Compute correlation
    from scipy.stats import pearsonr, spearmanr
    r_pearson, p_pearson = pearsonr(df_valid['target'], df_valid['log2fc'])
    r_spearman, p_spearman = spearmanr(df_valid['target'], df_valid['log2fc'])
    ax.text(0.05, 0.95,
            f'Pearson r={r_pearson:.3f} (p={p_pearson:.2e})\nSpearman ρ={r_spearman:.3f} (p={p_spearman:.2e})',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    label = 'EUR' if source == 'caqtl_eur' else 'AFR'
    ax.set_xlabel(f'Measured caQTL Coefficient ({label})', fontsize=12)
    ax.set_ylabel('ChromBPNet log2FC', fontsize=12)
    ax.set_title(f'Predicted vs Actual caQTL Effect — {label} (k={k})', fontsize=13)
    ax.legend(fontsize=7, ncol=2, loc='best', markerscale=1.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(INJECT_RESULTS_DIR, f'{source}_pred_vs_actual_k{k}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {source} plot to {out_path}')

    # Save data
    df_valid.to_csv(os.path.join(INJECT_RESULTS_DIR, f'{source}_variant_results_k{k}.csv'), index=False)


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description='SEAM pipeline for ChromBPNet LCL population analysis')
    parser.add_argument('--step', required=True,
                        choices=['all', 'attribute', 'cluster', 'results', 'final', 'inject'],
                        help='Pipeline step to run')
    parser.add_argument('--locus', default=None,
                        help='Comma-separated locus names (default: all 34)')
    parser.add_argument('--k', type=int, default=K_DEFAULT,
                        help=f'Number of clusters for k-means (default: {K_DEFAULT})')
    parser.add_argument('--source', default=None,
                        choices=['gnomad', 'caqtl_eur', 'caqtl_afr'],
                        help='Variant source for attribute step (uses variant_libs/{source}/)')
    parser.add_argument('--inject', default=None, nargs='+',
                        choices=['gnomad', 'caqtl_eur', 'caqtl_afr'],
                        help='Inject variant maps into 25k SEAM library before clustering. '
                             'Specify one or more sources (e.g. --inject gnomad caqtl_eur)')
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
        step_attribute(loci, device=args.device, source=args.source)

    if args.step in ('all', 'cluster'):
        print(f'\n>>> Step 2: K-means Clustering + CSM (k={args.k}) <<<')
        step_cluster(loci, args.k)

    if args.step in ('all', 'results'):
        print(f'\n>>> Step 3: Per-locus Results <<<')
        step_seq_results(loci, args.k)

    if args.step in ('all', 'final'):
        print(f'\n>>> Step 4: Cross-locus Final Results <<<')
        step_final_results(args.k)

    if args.step == 'inject' or args.inject:
        inject_sources = args.inject if args.inject else ['gnomad', 'caqtl_eur', 'caqtl_afr']
        for src in inject_sources:
            print(f'\n>>> Variant Injection: {src} (k={args.k}) <<<')
            step_inject(loci, args.k, src)
        print(f'\n>>> Variant Injection Final Plots <<<')
        step_inject_final_plots(args.k, sources=inject_sources)

    print('\nPipeline complete.')


if __name__ == '__main__':
    main()
