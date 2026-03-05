#!/usr/bin/env python
"""
Full SEAM pipeline for AlphaGenome eQTL LCL variant analysis.

Adapted from gnomAD 25-loci pipeline. Instead of gnomAD SNVs, injects
EBV-transformed lymphocyte eQTL variants from AlphaGenome evaluation data.

Steps per locus:
  1. Extract 1000bp sequence from hg38, twohot encode
  2. Generate mutagenesis library (25K seqs, 1% mutation rate) with eQTL SNVs injected
  3. DeepSHAP attribution maps (9-fold CLIPNET, 50 dinuc shuffles)
  4. K-means clustering on attribution maps
  5. CSM percent-mismatch from WT

Usage:
  python run_seam_pipeline.py --step all
  python run_seam_pipeline.py --step extract
  python run_seam_pipeline.py --step attribute --locus IRF7
  python run_seam_pipeline.py --step cluster --k 5
"""

import os
import sys
import gc
import glob
import argparse
import logging

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = '/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis/SEAM_CLIPNET/LCL_variants_analysis'
REVO_DIR = '/grid/wsbs/home_norepl/pmantill/SEAM_revisions/SEAM_revisions/seam+REVO_exploration'

DATA_DIR = os.path.join(BASE_DIR, 'data')
ATTRIBUTION_DIR = os.path.join(BASE_DIR, 'DeepSHAP_maps')
RESULTS_DIR = os.path.join(BASE_DIR, 'SEAM_results')
MODEL_DIR = os.path.join(REVO_DIR, 'pytorch_test_run', 'clipnet_models')
GENOME_FASTA = os.path.join(REVO_DIR, 'pytorch_test_run', 'hg38_genome', 'hg38.fa')

# AlphaGenome eQTL variant data (Cells_EBV-transformed_lymphocytes)
EQTL_DATA_DIR = '/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis/variant_data/Alphagenome_data/eqtl_variants'

# ── Loci ───────────────────────────────────────────────────────────────────
LOCI = pd.DataFrame({
    'name':     ['IRF7', 'HLA-A', 'HLA-B', 'HLA-C', 'HLA-G',
                 'HOXA1', 'HOXA13', 'HOXC13', 'B-ACTIN', 'TBP', 'GAPDH',
                 'YAP1', 'TAZ', 'PIK3R3', 'MYC', 'TNF', 'BCL2',
                 'KRAS', 'EGFR', 'ERBB2', 'PIK3CA', 'CCND1', 'BRAF', 'VEGFA', 'MDM2'],
    'chrom':    ['chr11', 'chr6', 'chr6', 'chr6', 'chr6',
                 'chr7', 'chr7', 'chr12', 'chr7', 'chr6', 'chr12',
                 'chr11', 'chr3', 'chr1', 'chr8', 'chr6', 'chr18',
                 'chr12', 'chr7', 'chr17', 'chr3', 'chr11', 'chr7', 'chr6', 'chr12'],
    'tss':      [616000, 29942532, 31357179, 31272092, 29827825,
                 27095025, 27209044, 53976181, 5530601, 170554302, 6534512,
                 102110461, 149658025, 46132640, 127736231, 31575565, 63319769,
                 25250929, 55019017, 39700064, 179148357, 69641156, 140924929, 43770211, 68808177],
    'category': ['Oncogene', 'HLA', 'HLA', 'HLA', 'HLA',
                 'Hox', 'Hox', 'Hox', 'Housekeeping', 'Housekeeping', 'Housekeeping',
                 'Oncogene', 'Oncogene', 'Oncogene', 'Oncogene', 'Oncogene', 'Oncogene',
                 'Oncogene', 'Oncogene', 'Oncogene', 'Oncogene', 'Oncogene', 'Oncogene', 'Oncogene', 'Oncogene'],
})
LOCI['start'] = LOCI['tss'] - 500
LOCI['end']   = LOCI['tss'] + 500

# ── Constants ──────────────────────────────────────────────────────────────
NUM_SEQS = 25000
MUT_RATE = 0.01
N_FOLDS = 9
NUM_SHUFS = 50
BATCH_SIZE = 256

# ── TwoHot encoding ───────────────────────────────────────────────────────
TWOHOT_MAP = {
    'A': np.array([2,0,0,0]), 'C': np.array([0,2,0,0]),
    'G': np.array([0,0,2,0]), 'T': np.array([0,0,0,2]),
    'N': np.array([0,0,0,0]),
}
TWOHOT_IUPAC = {
    **TWOHOT_MAP,
    'M': np.array([1,1,0,0]), 'R': np.array([1,0,1,0]),
    'W': np.array([1,0,0,1]), 'S': np.array([0,1,1,0]),
    'Y': np.array([0,1,0,1]), 'K': np.array([0,0,1,1]),
}

def twohot_encode(seq):
    return np.array([TWOHOT_MAP[c] for c in seq.upper()])

def twohot_encode_iupac(seq):
    return np.array([TWOHOT_IUPAC[c] for c in seq.upper()])

def twohot2seq(twohot):
    rev = {tuple(v): k for k, v in TWOHOT_IUPAC.items()}
    return ''.join(rev[tuple(twohot[i])] for i in range(twohot.shape[0]))


# =========================================================================
# Step 1: Extract sequences
# =========================================================================
def step_extract(loci):
    import pysam
    os.makedirs(DATA_DIR, exist_ok=True)

    npz_path = os.path.join(DATA_DIR, 'x_twohot_25loci.npz')
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        existing = set(data.files)
        needed = set(loci['name']) - existing
        if not needed:
            print(f'All {len(loci)} loci already extracted in {npz_path}')
            return
        print(f'{len(needed)} new loci to extract: {needed}')
        x_twohot = {name: data[name] for name in data.files}
    else:
        x_twohot = {}

    fasta = pysam.Fastafile(GENOME_FASTA)
    for _, row in loci.iterrows():
        if row['name'] in x_twohot:
            continue
        seq = fasta.fetch(row['chrom'], row['start'], row['end']).upper()
        assert len(seq) == 1000, f"{row['name']}: expected 1000bp, got {len(seq)}"
        x_twohot[row['name']] = twohot_encode(seq)

    np.savez(npz_path, **x_twohot)
    print(f'Saved {len(x_twohot)} loci to {npz_path}')


# =========================================================================
# Step 2: Mutagenesis library (with eQTL LCL SNVs injected)
# =========================================================================

_BASE_TO_TWOHOT = {
    'A': np.array([2,0,0,0]), 'C': np.array([0,2,0,0]),
    'G': np.array([0,0,2,0]), 'T': np.array([0,0,0,2]),
}


def _parse_variant_id(variant_id):
    """Parse AlphaGenome variant_id like 'chr6_31575565_A_G_b38' into chrom, pos, ref, alt."""
    parts = variant_id.split('_')
    chrom = parts[0]
    pos = int(parts[1])
    ref = parts[2]
    alt = parts[3]
    return chrom, pos, ref, alt


def load_eqtl_lcl_snvs(name, locus_chrom, locus_start, locus_end):
    """Load eQTL SNVs for Cells_EBV-transformed_lymphocytes that fall within a locus window.

    Searches all eQTL feather files for variants in the locus region,
    filtered to Cells_EBV-transformed_lymphocytes tissue.
    Returns DataFrame with pos, ref, alt, variant_id, prediction, target, gene_id.
    """
    all_snvs = []

    for fname in os.listdir(EQTL_DATA_DIR):
        if not fname.endswith('.feather'):
            continue
        fpath = os.path.join(EQTL_DATA_DIR, fname)
        df = pd.read_feather(fpath)

        # Filter to LCL tissue
        lcl = df[df['tissue'] == 'Cells_EBV-transformed_lymphocytes'].copy()
        if lcl.empty:
            continue

        # Parse variant_id to get genomic coords
        parsed = lcl['variant_id'].apply(_parse_variant_id)
        lcl['chrom'] = parsed.apply(lambda x: x[0])
        lcl['pos'] = parsed.apply(lambda x: x[1])
        lcl['ref'] = parsed.apply(lambda x: x[2])
        lcl['alt'] = parsed.apply(lambda x: x[3])

        # Filter to SNVs in the locus window
        mask = (
            (lcl['chrom'] == locus_chrom) &
            (lcl['pos'] >= locus_start) &
            (lcl['pos'] < locus_end) &
            (lcl['ref'].str.len() == 1) &
            (lcl['alt'].str.len() == 1)
        )
        hits = lcl[mask]
        if not hits.empty:
            all_snvs.append(hits)

    if all_snvs:
        combined = pd.concat(all_snvs, ignore_index=True)
        # Deduplicate by variant_id (same SNP may appear in multiple eQTL files)
        # Keep the row with the highest absolute prediction score
        combined['abs_pred'] = combined['prediction'].abs()
        combined = combined.sort_values('abs_pred', ascending=False).drop_duplicates(
            subset='variant_id', keep='first'
        ).drop(columns='abs_pred')
        return combined

    return pd.DataFrame()


def create_variant_sequences(x_ref, snvs, locus_start):
    """Create mutant sequences by introducing each eQTL SNV into the WT."""
    variant_seqs = []
    for _, snv in snvs.iterrows():
        idx = int(snv['pos']) - locus_start
        if idx < 0 or idx >= 1000:
            continue
        alt = snv['alt'].upper()
        if alt not in _BASE_TO_TWOHOT:
            continue
        seq = x_ref.copy()
        seq[idx] = _BASE_TO_TWOHOT[alt]
        variant_seqs.append(seq)

    if variant_seqs:
        return np.array(variant_seqs)
    return np.empty((0, 1000, 4), dtype=x_ref.dtype)


def step_mutagenize(loci):
    import squid
    os.makedirs(DATA_DIR, exist_ok=True)

    ref_data = np.load(os.path.join(DATA_DIR, 'x_twohot_25loci.npz'))
    mut_generator = squid.mutagenizer.TwoHotMutagenesis(mut_rate=MUT_RATE)

    for _, row in loci.iterrows():
        name = row['name']
        outpath = os.path.join(DATA_DIR, f'x_mut_{name}_{NUM_SEQS}.npy')
        meta_path = os.path.join(DATA_DIR, f'x_mut_{name}_{NUM_SEQS}_metadata.csv')
        if os.path.exists(outpath) and os.path.exists(meta_path):
            print(f'{name}: mutagenesis library + metadata exist, skipping')
            continue

        x_ref = ref_data[name]

        # --- Load and inject eQTL LCL SNVs ---
        snvs = load_eqtl_lcl_snvs(name, row['chrom'], row['start'], row['end'])
        snv_seqs = create_variant_sequences(x_ref, snvs, row['start'])
        n_variants = snv_seqs.shape[0]
        print(f'{name}: {n_variants} eQTL LCL SNVs injected')

        # --- Build metadata ---
        # Row 0 = WT, rows 1..n_variants = eQTL SNVs, rest = random mutagenesis
        meta_rows = [{'seq_idx': 0, 'source': 'WT', 'variant_id': '', 'pos': '',
                      'ref': '', 'alt': '', 'gene_id': '', 'prediction': '',
                      'target': ''}]

        valid_idx = 0
        for _, snv in snvs.iterrows():
            idx = int(snv['pos']) - row['start']
            if idx < 0 or idx >= 1000:
                continue
            alt = snv['alt'].upper()
            if alt not in _BASE_TO_TWOHOT:
                continue
            valid_idx += 1
            meta_rows.append({
                'seq_idx': valid_idx,
                'source': 'eQTL_LCL',
                'variant_id': snv.get('variant_id', ''),
                'pos': int(snv['pos']),
                'ref': snv['ref'],
                'alt': snv['alt'],
                'gene_id': snv.get('gene_id', ''),
                'prediction': snv.get('prediction', ''),
                'target': snv.get('target', ''),
            })

        # --- Fill remainder with random mutagenesis ---
        n_random = NUM_SEQS - 1 - n_variants
        if n_random < 0:
            print(f'  WARNING: {n_variants} SNVs exceed {NUM_SEQS}, truncating random to 0')
            n_random = 0

        if n_random > 0:
            mave = squid.mave.InSilicoMAVE(
                mut_generator, mut_predictor=None,
                seq_length=1000, mut_window=None, save_window=None,
            )
            x_random, _ = mave.generate(x_ref, num_sim=n_random)
            x_random = x_random[1:]  # drop the WT duplicate
            n_random_actual = x_random.shape[0]
        else:
            x_random = np.empty((0, 1000, 4), dtype=x_ref.dtype)
            n_random_actual = 0

        for i in range(n_random_actual):
            meta_rows.append({
                'seq_idx': 1 + n_variants + i,
                'source': 'random_mutagenesis',
                'variant_id': '', 'pos': '', 'ref': '', 'alt': '',
                'gene_id': '', 'prediction': '', 'target': '',
            })

        # --- Concatenate: [WT, eQTL_SNVs, random_mutagenesis] ---
        parts = [x_ref[np.newaxis, :]]
        if n_variants > 0:
            parts.append(snv_seqs)
        if n_random_actual > 0:
            parts.append(x_random)
        x_mut = np.concatenate(parts, axis=0).astype(np.int8)

        np.save(outpath, x_mut)
        meta_df = pd.DataFrame(meta_rows)
        meta_df.to_csv(meta_path, index=False)
        print(f'{name}: saved {x_mut.shape} to {outpath}')
        print(f'  {n_variants} eQTL LCL + {n_random_actual} random + 1 WT = {x_mut.shape[0]} total')
        print(f'  Metadata: {meta_path}')


# =========================================================================
# Step 3: DeepSHAP attribution
# =========================================================================

def string_to_char_array(seq):
    return np.frombuffer(bytes(seq, 'utf8'), dtype=np.int8)

def char_array_to_string(arr):
    return arr.tobytes().decode('ascii')

def kshuffle(seq, num_shufs=1, k=2, random_seed=None):
    arr = string_to_char_array(seq)
    rng = np.random.RandomState(random_seed)
    if k == 1:
        return [char_array_to_string(rng.permutation(arr)) for _ in range(num_shufs)]
    arr_shortmers = np.empty((len(arr), k - 1), dtype=arr.dtype)
    arr_shortmers[:] = -1
    for i in range(k - 1):
        arr_shortmers[:len(arr) - i, i] = arr[i:]
    shortmers, tokens = np.unique(arr_shortmers, return_inverse=True, axis=0)
    shuf_next_inds = []
    for token in range(len(shortmers)):
        inds = np.where(tokens == token)[0]
        shuf_next_inds.append(inds + 1)
    all_results = []
    for _ in range(num_shufs):
        for t in range(len(shortmers)):
            inds = np.arange(len(shuf_next_inds[t]))
            inds[:-1] = rng.permutation(len(inds) - 1)
            shuf_next_inds[t] = shuf_next_inds[t][inds]
        counters = [0] * len(shortmers)
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]
        shuffled_arr = shortmers[result][:, 0]
        all_results.append(char_array_to_string(shuffled_arr))
    return all_results

def dinuc_shuffle(x, num_shufs=NUM_SHUFS, random_seed=1234):
    shuffled_seqs = kshuffle(twohot2seq(x[0]), num_shufs=num_shufs, k=2, random_seed=random_seed)
    return np.array([twohot_encode_iupac(seq) for seq in shuffled_seqs])


def step_attribute(loci):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
    logging.getLogger('tensorflow').setLevel(logging.FATAL)

    import tensorflow as tf
    import shap
    from shap.explainers.deep.deep_tf import op_handlers, passthrough
    op_handlers['AddV2'] = passthrough
    op_handlers['SpaceToBatchND'] = passthrough
    op_handlers['BatchToSpaceND'] = passthrough
    tf.compat.v1.disable_v2_behavior()

    os.makedirs(ATTRIBUTION_DIR, exist_ok=True)

    print(f'Loading {N_FOLDS} CLIPNET folds...')
    models = [
        tf.keras.models.load_model(os.path.join(MODEL_DIR, f'fold_{i}.h5'), compile=False)
        for i in range(1, N_FOLDS + 1)
    ]
    for i in range(len(models)):
        models[i]._name = f'model_{i}'

    contrib = [model.output[1] for model in models]
    ref_data = np.load(os.path.join(DATA_DIR, 'x_twohot_25loci.npz'))

    for _, row in loci.iterrows():
        name = row['name']
        out_path = os.path.join(ATTRIBUTION_DIR, f'maps_quantity_{name}_{NUM_SEQS}.npy')

        if os.path.exists(out_path):
            print(f'{name}: attribution maps exist, skipping')
            continue

        mut_path = os.path.join(DATA_DIR, f'x_mut_{name}_{NUM_SEQS}.npy')
        if not os.path.exists(mut_path):
            print(f'{name}: mutagenesis library not found, skipping')
            continue

        print(f'\n{"="*50}')
        print(f'{name} (quantity)')
        print(f'{"="*50}')

        x_mut = np.load(mut_path)
        x_ref = ref_data[name]
        x_ref_expanded = x_ref[np.newaxis, :]

        print(f'Creating {NUM_SHUFS} dinucleotide shuffle backgrounds...')
        background = dinuc_shuffle(x_ref_expanded, num_shufs=NUM_SHUFS)

        print(f'Initializing {N_FOLDS} DeepExplainers...')
        explainers = [
            shap.DeepExplainer((model.input, c), data=background)
            for model, c in zip(models, contrib)
        ]

        fold_results = []
        for i in range(N_FOLDS):
            ckpt_path = os.path.join(ATTRIBUTION_DIR, f'_ckpt_quantity_{name}_{NUM_SEQS}_fold{i}.npy')
            if os.path.exists(ckpt_path):
                print(f'  Fold {i+1}/{N_FOLDS}: loading checkpoint')
                fold_results.append(np.load(ckpt_path))
                continue

            fold_batches = []
            for j in range(0, x_mut.shape[0], BATCH_SIZE):
                sv = explainers[i].shap_values(x_mut[j:j + BATCH_SIZE])
                fold_batches.append(sv)
                gc.collect()

            fold_concat = np.concatenate(fold_batches, axis=1).sum(axis=0)
            np.save(ckpt_path, fold_concat)
            print(f'  Fold {i+1}/{N_FOLDS}: checkpoint saved')
            fold_results.append(fold_concat)
            gc.collect()

        attrs = np.array(fold_results).mean(axis=0)
        np.save(out_path, attrs)
        print(f'Saved {attrs.shape} to {out_path}')

        for ckpt in glob.glob(os.path.join(ATTRIBUTION_DIR, f'_ckpt_quantity_{name}_{NUM_SEQS}_fold*.npy')):
            os.remove(ckpt)

        preds_path = os.path.join(ATTRIBUTION_DIR, f'preds_quantity_{name}_{NUM_SEQS}.npy')
        if not os.path.exists(preds_path):
            print('Computing ensemble predictions...')
            fold_preds = []
            for model in models:
                batched_preds = []
                for j in range(0, len(x_mut), BATCH_SIZE):
                    out = model.predict(x_mut[j:j + BATCH_SIZE], verbose=0)
                    batched_preds.append(out[1].squeeze())
                fold_preds.append(np.concatenate(batched_preds))
            preds = np.mean(fold_preds, axis=0)
            np.save(preds_path, preds)
            print(f'Saved predictions {preds.shape} to {preds_path}')

        gc.collect()


# =========================================================================
# Step 4: Cluster + CSM
# =========================================================================
def step_cluster(loci, k):
    from sklearn.cluster import KMeans as KMeansCPU

    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
        import gc; gc.collect()
    except Exception:
        pass

    try:
        from kmeanstf import KMeansTF
        _has_gpu_kmeans = True
    except (ImportError, Exception):
        print("KMeansTF not available, using CPU KMeans")
        _has_gpu_kmeans = False

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    sys.path.insert(0, os.path.join(REVO_DIR, '..', 'seam_repo', 'seam-nn', 'seam'))
    from logomaker_batch.batch_logo import BatchLogo

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ref_data = np.load(os.path.join(DATA_DIR, 'x_twohot_25loci.npz'))

    def fit_kmeans(data, n_clusters):
        if _has_gpu_kmeans:
            try:
                km = KMeansTF(n_clusters=n_clusters, init='k-means++',
                              n_init=10, max_iter=300, random_state=42)
                km.fit(data.astype(np.float32))
                labels = km.labels_
                if hasattr(labels, 'numpy'):
                    labels = labels.numpy()
                return labels, float(km.inertia_)
            except Exception:
                pass
        km = KMeansCPU(n_clusters=n_clusters, init='k-means++',
                       n_init=10, max_iter=300, random_state=42)
        km.fit(data)
        return km.labels_, float(km.inertia_)

    for _, row in loci.iterrows():
        name = row['name']
        maps_path = os.path.join(ATTRIBUTION_DIR, f'maps_quantity_{name}_{NUM_SEQS}.npy')
        mut_path = os.path.join(DATA_DIR, f'x_mut_{name}_{NUM_SEQS}.npy')

        if not os.path.exists(maps_path):
            print(f'{name}: attribution maps not found, skipping')
            continue

        print(f'\n{"="*50}')
        print(f'{name} (k={k})')
        print(f'{"="*50}')

        locus_dir = os.path.join(RESULTS_DIR, name, f'k{k}')
        os.makedirs(locus_dir, exist_ok=True)

        maps = np.load(maps_path)
        if maps.ndim == 3 and maps.shape[0] == 4 and maps.shape[2] != 4:
            maps = maps.transpose(1, 2, 0)

        x_mut = np.load(mut_path)
        x_ref = ref_data[name]

        maps_flat = maps.reshape(maps.shape[0], -1)
        print(f'Clustering {maps_flat.shape[0]} maps into {k} clusters...')
        labels, inertia = fit_kmeans(maps_flat, n_clusters=k)

        mismatch_mask = np.any(x_mut != x_ref[np.newaxis, :, :], axis=2)
        cluster_ids = sorted(np.unique(labels))
        L = x_mut.shape[1]
        pct_mismatch = np.zeros((len(cluster_ids), L))
        for ci, c in enumerate(cluster_ids):
            members = labels == c
            if np.sum(members) > 0:
                pct_mismatch[ci] = mismatch_mask[members].mean(axis=0) * 100

        np.save(os.path.join(locus_dir, f'cluster_labels.npy'), labels)
        np.save(os.path.join(locus_dir, f'csm_matrix.npy'), pct_mismatch)

        wt_map = maps[0]
        input_mask = (x_ref > 0).astype(np.float32)
        wt_logo = wt_map * input_mask

        fig = plt.figure(figsize=(20, max(5, 2.5 + k * 0.5)))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, max(1.5, k * 0.35)], hspace=0.35)

        ax_wt = fig.add_subplot(gs[0])
        logo = BatchLogo(wt_logo[np.newaxis, :, :], figsize=[20, 2.5], show_progress=False)
        logo.process_all()
        logo.draw_single(0, ax=ax_wt, fixed_ylim=False, border=True)
        ax_wt.set_ylabel('Attribution')
        ax_wt.set_title(f'WT DeepSHAP - {name} (quantity)')
        ax_wt.set_xticklabels([])

        ax_csm = fig.add_subplot(gs[1])
        im = ax_csm.pcolormesh(pct_mismatch, cmap='viridis', vmin=0, vmax=100)
        wt_cluster = labels[0]
        ytick_labels = []
        for c in cluster_ids:
            n = int(np.sum(labels == c))
            lbl = f'C{c} (n={n})'
            if c == wt_cluster:
                lbl += ' *WT*'
            ytick_labels.append(lbl)
        ax_csm.set_yticks(np.arange(len(cluster_ids)) + 0.5)
        ax_csm.set_yticklabels(ytick_labels, fontsize=8)
        ax_csm.set_xlabel('Position')
        ax_csm.set_title(f'CSM: % Mismatch from WT (k={k})')
        plt.colorbar(im, ax=ax_csm, label='% Mismatch', shrink=0.8)
        plt.savefig(os.path.join(locus_dir, 'wt_csm.png'), dpi=200, bbox_inches='tight')
        plt.close()

        preds_path = os.path.join(ATTRIBUTION_DIR, f'preds_quantity_{name}_{NUM_SEQS}.npy')
        if os.path.exists(preds_path):
            preds = np.load(preds_path)
            wt_pred = preds[0]
            fig, ax = plt.subplots(figsize=(8, max(3, k * 0.35)))
            box_data = [preds[labels == c] for c in cluster_ids]
            bp = ax.boxplot(box_data, patch_artist=True, showfliers=False, vert=False, widths=0.6)
            for patch in bp['boxes']:
                patch.set_facecolor('steelblue')
                patch.set_alpha(0.6)
            ax.axvline(x=wt_pred, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'WT pred = {wt_pred:.4f}')
            ax.set_yticklabels(ytick_labels, fontsize=8)
            ax.set_xlabel('DNN Quantity Prediction')
            ax.set_title(f'Cluster vs Prediction - {name} (k={k})')
            ax.legend(fontsize=9, loc='lower right')
            ax.grid(True, axis='x', alpha=0.3)
            plt.savefig(os.path.join(locus_dir, 'cluster_preds.png'), dpi=150, bbox_inches='tight')
            plt.close()

        rows = []
        for ci, c in enumerate(cluster_ids):
            members = labels == c
            n = int(np.sum(members))
            r = {
                'cluster': int(c), 'n_seqs': n,
                'pct_of_total': round(100 * n / len(labels), 2),
                'has_wt': c == wt_cluster,
                'mean_mismatch_pct': round(float(pct_mismatch[ci].mean()), 2),
                'category': row['category'],
            }
            if os.path.exists(preds_path):
                r['mean_pred'] = round(float(preds[members].mean()), 6)
                r['std_pred'] = round(float(preds[members].std()), 6)
            rows.append(r)

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(locus_dir, 'cluster_info.csv'), index=False)
        print(f'Saved results to {locus_dir}')
        gc.collect()


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description='SEAM pipeline for eQTL LCL 25 loci')
    parser.add_argument('--step', required=True,
                        choices=['all', 'extract', 'mutagenize', 'attribute', 'cluster'],
                        help='Pipeline step to run')
    parser.add_argument('--locus', default=None,
                        help='Comma-separated locus names (default: all 25)')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of clusters for k-means (default: 5)')
    args = parser.parse_args()

    loci = LOCI
    if args.locus:
        names = set(args.locus.split(','))
        loci = LOCI[LOCI['name'].isin(names)]
        if loci.empty:
            raise ValueError(f'No matching loci. Available: {LOCI["name"].tolist()}')
        print(f'Processing {len(loci)} loci: {loci["name"].tolist()}')

    if args.step in ('all', 'extract'):
        print('\n>>> Step 1: Extract sequences <<<')
        step_extract(loci)

    if args.step in ('all', 'mutagenize'):
        print('\n>>> Step 2: Generate mutagenesis libraries <<<')
        step_mutagenize(loci)

    if args.step in ('all', 'attribute'):
        print('\n>>> Step 3: DeepSHAP attribution <<<')
        step_attribute(loci)

    if args.step in ('all', 'cluster'):
        print(f'\n>>> Step 4: Cluster + CSM (k={args.k}) <<<')
        step_cluster(loci, args.k)

    print('\nPipeline complete.')


if __name__ == '__main__':
    main()
