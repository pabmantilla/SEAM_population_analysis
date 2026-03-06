#!/usr/bin/env python
"""
Generate SEAM mutagenesis libraries for ChromBPNet (2114bp windows centered on TSS).

Uses squid (SEAM) RandomMutagenesis with standard one-hot encoding (ChromBPNet format).
Extracts 2114bp windows (TSS ± 1057) from GRCh38.p13 reference genome.

Usage:
  python make_mutagenesis_library.py
  python make_mutagenesis_library.py --locus IRF7,HLA-A
  python make_mutagenesis_library.py --num-sim 50000 --mut-rate 0.01

Run with SEAM venv:
  source ~/SEAM_revisions/SEAM_revisions/.venv/bin/activate
"""

import os
import argparse
import numpy as np
import pandas as pd
import pysam
import squid

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = '/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis'
GENOME_FASTA = os.path.join(BASE_DIR, 'variant_data', 'hg38_reference', 'GRCh38.p13.genome.fa')
LOCI_TSV = os.path.join(BASE_DIR, 'variant_data', 'GnomAD_data', 'loci_backup_all34.tsv')
OUT_DIR = os.path.join(BASE_DIR, 'SEAM_ChromBPNet', 'LCL_population_variants', 'Mutagenisis_lib')

# ── Constants ──────────────────────────────────────────────────────────────
SEQ_LENGTH = 2114       # ChromBPNet input window
HALF_WINDOW = SEQ_LENGTH // 2  # 1057
NUM_SIM = 25000
MUT_RATE = 0.01

# ── One-hot encoding (standard, for ChromBPNet) ───────────────────────────
ONEHOT_MAP = {
    'A': np.array([1, 0, 0, 0], dtype=np.float32),
    'C': np.array([0, 1, 0, 0], dtype=np.float32),
    'G': np.array([0, 0, 1, 0], dtype=np.float32),
    'T': np.array([0, 0, 0, 1], dtype=np.float32),
    'N': np.array([0, 0, 0, 0], dtype=np.float32),
}


def onehot_encode(seq):
    """One-hot encode a DNA sequence. Shape: (L, 4)."""
    return np.array([ONEHOT_MAP[c] for c in seq.upper()], dtype=np.float32)


def load_loci(loci_tsv):
    """Load TSS loci from TSV file."""
    loci = pd.read_csv(loci_tsv, sep='\t')
    # pysam.fetch uses 0-based half-open coords; TSS is 1-based
    # TSS ± 1057 gives 2114bp window centered on TSS
    loci['start'] = loci['tss'] - HALF_WINDOW
    loci['end'] = loci['tss'] + HALF_WINDOW
    return loci


def extract_sequences(loci, genome_fasta):
    """Extract and one-hot encode sequences for all loci."""
    fasta = pysam.Fastafile(genome_fasta)
    sequences = {}

    for _, row in loci.iterrows():
        seq = fasta.fetch(row['chrom'], row['start'], row['end']).upper()
        assert len(seq) == SEQ_LENGTH, f"{row['name']}: expected {SEQ_LENGTH}bp, got {len(seq)}"
        sequences[row['name']] = onehot_encode(seq)

    fasta.close()
    return sequences


def make_mutagenesis_library(x_ref, num_sim, mut_rate):
    """Generate mutagenesis library using squid RandomMutagenesis.

    Returns (num_sim+1, L, 4) array: index 0 = WT, rest = mutants.
    """
    mut_generator = squid.mutagenizer.RandomMutagenesis(mut_rate=mut_rate)
    mave = squid.mave.InSilicoMAVE(
        mut_generator,
        mut_predictor=None,
        seq_length=SEQ_LENGTH,
        mut_window=None,
        save_window=None,
    )
    x_mut, _ = mave.generate(x_ref, num_sim=num_sim, verbose=0)
    return x_mut


def main():
    parser = argparse.ArgumentParser(description='Generate SEAM mutagenesis libraries for ChromBPNet')
    parser.add_argument('--locus', default=None,
                        help='Comma-separated locus names (default: all 34)')
    parser.add_argument('--num-sim', type=int, default=NUM_SIM,
                        help=f'Number of mutant sequences per locus (default: {NUM_SIM})')
    parser.add_argument('--mut-rate', type=float, default=MUT_RATE,
                        help=f'Mutation rate (default: {MUT_RATE})')
    parser.add_argument('--out-dir', default=OUT_DIR,
                        help=f'Output directory (default: {OUT_DIR})')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load loci
    loci = load_loci(LOCI_TSV)
    if args.locus:
        names = set(args.locus.split(','))
        loci = loci[loci['name'].isin(names)]
        if loci.empty:
            all_names = load_loci(LOCI_TSV)['name'].tolist()
            raise ValueError(f'No matching loci. Available: {all_names}')
    print(f'Processing {len(loci)} loci, {args.num_sim} seqs each, mut_rate={args.mut_rate}')

    # Extract sequences
    print(f'Extracting {SEQ_LENGTH}bp sequences from {os.path.basename(GENOME_FASTA)}...')
    sequences = extract_sequences(loci, GENOME_FASTA)

    # Save reference sequences
    ref_path = os.path.join(args.out_dir, 'x_onehot_ref_2114bp.npz')
    np.savez(ref_path, **sequences)
    print(f'Saved {len(sequences)} reference sequences to {ref_path}')

    # Generate mutagenesis libraries
    for _, row in loci.iterrows():
        name = row['name']
        outpath = os.path.join(args.out_dir, f'x_mut_{name}_{args.num_sim}.npy')
        meta_path = os.path.join(args.out_dir, f'x_mut_{name}_{args.num_sim}_metadata.csv')

        if os.path.exists(outpath) and os.path.exists(meta_path):
            print(f'{name}: library exists, skipping')
            continue

        print(f'{name}: generating {args.num_sim} mutants...', end=' ')
        x_ref = sequences[name]
        x_mut = make_mutagenesis_library(x_ref, args.num_sim, args.mut_rate)
        np.save(outpath, x_mut)

        # Metadata: index 0 = WT, rest = random mutagenesis
        meta_rows = [{'seq_idx': 0, 'source': 'WT'}]
        for i in range(1, x_mut.shape[0]):
            meta_rows.append({'seq_idx': i, 'source': 'random_mutagenesis'})
        pd.DataFrame(meta_rows).to_csv(meta_path, index=False)

        print(f'saved {x_mut.shape} to {outpath}')

    print('\nDone.')


if __name__ == '__main__':
    main()
