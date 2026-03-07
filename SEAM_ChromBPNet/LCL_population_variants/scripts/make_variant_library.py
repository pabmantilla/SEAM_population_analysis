#!/usr/bin/env python
"""
Generate variant-only mutagenesis libraries for ChromBPNet (2114bp windows).

Creates one-hot encoded sequences with each variant's ALT allele injected into
the WT reference sequence. No random mutagenesis — only observed variants.

Sources:
  --source gnomad     : per-locus TSVs from variant_data/GnomAD_data/
  --source caqtl_eur  : European caQTL feather from AlphaGenome
  --source caqtl_afr  : African caQTL feather from AlphaGenome

Output per locus:
  variant_libs/{source}/x_var_{LOCUS}.npy        — (N+1, 2114, 4) one-hot array
  variant_libs/{source}/x_var_{LOCUS}_metadata.csv — variant metadata

Usage:
  python make_variant_library.py --source gnomad --locus IRF7
  python make_variant_library.py --source caqtl_eur
"""

import os
import argparse
import numpy as np
import pandas as pd
import pysam

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = '/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis'
GENOME_FASTA = os.path.join(BASE_DIR, 'variant_data', 'hg38_reference', 'GRCh38.p13.genome.fa')
LOCI_TSV = os.path.join(BASE_DIR, 'variant_data', 'GnomAD_data', 'loci_backup_all34.tsv')
LCL_DIR = os.path.join(BASE_DIR, 'SEAM_ChromBPNet', 'LCL_population_variants')
GNOMAD_DIR = os.path.join(BASE_DIR, 'variant_data', 'GnomAD_data')
CAQTL_DIR = os.path.join(BASE_DIR, 'variant_data', 'Alphagenome_data', 'chromatin_accessibility_qtl')

# ── Constants ──────────────────────────────────────────────────────────────
SEQ_LENGTH = 2114
HALF_WINDOW = SEQ_LENGTH // 2  # 1057

ONEHOT_MAP = {
    'A': np.array([1, 0, 0, 0], dtype=np.float32),
    'C': np.array([0, 1, 0, 0], dtype=np.float32),
    'G': np.array([0, 0, 1, 0], dtype=np.float32),
    'T': np.array([0, 0, 0, 1], dtype=np.float32),
    'N': np.array([0, 0, 0, 0], dtype=np.float32),
}

NUC_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def onehot_encode(seq):
    """One-hot encode a DNA sequence. Shape: (L, 4)."""
    return np.array([ONEHOT_MAP[c] for c in seq.upper()], dtype=np.float32)


def load_loci(loci_tsv=LOCI_TSV):
    loci = pd.read_csv(loci_tsv, sep='\t')
    loci['start'] = loci['tss'] - HALF_WINDOW
    loci['end'] = loci['tss'] + HALF_WINDOW
    return loci


# ── gnomAD variant loading ────────────────────────────────────────────────
def load_gnomad_variants(locus_name, chrom, start, end):
    """Load gnomAD SNVs within the 2114bp window for a locus."""
    tsv_path = os.path.join(GNOMAD_DIR, f'{locus_name}_gnomad_variants.tsv')
    if not os.path.exists(tsv_path):
        return pd.DataFrame()

    df = pd.read_csv(tsv_path, sep='\t')

    # Filter to SNVs only (single nucleotide ref and alt)
    df = df[(df['ref'].str.len() == 1) & (df['alt'].str.len() == 1)].copy()

    # Filter to variants within the 2114bp window
    # gnomAD positions are 1-based; our window is [start, end) in pysam 0-based
    # Convert: variant is in window if start < pos <= end (1-based pos)
    df = df[(df['pos'] > start) & (df['pos'] <= end)].copy()

    # Compute offset within the one-hot array (0-based)
    # pos is 1-based genomic coord, start is 0-based pysam coord
    # offset = pos - start - 1
    df['offset'] = df['pos'] - start - 1

    return df


# ── caQTL variant loading ────────────────────────────────────────────────
def load_caqtl_variants(population, chrom, start, end, fasta):
    """Load caQTL SNVs within the 2114bp window.

    Parses variant_id format: chr_pos_allele1_allele2_hg38
    Checks genome to determine true REF/ALT (allele order may be swapped).
    """
    if population == 'caqtl_eur':
        feather_path = os.path.join(CAQTL_DIR, 'caqtl_european_variant_causality_human_predictions.feather')
    elif population == 'caqtl_afr':
        feather_path = os.path.join(CAQTL_DIR, 'caqtl_african_variant_causality_human_predictions.feather')
    else:
        raise ValueError(f'Unknown population: {population}')

    df = pd.read_feather(feather_path)

    # Parse variant_id: chr_pos_allele1_allele2_hg38
    parts = df['variant_id'].str.split('_', expand=True)
    df['chrom_parsed'] = parts[0]
    df['pos'] = parts[1].astype(int)  # 1-based
    df['allele1'] = parts[2]
    df['allele2'] = parts[3]

    # Filter to SNVs on the target chromosome
    df = df[
        (df['chrom_parsed'] == chrom) &
        (df['allele1'].str.len() == 1) &
        (df['allele2'].str.len() == 1)
    ].copy()

    # Filter to variants within the 2114bp window
    df = df[(df['pos'] > start) & (df['pos'] <= end)].copy()

    if df.empty:
        return df

    # Determine true REF/ALT by checking genome
    # pos is 1-based; pysam fetch is 0-based
    refs = []
    alts = []
    for _, row in df.iterrows():
        genome_base = fasta.fetch(chrom, row['pos'] - 1, row['pos']).upper()
        if genome_base == row['allele1']:
            refs.append(row['allele1'])
            alts.append(row['allele2'])
        elif genome_base == row['allele2']:
            refs.append(row['allele2'])
            alts.append(row['allele1'])
        else:
            # Neither allele matches genome — skip
            refs.append(None)
            alts.append(None)

    df['ref'] = refs
    df['alt'] = alts
    df = df.dropna(subset=['ref', 'alt'])

    # Compute offset within the one-hot array
    df['offset'] = df['pos'] - start - 1

    return df


# ── Library construction ──────────────────────────────────────────────────
def build_variant_library(wt_onehot, variants_df):
    """Build one-hot library: [WT, variant1, variant2, ...].

    Args:
        wt_onehot: (2114, 4) one-hot WT sequence
        variants_df: DataFrame with 'offset' and 'alt' columns

    Returns:
        x_lib: (N+1, 2114, 4) array
        metadata: list of dicts with variant info
    """
    n_variants = len(variants_df)
    x_lib = np.zeros((n_variants + 1, SEQ_LENGTH, 4), dtype=np.float32)
    x_lib[0] = wt_onehot  # index 0 = WT

    metadata = [{'seq_idx': 0, 'source': 'WT', 'variant_id': 'WT',
                 'offset': -1, 'ref': '', 'alt': ''}]

    for i, (_, var) in enumerate(variants_df.iterrows(), start=1):
        seq = wt_onehot.copy()
        offset = int(var['offset'])
        alt = var['alt'].upper()

        # Inject ALT allele
        seq[offset] = ONEHOT_MAP[alt]
        x_lib[i] = seq

        meta = {
            'seq_idx': i,
            'source': 'variant',
            'offset': offset,
            'ref': var['ref'],
            'alt': alt,
        }
        # Include variant_id if available
        if 'variant_id' in var.index:
            meta['variant_id'] = var['variant_id']
        else:
            meta['variant_id'] = f"{var.get('chrom', '')}_{var.get('pos', '')}_{var['ref']}_{alt}"

        # Include extra fields if present
        for col in ['pos', 'rsids', 'AF', 'consequence']:
            if col in var.index and pd.notna(var[col]):
                meta[col] = var[col]

        metadata.append(meta)

    return x_lib, metadata


def main():
    parser = argparse.ArgumentParser(description='Generate variant-only libraries for ChromBPNet')
    parser.add_argument('--source', required=True,
                        choices=['gnomad', 'caqtl_eur', 'caqtl_afr'],
                        help='Variant source')
    parser.add_argument('--locus', default=None,
                        help='Comma-separated locus names (default: all 34)')
    args = parser.parse_args()

    out_dir = os.path.join(LCL_DIR, 'variant_libs', args.source)
    os.makedirs(out_dir, exist_ok=True)

    loci = load_loci()
    if args.locus:
        names = set(args.locus.split(','))
        loci = loci[loci['name'].isin(names)]
        if loci.empty:
            raise ValueError(f'No matching loci. Available: {load_loci()["name"].tolist()}')

    print(f'Source: {args.source}')
    print(f'Processing {len(loci)} loci')
    print(f'Output: {out_dir}')

    fasta = pysam.Fastafile(GENOME_FASTA)

    for _, row in loci.iterrows():
        name = row['name']
        chrom = row['chrom']
        start = row['start']
        end = row['end']

        out_path = os.path.join(out_dir, f'x_var_{name}.npy')
        meta_path = os.path.join(out_dir, f'x_var_{name}_metadata.csv')

        if os.path.exists(out_path) and os.path.exists(meta_path):
            print(f'{name}: library exists, skipping')
            continue

        # Get WT sequence
        seq = fasta.fetch(chrom, start, end).upper()
        assert len(seq) == SEQ_LENGTH, f"{name}: expected {SEQ_LENGTH}bp, got {len(seq)}"
        wt_onehot = onehot_encode(seq)

        # Load variants
        if args.source == 'gnomad':
            variants = load_gnomad_variants(name, chrom, start, end)
        else:
            variants = load_caqtl_variants(args.source, chrom, start, end, fasta)

        if variants.empty:
            print(f'{name}: no variants found, skipping')
            continue

        # Build library
        x_lib, metadata = build_variant_library(wt_onehot, variants)
        np.save(out_path, x_lib)
        pd.DataFrame(metadata).to_csv(meta_path, index=False)

        print(f'{name}: {len(variants)} variants -> {x_lib.shape} saved')

    fasta.close()
    print('\nDone.')


if __name__ == '__main__':
    main()
