#!/bin/bash
#SBATCH --partition=cpuq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-33
#SBATCH --job-name=bpnet_mutlib
#SBATCH --output=/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis/SEAM_ChromBPNet/LCL_population_variants/scripts/slurm_scripts/slurm_logs/%x_%A_%a.out
#SBATCH --error=/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis/SEAM_ChromBPNet/LCL_population_variants/scripts/slurm_scripts/slurm_logs/%x_%A_%a.err
#
# SEAM mutagenesis library generation for ChromBPNet (2114bp, one-hot).
# 34 array tasks, one per locus.
#
# Usage:
#   sbatch run_mutagenesis_lib.sh
#   sbatch run_mutagenesis_lib.sh --num-sim 50000

PROJ_DIR=/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis
SCRIPT_DIR="$PROJ_DIR/SEAM_ChromBPNet/LCL_population_variants/scripts"
mkdir -p "$SCRIPT_DIR/slurm_scripts/slurm_logs"

source /grid/wsbs/home_norepl/pmantill/SEAM_revisions/SEAM_revisions/.venv/bin/activate
export PYTHONUNBUFFERED=1

# Read locus name from TSV by array task ID (skip header, 0-indexed)
LOCI_TSV="$PROJ_DIR/variant_data/GnomAD_data/loci_backup_all34.tsv"
LOCUS=$(awk -F'\t' -v idx="$SLURM_ARRAY_TASK_ID" 'NR==idx+2 {print $1}' "$LOCI_TSV")

EXTRA_ARGS="$@"

echo "=============================================="
echo "SEAM Mutagenesis Library (ChromBPNet 2114bp)"
echo "Job ID:    $SLURM_JOB_ID"
echo "Array:     $SLURM_ARRAY_TASK_ID/34"
echo "Locus:     $LOCUS"
echo "Args:      $EXTRA_ARGS"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo "=============================================="

python3 "$SCRIPT_DIR/make_mutagenesis_library.py" --locus "$LOCUS" $EXTRA_ARGS

echo ""
echo "Done: $(date)"
