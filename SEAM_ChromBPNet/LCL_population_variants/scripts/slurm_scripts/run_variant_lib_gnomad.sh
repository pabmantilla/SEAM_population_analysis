#!/bin/bash
#SBATCH --partition=cpuq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-33
#SBATCH --job-name=varlib_gnomad
#SBATCH --output=/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis/SEAM_ChromBPNet/LCL_population_variants/scripts/slurm_scripts/slurm_logs/%x_%A_%a.out
#SBATCH --error=/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis/SEAM_ChromBPNet/LCL_population_variants/scripts/slurm_scripts/slurm_logs/%x_%A_%a.err
#
# Variant-only library (gnomAD) per locus. CPU only.
#
# Usage:
#   sbatch run_variant_lib_gnomad.sh

PROJ_DIR=/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis
SCRIPT_DIR="$PROJ_DIR/SEAM_ChromBPNet/LCL_population_variants/scripts"
LOCI_TSV="$PROJ_DIR/variant_data/GnomAD_data/loci_backup_all34.tsv"
mkdir -p "$SCRIPT_DIR/slurm_scripts/slurm_logs"

source /grid/wsbs/home_norepl/pmantill/SEAM_revisions/SEAM_revisions/.venv/bin/activate
export PYTHONUNBUFFERED=1

LOCUS=$(awk -F'\t' -v idx="$SLURM_ARRAY_TASK_ID" 'NR==idx+2 {print $1}' "$LOCI_TSV")

echo "=============================================="
echo "Variant Library: gnomAD"
echo "Job ID:    $SLURM_JOB_ID"
echo "Array:     $SLURM_ARRAY_TASK_ID/34"
echo "Locus:     $LOCUS"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo "=============================================="

python3 "$SCRIPT_DIR/make_variant_library.py" --source gnomad --locus "$LOCUS"

echo ""
echo "Done: $(date)"
