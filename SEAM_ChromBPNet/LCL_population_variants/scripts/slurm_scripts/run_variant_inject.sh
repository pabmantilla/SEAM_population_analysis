#!/bin/bash
#SBATCH --partition=cpuq
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-33
#SBATCH --job-name=bpnet_inject
#SBATCH --output=/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis/SEAM_ChromBPNet/LCL_population_variants/scripts/slurm_scripts/slurm_logs/%x_%A_%a.out
#SBATCH --error=/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis/SEAM_ChromBPNet/LCL_population_variants/scripts/slurm_scripts/slurm_logs/%x_%A_%a.err
#
# Variant injection into 25k SEAM library + clustering for all sources.
# CPU-only. 34 array tasks, one per locus.
#
# Usage:
#   sbatch run_variant_inject.sh

PROJ_DIR=/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis
SCRIPT_DIR="$PROJ_DIR/SEAM_ChromBPNet/LCL_population_variants/scripts"
LOCI_TSV="$PROJ_DIR/variant_data/GnomAD_data/loci_backup_all34.tsv"
mkdir -p "$SCRIPT_DIR/slurm_scripts/slurm_logs"

source "$PROJ_DIR/SEAM_ChromBPNet/chrombpnet_torch_env/bin/activate"
export PYTHONUNBUFFERED=1

# Pin threading to allocated CPUs to avoid contention when sharing nodes
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Isolate temp files per array task
export TMPDIR="/tmp/bpnet_inject_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

LOCUS=$(awk -F'\t' -v idx="$SLURM_ARRAY_TASK_ID" 'NR==idx+2 {print $1}' "$LOCI_TSV")

echo "=============================================="
echo "Variant Injection + Clustering"
echo "Job ID:    $SLURM_JOB_ID"
echo "Array:     $SLURM_ARRAY_TASK_ID/34"
echo "Locus:     $LOCUS"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo "=============================================="

python3 "$SCRIPT_DIR/run_seam_pipeline.py" --step inject --inject gnomad caqtl_eur caqtl_afr --locus "$LOCUS"

rm -rf "$TMPDIR"
echo ""
echo "Done: $(date)"
