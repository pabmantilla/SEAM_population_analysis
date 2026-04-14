#!/bin/bash
#SBATCH --partition=cpuq
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-33
#SBATCH --job-name=bpnet_cluster
#SBATCH --output=/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis/SEAM_ChromBPNet/LCL_population_variants/scripts/slurm_scripts/slurm_logs/%x_%A_%a.out
#SBATCH --error=/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis/SEAM_ChromBPNet/LCL_population_variants/scripts/slurm_scripts/slurm_logs/%x_%A_%a.err
#
# K-means clustering + CSM on ChromBPNet attribution maps.
# CPU-only. 34 array tasks, one per locus.
#
# Usage:
#   sbatch run_seam_cluster.sh
#   sbatch run_seam_cluster.sh  # uses default k=100

PROJ_DIR=/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis
SCRIPT_DIR="$PROJ_DIR/SEAM_ChromBPNet/LCL_population_variants/scripts"
LOCI_TSV="$PROJ_DIR/variant_data/GnomAD_data/loci_backup_all34.tsv"
mkdir -p "$SCRIPT_DIR/slurm_scripts/slurm_logs"

source "$PROJ_DIR/SEAM_ChromBPNet/chrombpnet_torch_env/bin/activate"
export PYTHONUNBUFFERED=1

LOCUS=$(awk -F'\t' -v idx="$SLURM_ARRAY_TASK_ID" 'NR==idx+2 {print $1}' "$LOCI_TSV")

EXTRA_ARGS="$@"

echo "=============================================="
echo "ChromBPNet Clustering + CSM"
echo "Job ID:    $SLURM_JOB_ID"
echo "Array:     $SLURM_ARRAY_TASK_ID/34"
echo "Locus:     $LOCUS"
echo "Args:      $EXTRA_ARGS"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo "=============================================="

python3 "$SCRIPT_DIR/run_seam_pipeline.py" --step cluster --locus "$LOCUS" $EXTRA_ARGS

echo ""
echo "Done: $(date)"
