#!/bin/bash
#SBATCH --partition=cpuq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=bpnet_results
#SBATCH --output=/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis/SEAM_ChromBPNet/LCL_population_variants/scripts/slurm_scripts/slurm_logs/%x_%A_%a.out
#SBATCH --error=/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis/SEAM_ChromBPNet/LCL_population_variants/scripts/slurm_scripts/slurm_logs/%x_%A_%a.err
#
# Generate per-locus seq results + cross-locus final results.
# CPU-only, single job (processes all loci).
#
# Usage:
#   sbatch run_seam_results.sh

PROJ_DIR=/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis
SCRIPT_DIR="$PROJ_DIR/SEAM_ChromBPNet/LCL_population_variants/scripts"
mkdir -p "$SCRIPT_DIR/slurm_scripts/slurm_logs"

source "$PROJ_DIR/SEAM_ChromBPNet/chrombpnet_torch_env/bin/activate"
export PYTHONUNBUFFERED=1

EXTRA_ARGS="$@"

echo "=============================================="
echo "ChromBPNet SEAM Results"
echo "Job ID:    $SLURM_JOB_ID"
echo "Args:      $EXTRA_ARGS"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo "=============================================="

python3 "$SCRIPT_DIR/run_seam_pipeline.py" --step results $EXTRA_ARGS
python3 "$SCRIPT_DIR/run_seam_pipeline.py" --step final $EXTRA_ARGS

echo ""
echo "Done: $(date)"
