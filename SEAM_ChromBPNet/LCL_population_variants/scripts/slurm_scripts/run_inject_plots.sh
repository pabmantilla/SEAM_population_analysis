#!/bin/bash
#SBATCH --partition=cpuq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=inject_plots
#SBATCH --output=/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis/SEAM_ChromBPNet/LCL_population_variants/scripts/slurm_scripts/slurm_logs/%x_%j.out
#SBATCH --error=/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis/SEAM_ChromBPNet/LCL_population_variants/scripts/slurm_scripts/slurm_logs/%x_%j.err
#
# Compute mechanistic causality from existing injection results and generate plots.
# Run AFTER variant injection jobs complete.
#
# Usage:
#   sbatch run_inject_plots.sh

PROJ_DIR=/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis
SCRIPT_DIR="$PROJ_DIR/SEAM_ChromBPNet/LCL_population_variants/scripts"
mkdir -p "$SCRIPT_DIR/slurm_scripts/slurm_logs"

source "$PROJ_DIR/SEAM_ChromBPNet/chrombpnet_torch_env/bin/activate"
export PYTHONUNBUFFERED=1

echo "=============================================="
echo "Variant Injection Plots"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo "=============================================="

python3 "$SCRIPT_DIR/make_inject_plots.py" --sources gnomad caqtl_eur caqtl_afr

echo ""
echo "Done: $(date)"
