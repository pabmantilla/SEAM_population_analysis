#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=seam_eqtl_lcl
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/seam_eqtl_lcl_%j.out
#SBATCH --error=slurm_logs/seam_eqtl_lcl_%j.err
#SBATCH --mail-user pmantill@cshl.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# ── Environment: SEAM .venv (includes CLIPNET + SEAM + DeepSHAP) ──
cd /grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis/SEAM_CLIPNET/LCL_variants_analysis/scripts
mkdir -p slurm_logs

source /grid/wsbs/home_norepl/pmantill/SEAM_revisions/SEAM_revisions/.venv/bin/activate
module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163

echo "=============================================="
echo "SEAM eQTL LCL Pipeline"
echo "Env: SEAM .venv (CLIPNET + SEAM + DeepSHAP)"
echo "Job ID: $SLURM_JOB_ID"
echo "Args: $@"
echo "Start time: $(date)"
echo "=============================================="

python run_seam_pipeline.py "$@"

echo "=============================================="
echo "Pipeline Complete"
echo "End time: $(date)"
echo "=============================================="
