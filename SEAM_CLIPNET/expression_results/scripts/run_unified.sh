#!/bin/bash
#SBATCH --partition=cpuq
#SBATCH --job-name=unified_volcano
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_logs/unified_volcano_%j.out
#SBATCH --error=slurm_logs/unified_volcano_%j.err
#SBATCH --mail-user pmantill@cshl.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

cd /grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis/SEAM_CLIPNET/expression_results/scripts
mkdir -p slurm_logs

source /grid/wsbs/home_norepl/pmantill/SEAM_revisions/SEAM_revisions/.venv/bin/activate
module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163

echo "=============================================="
echo "Unified Volcano: gnomAD + eQTL + Causal"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=============================================="

python unified_volcano.py

echo "=============================================="
echo "Done"
echo "End time: $(date)"
echo "=============================================="
