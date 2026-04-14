#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=bio_ai
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-33
#SBATCH --job-name=bpnet_attr
#SBATCH --output=/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis/SEAM_ChromBPNet/LCL_population_variants/scripts/slurm_scripts/slurm_logs/%x_%A_%a.out
#SBATCH --error=/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis/SEAM_ChromBPNet/LCL_population_variants/scripts/slurm_scripts/slurm_logs/%x_%A_%a.err
#
# ChromBPNet DeepSHAP attributions + predictions per locus.
# 34 array tasks, one per locus. Requires GPU.
#
# Usage:
#   sbatch run_seam_attribute.sh

PROJ_DIR=/grid/wsbs/home_norepl/pmantill/Human_nc_variants/SEAM_population_analysis
SCRIPT_DIR="$PROJ_DIR/SEAM_ChromBPNet/LCL_population_variants/scripts"
LOCI_TSV="$PROJ_DIR/variant_data/GnomAD_data/loci_backup_all34.tsv"
mkdir -p "$SCRIPT_DIR/slurm_scripts/slurm_logs"

source "$PROJ_DIR/SEAM_ChromBPNet/chrombpnet_torch_env/bin/activate"
module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163
export PYTHONUNBUFFERED=1

LOCUS=$(awk -F'\t' -v idx="$SLURM_ARRAY_TASK_ID" 'NR==idx+2 {print $1}' "$LOCI_TSV")

echo "=============================================="
echo "ChromBPNet DeepSHAP Attributions"
echo "Job ID:    $SLURM_JOB_ID"
echo "Array:     $SLURM_ARRAY_TASK_ID/34"
echo "Locus:     $LOCUS"
echo "Node:      $SLURM_NODELIST"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo "Start:     $(date)"
echo "=============================================="

python3 "$SCRIPT_DIR/run_seam_pipeline.py" --step attribute --locus "$LOCUS"

echo ""
echo "Done: $(date)"
