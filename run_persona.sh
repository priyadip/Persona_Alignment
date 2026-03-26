#!/bin/bash
#SBATCH --job-name=sherlock_persona
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --qos=dgxqos
#SBATCH --output=/scratch/data/m25csa023/PersonaAllignment/sherlock_persona_%j.log

# Exit on error - stop pipeline if any step fails
set -e

# ============================================================================
# Persona Alignment — Sherlock Holmes Fine-tuning Pipeline
# Runs: fine_tune → extract_loss → evaluation → visualization
# Excludes: modal_deploy.py (cloud), app.py (web), test.py (interactive)
# ============================================================================

echo "=========================================="
echo "Persona Alignment: Sherlock Holmes LLM"
echo "=========================================="
echo "SLURM Job ID : $SLURM_JOB_ID"
echo "Running on   : $(hostname)"
echo "Start time   : $(date)"
echo "=========================================="

# ================= USER CONFIG =================
USERNAME="m25csa023"
# ===============================================

export WORK_DIR="/scratch/data/${USERNAME}/PersonaAllignment"
export CONDA_ENV="/scratch/data/${USERNAME}/conda/envs/dlops"
export CONDA_PIP="${CONDA_ENV}/bin/pip"
export CONDA_PYTHON="${CONDA_ENV}/bin/python"
export PYTHONUNBUFFERED=1

# Model is local — disable HF network calls entirely
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1
# export HF_HOME="/scratch/data/${USERNAME}/hf_cache"

# Conda setup
export CONDARC=/scratch/data/${USERNAME}/conda/condarc
module purge
module load anaconda3/2024
source ~/.bashrc
conda activate /scratch/data/${USERNAME}/conda/envs/dlops

# Upgrade all tightly coupled packages together to ensure compatibility
${CONDA_PIP} install --quiet --upgrade \
    transformers \
    peft \
    trl \
    accelerate

# Install remaining dependencies into conda env only if not already installed
if ! ${CONDA_PYTHON} -c "import bitsandbytes" 2>/dev/null; then
    echo "Installing remaining dependencies..."
    ${CONDA_PIP} install --quiet \
        bitsandbytes==0.48.2 \
        datasets \
        tqdm \
        numpy \
        matplotlib \
        seaborn
else
    echo "Dependencies already installed, skipping pip install."
fi

# GPU check
echo "=========================================="
nvidia-smi
${CONDA_PYTHON} - <<EOF
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Memory:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
EOF
echo "=========================================="

cd ${WORK_DIR}
echo "Working in: $(pwd)"
echo "Files present:"
ls -lh
echo "=========================================="

# ============================================================================
# STEP 1: Fine-tune (prepares data + trains LoRA on Qwen2.5-7B, pure persona style)
# Auto-detects latest checkpoint and resumes if a partial run exists
# ============================================================================
echo "[STEP 1/4] Fine-tuning model..."

# Auto-detect latest checkpoint
CHECKPOINT_DIR="${WORK_DIR}/sherlock-finetuned"
LATEST_CHECKPOINT=""
if [ -d "$CHECKPOINT_DIR" ]; then
    LATEST_CHECKPOINT=$(ls -d ${CHECKPOINT_DIR}/checkpoint-* 2>/dev/null | \
        sed 's/.*checkpoint-//' | sort -n | tail -1 | \
        xargs -I{} echo "${CHECKPOINT_DIR}/checkpoint-{}")
fi

if [ -n "$LATEST_CHECKPOINT" ] && [ -f "${LATEST_CHECKPOINT}/adapter_model.safetensors" ]; then
    echo "Partial run detected. Resuming from: ${LATEST_CHECKPOINT}"
    export RESUME_CHECKPOINT="${LATEST_CHECKPOINT}"
else
    echo "No previous checkpoint found. Starting fresh training."
    export RESUME_CHECKPOINT=""
fi

if ${CONDA_PYTHON} -u fine_tune.py; then
    echo "Fine-tuning done at: $(date)"
else
    echo "ERROR: Fine-tuning failed!"
    exit 1
fi
echo "=========================================="

# ============================================================================
# STEP 2: Extract training loss from latest checkpoint
# ============================================================================
echo "[STEP 2/4] Extracting training loss..."
if ${CONDA_PYTHON} -u extract_loss.py; then
    echo "Loss extraction done at: $(date)"
else
    echo "WARNING: Loss extraction failed, continuing..."
fi
echo "=========================================="

# ============================================================================
# STEP 3: Evaluate base model vs fine-tuned model
# ============================================================================
echo "[STEP 3/4] Evaluating models..."
if ${CONDA_PYTHON} -u evaluation.py; then
    echo "Evaluation done at: $(date)"
else
    echo "ERROR: Evaluation failed!"
    exit 1
fi
echo "=========================================="

# ============================================================================
# STEP 4: Generate visualizations
# ============================================================================
echo "[STEP 4/4] Generating visualizations..."
if ${CONDA_PYTHON} -u visual.py; then
    echo "Visualization done at: $(date)"
else
    echo "WARNING: Visualization failed, continuing..."
fi
echo "=========================================="

# ============================================================================
# STEP 5: Cleanup old checkpoints (keep only final model to save disk space)
# ============================================================================
echo "[CLEANUP] Removing intermediate checkpoints..."
CHECKPOINT_DIR="${WORK_DIR}/sherlock-finetuned"
if [ -d "$CHECKPOINT_DIR" ]; then
    # Remove checkpoint-* directories but keep the final adapter files
    find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "checkpoint-*" -exec rm -rf {} + 2>/dev/null || true
    echo "Intermediate checkpoints removed."
else
    echo "No checkpoint directory found, skipping cleanup."
fi
echo "=========================================="

echo "Results:"
ls -lh ${WORK_DIR}/result/ 2>/dev/null || echo "(no result dir yet)"
ls -lh ${WORK_DIR}/*.json  2>/dev/null
echo ""
echo "Job complete at: $(date)"
echo "=========================================="
