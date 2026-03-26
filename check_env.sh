#!/bin/bash
#SBATCH --job-name=check_env
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/data/m25csa023/PersonaAllignment/check_env_%j.log

echo "========== SYSTEM =========="
echo "Hostname : $(hostname)"
echo "Date     : $(date)"
echo ""

echo "========== GPU / DRIVER =========="
nvidia-smi
echo ""

echo "========== CUDA VERSION =========="
nvcc --version 2>/dev/null || echo "nvcc not in PATH"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-not set}"
echo ""

echo "========== PYTORCH =========="
CONDA_PYTHON="/scratch/data/m25csa023/conda/envs/dlops/bin/python"

${CONDA_PYTHON} - <<'EOF'
import torch
print(f"torch version       : {torch.__version__}")
print(f"CUDA available      : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA runtime ver    : {torch.version.cuda}")
    print(f"cuDNN version       : {torch.backends.cudnn.version()}")
    n = torch.cuda.device_count()
    print(f"GPU count           : {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}")
        print(f"    Total memory    : {p.total_memory / 1024**3:.1f} GB")
        print(f"    Compute cap     : {p.major}.{p.minor}")
else:
    print("  No CUDA GPUs detected")
EOF
echo ""

echo "========== KEY PACKAGES =========="
${CONDA_PYTHON} - <<'EOF'
pkgs = [
    "torch", "torchvision", "torchaudio",
    "transformers", "peft", "trl", "accelerate",
    "bitsandbytes", "triton", "datasets",
    "numpy", "scipy", "tokenizers"
]
import importlib, importlib.metadata
for pkg in pkgs:
    try:
        v = importlib.metadata.version(pkg)
        print(f"  {pkg:<20} {v}")
    except Exception:
        print(f"  {pkg:<20} NOT INSTALLED")
EOF
echo ""

echo "========== TORCH 2.6 COMPATIBILITY =========="
${CONDA_PYTHON} - <<'EOF'
import torch, re
from packaging.version import Version

v = Version(torch.__version__.split("+")[0])
if v >= Version("2.6.0"):
    print(f"  torch {v} >= 2.6 : OK — no upgrade needed")
else:
    print(f"  torch {v} < 2.6 : UPGRADE REQUIRED")

# Parse driver version from nvidia-smi to recommend correct wheel
import subprocess, re
try:
    out = subprocess.check_output(["nvidia-smi"], text=True)
    m = re.search(r"Driver Version:\s*([\d.]+)", out)
    if m:
        driver = m.group(1)
        major = int(driver.split(".")[0])
        print(f"  Driver version : {driver}")
        if major >= 550:
            print("  Recommended    : pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124")
        elif major >= 525:
            print("  Recommended    : pip install torch==2.6.0+cu121 -- (not available, driver may be too old for 2.6)")
            print("  ADVICE         : Ask sysadmin to update driver to >= 550 for CUDA 12.4 support")
        else:
            print(f"  Driver {driver} is too old (need >= 525). Contact sysadmin.")
except Exception as e:
    print(f"  Could not run nvidia-smi: {e}")
EOF
echo ""

echo "========== DONE =========="
