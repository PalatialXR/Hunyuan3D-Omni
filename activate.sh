#!/bin/bash
# Activation script for Hunyuan3D-Omni environment

# Get conda base path
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Activate environment
conda activate hunyuan3d-omni

echo "======================================================================"
echo "Hunyuan3D-Omni Environment Activated"
echo "======================================================================"
echo "Python: $(which python)"
echo "Python Version: $(python --version)"
echo ""
echo "To run inference:"
echo "  python inference.py --control_type <point|voxel|bbox|pose>"
echo ""
echo "Example:"
echo "  python inference.py --control_type point"
echo "  python inference.py --control_type point --use_ema"
echo "  python inference.py --control_type point --flashvdm"
echo "======================================================================"
