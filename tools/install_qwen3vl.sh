#!/bin/bash
set -e

echo "======================================================================"
echo "Qwen3-VL Installation Script (Optional)"
echo "======================================================================"
echo ""
echo "⚠️  WARNING: Transformers Version Conflict"
echo "--------------------------------------------------------------------"
echo "Qwen3-VL requires transformers from git (v4.47+)"
echo "Hunyuan3D-Omni requires transformers==4.46.0"
echo ""
echo "Choose your installation strategy:"
echo "--------------------------------------------------------------------"
echo ""
echo "Option 1: Separate Environment (RECOMMENDED)"
echo "  - Creates hunyuan3d-qwen3 environment with Qwen3-VL"
echo "  - Keeps hunyuan3d-omni environment intact"
echo "  - Use for detection, then pass results to Hunyuan"
echo ""
echo "Option 2: Upgrade Current Environment (RISKY)"
echo "  - Upgrades transformers in current environment"
echo "  - May break Hunyuan3D-Omni compatibility"
echo "  - Only if you want both in same environment"
echo ""
echo "======================================================================"
echo ""

read -p "Choose option (1=separate, 2=upgrade, q=quit): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Qq]$ ]]; then
    echo "Installation cancelled."
    exit 0
fi

if [[ $REPLY == "1" ]]; then
    # OPTION 1: Separate Environment
    echo ""
    echo "[INFO] Creating separate environment: hunyuan3d-qwen3"
    
    # Create new environment
    conda create -n hunyuan3d-qwen3 python=3.10 -y
    
    # Activate
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate hunyuan3d-qwen3
    
    # Install PyTorch
    echo "[INFO] Installing PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    
    # Install Qwen3-VL dependencies
    echo "[INFO] Installing Qwen3-VL and dependencies..."
    pip install git+https://github.com/huggingface/transformers
    pip install accelerate pillow opencv-python numpy
    
    # Optional: Flash Attention 2
    read -p "Install Flash Attention 2 for better performance? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "[INFO] Installing Flash Attention 2..."
        pip install flash-attn --no-build-isolation
    fi
    
    # Create activation script
    cat > "$(dirname "$0")/../activate_qwen3.sh" << 'EOF'
#!/bin/bash
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate hunyuan3d-qwen3
echo "======================================================================"
echo "Qwen3-VL Environment Activated"
echo "======================================================================"
echo "Use for detection only:"
echo "  python tools/bbox_detector.py --backend qwen3 --image input.jpg --output detections.json"
echo ""
echo "Then switch to Hunyuan environment for 3D generation:"
echo "  conda activate hunyuan3d-omni"
echo "  python tools/pipeline_bbox_inference.py --image_path input.jpg --detections_json detections.json --output_path ./output"
echo "======================================================================"
EOF
    chmod +x "$(dirname "$0")/../activate_qwen3.sh"
    
    echo ""
    echo "======================================================================"
    echo "✓ Qwen3-VL Environment Created!"
    echo "======================================================================"
    echo ""
    echo "To use Qwen3-VL for detection:"
    echo "  1. Activate Qwen3 environment:"
    echo "     conda activate hunyuan3d-qwen3"
    echo "     # Or: source activate_qwen3.sh"
    echo ""
    echo "  2. Run detection:"
    echo "     python tools/bbox_detector.py --backend qwen3 --image input.jpg --output detections.json"
    echo ""
    echo "  3. Switch to Hunyuan environment for 3D generation:"
    echo "     conda activate hunyuan3d-omni"
    echo "     python tools/pipeline_bbox_inference.py --image_path input.jpg --detections_json detections.json --output_path ./output"
    echo ""
    echo "======================================================================"
    
elif [[ $REPLY == "2" ]]; then
    # OPTION 2: Upgrade Current Environment
    echo ""
    echo "[WARNING] This will upgrade transformers in your current environment!"
    echo "          Hunyuan3D-Omni may not work correctly after this."
    echo ""
    read -p "Are you sure? (yes/no): " -r
    
    if [[ $REPLY != "yes" ]]; then
        echo "Installation cancelled."
        exit 0
    fi
    
    echo "[INFO] Upgrading transformers..."
    pip install --upgrade git+https://github.com/huggingface/transformers
    
    # Optional: Flash Attention 2
    read -p "Install Flash Attention 2 for better performance? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "[INFO] Installing Flash Attention 2..."
        pip install flash-attn --no-build-isolation
    fi
    
    echo ""
    echo "======================================================================"
    echo "✓ Transformers Upgraded"
    echo "======================================================================"
    echo ""
    echo "⚠️  WARNING: Test Hunyuan3D-Omni to ensure it still works!"
    echo ""
    echo "Test with:"
    echo "  python test_installation.py"
    echo ""
    echo "If broken, recreate environment:"
    echo "  bash install.sh"
    echo ""
    echo "======================================================================"
    
else
    echo "Invalid option. Exiting."
    exit 1
fi

echo ""
echo "To use Qwen3-VL:"
echo "  python tools/bbox_detector.py --backend qwen3 --image input.jpg"
echo ""

