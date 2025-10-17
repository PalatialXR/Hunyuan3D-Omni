#!/bin/bash
set -e  # Exit on error

echo "======================================================================"
echo "Hunyuan3D-Omni Installation Script"
echo "======================================================================"

# Configuration
ENV_NAME="hunyuan3d-omni"
PYTHON_VERSION="3.10"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

print_info "Conda found: $(which conda)"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    print_warning "Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove it and reinstall? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        print_info "Exiting installation."
        exit 0
    fi
fi

# Create conda environment
print_info "Creating conda environment: ${ENV_NAME} with Python ${PYTHON_VERSION}..."
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Activate environment
print_info "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Verify Python version
CURRENT_PYTHON=$(python --version 2>&1 | awk '{print $2}')
print_info "Python version: ${CURRENT_PYTHON}"

# Install PyTorch with CUDA 12.4 support
print_info "Installing PyTorch 2.5.1 with CUDA 12.4 support..."
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA availability
print_info "Verifying CUDA availability..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Install requirements
print_info "Installing package requirements..."
cd "${SCRIPT_DIR}"

# Remove the extra-index-url lines from requirements as they might cause issues
# Install from clean requirements
python -m pip install --upgrade pip wheel setuptools

# Install requirements.txt
if [ -f "requirements.txt" ]; then
    print_info "Installing from requirements.txt..."
    python -m pip install -r requirements.txt
else
    print_error "requirements.txt not found in ${SCRIPT_DIR}"
    exit 1
fi

# Create activation script
print_info "Creating activation script..."
cat > "${SCRIPT_DIR}/activate.sh" << 'EOF'
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
echo "Standard inference:"
echo "  python inference.py --control_type point"
echo ""
echo "Auto-scale with AI detection:"
echo "  python tools/pipeline_auto_scale.py --image_path input.jpg --output_path ./output"
echo ""
echo "Note: Auto-scale requires OPENAI_API_KEY environment variable"
echo "======================================================================"
EOF

chmod +x "${SCRIPT_DIR}/activate.sh"

# Create a simple test script
print_info "Creating test script..."
cat > "${SCRIPT_DIR}/test_installation.py" << 'EOF'
#!/usr/bin/env python
"""Test script to verify Hunyuan3D-Omni installation."""

import sys

def test_imports():
    """Test if all critical packages can be imported."""
    print("Testing package imports...")
    
    packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('torchaudio', 'TorchAudio'),
        ('transformers', 'Transformers'),
        ('diffusers', 'Diffusers'),
        ('trimesh', 'Trimesh'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('cv2', 'OpenCV'),
        ('gradio', 'Gradio'),
    ]
    
    failed = []
    for module, name in packages:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name} - {e}")
            failed.append(name)
    
    return failed

def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Device count: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name(0)}")
            
            # Test VRAM
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  VRAM - Total: {mem_total:.2f} GB, Reserved: {mem_reserved:.2f} GB, Allocated: {mem_allocated:.2f} GB")
            
            if mem_total < 10:
                print(f"  ⚠ Warning: Model requires ~10GB VRAM, you have {mem_total:.2f} GB")
        else:
            print("  ⚠ CUDA is not available. GPU inference will not work.")
    except Exception as e:
        print(f"  ✗ Error testing CUDA: {e}")
        return False
    
    return True

def test_hunyuan3d_modules():
    """Test if Hunyuan3D-Omni specific modules can be imported."""
    print("\nTesting Hunyuan3D-Omni modules...")
    try:
        from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline
        print("  ✓ Hunyuan3DOmniSiTFlowMatchingPipeline")
        
        from hy3dshape.preprocessors import ImageProcessorV2
        print("  ✓ ImageProcessorV2")
        
        from hy3dshape.postprocessors import FloaterRemover, DegenerateFaceRemover
        print("  ✓ FloaterRemover, DegenerateFaceRemover")
        
        return True
    except ImportError as e:
        print(f"  ✗ Failed to import Hunyuan3D modules: {e}")
        return False

def main():
    print("=" * 70)
    print("Hunyuan3D-Omni Installation Test")
    print("=" * 70)
    
    # Test imports
    failed_packages = test_imports()
    
    # Test CUDA
    cuda_ok = test_cuda()
    
    # Test Hunyuan3D modules
    hunyuan_ok = test_hunyuan3d_modules()
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    if failed_packages:
        print(f"✗ Failed packages: {', '.join(failed_packages)}")
    else:
        print("✓ All packages imported successfully")
    
    if not cuda_ok:
        print("✗ CUDA test failed")
    else:
        print("✓ CUDA is working")
    
    if not hunyuan_ok:
        print("✗ Hunyuan3D-Omni modules not found or failed to import")
    else:
        print("✓ Hunyuan3D-Omni modules loaded successfully")
    
    print("=" * 70)
    
    if failed_packages or not hunyuan_ok:
        print("\n⚠ Installation incomplete or has issues.")
        sys.exit(1)
    else:
        print("\n✓ Installation successful!")
        print("\nTo run inference, use:")
        print("  python inference.py --control_type <point|voxel|bbox|pose>")
        sys.exit(0)

if __name__ == "__main__":
    main()
EOF

chmod +x "${SCRIPT_DIR}/test_installation.py"

# Run test
print_info "Running installation tests..."
python "${SCRIPT_DIR}/test_installation.py"

# Final instructions
echo ""
echo "======================================================================"
echo "Installation Complete!"
echo "======================================================================"
echo ""
echo "To activate the environment in the future, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Or use the activation script:"
echo "  source ${SCRIPT_DIR}/activate.sh"
echo ""
echo "To test the installation:"
echo "  python ${SCRIPT_DIR}/test_installation.py"
echo ""
echo "======================================================================"
echo "Usage Options"
echo "======================================================================"
echo ""
echo "1. Standard Inference (Original Hunyuan3D-Omni):"
echo "  python inference.py --control_type <point|voxel|bbox|pose>"
echo ""
echo "2. Auto-Scale with AI Bbox Detection (NEW):"
echo "  python tools/pipeline_auto_scale.py --image_path input.jpg --output_path ./output"
echo ""
echo "3. Two-Stage Detection + Inference:"
echo "  python tools/bbox_detector.py --image input.jpg --output detections.json"
echo "  python tools/pipeline_bbox_inference.py \\"
echo "    --image_path input.jpg --detections_json detections.json --output_path ./output"
echo ""
echo "======================================================================"
echo "Auto-Scale Examples"
echo "======================================================================"
echo ""
echo "Basic auto-scale (with EMA + FlashVDM):"
echo "  python tools/pipeline_auto_scale.py \\"
echo "    --image_path demos/image/0.png \\"
echo "    --output_path ./auto_results \\"
echo "    --use_ema --flashvdm"
echo ""
echo "Note: Auto-scale with OpenAI requires OPENAI_API_KEY environment variable"
echo "  export OPENAI_API_KEY='your-key-here'"
echo ""
echo "======================================================================"
echo "Optional: Qwen3-VL Local Detection"
echo "======================================================================"
echo ""
echo "To use Qwen3-VL for free local detection (instead of OpenAI):"
echo "  1. Create separate environment (recommended due to transformers conflict):"
echo "     bash tools/install_qwen3vl.sh"
echo ""
echo "  2. Or upgrade transformers in current environment (may break Hunyuan):"
echo "     pip install git+https://github.com/huggingface/transformers"
echo ""
echo "  3. Use Qwen3-VL backend:"
echo "     python tools/bbox_detector.py --backend qwen3 --image input.jpg"
echo ""
echo "For full documentation, see:"
echo "  - QUICK_START.md (quick reference)"
echo "  - INTEGRATION_GUIDE.md (pipeline integration)"
echo "  - tools/README.md (tool details)"
echo ""
echo "======================================================================"

