#!/bin/bash
# Convenience wrapper for running Hunyuan3D-Omni inference

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="hunyuan3d-omni"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# Check if conda is available
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed."
    exit 1
fi

# Check if environment exists
if ! conda env list | grep -q "^${ENV_NAME} "; then
    print_error "Environment '${ENV_NAME}' not found."
    echo "Please run ./install.sh first."
    exit 1
fi

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

print_info "Environment activated: ${ENV_NAME}"
print_info "Working directory: ${SCRIPT_DIR}"

# Change to script directory
cd "${SCRIPT_DIR}"

# If no arguments provided, show help
if [ $# -eq 0 ]; then
    echo ""
    echo "Usage: $0 <control_type> [options]"
    echo ""
    echo "Control Types:"
    echo "  point   - Point cloud control"
    echo "  voxel   - Voxel control"
    echo "  bbox    - Bounding box control"
    echo "  pose    - Pose control"
    echo ""
    echo "Examples:"
    echo "  $0 point"
    echo "  $0 point --use_ema"
    echo "  $0 voxel --flashvdm"
    echo "  $0 bbox --save_dir ./my_results"
    echo ""
    exit 0
fi

# Run inference with all arguments
print_info "Running inference..."
python inference.py --control_type "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    print_info "Inference completed successfully!"
else
    print_error "Inference failed with exit code ${EXIT_CODE}"
fi

exit $EXIT_CODE

