# Installation Notes

## Standard Installation

```bash
bash install.sh
```

This installs Hunyuan3D-Omni with OpenAI API support for object detection.

## Detection Backend Options

### Option 1: OpenAI API (Default)
- ✅ **No conflicts** - Works with standard installation
- ✅ **High quality** - Excellent dimension estimation
- ⚠️ **Costs ~$0.01-0.02 per image**
- ⚠️ **Requires API key and internet**

```bash
export OPENAI_API_KEY="your-key-here"
python tools/bbox_detector.py --backend openai --image input.jpg
```

### Option 2: Qwen3-VL Local Model
- ✅ **Free** - No API costs
- ✅ **Local** - Runs on your GPU
- ✅ **Privacy** - No data sent to cloud
- ⚠️ **Transformers conflict** - Requires separate environment

**Installation:**
```bash
bash tools/install_qwen3vl.sh
# Choose Option 1 for separate environment (recommended)
```

**Usage:**
```bash
# Step 1: Detection (Qwen3 environment)
conda activate hunyuan3d-qwen3
python tools/bbox_detector.py --backend qwen3 --image input.jpg --output detections.json

# Step 2: 3D Generation (Hunyuan environment)
conda activate hunyuan3d-omni
python tools/pipeline_bbox_inference.py \
  --image_path input.jpg \
  --detections_json detections.json \
  --output_path ./output
```

## The Transformers Version Conflict

| Package | Required Transformers |
|---------|----------------------|
| Hunyuan3D-Omni | `==4.46.0` |
| Qwen3-VL | `>=4.47` (from git) |

**Solution:** Use separate conda environments or stick with OpenAI API.

See [tools/QWEN3_COMPATIBILITY.md](tools/QWEN3_COMPATIBILITY.md) for detailed guide.

## Quick Decision Matrix

| Your Situation | Recommended Backend |
|----------------|-------------------|
| Just testing | OpenAI API |
| Cost matters | Qwen3-VL (separate env) |
| Privacy required | Qwen3-VL (separate env) |
| Maximum quality | OpenAI API |
| At scale (1000s of images) | Qwen3-VL (separate env) |

## Standard Workflow

### Testing (OpenAI)
```bash
# One-step inference
python tools/pipeline_auto_scale.py \
  --image_path input.jpg \
  --output_path ./output \
  --use_ema --flashvdm
```

### Production (Qwen3-VL)
```bash
# Two-step inference (allows reuse of detections)
# Step 1: Detect once
conda activate hunyuan3d-qwen3
python tools/bbox_detector.py --backend qwen3 \
  --image input.jpg \
  --output detections.json

# Step 2: Generate with different settings (fast iteration)
conda activate hunyuan3d-omni
python tools/pipeline_bbox_inference.py \
  --image_path input.jpg \
  --detections_json detections.json \
  --output_path ./output \
  --use_ema --flashvdm
```

## Requirements Summary

### Standard Environment (hunyuan3d-omni)
- Python 3.10
- PyTorch 2.5.1 + CUDA 12.4
- transformers==4.46.0
- ~15GB disk space
- ~10GB VRAM for inference

### Qwen3-VL Environment (Optional)
- Python 3.10
- PyTorch 2.5.1 + CUDA 12.4
- transformers from git (v4.47+)
- Additional ~2GB disk space
- ~8GB VRAM for detection

## Testing Your Installation

```bash
# Test standard installation
conda activate hunyuan3d-omni
python test_installation.py

# Test Qwen3-VL (if installed)
conda activate hunyuan3d-qwen3
python -c "from transformers import Qwen3VLForConditionalGeneration; print('✓ Qwen3-VL OK')"
```

## Troubleshooting

### "Qwen3-VL not available" error
**Solution:** Either use `--backend openai` or run `bash tools/install_qwen3vl.sh`

### Hunyuan inference fails after Qwen3 install
**Solution:** You upgraded transformers in the same environment. Recreate:
```bash
conda env remove -n hunyuan3d-omni
bash install.sh
```

### CUDA out of memory
**Solution:** 
- Use lower resolution images
- Close other GPU applications
- Consider detection and generation in separate runs

## Further Reading

- [QUICK_START.md](QUICK_START.md) - Basic usage examples
- [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - Pipeline integration
- [tools/QWEN3_COMPATIBILITY.md](tools/QWEN3_COMPATIBILITY.md) - Detailed compatibility guide
- [tools/DEPTH_ESTIMATION_GUIDE.md](tools/DEPTH_ESTIMATION_GUIDE.md) - Depth estimation strategy

