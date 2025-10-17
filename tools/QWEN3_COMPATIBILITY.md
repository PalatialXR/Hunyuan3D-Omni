# Qwen3-VL Compatibility Guide

## The Version Conflict

**Problem:** Qwen3-VL and Hunyuan3D-Omni require different versions of `transformers`:

| Package | Required Transformers Version |
|---------|-------------------------------|
| Hunyuan3D-Omni | `transformers==4.46.0` |
| Qwen3-VL | `transformers>=4.47` (from git) |

## Solutions

### ✅ Option 1: Separate Environments (Recommended)

**Best for:** Production use, reliability

Create two conda environments:
- `hunyuan3d-omni` - For 3D generation (main environment)
- `hunyuan3d-qwen3` - For detection only (optional)

```bash
# Install Qwen3-VL in separate environment
bash tools/install_qwen3vl.sh
# Choose option 1 when prompted

# Workflow:
# Step 1: Detection (Qwen3 environment)
conda activate hunyuan3d-qwen3
python tools/bbox_detector.py --backend qwen3 \
  --image input.jpg \
  --output detections.json

# Step 2: 3D Generation (Hunyuan environment)
conda activate hunyuan3d-omni
python tools/pipeline_bbox_inference.py \
  --image_path input.jpg \
  --detections_json detections.json \
  --output_path ./output
```

**Pros:**
- ✅ Both systems work perfectly
- ✅ No compatibility issues
- ✅ Easy to maintain

**Cons:**
- ❌ Need to switch environments
- ❌ Slightly more disk space (~2GB)

### ⚠️ Option 2: Upgrade Transformers (Risky)

**Best for:** Testing, experimentation

Upgrade transformers in current environment:

```bash
conda activate hunyuan3d-omni
pip install git+https://github.com/huggingface/transformers
```

**Then test Hunyuan:**
```bash
python test_installation.py
python inference.py --control_type point  # Try an example
```

**Pros:**
- ✅ Single environment
- ✅ Can use both in same script

**Cons:**
- ❌ May break Hunyuan3D-Omni
- ❌ Untested configuration
- ❌ Need to reinstall if it fails

### 🔄 Option 3: Use OpenAI API (No Conflict)

**Best for:** Quick start, no local model needed

Just use OpenAI for detection:

```bash
export OPENAI_API_KEY="your-key-here"

python tools/bbox_detector.py --backend openai \
  --image input.jpg \
  --output detections.json
```

**Pros:**
- ✅ No version conflicts
- ✅ No additional installation
- ✅ High quality detections

**Cons:**
- ❌ Costs ~$0.01-0.02 per image
- ❌ Requires API key
- ❌ Needs internet connection

## Recommended Workflow

### For Development / Testing
```bash
# Use OpenAI for quick testing
python tools/bbox_detector.py --backend openai --image test.jpg
```

### For Production / At Scale
```bash
# Setup Qwen3-VL once
bash tools/install_qwen3vl.sh  # Choose option 1

# Then use two-stage workflow:
# 1. Detect (Qwen3 env)
conda activate hunyuan3d-qwen3
python tools/bbox_detector.py --backend qwen3 --image input.jpg --output detections.json

# 2. Generate (Hunyuan env) 
conda activate hunyuan3d-omni
python tools/pipeline_bbox_inference.py --image_path input.jpg --detections_json detections.json --output_path ./output
```

## Comparison Table

| Feature | OpenAI API | Qwen3-VL (Separate) | Qwen3-VL (Upgrade) |
|---------|-----------|---------------------|-------------------|
| **Cost** | ~$0.01/image | Free | Free |
| **Setup Complexity** | Easy | Medium | Easy |
| **Reliability** | High | High | Unknown |
| **Speed** | 2-5s | 3-8s | 3-8s |
| **Privacy** | Cloud | Local | Local |
| **Conflicts** | None | None | Possible |
| **Quality** | Excellent | Very Good | Very Good |

## Testing Compatibility

After any transformers upgrade, test Hunyuan:

```bash
# 1. Check imports
python -c "from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline; print('✓ Import OK')"

# 2. Run installation test
python test_installation.py

# 3. Try inference
python inference.py --control_type point
```

If anything fails, reinstall clean environment:
```bash
conda env remove -n hunyuan3d-omni
bash install.sh
```

## FAQ

### Q: Can I use both in the same environment?
**A:** Maybe, but not recommended. The newer transformers *might* work with Hunyuan, but it's untested. Separate environments are safer.

### Q: Which is faster, OpenAI or Qwen3-VL?
**A:** Similar speeds (~2-5s). OpenAI has network latency, Qwen3 has model inference time.

### Q: Can I fine-tune Qwen3-VL for my objects?
**A:** Yes! With separate environment, you can customize Qwen3 without affecting Hunyuan.

### Q: What about other detection models (YOLO, etc)?
**A:** They can detect 2D bboxes but won't estimate real-world dimensions. You'd need a separate dimension estimation step.

### Q: Will this be fixed in the future?
**A:** Once transformers 4.47+ is officially released, Hunyuan may update to support it. For now, use separate environments.

## Summary

| Your Needs | Recommended Solution |
|------------|---------------------|
| Just testing | Use OpenAI API |
| Production, cost matters | Separate environments (Option 1) |
| Experimenting | Try upgrade (Option 2), fallback to Option 1 |
| Maximum quality | OpenAI API or Qwen3 (separate env) |
| Privacy required | Qwen3-VL (separate env) |

**Our recommendation: Start with OpenAI API for testing, then set up separate Qwen3-VL environment for production if you need free/local detection.**

