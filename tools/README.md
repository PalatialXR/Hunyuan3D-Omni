# Hunyuan3D-Omni Pipeline Tools

## Overview

Standalone tools for bounding box detection and 3D inference, designed for integration with the Palatial pipeline.

## Tools

### 1. `bbox_detector.py`
Detects objects and bounding boxes using OpenAI Vision API with structured outputs and chain-of-thought reasoning.

**Features:**
- Pydantic models for structured validation
- Chain-of-thought reasoning for better accuracy
- Real-world dimension estimation
- Confidence scores and detailed reasoning

**Usage:**
```bash
python bbox_detector.py \
  --image input.jpg \
  --output detections.json \
  --log_level INFO
```

### 2. `pipeline_bbox_inference.py`
Runs Hunyuan3D-Omni inference using detected bounding boxes.

**Usage:**
```bash
python pipeline_bbox_inference.py \
  --image_path input.jpg \
  --detections_json detections.json \
  --output_path ./output \
  --guidance_scale 4.5 \
  --num_inference_steps 50
```

### 3. `pipeline_auto_scale.py`
Combined pipeline that runs both detection and inference.

**Usage:**
```bash
python pipeline_auto_scale.py \
  --image_path input.jpg \
  --output_path ./output \
  --use_ema \
  --flashvdm
```

## Integration with qMeshGeneration

To integrate with `qMeshGeneration.py`:

```python
# In qMeshGeneration.py
if params.get("auto_scale", False):
    hunyuan_omni_path = QC.path("hunyuan_omni")
    auto_scale_script = os.path.join(hunyuan_omni_path, "tools", "pipeline_auto_scale.py")
    
    cmd = [
        "conda", "run", "--no-capture-output",
        "-n", "hunyuan3d-omni",
        "python", auto_scale_script,
        "--image_path", input_image,
        "--output_path", output_dir,
        "--guidance_scale", str(guidance_scale),
        "--num_inference_steps", str(num_inference_steps),
        "--seed", str(seed)
    ]
    
    if params.get("use_ema"):
        cmd.append("--use_ema")
    if params.get("flashvdm"):
        cmd.append("--flashvdm")
```

## Environment Variables

```bash
export OPENAI_API_KEY="your-key-here"
```

## Dependencies

- openai>=1.0.0
- pydantic>=2.0.0
- torch
- trimesh
- PIL

All installed via `requirements.txt` in parent directory.

