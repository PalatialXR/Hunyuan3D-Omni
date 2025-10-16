# Quick Start: Auto-Scale Integration

## Setup (One-time)

```bash
cd /opt/palatial/palatial_home/apps/Hunyuan3D-Omni
./install.sh
export OPENAI_API_KEY="your-key-here"
```

## Standalone Usage

### Single Command Pipeline
```bash
conda run -n hunyuan3d-omni python tools/pipeline_auto_scale.py \
  --image_path input.jpg \
  --output_path ./output \
  --guidance_scale 4.5 \
  --num_inference_steps 50 \
  --use_ema
```

### Two-Stage (Detection + Inference)
```bash
# Stage 1: Detect objects with chain-of-thought reasoning
conda run -n hunyuan3d-omni python tools/bbox_detector.py \
  --image input.jpg \
  --output detections.json

# Stage 2: Run 3D inference
conda run -n hunyuan3d-omni python tools/pipeline_bbox_inference.py \
  --image_path input.jpg \
  --detections_json detections.json \
  --output_path ./output
```

## qMeshGeneration Integration

Add to `apps/palatial-pipeline-queue/qProcess/processes/qRealToSim/Tools/qMeshGeneration.py`:

```python
# At top with other path registrations
QC.registerPath("hunyuan_omni", os.path.join(QC.path("apps",""), "Hunyuan3D-Omni"))

def command(self):
    """Build command based on mode parameter"""
    details = self.template(self._details)
    params = details.get("parameters", {})
    
    # ... existing project_id extraction ...
    
    mode = params.get("mode", "single_view")
    auto_scale = params.get("auto_scale", False)
    
    # Auto-scale mode: use Hunyuan3D-Omni with bbox detection
    if auto_scale and mode == "single_view":
        hunyuan_omni_path = QC.path("hunyuan_omni")
        auto_scale_script = os.path.join(hunyuan_omni_path, "tools", "pipeline_auto_scale.py")
        
        input_dir = QC.transient(project_id, "uploads/")
        output_dir = QC.transient(project_id, "mesh/")
        input_image = self._find_input_image(input_dir, params.get("asset", "image"))
        
        cmd = [
            "conda", "run", "--no-capture-output",
            "-n", "hunyuan3d-omni",
            "python", auto_scale_script,
            "--image_path", input_image,
            "--output_path", output_dir,
            "--guidance_scale", str(params.get("guidance_scale", 4.5)),
            "--num_inference_steps", str(params.get("num_inference_steps", 50)),
            "--octree_resolution", str(params.get("octree_resolution", 512)),
            "--seed", str(params.get("seed", 1234)),
            "--log_level", params.get("log_level", "INFO")
        ]
        
        if params.get("use_ema"):
            cmd.append("--use_ema")
        if params.get("flashvdm"):
            cmd.append("--flashvdm")
        
        return cmd
    
    # Standard mode: use existing Hunyuan pipeline
    # ... existing command building logic ...
```

Add parameters to template:

```python
@classmethod
def template(cls, details):
    tmpl = super().template(details)
    params = tmpl.get("parameters", {})
    
    params.update({
        # ... existing parameters ...
        "auto_scale": params.get("auto_scale", details.get("auto_scale", False)),
        "use_ema": params.get("use_ema", details.get("use_ema", False)),
        "flashvdm": params.get("flashvdm", details.get("flashvdm", False)),
    })
    
    # Add OpenAI API key to environment if auto_scale enabled
    env = tmpl.get("env", {})
    if params.get("auto_scale"):
        env["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    
    tmpl["env"] = env
    return tmpl
```

## qImageToObj Integration

Add to schema in `apps/palatial-pipeline-queue/qProcess/processes/qRealToSim/Conforms/qImageToObj.py`:

```python
@classmethod
def schema(cls):
    sch = super().schema()
    
    sch.update({
        "spacer150": {"order": 150, "type": "break"},
        "header151": {"order": 151, "type": "header", "value": "AI Auto-Scale Detection"},
        
        "auto_scale": {
            "order": 152,
            "type": "boolean",
            "label": "Enable Auto-Scale",
            "tooltip": "Use AI to detect and scale objects automatically",
            "value": False,
        },
        
        "use_ema": {
            "order": 153,
            "type": "boolean",
            "label": "Use EMA Model",
            "tooltip": "Higher quality but slower (only with auto-scale)",
            "value": False,
        },
    })
```

## Key Features

### Chain-of-Thought Reasoning
The detector uses multi-stage reasoning:
1. **Identification**: What objects are visible?
2. **Spatial Analysis**: Where are they positioned?
3. **Dimension Estimation**: What are their real-world sizes?
4. **Depth Reasoning**: How far away are they?
5. **Bbox Placement**: Precise normalized coordinates

### Structured Outputs
Uses Pydantic models for validation:
- `DetectedObject`: Full object with bbox, depth, dimensions
- `BoundingBox2D`: Validated 2D coordinates
- `DepthEstimate`: Depth range with visual cues
- `RealWorldDimensions`: Size in centimeters

### Output Format
```json
{
  "name": "wooden_chair",
  "description": "Brown wooden dining chair with cushioned seat",
  "bbox_2d": [0.2, 0.3, 0.5, 0.8],
  "depth": [0.4, 0.6],
  "real_world_dimensions": {
    "length_cm": 45,
    "width_cm": 50,
    "height_cm": 90
  },
  "confidence": 0.92,
  "reasoning": "Identified as dining chair based on typical proportions and design. Positioned in mid-ground based on perspective cues. Estimated depth from shadow direction and relative size to other objects. Dimensions based on standard dining chair measurements."
}
```

## Requirements

- Python 3.10+
- OpenAI API key
- CUDA-capable GPU (~10GB VRAM)
- Conda environment: `hunyuan3d-omni`

## Comparison: Standard vs Auto-Scale

| Feature | Standard Mode | Auto-Scale Mode |
|---------|--------------|-----------------|
| Object Detection | Manual | Automatic (AI) |
| Bounding Box | Not used | AI-predicted |
| Scaling | Default | Real-world dimensions |
| Multi-Object | Single | Multiple detected |
| API Cost | $0 | ~$0.01-0.02/image |
| Accuracy | Good | Better for scale |

## Performance

- **Detection**: ~2-5 sec/image
- **Inference**: ~10-30 sec/object
- **Total**: ~12-35 sec for single object
- **With FlashVDM**: 30-50% faster

## Cost Estimation

- OpenAI (gpt-4o): ~$0.015/image
- GPU compute: ~$0.01/minute
- Total: ~$0.02-0.05/object

Use `gpt-4o-mini` for 10x cost reduction with slightly lower accuracy.

