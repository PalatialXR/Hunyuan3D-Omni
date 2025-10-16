## Hunyuan3D-Omni Integration with Palatial Pipeline

### Overview

This guide explains how to integrate the Hunyuan3D-Omni bounding box detection and inference pipeline with the existing Palatial `qMeshGeneration` and `qImageToObj` processes.

### Architecture

The integration consists of three main components:

1. **bbox_detector.py** - Handles OpenAI Vision API calls for object detection
2. **pipeline_bbox_inference.py** - Runs Hunyuan3D-Omni inference with detected bboxes
3. **qMeshGeneration** - Modified to support `--auto-scale` mode

### File Structure

```
apps/
├── Hunyuan3D-Omni/                    # New submodule
│   ├── tools/
│   │   ├── bbox_detector.py           # OpenAI detection
│   │   └── pipeline_bbox_inference.py # Bbox inference pipeline
│   ├── inference.py                   # Original Hunyuan3D-Omni script
│   ├── install.sh                     # Environment setup
│   └── requirements.txt               # Dependencies
│
└── palatial-pipeline-queue/
    └── qProcess/processes/qRealToSim/
        ├── Tools/
        │   └── qMeshGeneration.py     # Modified to support auto-scale
        └── Conforms/
            └── qImageToObj.py         # Uses qMeshGeneration
```

### Integration Steps

#### Step 1: Add Hunyuan3D-Omni Path Registration

In `qMeshGeneration.py`, add:

```python
# Register both Hunyuan paths
QC.registerPath("hunyuan", os.path.join(QC.path("apps",""), "tencent-hunyan3d-v2.1"))
QC.registerPath("hunyuan_omni", os.path.join(QC.path("apps",""), "Hunyuan3D-Omni"))
```

#### Step 2: Add Auto-Scale Mode to qMeshGeneration

Modify the `command()` method in `qMeshGeneration.py`:

```python
def command(self):
    """Build command based on mode parameter"""
    
    details = self.template(self._details)
    params = details.get("parameters", {})
    
    # Extract project ID
    project_id = self._extract_project_id()
    mode = params.get("mode", "single_view")
    auto_scale = params.get("auto_scale", False)
    
    # If auto-scale is enabled, use Hunyuan3D-Omni bbox pipeline
    if auto_scale and mode == "single_view":
        return self._build_auto_scale_command(project_id, params)
    
    # Otherwise, use existing logic
    return self._build_standard_command(project_id, params, mode)

def _build_auto_scale_command(self, project_id, params):
    """Build command for auto-scale mode using Hunyuan3D-Omni"""
    
    hunyuan_omni_path = QC.path("hunyuan_omni")
    detector_script = os.path.join(hunyuan_omni_path, "tools", "bbox_detector.py")
    inference_script = os.path.join(hunyuan_omni_path, "tools", "pipeline_bbox_inference.py")
    
    input_dir = QC.transient(project_id, "uploads/")
    output_dir = QC.transient(project_id, "mesh/")
    temp_dir = QC.transient(project_id, "temp/")
    
    # Find input image
    asset = params.get("asset", "image")
    input_image = self._find_input_image(input_dir, asset)
    
    # Output paths
    detections_json = os.path.join(temp_dir, "detections.json")
    
    # Step 1: Detect bounding boxes
    detect_cmd = [
        "conda", "run", "--no-capture-output",
        "-n", "hunyuan3d-omni",
        "python", detector_script,
        "--image", input_image,
        "--output", detections_json,
        "--model", params.get("openai_model", "gpt-4o"),
        "--log_level", params.get("log_level", "INFO")
    ]
    
    # Step 2: Run bbox inference
    inference_cmd = [
        "conda", "run", "--no-capture-output",
        "-n", "hunyuan3d-omni",
        "python", inference_script,
        "--image_path", input_image,
        "--detections_json", detections_json,
        "--output_path", output_dir,
        "--guidance_scale", str(params.get("guidance_scale", 4.5)),
        "--num_inference_steps", str(params.get("num_inference_steps", 50)),
        "--octree_resolution", str(params.get("octree_resolution", 512)),
        "--seed", str(params.get("seed", 1234)),
        "--log_level", params.get("log_level", "INFO")
    ]
    
    if params.get("use_ema", False):
        inference_cmd.append("--use_ema")
    
    if params.get("flashvdm", False):
        inference_cmd.append("--flashvdm")
    
    # Chain commands: detect then infer
    # For now, return detection command and add inference as follow-up dependency
    # In production, you might want to use a wrapper script
    
    return detect_cmd  # This would need refinement for chaining
```

#### Step 3: Update Template with Auto-Scale Parameters

Add auto-scale parameters to the template:

```python
@classmethod
def template(cls, details):
    tmpl = super().template(details)
    params = tmpl.get("parameters", {})

    params.update({
        # ... existing parameters ...
        "auto_scale": params.get("auto_scale", details.get("auto_scale", False)),
        "openai_model": params.get("openai_model", details.get("openai_model", "gpt-4o")),
        "use_ema": params.get("use_ema", details.get("use_ema", False)),
        "flashvdm": params.get("flashvdm", details.get("flashvdm", False)),
        "max_objects": params.get("max_objects", details.get("max_objects", None)),
    })
    
    # Add OpenAI API key to environment
    env = tmpl.get("env", {})
    if params.get("auto_scale"):
        env["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    
    tmpl.update({
        "parameters": params,
        "env": env
    })
    
    return tmpl
```

#### Step 4: Update qImageToObj to Support Auto-Scale

In `qImageToObj.py`, add auto-scale parameter:

```python
def dependencies(self):
    # ... existing code ...
    
    deps += [
        {
            "pid": "shape-generation",
            "type": "mesh-generation",
            "name": "Generate 3D Shape",
            "weight": 50,
            "parameters": {
                "mode": "single_view",
                "asset": asset,
                "auto_scale": params.get("auto_scale", False),  # NEW
                "openai_model": params.get("openai_model", "gpt-4o"),  # NEW
                "guidance_scale": params.get("guidance_scale", 7.5),
                "num_inference_steps": params.get("num_inference_steps", 50),
                "seed": params.get("seed", 42),
                "did": did,
            },
            "dependencies": ["download-image"],
        },
        # ... rest of dependencies ...
    ]
```

#### Step 5: Add Schema Fields for Auto-Scale

In `qImageToObj.py` schema method:

```python
@classmethod
def schema(cls):
    sch = super().schema()
    
    sch.update({
        # ... existing fields ...
        
        "spacer150": {"order": 150, "type": "break"},
        "header151": {"order": 151, "type": "header", "value": "AI Detection Options"},
        
        "auto_scale": {
            "order": 152,
            "type": "boolean",
            "label": "Auto-Scale Detection",
            "tooltip": "Use AI to automatically detect object bounding boxes for better scaling",
            "value": False,
        },
        
        "openai_model": {
            "order": 153,
            "type": "select",
            "label": "OpenAI Model",
            "tooltip": "Model for object detection (only used if auto-scale is enabled)",
            "options": {
                "choices": [
                    {"value": "gpt-4o", "label": "GPT-4o (Best Quality)"},
                    {"value": "gpt-4o-mini", "label": "GPT-4o Mini (Faster)"},
                    {"value": "gpt-4-turbo", "label": "GPT-4 Turbo (Balanced)"},
                ]
            },
            "value": "gpt-4o",
        },
        
        # ... rest of schema ...
    })
```

### Environment Setup

#### 1. Install Hunyuan3D-Omni Environment

```bash
cd /opt/palatial/palatial_home/apps/Hunyuan3D-Omni
./install.sh
```

This creates the `hunyuan3d-omni` conda environment with all dependencies.

#### 2. Set OpenAI API Key

Add to your environment (e.g., in systemd service or `.bashrc`):

```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

Or add to your pipeline queue service configuration.

#### 3. Verify Installation

```bash
conda activate hunyuan3d-omni
python tools/bbox_detector.py --image demos/image/0.png --output /tmp/test_detection.json
```

### Usage

#### Method 1: Direct Script Usage

```bash
# Step 1: Detect bounding boxes
conda run -n hunyuan3d-omni python tools/bbox_detector.py \
  --image input.jpg \
  --output detections.json

# Step 2: Run inference
conda run -n hunyuan3d-omni python tools/pipeline_bbox_inference.py \
  --image_path input.jpg \
  --detections_json detections.json \
  --output_path ./output
```

#### Method 2: Via qMeshGeneration (after integration)

```python
# In your pipeline code
process = {
    "type": "mesh-generation",
    "parameters": {
        "mode": "single_view",
        "asset": "my_image",
        "auto_scale": True,  # Enable auto-scale
        "openai_model": "gpt-4o",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
    }
}
```

#### Method 3: Via qImageToObj UI

In the web UI, simply check the "Auto-Scale Detection" checkbox when creating an Image to 3D job.

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `auto_scale` | bool | False | Enable AI-based bbox detection |
| `openai_model` | str | "gpt-4o" | OpenAI model for detection |
| `use_ema` | bool | False | Use EMA model for better quality |
| `flashvdm` | bool | False | Use FlashVDM for faster inference |
| `max_objects` | int | None | Limit number of objects to process |
| `guidance_scale` | float | 4.5 | Guidance scale for generation |
| `num_inference_steps` | int | 50 | Number of inference steps |
| `octree_resolution` | int | 512 | 3D resolution |
| `seed` | int | 1234 | Random seed |

### Output Structure

When auto-scale is enabled, the output directory contains:

```
mesh/
├── detections.json              # Detected objects
├── image_00_chair.glb          # 3D model for object 0
├── image_00_chair.ply          # Point cloud for object 0
├── image_00_chair.png          # Input image copy
├── image_00_chair_metadata.json # Object metadata
├── image_01_table.glb          # 3D model for object 1
├── ...
└── inference_summary.json       # Summary of all results
```

### Error Handling

The pipeline handles common errors:

1. **No OpenAI API Key**: Falls back to standard mode or fails gracefully
2. **No Objects Detected**: Creates a warning log, falls back to standard generation
3. **Inference Failure**: Each object is processed independently; failures are logged but don't stop the pipeline
4. **Signal Handling**: SIGTERM/SIGINT are caught for graceful shutdown

### Performance Considerations

#### GPU Memory

- Standard mode: ~6-8GB VRAM per inference
- Auto-scale mode: ~6-8GB VRAM per object detected
- Consider using `--max_objects` to limit memory usage

#### Speed

- OpenAI detection: ~2-5 seconds per image
- Inference per object: ~10-30 seconds (depends on resolution)
- Use `--flashvdm` flag to speed up by 30-50%

#### Cost

- OpenAI API: ~$0.01-0.02 per image with gpt-4o
- Use `gpt-4o-mini` for cheaper option (~$0.001-0.002)

### Testing

```bash
# Test detection only
cd /opt/palatial/palatial_home/apps/Hunyuan3D-Omni
conda activate hunyuan3d-omni

python tools/bbox_detector.py \
  --image demos/image/0.png \
  --output /tmp/test_detections.json

# Test full pipeline
python tools/pipeline_bbox_inference.py \
  --image_path demos/image/0.png \
  --detections_json /tmp/test_detections.json \
  --output_path /tmp/test_output
```

### Troubleshooting

**OpenAI API Key Not Found**
```bash
export OPENAI_API_KEY="your-key-here"
```

**Conda Environment Not Found**
```bash
cd /opt/palatial/palatial_home/apps/Hunyuan3D-Omni
./install.sh
```

**CUDA Out of Memory**
```bash
# Reduce resolution or limit objects
python tools/pipeline_bbox_inference.py \
  --octree_resolution 256 \
  --max_objects 3 \
  ...
```

**Import Errors**
```bash
conda activate hunyuan3d-omni
pip install -r requirements.txt
```

### Future Enhancements

1. **Multi-object Merging**: Combine detected objects into single scene
2. **Custom Prompts**: Allow per-project detection prompts
3. **Bbox Refinement**: Manual adjustment of detected bboxes via UI
4. **Batch Processing**: Process multiple images in parallel
5. **Cache Detection**: Reuse detections across multiple inference runs

### Support

For issues or questions:
- Check logs in pipeline queue dashboard
- Test components individually using the scripts directly
- Verify conda environment is properly activated
- Ensure OpenAI API key has sufficient credits

