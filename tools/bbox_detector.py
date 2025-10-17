#!/usr/bin/env python
"""
Bounding Box Detector using OpenAI Vision API with Structured Outputs
Uses chain-of-thought reasoning and web search for accurate bbox prediction
Part of the Palatial Pipeline for Hunyuan3D-Omni integration
"""

import os
import sys
import json
import base64
import logging
from typing import List, Dict, Optional
from pathlib import Path

try:
    from openai import OpenAI
    from pydantic import BaseModel, Field, field_validator
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Install with: pip install openai pydantic")
    sys.exit(1)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Pydantic Models for Structured Outputs
# -----------------------------------------------------------------------------

class ObjectIdentification(BaseModel):
    """First stage: Identify the object"""
    object_name: str = Field(description="Short, descriptive name (lowercase with underscores)")
    category: str = Field(description="Object category (e.g., furniture, electronics, kitchenware)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in identification (0-1)")
    search_query: str = Field(description="Web search query to find real-world dimensions")
    reasoning: str = Field(description="Why you identified this object")


class RealWorldDimensions(BaseModel):
    """Dimensions found from web search or estimation (in meters)"""
    length_m: float = Field(gt=0, description="Length in meters")
    width_m: float = Field(gt=0, description="Width in meters") 
    height_m: float = Field(gt=0, description="Height in meters")
    source: str = Field(description="Source of dimensions (web_search, standard_size, or visual_estimate)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in dimensions")


class BoundingBox2D(BaseModel):
    """2D bounding box in normalized coordinates"""
    x_min: float = Field(ge=0.0, le=1.0, description="Left edge (0-1)")
    y_min: float = Field(ge=0.0, le=1.0, description="Top edge (0-1)")
    x_max: float = Field(ge=0.0, le=1.0, description="Right edge (0-1)")
    y_max: float = Field(ge=0.0, le=1.0, description="Bottom edge (0-1)")
    
    @field_validator('x_max')
    @classmethod
    def x_max_greater_than_min(cls, v, info):
        if 'x_min' in info.data and v <= info.data['x_min']:
            raise ValueError('x_max must be greater than x_min')
        return v
    
    @field_validator('y_max')
    @classmethod
    def y_max_greater_than_min(cls, v, info):
        if 'y_min' in info.data and v <= info.data['y_min']:
            raise ValueError('y_max must be greater than y_min')
        return v


class DepthEstimate(BaseModel):
    """Depth estimation in normalized space"""
    z_min: float = Field(ge=0.0, le=1.0, description="Near depth (0=close, 1=far)")
    z_max: float = Field(ge=0.0, le=1.0, description="Far depth (0=close, 1=far)")
    cues_used: List[str] = Field(description="Visual cues used (perspective, shadows, size, occlusion)")
    
    @field_validator('z_max')
    @classmethod
    def z_max_greater_than_min(cls, v, info):
        if 'z_min' in info.data and v <= info.data['z_min']:
            raise ValueError('z_max must be greater than z_min')
        return v


class DetectedObject(BaseModel):
    """Complete detected object with all information"""
    name: str = Field(description="Object identifier")
    description: str = Field(description="Brief description")
    bbox_2d: List[float] = Field(min_length=4, max_length=4, description="[x_min, y_min, x_max, y_max]")
    depth: List[float] = Field(min_length=2, max_length=2, description="[z_min, z_max]")
    real_world_dimensions: Optional[Dict[str, float]] = Field(default=None, description="Real dimensions in meters")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall detection confidence")
    reasoning: str = Field(description="Chain-of-thought reasoning for bbox placement")


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image to base64 for OpenAI API.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def identify_objects_with_reasoning(
    client: OpenAI,
    image_path: str,
    base64_image: str,
    custom_prompt: Optional[str] = None
) -> List[ObjectIdentification]:
    """Stage 1: Identify objects in the image with reasoning"""
    
    system_prompt = """You are an expert object recognition system. Analyze the image and identify all distinct, well-defined objects suitable for 3D reconstruction.

For each object:
1. Identify what it is (be specific but concise)
2. Categorize it
3. Assess your confidence
4. Create a web search query to find its typical dimensions
5. Explain your reasoning

Guidelines:
- Only detect clear, unoccluded objects
- Avoid tiny objects or background elements
- Use descriptive names (e.g., "wooden_dining_chair" not "chair1")
- Focus on objects that would benefit from 3D modeling"""

    user_text = custom_prompt or "Identify all distinct objects in this image that would be good candidates for 3D reconstruction."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=1500
        )
        
        result = json.loads(response.choices[0].message.content)
        objects = result.get("objects", [])
        
        return [ObjectIdentification(**obj) for obj in objects]
    except Exception as e:
        logger.error(f"Object identification failed: {e}")
        return []


def estimate_depth_from_image(
    image_path: str,
    bbox: List[float],
    use_depth_model: bool = False
) -> List[float]:
    """
    Estimate depth range for an object in the image.
    
    IMPORTANT: For single-object images with volumetric scaling approach,
    depth is LESS CRITICAL than you might think!
    
    Why depth matters less:
    1. 2D bbox captures the VISUAL SHAPE (what we see) - this is most important
    2. Real-world dimensions provide ACTUAL SIZE (meters)
    3. Volumetric scaling ensures CORRECT PROPORTIONS
    4. Depth mainly affects the Z-dimension of the initial bbox, which gets
       scaled uniformly anyway
    
    Depth is more important for:
    - Multi-object scenes (relative positioning)
    - Complex spatial relationships
    - When no real-world dimensions available
    
    For single objects (bottle, cup, etc.), a reasonable default depth
    works perfectly fine with our volumetric scaling approach.
    
    Args:
        image_path: Path to image
        bbox: Normalized bbox [x1, y1, x2, y2] in 0-1 range
        use_depth_model: Use monocular depth estimation model (slower)
        
    Returns:
        [z_min, z_max] in normalized 0-1 range (0=close, 1=far)
    """
    
    # For most single-object images, use sensible defaults
    # The 3D generation relies more on bbox shape than depth
    
    if use_depth_model:
        try:
            # Optional: Use Depth-Anything V2 or MiDaS for better depth estimation
            # pip install depth-anything-v2
            from depth_anything_v2.dpt import DepthAnythingV2
            import torch
            import cv2
            import numpy as np
            
            if not hasattr(estimate_depth_from_image, '_depth_model'):
                logger.info("Loading Depth-Anything V2 model...")
                estimate_depth_from_image._depth_model = DepthAnythingV2(
                    encoder='vitl',
                    features=256,
                    out_channels=[256, 512, 1024, 1024]
                )
                estimate_depth_from_image._depth_model.load_state_dict(
                    torch.load('checkpoints/depth_anything_v2_vitl.pth')
                )
                estimate_depth_from_image._depth_model.eval()
            
            # Load image
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            
            # Get depth map
            depth = estimate_depth_from_image._depth_model.infer_image(image)
            
            # Extract depth in bbox region
            x1, y1, x2, y2 = bbox
            x1_px, y1_px = int(x1 * w), int(y1 * h)
            x2_px, y2_px = int(x2 * w), int(y2 * h)
            
            bbox_depth = depth[y1_px:y2_px, x1_px:x2_px]
            
            # Normalize to 0-1 range
            min_depth = np.percentile(bbox_depth, 10)
            max_depth = np.percentile(bbox_depth, 90)
            
            # Normalize to scene
            scene_min = depth.min()
            scene_max = depth.max()
            
            z_min = (min_depth - scene_min) / (scene_max - scene_min)
            z_max = (max_depth - scene_min) / (scene_max - scene_min)
            
            return [round(float(z_min), 2), round(float(z_max), 2)]
            
        except Exception as e:
            logger.warning(f"Depth model failed: {e}, using defaults")
    
    # Default depth estimation strategies
    
    # Strategy 1: Use vertical position (lower in image = closer for ground objects)
    y_center = (bbox[1] + bbox[3]) / 2
    if y_center > 0.6:  # Lower in frame
        return [0.2, 0.5]  # Closer
    elif y_center < 0.4:  # Upper in frame  
        return [0.5, 0.8]  # Farther
    else:
        return [0.3, 0.7]  # Mid-range (most common for single objects)


def detect_bounding_boxes_qwen3(
    image_path: str,
    prompt: Optional[str] = None,
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    output_json: Optional[str] = None,
    use_fp8: bool = False,
    use_flash_attn: bool = True
) -> List[Dict]:
    """
    Detect objects using local Qwen3-VL model.
    
    Note: Qwen3-VL uses 0-1000 coordinate range (not 0-1)
    
    Args:
        image_path: Path to input image
        prompt: Custom prompt (optional, uses default if None)
        model: Qwen3-VL model name
        output_json: Optional path to save detection results
        use_fp8: Use FP8 quantization for lower VRAM usage
        use_flash_attn: Use flash attention 2 (recommended)
        
    Returns:
        List of detected objects with bbox information
    """
    logger.info(f"Analyzing image with Qwen3-VL: {image_path}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        import torch
    except ImportError as e:
        logger.error("="*70)
        logger.error("Qwen3-VL Support Not Available")
        logger.error("="*70)
        logger.error("Qwen3-VL requires newer transformers version (conflict with Hunyuan)")
        logger.error("")
        logger.error("Options:")
        logger.error("  1. Use OpenAI backend instead:")
        logger.error("     python tools/bbox_detector.py --backend openai --image ...")
        logger.error("")
        logger.error("  2. Install Qwen3-VL in separate environment:")
        logger.error("     bash tools/install_qwen3vl.sh")
        logger.error("")
        logger.error("  3. See full compatibility guide:")
        logger.error("     cat tools/QWEN3_COMPATIBILITY.md")
        logger.error("")
        logger.error(f"Error details: {e}")
        logger.error("="*70)
        raise ImportError(
            "Qwen3-VL not available. Use --backend openai or see tools/QWEN3_COMPATIBILITY.md"
        ) from e
    
    # Initialize model (cached after first load)
    if not hasattr(detect_bounding_boxes_qwen3, '_model'):
        logger.info(f"Loading Qwen3-VL model: {model}")
        
        # Model initialization with recommended settings
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True
        }
        
        if use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif use_fp8:
            model_kwargs["torch_dtype"] = torch.float8_e4m3fn
        else:
            model_kwargs["dtype"] = "auto"
        
        detect_bounding_boxes_qwen3._model = Qwen3VLForConditionalGeneration.from_pretrained(
            model, **model_kwargs
        )
        detect_bounding_boxes_qwen3._processor = AutoProcessor.from_pretrained(model)
        logger.info("Model loaded successfully")
    
    model_obj = detect_bounding_boxes_qwen3._model
    processor = detect_bounding_boxes_qwen3._processor
    
    # Qwen3-VL optimized prompt for bbox detection with depth reasoning
    # Key: Qwen3 expects bbox in [x1, y1, x2, y2] format with 0-1000 range
    detection_prompt = """Analyze this image and locate every object suitable for 3D reconstruction.

Use MULTI-STAGE REASONING:

1. OBJECT IDENTIFICATION: What objects do you see?

2. DEPTH ANALYSIS: For each object, estimate depth (0-1 scale where 0=very close, 1=very far)
   Look for these depth cues:
   - Size: Larger objects typically closer
   - Position: Objects lower in frame often closer (if on ground plane)
   - Occlusion: Objects that are partially hidden are farther
   - Shadows: Direction and sharpness indicate depth
   - Focus: Blurrier objects may be farther
   - Atmospheric perspective: Hazier objects are farther
   - Overlap: What's in front vs behind
   
   Depth examples:
   - Handheld object (phone, cup): 0.1-0.3 (very close)
   - Tabletop object: 0.3-0.5 (close)
   - Standing person/furniture: 0.4-0.7 (mid-range)
   - Room background: 0.7-0.9 (far)
   - Distant landscape: 0.9-1.0 (very far)

3. BOUNDING BOX: Draw TIGHT boxes preserving visual aspect ratio
   - Coordinates in 0-1000 range
   - No extra padding
   - Capture the object's silhouette

4. REAL-WORLD SIZE: Estimate dimensions in METERS
   - For cylindrical objects (bottles, cans): length_m and width_m = DIAMETER
   - For spherical objects (balls, fruits): all dimensions = DIAMETER
   - Common sizes:
     * Water bottle: 0.08m dia × 0.25m tall
     * Coffee mug: 0.09m dia × 0.10m tall
     * Smartphone: 0.15m × 0.08m × 0.01m
     * Dining chair: 0.45m × 0.90m × 0.50m
     * Basketball: 0.24m diameter (all dims)
     * Banana: 0.18m × 0.04m × 0.04m

Output JSON format:
{
  "objects": [
    {
      "bbox_2d": [x1, y1, x2, y2],
      "label": "object_name",
      "depth": [z_min, z_max],
      "real_world_dimensions": {"length_m": L, "width_m": W, "height_m": H},
      "confidence": 0.85,
      "depth_reasoning": "brief explanation of depth cues observed"
    }
  ]
}

Be precise and use the depth cues to make informed estimates!"""
    
    user_request = prompt or "Detect all objects."
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": f"{detection_prompt}\n\n{user_request}"}
        ]
    }]
    
    # Prepare inputs using Qwen3's format
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model_obj.device)
    
    # Generate
    logger.info("Running Qwen3-VL inference...")
    generated_ids = model_obj.generate(
        **inputs,
        max_new_tokens=2500,
        temperature=0.2
    )
    
    # Trim input from output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    logger.debug(f"Qwen3-VL response: {output_text[:500]}...")
    
    # Parse JSON response
    try:
        # Handle markdown fencing
        if "```json" in output_text:
            output_text = output_text.split("```json")[1].split("```")[0]
        
        json_start = output_text.find('{')
        json_end = output_text.rfind('}') + 1
        if json_start == -1 or json_end == 0:
            logger.error("No JSON found in response")
            return []
        
        json_str = output_text[json_start:json_end]
        result = json.loads(json_str)
        detections = result.get("objects", [])
        
        # Get image dimensions for coordinate conversion
        image = Image.open(image_path)
        width, height = image.size
        
        # Convert and validate detections
        validated_detections = []
        for det in detections:
            try:
                if 'bbox_2d' not in det or 'label' not in det:
                    logger.warning(f"Skipping detection with missing fields")
                    continue
                
                # Convert Qwen3 bbox (0-1000 range) to normalized (0-1 range)
                bbox_qwen = det['bbox_2d']
                if len(bbox_qwen) == 4:
                    bbox_normalized = [
                        round(bbox_qwen[0] / 1000, 2),  # x1
                        round(bbox_qwen[1] / 1000, 2),  # y1
                        round(bbox_qwen[2] / 1000, 2),  # x2
                        round(bbox_qwen[3] / 1000, 2)   # y2
                    ]
                    det['bbox_2d'] = bbox_normalized
                
                # Add missing fields
                det['name'] = det.get('label', 'object')
                det.setdefault('description', det['name'])
                det.setdefault('confidence', 0.85)
                det.setdefault('reasoning', 'Detected with Qwen3-VL spatial grounding')
                
                # Estimate depth if not provided
                if 'depth' not in det:
                    det['depth'] = estimate_depth_from_image(
                        image_path, 
                        bbox_normalized,
                        use_depth_model=False  # Set to True to use Depth-Anything V2
                    )
                
                # Normalize real-world dimensions
                if 'real_world_dimensions' in det:
                    for key in ['length_m', 'width_m', 'height_m']:
                        if key in det['real_world_dimensions']:
                            det['real_world_dimensions'][key] = round(
                                float(det['real_world_dimensions'][key]), 2
                            )
                
                det['confidence'] = round(float(det.get('confidence', 0.85)), 2)
                
                validated_detections.append(det)
            except Exception as e:
                logger.warning(f"Failed to validate detection: {e}")
                continue
        
        logger.info(f"Successfully detected {len(validated_detections)} objects with Qwen3-VL")
        
        # Save results
        if output_json:
            os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else '.', exist_ok=True)
            with open(output_json, 'w') as f:
                json.dump(validated_detections, f, indent=2)
            logger.info(f"Saved detections to: {output_json}")
        
        return validated_detections
        
    except Exception as e:
        logger.error(f"Failed to parse Qwen3-VL response: {e}")
        logger.debug(f"Full response: {output_text}")
        return []


def detect_bounding_boxes_with_cot(
    image_path: str,
    prompt: Optional[str] = None,
    model: str = "gpt-4o",
    output_json: Optional[str] = None,
    use_web_search: bool = False
) -> List[Dict]:
    """
    Detect objects and bounding boxes using chain-of-thought reasoning.
    
    Multi-stage process:
    1. Identify objects in the image
    2. Optionally search web for real-world dimensions
    3. Estimate 2D bounding boxes with spatial reasoning
    4. Estimate depth using visual cues
    5. Combine all information with confidence scores
    
    Args:
        image_path: Path to input image
        prompt: Custom prompt for detection (optional)
        model: OpenAI model to use (gpt-4o recommended)
        output_json: Optional path to save detection results
        use_web_search: Enable web search for dimension lookup (experimental)
        
    Returns:
        List of detected objects with comprehensive bbox information
        
    Note:
        Requires OPENAI_API_KEY environment variable to be set
    """
    logger.info(f"Analyzing image with chain-of-thought reasoning: {image_path}")
    
    # Validate image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Multi-stage chain-of-thought detection
    system_prompt = """You are an expert computer vision system specializing in object detection and 3D reconstruction.

Use chain-of-thought reasoning to analyze objects:

1. IDENTIFICATION: What objects do you see? What are their key characteristics?
2. SPATIAL ANALYSIS: Where are they in the image? How do they relate to each other?
3. BBOX PLACEMENT: Draw TIGHT bounding boxes that capture the object's VISUAL PROPORTIONS
4. DEPTH REASONING: Use perspective, shadows, occlusion, and relative size to estimate depth
5. DIMENSION ESTIMATION: Based on object type and context, estimate real-world size IN METERS

Return your analysis as JSON with this structure:
{
  "objects": [
    {
      "name": "object_identifier",
      "description": "what it is and key visual features",
      "bbox_2d": [x_min, y_min, x_max, y_max],
      "depth": [z_min, z_max],
      "real_world_dimensions": {"length_m": L, "width_m": W, "height_m": H},
      "confidence": 0.0-1.0,
      "reasoning": "Your step-by-step thought process for this detection"
    }
  ]
}

CRITICAL GUIDELINES FOR 2D BOUNDING BOX:
- bbox_2d should capture the VISUAL SHAPE as seen in the image
- Draw TIGHT boxes - no extra padding
- Preserve the aspect ratio you see (if it looks 2:1 in image, bbox should be ~2:1)
- Use normalized coordinates: 0.0 = left/top edge, 1.0 = right/bottom edge
- Example: Tall bottle taking 20% width, 60% height → bbox width ≈ 0.2, height ≈ 0.6

GUIDELINES FOR REAL-WORLD DIMENSIONS:
- ALL dimensions must be in METERS (e.g., banana: ~0.18m long, coffee cup: ~0.10m tall)
- For cylindrical objects (bottles, cans, cups): length_m and width_m should be the DIAMETER
- For spherical objects (balls, fruits): all three dimensions should be equal (the diameter)
- Be realistic: typical water bottle is ~0.08m diameter × 0.25m tall, not 0.25m × 0.25m
- Consider typical object sizes in your reasoning

OTHER GUIDELINES:
- Only detect clear, reconstructible objects
- Use visual cues: perspective lines, shadows, occlusion, relative sizes
- Be explicit about uncertainty in your reasoning field"""

    # Get API key from environment and initialize client
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)
    base64_image = encode_image_to_base64(image_path)
    
    # Prepare user message
    user_text = prompt or "Detect and analyze all objects in this image suitable for 3D reconstruction. Use detailed chain-of-thought reasoning."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}
            ]
        }
    ]
    
    try:
        logger.info("Sending request to OpenAI with structured output...")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=2500
        )
        
        result = json.loads(response.choices[0].message.content)
        detections = result.get("objects", [])
        
        if not detections:
            logger.warning("No objects detected in image")
            return []
        
        # Validate and structure detections
        validated_detections = []
        for det in detections:
            try:
                # Ensure required fields exist
                if not all(k in det for k in ['name', 'bbox_2d', 'depth']):
                    logger.warning(f"Skipping detection with missing fields: {det.get('name', 'unknown')}")
                    continue
                
                # Add defaults for optional fields first
                det.setdefault('description', det.get('name', 'object'))
                det.setdefault('confidence', 0.8)
                det.setdefault('reasoning', 'Detected with standard analysis')
                
                # Normalize bbox coordinates to 2 decimal places
                if 'bbox_2d' in det and isinstance(det['bbox_2d'], list) and len(det['bbox_2d']) == 4:
                    det['bbox_2d'] = [round(float(coord), 2) for coord in det['bbox_2d']]
                
                # Normalize depth to 2 decimal places
                if 'depth' in det and isinstance(det['depth'], list) and len(det['depth']) == 2:
                    det['depth'] = [round(float(coord), 2) for coord in det['depth']]
                
                # Normalize real-world dimensions to 2 decimal places
                if 'real_world_dimensions' in det and isinstance(det['real_world_dimensions'], dict):
                    for key in ['length_m', 'width_m', 'height_m']:
                        if key in det['real_world_dimensions']:
                            det['real_world_dimensions'][key] = round(float(det['real_world_dimensions'][key]), 2)
                
                # Normalize confidence to 2 decimal places
                det['confidence'] = round(float(det['confidence']), 2)
                
                validated_detections.append(det)
            except Exception as e:
                logger.warning(f"Failed to validate detection: {e}")
                continue
        
        logger.info(f"Successfully detected {len(validated_detections)} objects")
        
        # Log reasoning for debugging
        for i, det in enumerate(validated_detections):
            logger.debug(f"Object {i+1} ({det['name']}): {det.get('reasoning', 'N/A')[:100]}...")
        
        # Save results
        if output_json:
            os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else '.', exist_ok=True)
            with open(output_json, 'w') as f:
                json.dump(validated_detections, f, indent=2)
            logger.info(f"Saved detections to: {output_json}")
        
        return validated_detections
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise


# Maintain backward compatibility
def detect_bounding_boxes(*args, **kwargs):
    """Backward compatible wrapper"""
    return detect_bounding_boxes_with_cot(*args, **kwargs)


def apply_geometry_heuristics(
    length: float,
    width: float, 
    height: float,
    width_2d: float,
    height_2d: float,
    depth_2d: float
) -> tuple:
    """
    Apply geometric heuristics to correct common AI detection errors.
    
    Heuristics:
    1. Cylindrical objects (bottles, cans, cups) - enforce circular cross-section
    2. Spherical objects (balls, fruits) - enforce equal dimensions
    3. Flat circular objects (plates, coins) - enforce length ≈ width, small height
    4. Correct obvious axis confusion (height confused with horizontal dimension)
    
    Args:
        length, width, height: Real-world dimensions in meters
        width_2d, height_2d, depth_2d: 2D bbox dimensions (0-1 range)
        
    Returns:
        Corrected (length, width, height) tuple
    """
    # Safety check for zero dimensions
    if min(length, width, height) <= 0:
        return length, width, height
    
    # Calculate dimension ratios
    horizontal_ratio = max(length, width) / min(length, width)
    max_dim = max(length, width, height)
    min_dim = min(length, width, height)
    
    # --- HEURISTIC 1: Vertical Cylinder Detection ---
    # If one horizontal dimension is much larger than the other AND close to height,
    # likely a vertical cylinder with confused orientation
    if horizontal_ratio > 2.5:  # Significant asymmetry in horizontal dimensions
        larger_h = max(length, width)
        smaller_h = min(length, width)
        
        # Check if larger horizontal dimension is suspiciously close to height
        if abs(larger_h - height) / height < 0.15:  # Within 15%
            logger.debug(f"  [Heuristic] Vertical cylinder detected: "
                        f"correcting ({length:.3f}, {width:.3f}, {height:.3f}) "
                        f"→ ({smaller_h:.3f}, {smaller_h:.3f}, {height:.3f})")
            return smaller_h, smaller_h, height
    
    # --- HEURISTIC 2: Spherical Object Detection ---
    # All three dimensions should be roughly equal
    all_dims = sorted([length, width, height])
    ratio_min_to_max = all_dims[0] / all_dims[2]
    ratio_mid_to_max = all_dims[1] / all_dims[2]
    
    if ratio_min_to_max > 0.7 and ratio_mid_to_max > 0.85:  # Very similar dimensions
        # If 2D bbox also shows similar proportions, enforce perfect sphere
        bbox_2d_ratios = sorted([width_2d, height_2d, depth_2d])
        if bbox_2d_ratios[0] / bbox_2d_ratios[2] > 0.6:  # 2D also looks roundish
            avg_dim = sum(all_dims) / 3
            logger.debug(f"  [Heuristic] Spherical object detected: "
                        f"correcting ({length:.3f}, {width:.3f}, {height:.3f}) "
                        f"→ ({avg_dim:.3f}, {avg_dim:.3f}, {avg_dim:.3f})")
            return avg_dim, avg_dim, avg_dim
    
    # --- HEURISTIC 3: Flat Circular Object Detection ---
    # Length ≈ width (circular), but height << diameter
    if horizontal_ratio < 1.3:  # Horizontal dimensions similar
        avg_horizontal = (length + width) / 2
        if height / avg_horizontal < 0.15:  # Height < 15% of diameter
            logger.debug(f"  [Heuristic] Flat circular object detected: "
                        f"correcting ({length:.3f}, {width:.3f}, {height:.3f}) "
                        f"→ ({avg_horizontal:.3f}, {avg_horizontal:.3f}, {height:.3f})")
            return avg_horizontal, avg_horizontal, height
    
    # --- HEURISTIC 4: Horizontal Cylinder Detection ---
    # If height and one horizontal dimension are similar, but other is much larger
    # Could be a horizontal cylinder (bottle lying down)
    if abs(height - width) / max(height, width) < 0.15:  # height ≈ width
        if length / height > 2.5:  # length much larger
            smaller_h = min(height, width)
            logger.debug(f"  [Heuristic] Horizontal cylinder detected: "
                        f"correcting ({length:.3f}, {width:.3f}, {height:.3f}) "
                        f"→ ({length:.3f}, {smaller_h:.3f}, {smaller_h:.3f})")
            return length, smaller_h, smaller_h
    
    if abs(height - length) / max(height, length) < 0.15:  # height ≈ length
        if width / height > 2.5:  # width much larger
            smaller_h = min(height, length)
            logger.debug(f"  [Heuristic] Horizontal cylinder detected: "
                        f"correcting ({length:.3f}, {width:.3f}, {height:.3f}) "
                        f"→ ({smaller_h:.3f}, {width:.3f}, {smaller_h:.3f})")
            return smaller_h, width, smaller_h
    
    # --- HEURISTIC 5: Aspect Ratio Sanity Check ---
    # If any dimension is > 10x another (excluding elongated objects like pens/bats)
    # and the 2D bbox doesn't show this, likely an error
    if max_dim / min_dim > 10:
        bbox_max_ratio = max(width_2d, height_2d, depth_2d) / min(width_2d, height_2d, depth_2d)
        if bbox_max_ratio < 5:  # 2D doesn't show extreme elongation
            # Cap the ratio to 5:1
            scale = 5.0 / (max_dim / min_dim)
            logger.debug(f"  [Heuristic] Extreme aspect ratio detected: "
                        f"scaling dimensions by {scale:.3f}")
            return length * scale, width * scale, height * scale
    
    # No corrections applied
    return length, width, height


def convert_to_3d_bbox(
    bbox_2d: List[float],
    depth_range: List[float],
    real_world_dims: Optional[Dict[str, float]] = None
) -> tuple[List[float], Optional[float]]:
    """
    Convert 2D bounding box with depth to 3D scale vector for Hunyuan3D-Omni.
    
    NEW STRATEGY (Volumetric Scaling):
    1. Always use 2D bbox proportions for shape (captures visual appearance)
    2. If real-world dims available, return scale factor for post-generation scaling
    3. This separates shape (from image) and size (from real-world knowledge)
    
    Hunyuan3D-Omni expects bbox as [length, height, width] in 0-1 range.
    Internally converts to 8 corner points in [-1,1] range.
    
    Args:
        bbox_2d: [x_min, y_min, x_max, y_max] in normalized coords (0-1)
        depth_range: [z_min, z_max] in normalized coords (0-1)
        real_world_dims: Optional real-world dimensions in meters
        
    Returns:
        Tuple of:
        - bbox_3d: [width_scale, height_scale, depth_scale] - 3 values in 0-1 range
        - scale_factor: Optional uniform scale to apply to generated mesh (None if no real dims)
        
    Examples:
        Water bottle with 2D bbox [0.2,0.1,0.8,0.9], depth [0.0,0.2]:
        → bbox_3d: [0.75, 1.0, 0.25] (normalized 2D proportions)
        → scale_factor: 0.267 (to achieve 0.08m × 0.25m × 0.08m final size)
    """
    x_min, y_min, x_max, y_max = bbox_2d
    z_min, z_max = depth_range
    
    # Calculate bbox dimensions in normalized image space
    width_2d = x_max - x_min  # 0-1 range (horizontal in image)
    height_2d = y_max - y_min  # 0-1 range (vertical in image)
    depth_2d = z_max - z_min  # 0-1 range (depth/distance)
    
    # Always use 2D bbox proportions for shape (normalized to largest dim)
    max_dim_2d = max(width_2d, height_2d, depth_2d)
    
    if max_dim_2d > 0:
        width_scale = width_2d / max_dim_2d
        height_scale = height_2d / max_dim_2d
        depth_scale = depth_2d / max_dim_2d
    else:
        width_scale = height_scale = depth_scale = 1.0
    
    bbox_3d = [width_scale, height_scale, depth_scale]
    
    # Calculate volumetric scale factor if real-world dimensions available
    scale_factor = None
    if real_world_dims and all(k in real_world_dims for k in ['length_m', 'width_m', 'height_m']):
        # Get real-world dimensions (in meters)
        real_length = real_world_dims['length_m']
        real_width = real_world_dims['width_m']
        real_height = real_world_dims['height_m']
        
        # Calculate volumes
        # Current volume (from normalized 2D bbox)
        current_volume = width_scale * height_scale * depth_scale
        
        # Target volume (from real-world dimensions)
        target_volume = real_length * real_height * real_width
        
        # Uniform scale factor (cube root to preserve proportions)
        if current_volume > 0:
            scale_factor = (target_volume / current_volume) ** (1.0 / 3.0)
            logger.debug(f"  [Volumetric] Current vol: {current_volume:.6f}, "
                        f"Target vol: {target_volume:.6f}, Scale: {scale_factor:.3f}")
    
    return bbox_3d, scale_factor


def main():
    """Command-line interface for bbox detection"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Detect bounding boxes using OpenAI Vision API',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save detection JSON (default: {image_dir}/detections.json)')
    
    # Model selection
    parser.add_argument('--backend', type=str, default='openai',
                        choices=['openai', 'qwen3'],
                        help='Detection backend to use (openai or qwen3)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name (default: gpt-4o for OpenAI, Qwen/Qwen3-VL-8B-Instruct for Qwen3)')
    
    # Qwen3-specific options
    parser.add_argument('--flash_attn', action='store_true', default=True,
                        help='Use Flash Attention 2 for Qwen3 (recommended)')
    parser.add_argument('--no_flash_attn', action='store_false', dest='flash_attn',
                        help='Disable Flash Attention 2')
    parser.add_argument('--use_fp8', action='store_true', default=False,
                        help='Use FP8 quantization for Qwen3 (ultra low VRAM)')
    
    # Common options
    parser.add_argument('--prompt', type=str, default=None,
                        help='Custom detection prompt')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='[%(levelname)s] %(message)s'
    )
    
    # Default output path
    if args.output is None:
        image_dir = os.path.dirname(args.image) or '.'
        args.output = os.path.join(image_dir, 'detections.json')
    
    # Set default model if not specified
    if args.model is None:
        args.model = 'gpt-4o' if args.backend == 'openai' else 'Qwen/Qwen3-VL-8B-Instruct'
    
    # Run detection with selected backend
    try:
        if args.backend == 'qwen3':
            logger.info("Using Qwen3-VL (local) for detection")
            detections = detect_bounding_boxes_qwen3(
                image_path=args.image,
                prompt=args.prompt,
                model=args.model,
                output_json=args.output,
                use_fp8=args.use_fp8,
                use_flash_attn=args.flash_attn
            )
        else:
            logger.info("Using OpenAI API for detection")
            detections = detect_bounding_boxes(
                image_path=args.image,
                prompt=args.prompt,
                model=args.model,
                output_json=args.output
            )
        
        print(f"\nDetected {len(detections)} objects:")
        for i, det in enumerate(detections):
            print(f"  {i+1}. {det['name']}: bbox={det['bbox_2d']}, depth={det['depth']}")
        
        print(f"\nResults saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

