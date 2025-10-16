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
3. DIMENSION ESTIMATION: Based on context clues, what are likely real-world sizes IN METERS?
4. DEPTH REASONING: Use perspective, shadows, occlusion, and relative size to estimate depth
5. BBOX PLACEMENT: Precisely locate each object in normalized coordinates

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

Guidelines:
- Only detect clear, reconstructible objects
- Use visual cues: perspective lines, shadows, occlusion, relative sizes
- ALL dimensions must be in METERS (e.g., banana: ~0.18m long, coffee cup: ~0.15m tall)
- Consider typical object dimensions in your reasoning
- Be explicit about uncertainty in your reasoning field
- Tight bounding boxes around each object"""

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


def convert_to_3d_bbox(
    bbox_2d: List[float],
    depth_range: List[float],
    real_world_dims: Optional[Dict[str, float]] = None
) -> List[float]:
    """
    Convert 2D bounding box with depth to 3D scale vector for Hunyuan3D-Omni.
    
    Hunyuan3D-Omni bbox format is [width, height, depth] representing relative scales,
    NOT corner coordinates.
    
    Args:
        bbox_2d: [x_min, y_min, x_max, y_max] in normalized coords (0-1)
        depth_range: [z_min, z_max] in normalized coords (0-1)
        real_world_dims: Optional real-world dimensions in meters
        
    Returns:
        [width_scale, height_scale, depth_scale] - 3 values representing object dimensions
    """
    x_min, y_min, x_max, y_max = bbox_2d
    z_min, z_max = depth_range
    
    # Calculate bbox dimensions in normalized image space
    width_2d = x_max - x_min  # 0-1 range
    height_2d = y_max - y_min  # 0-1 range
    depth_2d = z_max - z_min  # 0-1 range
    
    # If we have real-world dimensions, use aspect ratios
    if real_world_dims and all(k in real_world_dims for k in ['length_m', 'width_m', 'height_m']):
        # Normalize to largest dimension
        max_dim = max(real_world_dims['length_m'], real_world_dims['width_m'], real_world_dims['height_m'])
        width_scale = real_world_dims['length_m'] / max_dim
        height_scale = real_world_dims['height_m'] / max_dim
        depth_scale = real_world_dims['width_m'] / max_dim
    else:
        # Use bbox dimensions directly, normalized
        # Scale to make largest dimension = 1.0
        max_dim = max(width_2d, height_2d, depth_2d)
        if max_dim > 0:
            width_scale = width_2d / max_dim
            height_scale = height_2d / max_dim
            depth_scale = depth_2d / max_dim
        else:
            width_scale = height_scale = depth_scale = 1.0
    
    return [width_scale, height_scale, depth_scale]


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
    # Note: API key is now required via OPENAI_API_KEY environment variable
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='OpenAI model to use')
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
    
    # Run detection
    try:
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

