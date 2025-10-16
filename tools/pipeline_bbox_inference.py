#!/usr/bin/env python
"""
Pipeline for Bounding Box Inference with Hunyuan3D-Omni
Follows the Palatial pipeline pattern with supervisor-style process management
"""

import os
import sys
import json
import argparse
import logging
import signal
import time
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import List, Dict, Optional
import torch
import trimesh
import shutil
import numpy as np
from PIL import Image

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Hunyuan3D imports
from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline
from hy3dshape.postprocessors import FloaterRemover, DegenerateFaceRemover

# Import bbox detector
from bbox_detector import detect_bounding_boxes, convert_to_3d_bbox

logger = logging.getLogger(__name__)


# Signal handler for graceful shutdown
class GracefulKiller:
    """Handle shutdown signals gracefully"""
    kill_now = False
    
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
    
    def exit_gracefully(self, *args):
        logger.warning("Received shutdown signal, cleaning up...")
        self.kill_now = True


def save_ply_points(filename: str, points: np.ndarray) -> None:
    """Save 3D points to PLY format."""
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n' % len(points))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        for point in points:
            f.write('%f %f %f\n' % (point[0], point[1], point[2]))


def postprocess_mesh(
    mesh,
    file_name: str,
    save_dir: str,
    sampled_point,
    image_file: str,
    apply_postprocessing: bool = True
):
    """Post-process and save mesh outputs."""
    if apply_postprocessing:
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save mesh
    mesh_path = os.path.join(save_dir, f'{file_name}.glb')
    mesh.export(mesh_path)
    logger.info(f"Saved mesh: {mesh_path}")
    
    # Save point cloud
    ply_path = os.path.join(save_dir, f'{file_name}.ply')
    save_ply_points(ply_path, sampled_point.cpu().numpy())
    
    # Copy input image
    image_copy = os.path.join(save_dir, f'{file_name}.png')
    shutil.copy(image_file, image_copy)


def run_bbox_inference(
    image_path: str,
    detections_json: str,
    output_dir: str,
    repo_id: str = "tencent/Hunyuan3D-Omni",
    guidance_scale: float = 4.5,
    num_inference_steps: int = 50,
    octree_resolution: int = 512,
    mc_level: int = 0,
    seed: int = 1234,
    use_ema: bool = False,
    flashvdm: bool = False,
    log_level: str = "INFO"
) -> Dict:
    """
    Run bounding box controlled 3D inference for detected objects.
    
    Args:
        image_path: Path to input image
        detections_json: Path to JSON file with detected bounding boxes
        output_dir: Directory to save generated 3D models
        repo_id: Hunyuan3D model repository ID
        guidance_scale: Guidance scale for generation
        num_inference_steps: Number of inference steps
        octree_resolution: Resolution for 3D representation
        mc_level: Marching cubes iso-level
        seed: Random seed for reproducibility
        use_ema: Use EMA model for inference
        flashvdm: Use FlashVDM for faster inference
        log_level: Logging level
        
    Returns:
        Dictionary with results summary
    """
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='[%(levelname)s] %(message)s'
    )
    
    killer = GracefulKiller()
    
    # Validate inputs
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if not os.path.exists(detections_json):
        raise FileNotFoundError(f"Detections JSON not found: {detections_json}")
    
    # Load detections
    with open(detections_json, 'r') as f:
        detections = json.load(f)
    
    if not detections:
        logger.warning("No objects detected in JSON")
        return {"status": "no_objects", "processed": 0, "failed": 0}
    
    logger.info(f"Loaded {len(detections)} detected objects")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize pipeline
    logger.info(f"Initializing Hunyuan3D-Omni pipeline from: {repo_id}")
    try:
        pipeline = Hunyuan3DOmniSiTFlowMatchingPipeline.from_pretrained(
            repo_id,
            fast_decode=flashvdm
        )
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise
    
    # Get image info
    image = Image.open(image_path)
    image_size = image.size
    base_name = Path(image_path).stem
    
    # Process each detected object
    results = {
        "status": "success",
        "processed": 0,
        "failed": 0,
        "objects": []
    }
    
    for idx, detection in enumerate(detections):
        if killer.kill_now:
            logger.warning("Shutdown signal received, stopping...")
            results["status"] = "interrupted"
            break
        
        obj_name = detection['name'].replace(' ', '_')
        bbox_2d = detection['bbox_2d']
        depth_range = detection.get('depth', [0.3, 0.7])
        
        # Convert to 3D bbox
        # Get real-world dimensions if available
        real_world_dims = detection.get('real_world_dimensions')
        bbox_3d = convert_to_3d_bbox(bbox_2d, depth_range, real_world_dims)
        
        logger.info(f"\n[{idx+1}/{len(detections)}] Processing: {detection['name']}")
        logger.info(f"  2D BBox: {bbox_2d}")
        logger.info(f"  Depth: {depth_range}")
        logger.info(f"  3D Scale: {[f'{v:.3f}' for v in bbox_3d]} (width, height, depth)")
        if real_world_dims:
            logger.info(f"  Real dims: {real_world_dims}")
        
        # Prepare bbox tensor [1, 1, 3] - Hunyuan3D-Omni expects 3 values
        bbox_tensor = torch.FloatTensor(bbox_3d).unsqueeze(0).unsqueeze(0)
        bbox_tensor = bbox_tensor.to(pipeline.device).to(pipeline.dtype)
        
        try:
            # Run inference
            result = pipeline(
                image=image_path,
                bbox=bbox_tensor,
                num_inference_steps=num_inference_steps,
                octree_resolution=octree_resolution,
                mc_level=mc_level,
                guidance_scale=guidance_scale,
                generator=torch.Generator('cuda').manual_seed(seed),
            )
            
            # Extract results
            mesh = result['shapes'][0][0]
            sampled_point = result['sampled_point'][0]
            
            # Save outputs
            file_name = f"{base_name}_{idx:02d}_{obj_name}"
            postprocess_mesh(mesh, file_name, output_dir, sampled_point, image_path)
            
            # Save detection metadata
            metadata = {
                'object_name': detection['name'],
                'description': detection.get('description', ''),
                'bbox_2d': bbox_2d,
                'depth_range': depth_range,
                'bbox_3d_scale': bbox_3d,
                'real_world_dimensions': real_world_dims,
                'inference_params': {
                    'guidance_scale': guidance_scale,
                    'num_inference_steps': num_inference_steps,
                    'octree_resolution': octree_resolution,
                    'seed': seed,
                    'use_ema': use_ema,
                    'flashvdm': flashvdm
                }
            }
            
            metadata_path = os.path.join(output_dir, f'{file_name}_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"  ✓ Saved: {file_name}.glb")
            
            results["processed"] += 1
            results["objects"].append({
                "name": obj_name,
                "file": f"{file_name}.glb",
                "status": "success"
            })
            
        except Exception as e:
            logger.error(f"  ✗ Error processing {detection['name']}: {e}")
            results["failed"] += 1
            results["objects"].append({
                "name": obj_name,
                "status": "failed",
                "error": str(e)
            })
            continue
    
    # Save summary
    summary_path = os.path.join(output_dir, 'inference_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Inference complete!")
    logger.info(f"  Processed: {results['processed']}/{len(detections)}")
    logger.info(f"  Failed: {results['failed']}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"{'='*80}")
    
    return results


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Bounding Box Controlled 3D Inference Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--detections_json', type=str, required=True,
                        help='Path to JSON file with detected bounding boxes')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output directory for generated 3D models')
    
    # Model arguments
    parser.add_argument('--repo_id', type=str, default='tencent/Hunyuan3D-Omni',
                        help='Hunyuan3D model repository ID')
    parser.add_argument('--use_ema', action='store_true',
                        help='Use EMA model for inference')
    parser.add_argument('--flashvdm', action='store_true',
                        help='Use FlashVDM for faster decoding')
    
    # Inference parameters
    parser.add_argument('--guidance_scale', type=float, default=4.5,
                        help='Guidance scale for generation')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Number of inference steps')
    parser.add_argument('--octree_resolution', type=int, default=512,
                        help='Resolution for 3D representation')
    parser.add_argument('--mc_level', type=int, default=0,
                        help='Marching cubes iso-level')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed for reproducibility')
    
    # Processing options
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    args = parser.parse_args()
    
    try:
        results = run_bbox_inference(
            image_path=args.image_path,
            detections_json=args.detections_json,
            output_dir=args.output_path,
            repo_id=args.repo_id,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            octree_resolution=args.octree_resolution,
            mc_level=args.mc_level,
            seed=args.seed,
            use_ema=args.use_ema,
            flashvdm=args.flashvdm,
            log_level=args.log_level
        )
        
        # Exit with appropriate code
        if results["status"] == "success" and results["failed"] == 0:
            sys.exit(0)
        elif results["status"] == "interrupted":
            sys.exit(130)  # SIGINT exit code
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

