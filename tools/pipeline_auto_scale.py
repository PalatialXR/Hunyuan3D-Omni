#!/usr/bin/env python
"""
Auto-Scale Pipeline Wrapper
Combines bbox detection and inference into a single command
Compatible with Palatial pipeline architecture

Usage:
    python pipeline_auto_scale.py --image_path input.jpg --output_path ./output
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def run_auto_scale_pipeline(
    image_path: str,
    output_dir: str,
    project_id: Optional[str] = None,
    openai_model: str = "gpt-4o",
    custom_prompt: Optional[str] = None,
    guidance_scale: float = 4.5,
    num_inference_steps: int = 50,
    octree_resolution: int = 512,
    seed: int = 1234,
    use_ema: bool = False,
    flashvdm: bool = False,
    log_level: str = "INFO",
    keep_detections: bool = True
) -> int:
    """
    Run complete auto-scale pipeline: detection + inference
    
    Returns:
        Exit code (0 = success, non-zero = failure)
    """
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='[%(levelname)s] %(message)s'
    )
    
    # Get script directory
    script_dir = Path(__file__).parent
    detector_script = script_dir / "bbox_detector.py"
    inference_script = script_dir / "pipeline_bbox_inference.py"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Detections JSON path
    detections_json = os.path.join(output_dir, "detections.json")
    
    logger.info("="*80)
    logger.info("AUTO-SCALE PIPELINE")
    logger.info("="*80)
    
    # Step 1: Detect bounding boxes
    logger.info("\nStep 1: Detecting objects with OpenAI...")
    
    detect_cmd = [
        sys.executable,  # Use same Python interpreter
        str(detector_script),
        "--image", image_path,
        "--output", detections_json,
        "--model", openai_model,
        "--log_level", log_level
    ]
    
    if custom_prompt:
        detect_cmd.extend(["--prompt", custom_prompt])
    
    logger.debug(f"Detection command: {' '.join(detect_cmd)}")
    
    try:
        # Explicitly pass environment to ensure OPENAI_API_KEY is inherited
        result = subprocess.run(detect_cmd, check=True, capture_output=False, env=os.environ)
    except subprocess.CalledProcessError as e:
        logger.error(f"Detection failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return 1
    
    # Verify detections file exists
    if not os.path.exists(detections_json):
        logger.error(f"Detections file not created: {detections_json}")
        return 1
    
    # Step 2: Run bbox inference
    logger.info("\nStep 2: Running 3D inference with detected bboxes...")
    
    # Build command as list (proper subprocess handling)
    infer_cmd = [
        sys.executable,
        str(inference_script),
        "--image_path", str(image_path),
        "--detections_json", str(detections_json),
        "--output_path", str(output_dir),
        "--guidance_scale", str(guidance_scale),
        "--num_inference_steps", str(num_inference_steps),
        "--octree_resolution", str(octree_resolution),
        "--seed", str(seed),
        "--log_level", str(log_level)
    ]
    
    # Pass output_file if project_id provided for standard naming convention
    if project_id:
        output_file = f"shape_{project_id}.glb"
        infer_cmd.extend(["--output_file", output_file])
    
    if use_ema:
        infer_cmd.append("--use_ema")
    
    if flashvdm:
        infer_cmd.append("--flashvdm")
    
    logger.info(f"Running inference command...")
    logger.debug(f"Command: {' '.join(infer_cmd)}")
    
    try:
        # Explicitly pass environment to ensure all env vars are inherited
        result = subprocess.run(infer_cmd, check=True, capture_output=False, env=os.environ)
    except subprocess.CalledProcessError as e:
        logger.error(f"Inference failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return 1
    
    # Cleanup if requested
    if not keep_detections:
        try:
            os.remove(detections_json)
            logger.info(f"Removed detections file: {detections_json}")
        except Exception as e:
            logger.warning(f"Could not remove detections file: {e}")
    
    logger.info("\n" + "="*80)
    logger.info("AUTO-SCALE PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    
    return 0


def main():
    """Command-line interface"""
    from typing import Optional  # Import here for CLI
    
    parser = argparse.ArgumentParser(
        description='Auto-Scale Pipeline: Automatic bbox detection and 3D inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--project_id', type=str, default=None,
                        help='Project ID for standard naming (e.g., shape_{project_id}.glb)')
    
    # OpenAI options
    # Note: API key is now required via OPENAI_API_KEY environment variable
    parser.add_argument('--openai_model', type=str, default='gpt-4o',
                        help='OpenAI model for detection')
    parser.add_argument('--custom_prompt', type=str, default=None,
                        help='Custom detection prompt')
    
    # Inference options
    parser.add_argument('--guidance_scale', type=float, default=4.5,
                        help='Guidance scale for generation')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Number of inference steps')
    parser.add_argument('--octree_resolution', type=int, default=512,
                        help='Resolution for 3D representation')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')
    parser.add_argument('--use_ema', action='store_true',
                        help='Use EMA model')
    parser.add_argument('--flashvdm', action='store_true',
                        help='Use FlashVDM for faster inference')
    
    # Other options
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--remove_detections', action='store_true',
                        help='Remove detections.json after processing')
    
    args = parser.parse_args()
    
    exit_code = run_auto_scale_pipeline(
        image_path=args.image_path,
        output_dir=args.output_path,
        project_id=args.project_id,
        openai_model=args.openai_model,
        custom_prompt=args.custom_prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        octree_resolution=args.octree_resolution,
        seed=args.seed,
        use_ema=args.use_ema,
        flashvdm=args.flashvdm,
        log_level=args.log_level,
        keep_detections=not args.remove_detections
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

