#!/usr/bin/env python
"""Test script to verify Hunyuan3D-Omni installation."""

import sys

def test_imports():
    """Test if all critical packages can be imported."""
    print("Testing package imports...")
    
    packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('torchaudio', 'TorchAudio'),
        ('transformers', 'Transformers'),
        ('diffusers', 'Diffusers'),
        ('trimesh', 'Trimesh'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('cv2', 'OpenCV'),
        ('gradio', 'Gradio'),
    ]
    
    failed = []
    for module, name in packages:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name} - {e}")
            failed.append(name)
    
    return failed

def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Device count: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name(0)}")
            
            # Test VRAM
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  VRAM - Total: {mem_total:.2f} GB, Reserved: {mem_reserved:.2f} GB, Allocated: {mem_allocated:.2f} GB")
            
            if mem_total < 10:
                print(f"  ⚠ Warning: Model requires ~10GB VRAM, you have {mem_total:.2f} GB")
        else:
            print("  ⚠ CUDA is not available. GPU inference will not work.")
    except Exception as e:
        print(f"  ✗ Error testing CUDA: {e}")
        return False
    
    return True

def test_hunyuan3d_modules():
    """Test if Hunyuan3D-Omni specific modules can be imported."""
    print("\nTesting Hunyuan3D-Omni modules...")
    try:
        from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline
        print("  ✓ Hunyuan3DOmniSiTFlowMatchingPipeline")
        
        from hy3dshape.preprocessors import ImageProcessorV2
        print("  ✓ ImageProcessorV2")
        
        from hy3dshape.postprocessors import FloaterRemover, DegenerateFaceRemover
        print("  ✓ FloaterRemover, DegenerateFaceRemover")
        
        return True
    except ImportError as e:
        print(f"  ✗ Failed to import Hunyuan3D modules: {e}")
        return False

def main():
    print("=" * 70)
    print("Hunyuan3D-Omni Installation Test")
    print("=" * 70)
    
    # Test imports
    failed_packages = test_imports()
    
    # Test CUDA
    cuda_ok = test_cuda()
    
    # Test Hunyuan3D modules
    hunyuan_ok = test_hunyuan3d_modules()
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    if failed_packages:
        print(f"✗ Failed packages: {', '.join(failed_packages)}")
    else:
        print("✓ All packages imported successfully")
    
    if not cuda_ok:
        print("✗ CUDA test failed")
    else:
        print("✓ CUDA is working")
    
    if not hunyuan_ok:
        print("✗ Hunyuan3D-Omni modules not found or failed to import")
    else:
        print("✓ Hunyuan3D-Omni modules loaded successfully")
    
    print("=" * 70)
    
    if failed_packages or not hunyuan_ok:
        print("\n⚠ Installation incomplete or has issues.")
        sys.exit(1)
    else:
        print("\n✓ Installation successful!")
        print("\nTo run inference, use:")
        print("  python inference.py --control_type <point|voxel|bbox|pose>")
        sys.exit(0)

if __name__ == "__main__":
    main()
