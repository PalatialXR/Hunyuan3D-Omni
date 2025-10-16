# Direct Real-World Dimensions Approach

## The Strategy

**Simple and direct**: Pass real-world dimensions (in meters) directly to Hunyuan3D-Omni, with minimal processing.

## How It Works

### When Real-World Dimensions Are Available

1. **Use them directly** (in meters)
2. **Only normalize** if any dimension exceeds 1 meter

### Example: Water Bottle

```python
Input dimensions:
  length_m: 0.08  (8cm diameter)
  width_m: 0.08   (8cm diameter) 
  height_m: 0.25  (25cm tall)

Output to Hunyuan3D:
  [0.08, 0.25, 0.08]  # Direct pass-through
```

### Example: Banana

```python
Input dimensions:
  length_m: 0.18  (18cm long)
  width_m: 0.04   (4cm thick)
  height_m: 0.04  (4cm tall)

Output to Hunyuan3D:
  [0.18, 0.04, 0.04]  # Direct pass-through
```

### Example: Dining Table (> 1 meter)

```python
Input dimensions:
  length_m: 1.5   (150cm long)
  width_m: 0.9    (90cm wide)
  height_m: 0.75  (75cm tall)

Max dimension: 1.5m (exceeds 1.0)

Normalized output:
  [1.5/1.5, 0.75/1.5, 0.9/1.5] = [1.0, 0.5, 0.6]
```

## Code Implementation

```python
def convert_to_3d_bbox(bbox_2d, depth_range, real_world_dims=None):
    if real_world_dims:
        # Use real-world dimensions directly
        width_scale = real_world_dims['length_m']
        height_scale = real_world_dims['height_m']
        depth_scale = real_world_dims['width_m']
        
        # Only normalize if any dimension > 1.0
        max_result = max(width_scale, height_scale, depth_scale)
        if max_result > 1.0:
            width_scale /= max_result
            height_scale /= max_result
            depth_scale /= max_result
    else:
        # Fallback: use 2D bbox proportions
        width_2d = bbox_2d[2] - bbox_2d[0]
        height_2d = bbox_2d[3] - bbox_2d[1]
        depth_2d = depth_range[1] - depth_range[0]
        
        max_dim = max(width_2d, height_2d, depth_2d)
        width_scale = width_2d / max_dim
        height_scale = height_2d / max_dim
        depth_scale = depth_2d / max_dim
    
    return [width_scale, height_scale, depth_scale]
```

## Why This Works

### ✅ Advantages

1. **Simple**: No complex scaling calculations
2. **Intuitive**: Meters in → normalized meters out
3. **Preserves proportions**: Real aspect ratios maintained
4. **Natural fit**: Most objects < 1m fit naturally in 0-1 range
5. **Model decides**: Let Hunyuan3D interpret the scale

### 🤔 What About Image Visibility?

The 2D bbox information is **still used** by Hunyuan3D-Omni internally to:
- Determine which part of the image to focus on
- Understand object placement and context
- Guide the generation process

The bbox scale values (0-1 range) provide **relative size information** that the model uses alongside the image features.

## Hunyuan3D-Omni Requirements

From the source code (`omni_encoder.py`):

```python
def bbox_to_corners(self, bbox):
    """
    Convert bbox (B,1,3) to 8 corner points (range[-1,1])
    
    Parameters:
        bbox: torch.Tensor, shape (B,1,3), [length, height, width] (range 0~1)
    
    Returns:
        corners: torch.Tensor, shape (B,8,3), 8 corner xyz coordinates
    """
```

**Key facts:**
- Input format: 3 values `[length, height, width]`
- Expected range: 0 to 1
- Internally converted to 8 corner points in [-1, 1] range

## Comparison to Previous Approaches

### ❌ Complex Scaling with Correspondence
```python
# OLD: Complex calculation
max_2d = max(width_2d, height_2d, depth_2d)
max_real = max(real_length, real_height, real_width)
scale_factor = max_2d / max_real
result = [real_length * scale_factor, ...]
```
**Problem**: Overthinking it, loses direct size information

### ✅ Direct Dimensions (Current)
```python
# NEW: Simple and direct
result = [real_length, real_height, real_width]
# Only normalize if > 1.0
```
**Benefits**: Simple, intuitive, preserves actual scale

## Expected Behavior

### Small Objects (< 1 meter)
- Input: Real dimensions in meters
- Output: Same values (direct pass-through)
- Result: Mesh should be close to real-world size

### Large Objects (> 1 meter)
- Input: Real dimensions in meters
- Output: Normalized to largest dimension = 1.0
- Result: Mesh maintains proportions, scaled to fit

### No Real-World Data
- Input: 2D bbox only
- Output: Proportions from image bbox
- Result: Reasonable shape, arbitrary absolute scale

## Testing Examples

### ✅ Water Bottle
```
Real: 0.08m × 0.25m × 0.08m
Output: [0.08, 0.25, 0.08]
Expected result: ~25cm tall bottle with circular cross-section
```

### ✅ Banana
```
Real: 0.18m × 0.04m × 0.04m
Output: [0.18, 0.04, 0.04]
Expected result: ~18cm long banana, 4.5:1 length:thickness ratio
```

### ✅ Coffee Mug
```
Real: 0.09m × 0.10m × 0.09m
Output: [0.09, 0.10, 0.09]
Expected result: ~10cm tall mug with circular base
```

### ✅ Dining Chair
```
Real: 0.45m × 0.90m × 0.50m
Output: [0.45, 0.90, 0.50]
Expected result: ~90cm tall chair with correct proportions
```

## Summary

The direct approach:
1. **Trusts the AI** to provide accurate real-world dimensions
2. **Passes them directly** to the model (in meters)
3. **Only normalizes** when dimensions exceed 1 meter
4. **Preserves proportions** naturally
5. **Keeps it simple** - no complex calculations

This should give you meshes that are:
- ✅ Correct shape (proper aspect ratios)
- ✅ Appropriate size (based on real-world dimensions)
- ✅ Ready to use (minimal post-processing needed)
