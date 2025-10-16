# Uniform Scaling Approach - Explanation

## The Problem
When generating 3D meshes from images, we need to maintain correct object proportions while sizing them appropriately based on their visibility in the image. Applying different scale factors to each axis would distort the object.

## The Solution
**Uniform scaling with correspondence**: Scale ONE axis based on the relationship between image visibility and real-world size, then apply that same scale to all axes.

## How It Works

### Step 1: Find Dominant Dimensions
- **2D bbox**: Find the largest dimension (width, height, or depth)
- **Real-world**: Find the largest dimension (length, width, or height)

### Step 2: Establish Correspondence
Calculate scale factor: `scale = max_2d_dimension / max_real_dimension`

### Step 3: Apply Uniformly
Apply the same scale factor to ALL three dimensions to maintain aspect ratios.

## Example: Banana

### Input Data
```
2D BBox: [0.2, 0.3, 0.8, 0.7]
  → width_2d  = 0.8 - 0.2 = 0.6  (60% of image width)
  → height_2d = 0.7 - 0.3 = 0.4  (40% of image height)

Depth: [0.0, 0.18]
  → depth_2d = 0.18 - 0.0 = 0.18

Real-world dimensions:
  → length = 0.18m (banana is 18cm long)
  → width  = 0.04m (4cm thick)
  → height = 0.04m (4cm tall)
```

### Calculation
```
1. Find dominant dimensions:
   max_2d_dim = max(0.6, 0.4, 0.18) = 0.6  (width is most visible)
   max_real_dim = max(0.18, 0.04, 0.04) = 0.18m  (length is longest)

2. Calculate uniform scale factor:
   scale_factor = 0.6 / 0.18 = 3.333

3. Apply to all dimensions:
   width_scale  = 0.18m × 3.333 = 0.60
   height_scale = 0.04m × 3.333 = 0.133
   depth_scale  = 0.04m × 3.333 = 0.133

4. Result: [0.60, 0.133, 0.133]
```

### Why This Works
- ✅ **Maintains aspect ratio**: 0.60 : 0.133 : 0.133 = 4.5 : 1 : 1 (correct banana proportions)
- ✅ **Respects visibility**: Dominant visible dimension (0.6) corresponds to dominant real dimension
- ✅ **In valid range**: All values are in [0, 1] as required by Hunyuan3D-Omni
- ✅ **No distortion**: Single scale factor applied uniformly

## Example: Water Bottle

### Input Data
```
2D BBox: [0.4, 0.2, 0.6, 0.8]
  → width_2d  = 0.2  (20% of image)
  → height_2d = 0.6  (60% of image - dominant!)

Depth: [0.3, 0.4]
  → depth_2d = 0.1

Real-world dimensions:
  → length = 0.08m (8cm diameter)
  → width  = 0.08m (8cm diameter)
  → height = 0.25m (25cm tall - dominant!)
```

### Calculation
```
1. Find dominant dimensions:
   max_2d_dim = max(0.2, 0.6, 0.1) = 0.6  (height is most visible)
   max_real_dim = max(0.08, 0.08, 0.25) = 0.25m  (height is tallest)

2. Calculate uniform scale factor:
   scale_factor = 0.6 / 0.25 = 2.4

3. Apply to all dimensions:
   width_scale  = 0.08m × 2.4 = 0.192
   height_scale = 0.25m × 2.4 = 0.60
   depth_scale  = 0.08m × 2.4 = 0.192

4. Result: [0.192, 0.60, 0.192]
```

### Why This Works
- ✅ **Correct correspondence**: Tallest real dimension (height) matches tallest visible dimension
- ✅ **Cylindrical shape preserved**: width = depth (maintains circular cross-section)
- ✅ **Appropriate scale**: 60% visibility in height translates to 60% in the output

## Key Benefits

1. **One Axis Reference**: Uses single dominant axis to determine scale
2. **Proportions Locked**: All other dimensions scale proportionally
3. **Smart Correspondence**: Matches dominant real dimension to dominant visible dimension
4. **No Distortion**: Uniform scaling preserves object shape
5. **Approximate Real-World Scale**: Output reflects both visibility and actual size

## Technical Details

### Hunyuan3D-Omni Requirements
- **Input format**: 3 values `[length, height, width]`
- **Value range**: 0-1 (normalized)
- **Internal processing**: Converts to 8 corner points in [-1, 1] range
- **Shape**: Tensor of shape `[batch, 1, 3]`

### Fallback Behavior
If no real-world dimensions are provided:
```python
# Simply normalize 2D bbox dimensions
max_dim = max(width_2d, height_2d, depth_2d)
result = [width_2d/max_dim, height_2d/max_dim, depth_2d/max_dim]
```

This maintains the proportions from the 2D bbox itself.

## Comparison to Other Approaches

### ❌ Independent Scaling Per Axis
```python
# BAD: Each axis scaled independently
width_scale = width_2d * some_factor_x
height_scale = height_2d * some_factor_y
depth_scale = depth_2d * some_factor_z
```
**Problem**: Different factors distort the object shape

### ❌ Direct Real Dimensions
```python
# BAD: Using raw meter values
width_scale = real_length  # e.g., 0.18
height_scale = real_height # e.g., 0.04
```
**Problem**: Ignores image visibility and context

### ✅ Uniform Scaling with Correspondence (Current)
```python
# GOOD: Single scale from dominant dimensions
scale = max_visible / max_real
width_scale = real_length * scale
height_scale = real_height * scale
depth_scale = real_width * scale
```
**Benefits**: Maintains shape, respects visibility, uses real proportions

## Summary

The uniform scaling approach ensures that:
1. Objects maintain their **real-world proportions**
2. Scale is informed by **image visibility** (how prominent they appear)
3. A **single reference axis** determines the scale factor
4. All dimensions scale **uniformly** (no distortion)
5. Results are in the **required 0-1 range**

This gives you meshes with correct shapes at appropriate sizes!

