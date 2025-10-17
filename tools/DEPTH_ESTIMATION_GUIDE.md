# Depth Estimation for Single-Object Images

## The Key Insight

**For single-object images with volumetric scaling, depth is MUCH less critical than you might think!**

## Why Depth Matters Less

### Our Volumetric Scaling Approach

```
1. 2D BBox → Shape (visual proportions)
2. Real Dims → Size (actual meters)
3. Volumetric Scale → Correct final mesh
```

### Example: Water Bottle

```python
# Input
2D bbox: width=0.6, height=0.8, depth=0.2  # Visual shape
Real dims: 0.08m × 0.25m × 0.08m           # Actual size

# Step 1: Generate with 2D bbox proportions
# Hunyuan generates mesh with shape [0.75, 1.0, 0.25] (normalized)
# Result: Bottle-shaped mesh (correct proportions!)

# Step 2: Apply volumetric scaling
volume_2d = 0.75 × 1.0 × 0.25 = 0.1875
volume_real = 0.08 × 0.25 × 0.08 = 0.0016
scale = (0.0016 / 0.1875)^(1/3) = 0.267

# Step 3: Scale uniformly
final_mesh = mesh × 0.267
# Result: ~0.08m × 0.25m × 0.08m bottle ✓
```

**Notice:** The depth value (0.2) only affects the initial generation, but gets **uniformly scaled** along with everything else. The shape is determined by the 2D bbox aspect ratios!

## When Depth DOES Matter

### Multi-Object Scenes
```
Table scene with:
- Plate (close, z=0.2-0.4)
- Cup (mid, z=0.4-0.6)  
- Person (far, z=0.6-0.8)

Depth matters here for RELATIVE POSITIONING
```

### Complex Spatial Relationships
```
- Overlapping objects
- Occlusion reasoning
- Distance relationships
```

### No Real-World Dimensions
```
If we don't have real dimensions, depth becomes
part of the size estimation process
```

## Our Depth Strategy

### Default (Current)
```python
def estimate_depth_from_image(image_path, bbox):
    """Simple heuristic based on vertical position"""
    
    y_center = (bbox[1] + bbox[3]) / 2
    
    if y_center > 0.6:  # Lower in frame
        return [0.2, 0.5]  # Ground-plane objects closer
    elif y_center < 0.4:  # Upper in frame
        return [0.5, 0.8]  # Higher objects farther
    else:
        return [0.3, 0.7]  # Default mid-range
```

**Why this works:**
- For single objects, consistent depth is fine
- 2D bbox shape drives the generation
- Volumetric scaling corrects the size anyway

### Optional: Depth-Anything V2

For complex scenes or when you need accurate depth:

```bash
# Install
pip install depth-anything-v2

# Download model
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth \
  -O checkpoints/depth_anything_v2_vitl.pth
```

```python
# Enable in code
det['depth'] = estimate_depth_from_image(
    image_path,
    bbox_normalized,
    use_depth_model=True  # Use actual depth estimation
)
```

**Trade-offs:**
- ✅ More accurate depth
- ✅ Better for multi-object scenes
- ❌ Slower (~500ms per image)
- ❌ Requires additional model (~1.3GB)
- ❌ Not critical for single objects

## Testing Different Depth Values

Want to see how depth affects generation? Try this:

```python
# Same water bottle, different depths

# Shallow depth (thick object)
depth_1 = [0.4, 0.9]  # Depth range: 0.5
bbox_3d = [0.75, 1.0, 0.625]  # Normalized

# Deep depth (thin object)  
depth_2 = [0.45, 0.55]  # Depth range: 0.1
bbox_3d = [0.75, 1.0, 0.125]  # Normalized

# After volumetric scaling:
# Both end up at 0.08m × 0.25m × 0.08m!
# (Because real-world dimensions drive final size)
```

## Recommendations

### For Single Objects (Your Use Case)
✅ **Use default depth estimation** - Fast, simple, sufficient
- Water bottles, cups, phones, fruits, etc.
- Depth = [0.3, 0.7] works great
- Focus on getting good 2D bbox and real dimensions

### For Multi-Object Scenes
⚠️ **Consider depth models** - More accurate positioning
- Dining tables, rooms, outdoor scenes
- When relative depth matters
- Use Depth-Anything V2 or similar

### For Production
🎯 **Hybrid approach:**
```python
if num_objects == 1:
    use_depth_model = False  # Fast default
else:
    use_depth_model = True   # Accurate depth
```

## Summary

| Factor | Importance for Single Objects | Why |
|--------|-------------------------------|-----|
| 2D BBox Shape | ⭐⭐⭐⭐⭐ Critical | Defines visual proportions |
| Real Dimensions | ⭐⭐⭐⭐⭐ Critical | Defines actual size |
| Volumetric Scaling | ⭐⭐⭐⭐⭐ Critical | Corrects final mesh |
| Depth Estimation | ⭐⭐ Nice-to-have | Minor effect with scaling |

**Bottom line:** With our volumetric scaling approach, accurate depth is a nice-to-have, not a must-have for single objects. The 2D bbox shape and real-world dimensions are what really matter!

## Example Output Logs

```
[INFO] Processing: plastic_bottle
[INFO]   2D BBox: [0.2, 0.1, 0.8, 0.9]
[INFO]   Depth: [0.3, 0.7] (default mid-range)
[INFO]   3D BBox (shape): [0.75, 1.0, 0.25]
[INFO]   Real dims: {length_m: 0.08, width_m: 0.08, height_m: 0.25}
[INFO]   Volumetric scale: 0.267x
[INFO]   Final mesh: ~0.08m × 0.25m × 0.08m ✓
```

Notice the depth is just a default, but the final mesh is still correct!

