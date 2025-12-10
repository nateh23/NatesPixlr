from PIL import Image
import numpy as np
from ao_baker import SurfaceEffectsBaker

# Create a simple test texture
print("Creating test texture...")
width, height = 512, 512
img_array = np.zeros((height, width, 3), dtype=np.uint8)
# Red base color
img_array[:, :, 0] = 180
img_array[:, :, 1] = 50
img_array[:, :, 2] = 50
test_img = Image.fromarray(img_array, mode='RGB')

# Apply curvature effects with MUCH stronger settings
print("Loading cube model...")
baker = SurfaceEffectsBaker()

print("Applying STRONG surface effects...")
result = baker.apply_surface_effects_to_texture(
    test_img,
    "/Users/nathanhenderson/Desktop/test.obj",
    curvature_strength=20.0,     # MUCH higher sensitivity
    edge_highlight=0.8,          # 80% brightness boost
    crevice_darken=0.5,          
    edge_hue_shift=60.0,         # Strong orange shift
    crevice_hue_shift=-60.0,     
    edge_saturation=0.5          # Big saturation boost
)

print("Saving result...")
result.save("/Users/nathanhenderson/Desktop/test_strong_output.png")
print("✓ Saved to ~/Desktop/test_strong_output.png")

# Also create a boosted curvature map
print("\nGenerating BOOSTED curvature map...")
curvature_map = baker.compute_curvature_map(width, height, strength=20.0)

print(f"Curvature stats with strength=20:")
print(f"  Min: {curvature_map.min():.4f}")
print(f"  Max: {curvature_map.max():.4f}")
print(f"  Mean: {curvature_map.mean():.4f}")

# Normalize and boost for visualization
curv_normalized = curvature_map / (curvature_map.max() + 1e-8)
curv_vis = (curv_normalized * 255).astype(np.uint8)
curv_img = Image.fromarray(curv_vis, mode='L')
curv_img.save("/Users/nathanhenderson/Desktop/test_strong_map.png")
print("✓ Saved to ~/Desktop/test_strong_map.png")
