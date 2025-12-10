from PIL import Image
import numpy as np
from ao_baker import SurfaceEffectsBaker

# Create a simple test texture (solid color with some variation)
print("Creating test texture...")
width, height = 512, 512
img_array = np.zeros((height, width, 3), dtype=np.uint8)
# Red base color
img_array[:, :, 0] = 180
img_array[:, :, 1] = 50
img_array[:, :, 2] = 50
test_img = Image.fromarray(img_array, mode='RGB')

# Apply curvature effects
print("Loading cube model...")
baker = SurfaceEffectsBaker()

print("Applying surface effects...")
result = baker.apply_surface_effects_to_texture(
    test_img,
    "/Users/nathanhenderson/Desktop/test.obj",
    curvature_strength=1.0,      # High sensitivity
    edge_highlight=0.5,          # 50% brightness on edges
    crevice_darken=0.3,          # 30% darkening in crevices
    edge_hue_shift=30.0,         # Shift edges toward orange/yellow
    crevice_hue_shift=-30.0,     # Shift crevices toward blue
    edge_saturation=0.3          # Boost edge saturation
)

print("Saving result...")
result.save("/Users/nathanhenderson/Desktop/test_curvature_output.png")
print("✓ Saved to ~/Desktop/test_curvature_output.png")

# Also create a visualization of the curvature map itself
print("\nGenerating curvature map visualization...")
curvature_map = baker.compute_curvature_map(width, height, strength=1.0)

# Normalize for visualization: -1 to 1 -> 0 to 255
curv_vis = ((curvature_map + 1.0) * 127.5).astype(np.uint8)
curv_img = Image.fromarray(curv_vis, mode='L')
curv_img.save("/Users/nathanhenderson/Desktop/test_curvature_map.png")
print("✓ Saved curvature map to ~/Desktop/test_curvature_map.png")
print("  (Gray=flat, White=convex edges, Black=concave crevices)")
