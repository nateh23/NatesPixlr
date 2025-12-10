from ao_baker import SurfaceEffectsBaker
import numpy as np

baker = SurfaceEffectsBaker()
baker.load_obj("/Users/nathanhenderson/Desktop/test.obj")

print("=== Mesh Info ===")
print(f"Vertices: {len(baker.vertices)}")
print(f"UVs: {len(baker.uvs)}")
print(f"Normals: {len(baker.normals)}")
print(f"Faces: {len(baker.faces)}")

print("\n=== First 3 Faces ===")
for i, face in enumerate(baker.faces[:3]):
    print(f"\nFace {i}:")
    for j, (v_idx, vt_idx, vn_idx) in enumerate(face):
        uv = baker.uvs[vt_idx] if vt_idx is not None else None
        normal = baker.normals[vn_idx] if vn_idx is not None else None
        print(f"  Vertex {j}: UV={uv}, Normal={normal}")

# Test curvature computation
print("\n=== Testing Curvature Map ===")
width, height = 128, 128
curv_map = baker.compute_curvature_map(width, height, strength=1.0)

print(f"Curvature map shape: {curv_map.shape}")
print(f"Min value: {curv_map.min():.4f}")
print(f"Max value: {curv_map.max():.4f}")
print(f"Mean value: {curv_map.mean():.4f}")
print(f"Non-zero pixels: {np.count_nonzero(curv_map)}")
print(f"Total pixels: {width * height}")

# Show some sample values
print(f"\nSample curvature values at different positions:")
for y in [32, 64, 96]:
    for x in [32, 64, 96]:
        print(f"  ({x}, {y}): {curv_map[y, x]:.4f}")
