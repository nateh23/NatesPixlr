"""
Debug script for AO baking - test with thirdtest.obj
"""
import sys
import numpy as np
from PIL import Image
from ao_baker import SurfaceEffectsBaker

def debug_ao_baking():
    """Test AO baking with debug output"""
    
    # Paths
    model_path = "/Users/nathanhenderson/Desktop/fourthtest.obj"
    output_dir = "/Users/nathanhenderson/Desktop"
    
    print("="*60)
    print("DEBUG: AO Baking Test")
    print("="*60)
    
    # Create baker
    baker = SurfaceEffectsBaker()
    
    # Load model
    print(f"\n1. Loading model: {model_path}")
    success = baker.load_obj(model_path)
    
    if not success:
        print("ERROR: Failed to load model")
        return
    
    print(f"✓ Model loaded successfully")
    print(f"  Vertices: {len(baker.vertices)}")
    print(f"  UVs: {len(baker.uvs) if baker.uvs is not None else 0}")
    print(f"  Normals: {len(baker.normals) if baker.normals is not None else 0}")
    print(f"  Faces: {len(baker.faces)}")
    
    # Show first few UVs and normals
    if baker.uvs is not None and len(baker.uvs) > 0:
        print(f"\n  First 3 UVs:")
        for i in range(min(3, len(baker.uvs))):
            print(f"    UV[{i}]: {baker.uvs[i]}")
    
    if baker.normals is not None and len(baker.normals) > 0:
        print(f"\n  First 3 Normals:")
        for i in range(min(3, len(baker.normals))):
            print(f"    Normal[{i}]: {baker.normals[i]}")
    
    # Show first face
    if len(baker.faces) > 0:
        print(f"\n  First face:")
        face = baker.faces[0]
        for i, (v_idx, vt_idx, vn_idx) in enumerate(face):
            print(f"    Vertex {i}: v={v_idx}, vt={vt_idx}, vn={vn_idx}")
    
    # Test curvature map
    print("\n" + "="*60)
    print("2. Testing CURVATURE map (UV-space)")
    print("="*60)
    
    resolution = 512
    strength = 5.0
    
    print(f"  Resolution: {resolution}x{resolution}")
    print(f"  Strength: {strength}")
    
    curv_map = baker.compute_curvature_map(resolution, resolution, strength)
    
    print(f"\n  Curvature map stats:")
    print(f"    Min: {curv_map.min():.6f}")
    print(f"    Max: {curv_map.max():.6f}")
    print(f"    Mean: {curv_map.mean():.6f}")
    print(f"    Non-zero pixels: {np.count_nonzero(curv_map)}/{resolution*resolution}")
    
    # Save curvature as edge map (positive values only)
    edge_map = np.maximum(0, curv_map)
    if edge_map.max() > 0:
        edge_img = np.clip(edge_map / edge_map.max() * 255, 0, 255).astype(np.uint8)
        edge_path = f"{output_dir}/debug_edge_map_uv.png"
        Image.fromarray(edge_img, mode='L').save(edge_path)
        print(f"  ✓ Saved edge map: {edge_path}")
    else:
        print(f"  ⚠ Edge map is all zeros!")
    
    # Test WORLD-SPACE curvature
    print("\n" + "="*60)
    print("3. Testing CURVATURE map (WORLD-SPACE)")
    print("="*60)
    
    ws_curv_map = baker.compute_curvature_map_worldspace(resolution, resolution, strength)
    
    print(f"\n  World-space curvature stats:")
    print(f"    Min: {ws_curv_map.min():.6f}")
    print(f"    Max: {ws_curv_map.max():.6f}")
    print(f"    Mean: {ws_curv_map.mean():.6f}")
    print(f"    Non-zero pixels: {np.count_nonzero(ws_curv_map)}/{resolution*resolution}")
    
    # Save world-space edge map
    ws_edge_map = np.maximum(0, ws_curv_map)
    if ws_edge_map.max() > 0:
        ws_edge_img = np.clip(ws_edge_map / ws_edge_map.max() * 255, 0, 255).astype(np.uint8)
        ws_edge_path = f"{output_dir}/debug_edge_map_worldspace.png"
        Image.fromarray(ws_edge_img, mode='L').save(ws_edge_path)
        print(f"  ✓ Saved world-space edge map: {ws_edge_path}")
    else:
        print(f"  ⚠ World-space edge map is all zeros!")
    
    # Test AO map
    print("\n" + "="*60)
    print("4. Testing AO map (UV-space crevices)")
    print("="*60)
    
    ao_map = baker.compute_ao_map(resolution, resolution, samples=32, distance=0.5)
    
    print(f"\n  AO map stats:")
    print(f"    Min: {ao_map.min():.6f}")
    print(f"    Max: {ao_map.max():.6f}")
    print(f"    Mean: {ao_map.mean():.6f}")
    print(f"    Non-zero pixels: {np.count_nonzero(ao_map < 1.0)}/{resolution*resolution}")
    
    # Save AO map
    ao_img = np.clip(ao_map * 255, 0, 255).astype(np.uint8)
    ao_path = f"{output_dir}/debug_ao_map.png"
    Image.fromarray(ao_img, mode='L').save(ao_path)
    print(f"  ✓ Saved AO map: {ao_path}")
    
    # Test with inverted curvature (crevices)
    print("\n" + "="*60)
    print("5. Testing INVERTED curvature (crevices direct)")
    print("="*60)
    
    crevice_map = np.maximum(0, -curv_map)
    
    print(f"\n  Crevice map stats:")
    print(f"    Min: {crevice_map.min():.6f}")
    print(f"    Max: {crevice_map.max():.6f}")
    print(f"    Mean: {crevice_map.mean():.6f}")
    print(f"    Non-zero pixels: {np.count_nonzero(crevice_map)}/{resolution*resolution}")
    
    if crevice_map.max() > 0:
        crevice_img = np.clip(crevice_map / crevice_map.max() * 255, 0, 255).astype(np.uint8)
        crevice_path = f"{output_dir}/debug_crevice_map.png"
        Image.fromarray(crevice_img, mode='L').save(crevice_path)
        print(f"  ✓ Saved crevice map: {crevice_path}")
    else:
        print(f"  ⚠ Crevice map is all zeros!")
    
    print("\n" + "="*60)
    print("DONE - Check desktop for debug images")
    print("="*60)

if __name__ == "__main__":
    debug_ao_baking()
