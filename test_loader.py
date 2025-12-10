from ao_baker import SurfaceEffectsBaker

baker = SurfaceEffectsBaker()
success = baker.load_obj("/Users/nathanhenderson/Desktop/test.obj")

print(f"Load successful: {success}")
if success:
    print(f"Vertices: {len(baker.vertices)}")
    print(f"UVs: {len(baker.uvs) if baker.uvs is not None else 'None'}")
    print(f"Normals: {len(baker.normals) if baker.normals is not None else 'None'}")
    print(f"Faces: {len(baker.faces)}")
    print(f"\nFirst face: {baker.faces[0]}")
    print(f"First UV: {baker.uvs[0] if baker.uvs is not None else 'None'}")
    print(f"First normal: {baker.normals[0] if baker.normals is not None else 'None'}")
