"""
Blender baking script - called by ao_baker.py via subprocess
Run with: blender --background --python blender_baker.py -- <args>
"""
import bpy
import sys
import os

def bake_edge_map(obj_path, output_path, resolution):
    """Bake edge/normal map using Blender"""
    # Clear scene
    bpy.ops.wm.read_homefile(use_empty=True)
    
    # Import OBJ
    bpy.ops.wm.obj_import(filepath=obj_path)
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    
    # Ensure mesh has UVs, auto-unwrap if needed
    if not obj.data.uv_layers:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # Create image for baking
    img = bpy.data.images.new("BakeImage", width=resolution, height=resolution, alpha=False)
    
    # Setup material with image texture node
    if not obj.data.materials:
        mat = bpy.data.materials.new(name="BakeMaterial")
        obj.data.materials.append(mat)
    else:
        mat = obj.data.materials[0]
    
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    # Add image texture node
    img_node = nodes.new('ShaderNodeTexImage')
    img_node.image = img
    img_node.select = True
    nodes.active = img_node
    
    # Bake normal map (gives us edge information)
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.bake_type = 'NORMAL'
    bpy.context.scene.render.bake.use_selected_to_active = False
    
    bpy.ops.object.bake(type='NORMAL')
    
    # Save image
    img.filepath_raw = output_path
    img.file_format = 'PNG'
    img.save()
    
    print(f"Edge map saved to: {output_path}")


def bake_ao_map(obj_path, output_path, resolution, samples=32):
    """Bake ambient occlusion using Blender"""
    # Clear scene
    bpy.ops.wm.read_homefile(use_empty=True)
    
    # Import OBJ
    bpy.ops.wm.obj_import(filepath=obj_path)
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    
    # Ensure mesh has UVs, auto-unwrap if needed
    if not obj.data.uv_layers:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # Create image for baking
    img = bpy.data.images.new("BakeImage", width=resolution, height=resolution, alpha=False)
    
    # Setup material with image texture node
    if not obj.data.materials:
        mat = bpy.data.materials.new(name="BakeMaterial")
        obj.data.materials.append(mat)
    else:
        mat = obj.data.materials[0]
    
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    # Add image texture node
    img_node = nodes.new('ShaderNodeTexImage')
    img_node.image = img
    img_node.select = True
    nodes.active = img_node
    
    # Bake AO
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.bake_type = 'AO'
    bpy.context.scene.cycles.samples = samples
    bpy.context.scene.render.bake.use_selected_to_active = False
    
    bpy.ops.object.bake(type='AO')
    
    # Save image
    img.filepath_raw = output_path
    img.file_format = 'PNG'
    img.save()
    
    print(f"AO map saved to: {output_path}")


if __name__ == "__main__":
    # Parse command line args after '--'
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]  # Get args after '--'
    
    if len(argv) < 4:
        print("Usage: blender --background --python blender_baker.py -- <mode> <obj_path> <output_path> <resolution> [samples]")
        sys.exit(1)
    
    mode = argv[0]  # 'edge' or 'ao'
    obj_path = argv[1]
    output_path = argv[2]
    resolution = int(argv[3])
    samples = int(argv[4]) if len(argv) > 4 else 32
    
    if mode == 'edge':
        bake_edge_map(obj_path, output_path, resolution)
    elif mode == 'ao':
        bake_ao_map(obj_path, output_path, resolution, samples)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
