"""
Blender baking script - called by ao_baker.py via subprocess
Run with: blender --background --python blender_baker.py -- <args>
"""
import bpy
import mathutils
import sys
import os

def bake_edge_map(obj_path, output_path, resolution):
    """Bake edge/curvature map by detecting mesh edges in world space"""
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
    
    # Mark edges in edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    
    # Mark sharp edges based on angle
    bpy.ops.mesh.edges_select_sharp(sharpness=0.523599)  # ~30 degrees in radians
    
    # Mark freestyle edges
    bpy.ops.mesh.mark_freestyle_edge(clear=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Enable freestyle for edge rendering
    bpy.context.scene.render.use_freestyle = True
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    
    # Setup freestyle to render edges
    freestyle = bpy.context.scene.view_layers[0].freestyle_settings
    freestyle.linesets[0].select_edge_mark = True
    freestyle.linesets[0].linestyle.color = (1, 1, 1)
    freestyle.linesets[0].linestyle.thickness = 2.0
    
    # Setup camera for orthographic top-down view
    bpy.ops.object.camera_add()
    cam = bpy.context.object
    cam.data.type = 'ORTHO'
    
    # Position camera to frame object
    bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    min_x = min(v.x for v in bbox)
    max_x = max(v.x for v in bbox)
    min_y = min(v.y for v in bbox)
    max_y = max(v.y for v in bbox)
    max_z = max(v.z for v in bbox)
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    cam.location = (center_x, center_y, max_z + 10)
    cam.rotation_euler = (0, 0, 0)
    
    size = max(max_x - min_x, max_y - min_y)
    cam.data.ortho_scale = size * 1.1
    
    bpy.context.scene.camera = cam
    
    # Render to image
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = output_path
    
    bpy.ops.render.render(write_still=True)
    
    print(f"Edge map saved to: {output_path}")


def bake_ao_map(obj_path, output_path, resolution, samples=32):
    """Bake ambient occlusion as a grayscale mask using Blender"""
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
    
    # Create grayscale image for baking (L mode)
    img = bpy.data.images.new("BakeImage", width=resolution, height=resolution, alpha=False, is_data=True)
    img.colorspace_settings.name = 'Non-Color'
    
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
    
    # Bake AO as grayscale mask
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.bake_type = 'AO'
    bpy.context.scene.cycles.samples = samples
    bpy.context.scene.render.bake.use_selected_to_active = False
    
    # AO settings for better crevice detection
    # Set AO distance on the world shader nodes
    if bpy.context.scene.world:
        bpy.context.scene.world.use_nodes = True
        world_nodes = bpy.context.scene.world.node_tree.nodes
        for node in world_nodes:
            if node.type == 'AMBIENT_OCCLUSION' or node.type == 'OUTPUT_WORLD':
                pass  # AO distance is set in render settings below
    
    # Set AO distance in render settings (Cycles)
    bpy.context.scene.cycles.ao_bounces = 1
    bpy.context.scene.cycles.ao_bounces_render = 1
    
    bpy.ops.object.bake(type='AO')
    
    # Save as grayscale PNG
    img.filepath_raw = output_path
    img.file_format = 'PNG'
    img.colorspace_settings.name = 'Non-Color'
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
