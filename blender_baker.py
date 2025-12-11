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
    
    # CRITICAL: Add a Diffuse BSDF shader to the material
    # Cycles might need an actual surface shader to calculate AO properly
    diffuse_node = nodes.new('ShaderNodeBsdfDiffuse')
    diffuse_node.location = (0, 100)
    
    # Add Material Output
    output_node = nodes.new('ShaderNodeOutputMaterial')
    output_node.location = (300, 100)
    
    # Connect diffuse to output
    links = mat.node_tree.links
    links.new(diffuse_node.outputs['BSDF'], output_node.inputs['Surface'])
    
    # Add image texture node (target for baking)
    img_node = nodes.new('ShaderNodeTexImage')
    img_node.image = img
    img_node.select = True
    img_node.location = (0, -100)
    nodes.active = img_node  # Must be active for baking
    
    # For AO baking, the image node doesn't need to be connected
    # It just needs to be selected/active
    
    # Bake AO directly with Cycles (confirmed working for simple geometry)
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = samples
    bpy.context.scene.render.bake.use_selected_to_active = False
    bpy.context.scene.render.bake.margin = 5  # No margin to match UV boundaries exactly
    
    # Recalculate normals and smooth shading for better AO
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)  # Recalculate normals outward
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.shade_smooth()
    
    # Add ground plane for better AO occlusion
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, -2))
    
    # Re-select the original object for baking
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # Set world to simple white background
    if bpy.context.scene.world:
        bpy.context.scene.world.use_nodes = True
        world_nodes = bpy.context.scene.world.node_tree.nodes
        world_nodes.clear()
        
        bg_node = world_nodes.new(type='ShaderNodeBackground')
        bg_node.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
        
        output_node = world_nodes.new(type='ShaderNodeOutputWorld')
        output_node.location = (200, 0)
        
        world_tree = bpy.context.scene.world.node_tree
        world_tree.links.new(bg_node.outputs[0], output_node.inputs[0])
    
    # CRITICAL FIX: Bake with AO type
    # Use margin=0 to prevent white bleeding beyond UV boundaries
    bpy.ops.object.bake(
        type='AO',
        margin=0,
        use_clear=True,
        use_selected_to_active=False,
        use_cage=False
    )
    
    # Get the baked pixel data
    pixels = list(img.pixels)
    print(f"After baking: {len(pixels)} RGBA values")
    
    # CRITICAL FIX: The AO bake puts data in R channel, but we need it in all RGB channels
    # Manually copy R to G and B for proper grayscale display
    for i in range(0, len(pixels), 4):
        ao_value = pixels[i]  # R channel contains AO
        pixels[i+1] = ao_value  # Copy to G
        pixels[i+2] = ao_value  # Copy to B
        # pixels[i+3] remains as alpha
    
    img.pixels = pixels
    img.update()  # Force update the image buffer
    print(f"Copied R channel to RGB for grayscale output")
    
    # Save as grayscale PNG
    # Try using pack() which forces data into the image, then save
    img.pack()
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
