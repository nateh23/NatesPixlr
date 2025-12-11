"""
Surface Effects Baker - Bake curvature maps from 3D models onto textures
"""
import numpy as np
from PIL import Image, ImageFilter
import colorsys
from pathlib import Path
from scipy import ndimage
import trimesh
import subprocess
import os


class SurfaceEffectsBaker:
    """Handles AO baking from 3D models onto textures"""
    
    def __init__(self):
        self.blender_path = "/Applications/Blender.app/Contents/MacOS/Blender"
        self.blender_script = str(Path(__file__).parent / "blender_baker.py")
        self.mesh_loaded = False
        self.vertices = None
        self.uvs = None
        self.normals = None
        self.faces = None
        self.trimesh_obj = None  # Store trimesh object for world-space baking
    
    def load_obj(self, obj_path: str) -> bool:
        """
        Load OBJ file and extract mesh data
        
        Args:
            obj_path: Path to OBJ file
            
        Returns:
            True if loaded successfully
        """
        try:
            vertices = []
            uvs = []
            normals = []
            faces = []
            
            with open(obj_path, 'r') as f:
                for line in f:
                    if line.startswith('v '):
                        # Vertex position
                        parts = line.split()
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    elif line.startswith('vt '):
                        # UV coordinate
                        parts = line.split()
                        uvs.append([float(parts[1]), float(parts[2])])
                    elif line.startswith('vn '):
                        # Vertex normal
                        parts = line.split()
                        normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    elif line.startswith('f '):
                        # Face (can be v, v/vt, v/vt/vn, or v//vn)
                        parts = line.split()[1:]
                        face = []
                        for part in parts:
                            indices = part.split('/')
                            v_idx = int(indices[0]) - 1  # OBJ is 1-indexed
                            vt_idx = int(indices[1]) - 1 if len(indices) > 1 and indices[1] else None
                            vn_idx = int(indices[2]) - 1 if len(indices) > 2 and indices[2] else None
                            face.append((v_idx, vt_idx, vn_idx))
                        faces.append(face)
            
            self.vertices = np.array(vertices, dtype=np.float32)
            self.uvs = np.array(uvs, dtype=np.float32) if uvs else None
            self.normals = np.array(normals, dtype=np.float32) if normals else None
            self.faces = faces
            self.mesh_loaded = True
            
            # Also load with trimesh for world-space operations
            try:
                self.trimesh_obj = trimesh.load_mesh(obj_path)
            except:
                self.trimesh_obj = None
            
            return True
        except Exception as e:
            print(f"Error loading OBJ: {e}")
            return False
    
    def compute_curvature_map(self, width: int, height: int, strength: float = 0.5) -> np.ndarray:
        """
        Compute curvature map from mesh normals
        
        Args:
            width: Texture width
            height: Texture height
            strength: Curvature detection sensitivity
            
        Returns:
            Curvature map as numpy array (-1 = concave/crevice, 0 = flat, +1 = convex/edge)
        """
        if not self.mesh_loaded or self.uvs is None or self.normals is None:
            return np.zeros((height, width), dtype=np.float32)
        
        # Create edge map by drawing lines along face boundaries
        edge_map = np.zeros((height, width), dtype=np.float32)
        faces_drawn = 0
        edges_drawn = 0
        
        # For each face, draw its edges in UV space
        for face in self.faces:
            if len(face) < 3:
                continue
            
            # Get UV coordinates and normals for this face
            uvs_face = []
            normals_face = []
            
            for v_idx, vt_idx, vn_idx in face:
                if vt_idx is not None and vn_idx is not None:
                    uvs_face.append(self.uvs[vt_idx])
                    normals_face.append(self.normals[vn_idx])
            
            if len(uvs_face) < 3:
                continue
            
            faces_drawn += 1
            
            # Get average normal for this face
            face_normal = np.mean(normals_face, axis=0)
            face_normal_mag = np.linalg.norm(face_normal)
            if face_normal_mag > 1e-8:
                face_normal /= face_normal_mag
            
            # Draw edges between consecutive vertices
            num_verts = len(uvs_face)
            for i in range(num_verts):
                uv0 = uvs_face[i]
                uv1 = uvs_face[(i + 1) % num_verts]
                
                # Convert UV to pixel coordinates
                x0 = int(uv0[0] * width)
                y0 = int((1 - uv0[1]) * height)
                x1 = int(uv1[0] * width)
                y1 = int((1 - uv1[1]) * height)
                
                # Draw line using Bresenham's algorithm
                self._draw_line(edge_map, x0, y0, x1, y1, strength)
                edges_drawn += 1
        
        # Apply Gaussian blur to spread edge influence
        if strength > 0:
            edge_map = ndimage.gaussian_filter(edge_map, sigma=3.0)
        
        return edge_map
    
    def compute_curvature_map_worldspace(self, width: int, height: int, strength: float = 5.0) -> np.ndarray:
        """
        Compute curvature map using world-space projection (no UVs needed!)
        Projects mesh from 6 orthographic directions and detects edges from normal changes.
        
        Args:
            width: Texture width
            height: Texture height
            strength: Edge detection sensitivity
            
        Returns:
            Curvature map as numpy array
        """
        if not self.mesh_loaded or self.trimesh_obj is None:
            print("ERROR: No trimesh mesh loaded")
            return np.zeros((height, width), dtype=np.float32)
        
        print(f"Computing world-space curvature map...")
        print(f"  Vertices: {len(self.trimesh_obj.vertices)}")
        print(f"  Faces: {len(self.trimesh_obj.faces)}")
        
        # Get mesh bounds
        bounds = self.trimesh_obj.bounds
        mesh_size = bounds[1] - bounds[0]
        max_dim = max(mesh_size)
        
        # Composite map from multiple angles
        final_map = np.zeros((height, width), dtype=np.float32)
        
        # Project from 6 orthographic directions (top, bottom, left, right, front, back)
        directions = [
            ('Z+', [0, 0, 1]),   # Top view
            ('Z-', [0, 0, -1]),  # Bottom view
            ('Y+', [0, 1, 0]),   # Front view
            ('Y-', [0, -1, 0]),  # Back view
            ('X+', [1, 0, 0]),   # Right view
            ('X-', [-1, 0, 0])   # Left view
        ]
        
        for name, direction in directions:
            # Create projection map for this direction
            proj_map = self._project_curvature_from_direction(
                direction, width, height, strength, max_dim
            )
            # Accumulate (use max to keep strongest edges)
            final_map = np.maximum(final_map, proj_map)
        
        print(f"  World-space map range: {final_map.min():.3f} to {final_map.max():.3f}")
        
        return final_map
    
    def _project_curvature_from_direction(self, direction, width, height, strength, mesh_size):
        """Project mesh from one direction and detect edges"""
        direction = np.array(direction, dtype=np.float32)
        direction = direction / np.linalg.norm(direction)
        
        # Get face normals
        face_normals = self.trimesh_obj.face_normals
        
        # Only consider faces facing this direction (dot product > 0)
        facing = np.dot(face_normals, direction) > 0.1
        
        # Create edge map
        edge_map = np.zeros((height, width), dtype=np.float32)
        
        # Get mesh bounds for projection
        bounds = self.trimesh_obj.bounds
        center = (bounds[0] + bounds[1]) / 2
        
        # Determine which axes to use for projection based on direction
        abs_dir = np.abs(direction)
        main_axis = np.argmax(abs_dir)
        
        # Choose perpendicular axes for 2D projection
        if main_axis == 0:  # X direction
            axes = [1, 2]  # Y, Z
        elif main_axis == 1:  # Y direction
            axes = [0, 2]  # X, Z
        else:  # Z direction
            axes = [0, 1]  # X, Y
        
        # Project visible faces onto 2D plane
        for face_idx, face in enumerate(self.trimesh_obj.faces):
            if not facing[face_idx]:
                continue
            
            # Get face vertices
            verts = self.trimesh_obj.vertices[face]
            
            # Project to 2D
            coords_2d = verts[:, axes]
            
            # Normalize to 0-1 range
            min_vals = bounds[0][axes]
            max_vals = bounds[1][axes]
            range_vals = max_vals - min_vals
            
            if range_vals[0] > 1e-6 and range_vals[1] > 1e-6:
                coords_norm = (coords_2d - min_vals) / range_vals
                
                # Convert to pixel coordinates
                px = (coords_norm[:, 0] * (width - 1)).astype(int)
                py = ((1 - coords_norm[:, 1]) * (height - 1)).astype(int)
                
                # Draw triangle edges
                for i in range(3):
                    x0, y0 = px[i], py[i]
                    x1, y1 = px[(i + 1) % 3], py[(i + 1) % 3]
                    self._draw_line(edge_map, x0, y0, x1, y1, strength)
        
        # Apply Gaussian blur
        if edge_map.max() > 0:
            edge_map = ndimage.gaussian_filter(edge_map, sigma=2.0)
        
        return edge_map
    
    def _draw_line(self, img, x0, y0, x1, y1, value):
        """Draw a line on the image using Bresenham's algorithm"""
        height, width = img.shape
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            # Set pixel if within bounds
            if 0 <= x < width and 0 <= y < height:
                img[y, x] = value
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
    
    def _barycentric(self, px, py, p0, p1, p2):
        """Compute barycentric coordinates of point (px, py) in triangle p0, p1, p2"""
        v0 = [p1[0] - p0[0], p1[1] - p0[1]]
        v1 = [p2[0] - p0[0], p2[1] - p0[1]]
        v2 = [px - p0[0], py - p0[1]]
        
        d00 = v0[0] * v0[0] + v0[1] * v0[1]
        d01 = v0[0] * v1[0] + v0[1] * v1[1]
        d11 = v1[0] * v1[0] + v1[1] * v1[1]
        d20 = v2[0] * v0[0] + v2[1] * v0[1]
        d21 = v2[0] * v1[0] + v2[1] * v1[1]
        
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-8:
            return [1.0, 0.0, 0.0]
        
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        
        return [u, v, w]
    
    def compute_ao_map(self, width: int, height: int, samples: int = 32, 
                      distance: float = 0.5) -> np.ndarray:
        """
        Compute AO map from negative curvature (concave areas) using WORLD-SPACE projection
        
        Args:
            width: Texture width
            height: Texture height
            samples: Unused
            distance: Unused
            
        Returns:
            AO map as numpy array (0 = fully occluded/dark, 1 = no occlusion/bright)
        """
        if not self.mesh_loaded or self.trimesh_obj is None:
            print("ERROR: No mesh loaded for AO baking")
            return np.ones((height, width), dtype=np.float32)
        
        print(f"Computing AO map from world-space curvature: {len(self.vertices)} vertices, {len(self.faces)} faces")
        
        # Always use world-space curvature for AO
        curvature_map = self.compute_curvature_map_worldspace(width, height, strength=5.0)
        
        # Extract negative curvature (concave/crevice areas)
        # Positive = convex/edges (ignore), Negative = concave/crevices (use for AO)
        ao_map = np.maximum(0, -curvature_map)  # Flip and clamp
        
        # Normalize to 0-1 range
        if ao_map.max() > 0:
            ao_map = ao_map / ao_map.max()
        
        print(f"AO from curvature range: {ao_map.min():.3f} to {ao_map.max():.3f}")
        
        # Invert so 1=bright, 0=dark
        ao_map = 1.0 - ao_map
        
        # Apply blur for smoothness
        ao_map = ndimage.gaussian_filter(ao_map, sigma=3.0)
        
        print(f"Final AO map range: {ao_map.min():.3f} to {ao_map.max():.3f}")
        
        return ao_map
    
    def _barycentric(self, px, py, p0, p1, p2):
        """Compute barycentric coordinates, return None if outside triangle"""
        v0 = np.array([p2[0] - p0[0], p2[1] - p0[1]])
        v1 = np.array([p1[0] - p0[0], p1[1] - p0[1]])
        v2 = np.array([px - p0[0], py - p0[1]])
        
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)
        
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-8:
            return None
            
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        
        # Check if inside triangle
        if u >= -0.001 and v >= -0.001 and w >= -0.001:
            return (u, v, w)
        return None
    
    def apply_surface_effects_to_texture(self, img: Image.Image,
                                        edge_highlight: float = 0.3,
                                        edge_color: tuple = (255, 136, 0),
                                        edge_map_path: str = None,
                                        edge_blend: float = 0.7,
                                        enable_edge: bool = True,
                                        ao_darken: float = 0.5,
                                        ao_color: tuple = (50, 30, 20),
                                        ao_map_path: str = None,
                                        ao_blend: float = 0.5,
                                        enable_ao: bool = True) -> Image.Image:
        """Apply surface effects using pre-baked edge and AO maps
        
        Args:
            img: Input texture image
            edge_highlight: Brightness boost on edges (0-1)
            edge_color: RGB tuple (0-255) to blend onto edges
            edge_map_path: Path to pre-baked edge map (REQUIRED)
            edge_blend: Strength of edge color blending (0-1)
            enable_edge: Whether to apply edge effects
            ao_darken: How much to darken crevices (0-1)
            ao_color: RGB tuple (0-255) to blend into occluded areas
            ao_map_path: Path to pre-baked AO map (optional)
            ao_blend: Strength of AO color blending (0-1)
            enable_ao: Whether to apply AO effects
            
        Returns:
            Texture with surface effects applied
        """
        width, height = img.size
        
        # Convert image to float array for blending
        img_array = np.array(img, dtype=np.float32)
        
        # FIRST: Apply AO effects (crevices) if enabled and map provided
        if enable_ao and ao_map_path and Path(ao_map_path).exists():
            try:
                # Load AO map (white=bright/no occlusion, black=dark/full occlusion)
                ao_img = Image.open(ao_map_path).convert('L')
                if ao_img.size != (width, height):
                    ao_img = ao_img.resize((width, height), Image.BILINEAR)
                ao_map = np.array(ao_img, dtype=np.float32) / 255.0
                print(f"Using AO map: {Path(ao_map_path).name}")
                
                # Invert: 0=bright (no occlusion), 1=dark (full occlusion in crevices)
                crevice_mask = 1.0 - ao_map
                
                # Mask AO color into crevices
                # Where crevice_mask=1 (black in AO map), use pure ao_color
                # Where crevice_mask=0 (white in AO map), keep original image
                ao_color_array = np.array(ao_color, dtype=np.float32)
                
                for c in range(3):
                    # Darken based on ao_darken, then blend color based on ao_blend
                    darkened = img_array[:, :, c] * (1.0 - crevice_mask * ao_darken)
                    img_array[:, :, c] = darkened * (1.0 - crevice_mask * ao_blend) + \
                                        ao_color_array[c] * crevice_mask * ao_blend
                
            except Exception as e:
                print(f"Failed to load AO map: {e}")
        
        # SECOND: Apply edge effects on top if enabled and map provided
        if enable_edge and edge_map_path and Path(edge_map_path).exists():
            try:
                # Load edge map (white=edges, black=no edges)
                edge_img = Image.open(edge_map_path).convert('L')
                if edge_img.size != (width, height):
                    edge_img = edge_img.resize((width, height), Image.BILINEAR)
                edge_mask = np.array(edge_img, dtype=np.float32) / 255.0
                print(f"Using edge map: {Path(edge_map_path).name}")
                
                # Apply brightness boost on edges
                for c in range(3):
                    img_array[:, :, c] += edge_highlight * edge_mask * 255
                
                img_array = np.clip(img_array, 0, 255)
                
                # Mask edge color onto edges
                # Where edge_mask=1 (white), blend in edge_color
                # Where edge_mask=0 (black), keep current image
                edge_color_array = np.array(edge_color, dtype=np.float32)
                
                for c in range(3):
                    img_array[:, :, c] = img_array[:, :, c] * (1.0 - edge_mask * edge_blend) + \
                                        edge_color_array[c] * edge_mask * edge_blend
                
            except Exception as e:
                print(f"Failed to load edge map: {e}")
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def bake_curvature_map(self, model_path: str, output_path: str, 
                          resolution: int = 1024, strength: float = 5.0,
                          use_worldspace: bool = True) -> bool:
        """
        Bake curvature map from 3D model to a PNG file (edges only)
        Uses original UV-space edge detection
        
        Args:
            model_path: Path to 3D model file
            output_path: Path to save curvature map PNG
            resolution: Map resolution (width, height will match aspect)
            strength: Curvature detection strength
            use_worldspace: Ignored, uses UV-space
            
        Returns:
            True if successful
        """
        try:
            # Load model
            if Path(model_path).suffix.lower() == '.obj':
                if not self.load_obj(model_path):
                    print("Failed to load model")
                    return False
            else:
                print(f"Unsupported model format: {Path(model_path).suffix}")
                return False
            
            # Use original UV-space edge detection
            print("Baking edge map (UV-space)...")
            curvature_map = self.compute_curvature_map(resolution, resolution, strength)
            
            # Normalize to 0-255 range (positive values only, edges)
            curvature_map = np.maximum(0, curvature_map)  # Only positive curvature
            if curvature_map.max() > 0:
                curvature_map = np.clip(curvature_map / curvature_map.max() * 255, 0, 255).astype(np.uint8)
            else:
                curvature_map = np.zeros((resolution, resolution), dtype=np.uint8)
            
            # Save as grayscale PNG
            img = Image.fromarray(curvature_map, mode='L')
            img.save(output_path)
            
            print(f"Edge map saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error baking edge map: {e}")
            return False
    
    def bake_ao_map(self, model_path: str, output_path: str,
                    resolution: int = 1024, samples: int = 32, 
                    distance: float = 0.5, use_worldspace: bool = True) -> bool:
        """
        Bake ambient occlusion map from 3D model to a PNG file (crevices)
        Uses Blender for proper AO baking
        
        Args:
            model_path: Path to 3D model file
            output_path: Path to save AO map PNG
            resolution: Map resolution
            samples: Number of AO ray samples
            distance: Ray distance for AO calculation (unused with Blender)
            use_worldspace: Unused, Blender handles this automatically
            
        Returns:
            True if successful
        """
        try:
            if not os.path.exists(self.blender_path):
                print(f"Blender not found at: {self.blender_path}")
                return False
            
            print(f"Baking AO map with Blender...")
            
            # Call Blender to bake AO
            cmd = [
                self.blender_path,
                "--background",
                "--python", self.blender_script,
                "--",
                "ao",
                model_path,
                output_path,
                str(resolution),
                str(samples)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and os.path.exists(output_path):
                print(f"AO map saved to: {output_path}")
                return True
            else:
                print(f"Blender baking failed:")
                print(result.stderr)
                return False
            
        except Exception as e:
            print(f"Error baking AO map: {e}")
            return False
    
    def apply_ao_to_texture(self, img: Image.Image, model_path: str,
                           samples: int = 32, distance: float = 0.5,
                           strength: float = 0.7, hue_shift: float = 0.0,
                           saturation_shift: float = 0.0) -> Image.Image:
        """
        Apply AO from 3D model to texture
        
        Args:
            img: Input texture image
            model_path: Path to 3D model file
            samples: Number of AO samples
            distance: Ray distance for AO
            strength: AO strength (0-1)
            hue_shift: Hue shift in occluded areas (-180 to 180 degrees)
            saturation_shift: Saturation shift in occluded areas (-1 to 1)
            
        Returns:
            Texture with AO applied
        """
        # Load model if needed
        if Path(model_path).suffix.lower() == '.obj':
            if not self.load_obj(model_path):
                print("Failed to load model, skipping AO")
                return img
        else:
            print(f"Unsupported model format: {Path(model_path).suffix}")
            print("Only OBJ is currently supported. FBX/glTF support coming soon!")
            return img
        
        # Compute AO map
        width, height = img.size
        ao_map = self.compute_ao_map(width, height, samples, distance)
        
        # Apply AO to image
        img_array = np.array(img, dtype=np.float32)
        
        # Apply darkening based on AO and strength
        for c in range(3):  # RGB channels
            img_array[:, :, c] *= (1.0 - strength * (1.0 - ao_map))
        
        # Apply color tinting in occluded areas
        if hue_shift != 0 or saturation_shift != 0:
            img_hsv = np.zeros_like(img_array)
            
            for y in range(height):
                for x in range(width):
                    r, g, b = img_array[y, x] / 255.0
                    h, s, v = colorsys.rgb_to_hsv(r, g, b)
                    
                    # Apply shifts proportional to occlusion
                    occlusion = 1.0 - ao_map[y, x]
                    
                    # Shift hue
                    if hue_shift != 0:
                        h = (h + (hue_shift / 360.0) * occlusion) % 1.0
                    
                    # Shift saturation
                    if saturation_shift != 0:
                        s = np.clip(s + saturation_shift * occlusion, 0, 1)
                    
                    r, g, b = colorsys.hsv_to_rgb(h, s, v)
                    img_hsv[y, x] = [r * 255, g * 255, b * 255]
            
            img_array = img_hsv
        
        # Convert back to image
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array, mode='RGB')
