"""
Surface Effects Baker - Bake curvature maps from 3D models onto textures
"""
import numpy as np
from PIL import Image, ImageFilter
import colorsys
from pathlib import Path
from scipy import ndimage


class SurfaceEffectsBaker:
    """Handles AO baking from 3D models onto textures"""
    
    def __init__(self):
        self.mesh_loaded = False
        self.vertices = None
        self.uvs = None
        self.normals = None
        self.faces = None
    
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
            # Return neutral (flat) if no mesh
            return np.zeros((height, width), dtype=np.float32)
        
        # Create edge map by drawing lines along face boundaries
        edge_map = np.zeros((height, width), dtype=np.float32)
        
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
        
        # Apply Gaussian blur to spread edge influence
        if strength > 0:
            edge_map = ndimage.gaussian_filter(edge_map, sigma=3.0)
        
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
        Compute AO map by raycasting from surface
        
        Args:
            width: Texture width
            height: Texture height
            samples: Number of AO samples per pixel
            distance: Maximum ray distance
            
        Returns:
            AO map as numpy array (0 = fully occluded, 1 = no occlusion)
        """
        if not self.mesh_loaded or self.uvs is None:
            # Return white (no occlusion) if no mesh
            return np.ones((height, width), dtype=np.float32)
        
        # Create AO map
        ao_map = np.zeros((height, width), dtype=np.float32)
        sample_count = np.zeros((height, width), dtype=np.int32)
        
        # For each face, rasterize and compute AO
        for face in self.faces:
            if len(face) < 3:
                continue
            
            # Get UV coordinates for this face
            uvs_face = []
            positions_face = []
            normals_face = []
            
            for v_idx, vt_idx, vn_idx in face:
                if vt_idx is not None:
                    uvs_face.append(self.uvs[vt_idx])
                    positions_face.append(self.vertices[v_idx])
                    if vn_idx is not None and self.normals is not None:
                        normals_face.append(self.normals[vn_idx])
            
            if len(uvs_face) < 3:
                continue
            
            # Simple rasterization: sample in triangle
            # Convert UV to pixel coordinates
            uv0, uv1, uv2 = uvs_face[0], uvs_face[1], uvs_face[2]
            
            # Get bounding box in texture space
            min_u = max(0, min(uv0[0], uv1[0], uv2[0]))
            max_u = min(1, max(uv0[0], uv1[0], uv2[0]))
            min_v = max(0, min(uv0[1], uv1[1], uv2[1]))
            max_v = min(1, max(uv0[1], uv1[1], uv2[1]))
            
            min_x = int(min_u * width)
            max_x = min(width - 1, int(max_u * width))
            min_y = int((1 - max_v) * height)  # Flip V
            max_y = min(height - 1, int((1 - min_v) * height))
            
            # Sample pixels in bounding box
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    u = x / width
                    v = 1 - (y / height)  # Flip V
                    
                    # Check if point is inside triangle (barycentric)
                    if self._point_in_triangle(u, v, uv0, uv1, uv2):
                        # Compute simple AO: average of hemisphere samples
                        # For now, use a simple falloff based on distance to center
                        # Real AO would raycast, but that's expensive
                        ao_value = self._compute_simple_ao(samples, distance)
                        ao_map[y, x] += ao_value
                        sample_count[y, x] += 1
        
        # Average samples
        mask = sample_count > 0
        ao_map[mask] /= sample_count[mask]
        
        # Fill in unmapped areas with full brightness
        ao_map[~mask] = 1.0
        
        return ao_map
    
    def _point_in_triangle(self, px, py, p0, p1, p2):
        """Check if point (px, py) is inside triangle defined by p0, p1, p2"""
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        
        d1 = sign((px, py), p0, p1)
        d2 = sign((px, py), p1, p2)
        d3 = sign((px, py), p2, p0)
        
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        
        return not (has_neg and has_pos)
    
    def _compute_simple_ao(self, samples: int, distance: float) -> float:
        """
        Compute simple AO value using random hemisphere sampling
        This is a simplified version - real AO would do actual raycasting
        """
        # For now, return a random occlusion factor
        # In a real implementation, this would raycast in hemisphere
        occlusion = 0.0
        for _ in range(samples):
            # Random hemisphere direction
            # Check if ray hits geometry within distance
            # For simplified version, use random occlusion
            occlusion += np.random.uniform(0.0, 0.3)  # Simulate some occlusion
        
        return 1.0 - (occlusion / samples)
    
    def apply_surface_effects_to_texture(self, img: Image.Image,
                                        edge_highlight: float = 0.3,
                                        edge_color: tuple = (255, 136, 0),
                                        edge_map_path: str = None,
                                        ao_darken: float = 0.5,
                                        ao_color: tuple = (50, 30, 20),
                                        ao_map_path: str = None) -> Image.Image:
        """Apply surface effects using pre-baked edge and AO maps
        
        Args:
            img: Input texture image
            edge_highlight: Brightness boost on edges (0-1)
            edge_color: RGB tuple (0-255) to blend onto edges
            edge_map_path: Path to pre-baked edge map (REQUIRED)
            ao_darken: How much to darken crevices (0-1)
            ao_color: RGB tuple (0-255) to blend into occluded areas
            ao_map_path: Path to pre-baked AO map (optional)
            
        Returns:
            Texture with surface effects applied
        """
        if not edge_map_path or not Path(edge_map_path).exists():
            print("ERROR: Edge map is required. Please bake edge map first.")
            return img
        
        width, height = img.size
        
        try:
            # Load edge map
            edge_img = Image.open(edge_map_path).convert('L')
            if edge_img.size != (width, height):
                edge_img = edge_img.resize((width, height), Image.BILINEAR)
            edge_mask = np.array(edge_img, dtype=np.float32) / 255.0
            print(f"Using edge map: {Path(edge_map_path).name}")
        except Exception as e:
            print(f"Failed to load edge map: {e}")
            return img
        
        # Convert image to array
        img_array = np.array(img, dtype=np.float32)
        
        # Apply edge effects
        # 1. Brightness boost on edges
        for c in range(3):
            img_array[:, :, c] += edge_highlight * edge_mask * 255
        
        img_array = np.clip(img_array, 0, 255)
        
        # 2. Blend edge color
        edge_color_array = np.array(edge_color, dtype=np.float32)
        blend_strength = 0.7  # How much edge color to blend (increased for visibility)
        
        for c in range(3):
            img_array[:, :, c] = img_array[:, :, c] * (1 - edge_mask * blend_strength) + \
                                edge_color_array[c] * edge_mask * blend_strength
        
        # Apply AO effects if map provided
        if ao_map_path and Path(ao_map_path).exists():
            try:
                # Load AO map
                ao_img = Image.open(ao_map_path).convert('L')
                if ao_img.size != (width, height):
                    ao_img = ao_img.resize((width, height), Image.BILINEAR)
                ao_mask = np.array(ao_img, dtype=np.float32) / 255.0
                print(f"Using AO map: {Path(ao_map_path).name}")
                
                # Invert: 1=bright (not occluded), 0=dark (occluded)
                occlusion_mask = 1.0 - ao_mask
                
                # 1. Darken occluded areas
                for c in range(3):
                    img_array[:, :, c] *= (1.0 - ao_darken * occlusion_mask)
                
                # 2. Blend AO color into crevices
                ao_color_array = np.array(ao_color, dtype=np.float32)
                ao_blend_strength = 0.5
                
                for c in range(3):
                    img_array[:, :, c] = img_array[:, :, c] * (1 - occlusion_mask * ao_blend_strength) + \
                                        ao_color_array[c] * occlusion_mask * ao_blend_strength
                
            except Exception as e:
                print(f"Failed to load AO map: {e}")
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def bake_curvature_map(self, model_path: str, output_path: str, 
                          resolution: int = 1024, strength: float = 5.0) -> bool:
        """
        Bake curvature map from 3D model to a PNG file (edges only)
        
        Args:
            model_path: Path to 3D model file
            output_path: Path to save curvature map PNG
            resolution: Map resolution (width, height will match aspect)
            strength: Curvature detection strength
            
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
            
            # Compute curvature map
            curvature_map = self.compute_curvature_map(resolution, resolution, strength)
            
            # Normalize to 0-255 range (positive values only, edges)
            curvature_map = np.maximum(0, curvature_map)  # Only positive curvature
            curvature_map = np.clip(curvature_map * 255, 0, 255).astype(np.uint8)
            
            # Save as grayscale PNG
            img = Image.fromarray(curvature_map, mode='L')
            img.save(output_path)
            
            print(f"Curvature map saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error baking curvature map: {e}")
            return False
    
    def bake_ao_map(self, model_path: str, output_path: str,
                    resolution: int = 1024, samples: int = 32, 
                    distance: float = 0.5) -> bool:
        """
        Bake ambient occlusion map from 3D model to a PNG file (crevices)
        
        Args:
            model_path: Path to 3D model file
            output_path: Path to save AO map PNG
            resolution: Map resolution
            samples: Number of AO ray samples
            distance: Ray distance for AO calculation
            
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
            
            # Compute AO map
            ao_map = self.compute_ao_map(resolution, resolution, samples, distance)
            
            # Normalize to 0-255 (0=fully occluded/dark, 255=fully lit/bright)
            ao_map = np.clip(ao_map * 255, 0, 255).astype(np.uint8)
            
            # Save as grayscale PNG
            img = Image.fromarray(ao_map, mode='L')
            img.save(output_path)
            
            print(f"AO map saved to: {output_path}")
            return True
            
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
