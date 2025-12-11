"""
Texture Pixelator - Core pixelation and dithering functionality
"""
import numpy as np
from PIL import Image, ImageFilter
from typing import Tuple, Optional, List
import os
import random
from scipy import ndimage
from ao_baker import SurfaceEffectsBaker


class TexturePixelator:
    """Handles texture pixelation, dithering, and color quantization"""
    
    # Bayer dithering matrices
    BAYER_2X2 = np.array([
        [0, 2],
        [3, 1]
    ]) / 4.0
    
    BAYER_4X4 = np.array([
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5]
    ]) / 16.0
    
    BAYER_8X8 = np.array([
        [0, 32, 8, 40, 2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44, 4, 36, 14, 46, 6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [3, 35, 11, 43, 1, 33, 9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47, 7, 39, 13, 45, 5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21]
    ]) / 64.0
    
    def __init__(self):
        self.bayer_matrices = {
            '2x2': self.BAYER_2X2,
            '4x4': self.BAYER_4X4,
            '8x8': self.BAYER_8X8
        }
        self.surface_baker = SurfaceEffectsBaker()
    
    def preprocess_image(self, img: Image.Image, blur_amount: float = 0.0, 
                        noise_amount: float = 0.0, color_variation: float = 0.0,
                        flood_fill_color: Tuple[int, int, int] = (255, 255, 255),
                        flood_fill_opacity: float = 0.0,
                        hue_shift: float = 0.0,
                        tint_strength: float = 0.0) -> Image.Image:
        """
        Apply preprocessing effects to add complexity before pixelation
        
        Args:
            img: Input image
            blur_amount: Gaussian blur radius (0 = no blur)
            noise_amount: Random noise intensity (0-50)
            color_variation: Random hue/saturation shift amount (0-30)
            flood_fill_color: RGB color for flood fill overlay
            flood_fill_opacity: Opacity of flood fill (0-1)
            hue_shift: Hue rotation in degrees (-180 to 180)
            tint_strength: How much to tint toward flood_fill_color (0-1)
        
        Returns:
            Preprocessed image
        """
        result = img.copy()
        
        # Apply blur
        if blur_amount > 0:
            result = result.filter(ImageFilter.GaussianBlur(radius=blur_amount))
        
        # Apply noise
        if noise_amount > 0:
            arr = np.array(result, dtype=np.float32)
            noise = np.random.normal(0, noise_amount, arr.shape)
            arr = np.clip(arr + noise, 0, 255)
            result = Image.fromarray(arr.astype(np.uint8), mode=result.mode)
        
        # Apply color variation
        if color_variation > 0 and result.mode == 'RGB':
            arr = np.array(result, dtype=np.float32)
            # Random hue shift per pixel
            shift = np.random.uniform(-color_variation, color_variation, arr.shape)
            arr = np.clip(arr + shift, 0, 255)
            result = Image.fromarray(arr.astype(np.uint8), mode='RGB')
        
        # Apply hue shift
        if hue_shift != 0.0 and result.mode == 'RGB':
            import colorsys
            arr = np.array(result, dtype=np.float32) / 255.0
            h, w = arr.shape[:2]
            
            for y in range(h):
                for x in range(w):
                    r, g, b = arr[y, x]
                    h_val, s, v = colorsys.rgb_to_hsv(r, g, b)
                    # Shift hue (wrap around at 1.0)
                    h_val = (h_val + hue_shift / 360.0) % 1.0
                    r, g, b = colorsys.hsv_to_rgb(h_val, s, v)
                    arr[y, x] = [r, g, b]
            
            result = Image.fromarray((arr * 255).astype(np.uint8), mode='RGB')
        
        # Apply tint toward flood fill color
        if tint_strength > 0.0 and result.mode == 'RGB':
            arr = np.array(result, dtype=np.float32)
            tint_color = np.array(flood_fill_color, dtype=np.float32)
            # Blend toward tint color
            arr = arr * (1.0 - tint_strength) + tint_color * tint_strength
            result = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode='RGB')
        
        # Apply flood fill overlay
        if flood_fill_opacity > 0.0 and result.mode == 'RGB':
            arr = np.array(result, dtype=np.float32)
            overlay_color = np.array(flood_fill_color, dtype=np.float32)
            # Blend with flood fill color
            arr = arr * (1.0 - flood_fill_opacity) + overlay_color * flood_fill_opacity
            result = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode='RGB')
        
        return result
    
    def greedy_expand_pixels(self, image: Image.Image, background_color: Tuple[int, int, int] = None,
                            iterations: int = 2, threshold: int = 10) -> Image.Image:
        """
        Expand non-background pixels outward to prevent background bleeding at seams.
        Uses a "push-pull" inpainting algorithm commonly used in texture atlases.
        
        Args:
            image: Input PIL Image
            background_color: RGB tuple of background color to detect. If None, uses edge detection.
            iterations: Number of expansion passes (more = thicker growth)
            threshold: Color difference threshold for background detection (0-255)
        
        Returns:
            Image with expanded foreground pixels
        """
        img_array = np.array(image)
        has_alpha = len(img_array.shape) == 3 and img_array.shape[2] == 4
        
        if has_alpha:
            # For images with alpha, expand based on alpha channel
            rgb = img_array[:, :, :3].copy().astype(np.float32)
            alpha = img_array[:, :, 3].astype(np.float32)
            
            # Pixels with low alpha are background
            mask = alpha >= 128  # True = foreground, False = background
            
            # Iteratively fill background pixels with weighted average of foreground neighbors
            for iteration in range(iterations):
                # Create padded versions for neighbor sampling
                mask_padded = np.pad(mask, 1, mode='edge')
                rgb_padded = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), mode='edge')
                
                # Find background pixels that have at least one foreground neighbor
                background_pixels = ~mask
                
                # For each background pixel, compute weighted average of foreground neighbors
                for y in range(rgb.shape[0]):
                    for x in range(rgb.shape[1]):
                        if not background_pixels[y, x]:
                            continue
                        
                        # Sample 8 neighbors (3x3 window)
                        neighbor_colors = []
                        neighbor_weights = []
                        
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                if dx == 0 and dy == 0:
                                    continue
                                
                                ny, nx = y + 1 + dy, x + 1 + dx
                                if mask_padded[ny, nx]:  # If neighbor is foreground
                                    neighbor_colors.append(rgb_padded[ny, nx])
                                    # Weight by distance (closer neighbors have more influence)
                                    dist = np.sqrt(dx*dx + dy*dy)
                                    neighbor_weights.append(1.0 / dist)
                        
                        # If we have foreground neighbors, fill with weighted average
                        if neighbor_colors:
                            neighbor_colors = np.array(neighbor_colors)
                            neighbor_weights = np.array(neighbor_weights)
                            neighbor_weights /= neighbor_weights.sum()
                            
                            rgb[y, x] = np.sum(neighbor_colors * neighbor_weights[:, np.newaxis], axis=0)
                            mask[y, x] = True  # Mark as filled for next iteration
                            alpha[y, x] = 255
            
            result_array = np.dstack([rgb.astype(np.uint8), alpha.astype(np.uint8)])
            return Image.fromarray(result_array, mode='RGBA')
        
        else:
            # For RGB images, detect background color from edge pixels
            if background_color is None:
                # Sample all edge pixels (top, bottom, left, right edges)
                h, w = img_array.shape[:2]
                edge_pixels = []
                
                # Top and bottom edges
                edge_pixels.extend([tuple(img_array[0, x]) for x in range(w)])
                edge_pixels.extend([tuple(img_array[h-1, x]) for x in range(w)])
                
                # Left and right edges (excluding corners already sampled)
                edge_pixels.extend([tuple(img_array[y, 0]) for y in range(1, h-1)])
                edge_pixels.extend([tuple(img_array[y, w-1]) for y in range(1, h-1)])
                
                # Find most common color among edge pixels
                from collections import Counter
                color_counts = Counter(edge_pixels)
                background_color = color_counts.most_common(1)[0][0]
            
            bg_array = np.array(background_color, dtype=np.float32)
            rgb = img_array.copy().astype(np.float32)
            
            # Calculate color distance from background
            color_diff = np.sqrt(np.sum((rgb - bg_array) ** 2, axis=2))
            mask = color_diff >= threshold  # True = foreground, False = background
            
            # Iteratively fill background pixels with weighted average of foreground neighbors
            for iteration in range(iterations):
                # Create padded versions for neighbor sampling
                mask_padded = np.pad(mask, 1, mode='edge')
                rgb_padded = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), mode='edge')
                
                # Find background pixels
                background_pixels = ~mask
                
                # For each background pixel, compute weighted average of foreground neighbors
                for y in range(rgb.shape[0]):
                    for x in range(rgb.shape[1]):
                        if not background_pixels[y, x]:
                            continue
                        
                        # Sample 8 neighbors (3x3 window)
                        neighbor_colors = []
                        neighbor_weights = []
                        
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                if dx == 0 and dy == 0:
                                    continue
                                
                                ny, nx = y + 1 + dy, x + 1 + dx
                                if mask_padded[ny, nx]:  # If neighbor is foreground
                                    neighbor_colors.append(rgb_padded[ny, nx])
                                    # Weight by distance (closer neighbors have more influence)
                                    dist = np.sqrt(dx*dx + dy*dy)
                                    neighbor_weights.append(1.0 / dist)
                        
                        # If we have foreground neighbors, fill with weighted average
                        if neighbor_colors:
                            neighbor_colors = np.array(neighbor_colors)
                            neighbor_weights = np.array(neighbor_weights)
                            neighbor_weights /= neighbor_weights.sum()
                            
                            rgb[y, x] = np.sum(neighbor_colors * neighbor_weights[:, np.newaxis], axis=0)
                            mask[y, x] = True  # Mark as filled for next iteration
            
            return Image.fromarray(rgb.astype(np.uint8), mode='RGB')
    
    def downsample_image(self, image: Image.Image, target_width: int, 
                        resample_mode: str = 'nearest') -> Image.Image:
        """
        Downsample image to target width while maintaining aspect ratio
        
        Args:
            image: Input PIL Image
            target_width: Target width in pixels
            resample_mode: 'nearest' or 'bilinear'
        
        Returns:
            Downsampled PIL Image
        """
        aspect_ratio = image.height / image.width
        target_height = int(target_width * aspect_ratio)
        
        resample_filter = Image.NEAREST if resample_mode == 'nearest' else Image.BILINEAR
        
        return image.resize((target_width, target_height), resample_filter)
    
    def quantize_colors_bit_depth(self, image: Image.Image, bits_per_channel: int) -> Image.Image:
        """
        Reduce color depth by limiting bits per channel
        
        Args:
            image: Input PIL Image
            bits_per_channel: Number of bits per color channel (1-8)
        
        Returns:
            Color quantized PIL Image
        """
        img_array = np.array(image).astype(float)
        
        # Calculate number of levels
        levels = 2 ** bits_per_channel
        
        # Quantize each channel
        quantized = np.floor(img_array / 256 * levels) / (levels - 1) * 255
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)
        
        return Image.fromarray(quantized)
    
    def quantize_colors_palette(self, image: Image.Image, palette_colors: int) -> Image.Image:
        """
        Reduce colors using adaptive palette quantization
        
        Args:
            image: Input PIL Image
            palette_colors: Number of colors in palette (2-256)
        
        Returns:
            Palette quantized PIL Image
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Use PIL's quantize with median cut algorithm
        quantized = image.quantize(colors=palette_colors, method=Image.MEDIANCUT)
        
        # Convert back to RGB
        return quantized.convert('RGB')
    
    def apply_bayer_dither(self, image: Image.Image, matrix_size: str = '4x4', 
                          strength: float = 1.0) -> Image.Image:
        """
        Apply Bayer matrix dithering
        
        Args:
            image: Input PIL Image
            matrix_size: '2x2', '4x4', or '8x8'
            strength: Dithering strength (0.0-1.0)
        
        Returns:
            Dithered PIL Image
        """
        if matrix_size not in self.bayer_matrices:
            raise ValueError(f"Invalid matrix size: {matrix_size}")
        
        img_array = np.array(image).astype(float) / 255.0
        matrix = self.bayer_matrices[matrix_size]
        
        # Tile the Bayer matrix to match image dimensions
        h, w = img_array.shape[:2]
        matrix_h, matrix_w = matrix.shape
        
        tiled_matrix = np.tile(matrix, (h // matrix_h + 1, w // matrix_w + 1))
        tiled_matrix = tiled_matrix[:h, :w]
        
        # Apply dithering with strength
        if len(img_array.shape) == 3:  # Color image
            tiled_matrix = np.expand_dims(tiled_matrix, axis=2)
        
        dithered = img_array + (tiled_matrix - 0.5) * strength * (1.0 / 255.0)
        dithered = np.clip(dithered * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(dithered)
    
    def apply_floyd_steinberg_dither(self, image: Image.Image, 
                                    color_levels: int = 8) -> Image.Image:
        """
        Apply Floyd-Steinberg error diffusion dithering
        
        Args:
            image: Input PIL Image
            color_levels: Number of levels per channel
        
        Returns:
            Dithered PIL Image
        """
        img_array = np.array(image).astype(float)
        h, w = img_array.shape[:2]
        
        # Process each channel
        for c in range(img_array.shape[2] if len(img_array.shape) == 3 else 1):
            channel = img_array[:, :, c] if len(img_array.shape) == 3 else img_array
            
            for y in range(h):
                for x in range(w):
                    old_pixel = channel[y, x]
                    new_pixel = np.round(old_pixel / 255 * (color_levels - 1)) / (color_levels - 1) * 255
                    channel[y, x] = new_pixel
                    error = old_pixel - new_pixel
                    
                    # Distribute error to neighboring pixels
                    if x + 1 < w:
                        channel[y, x + 1] += error * 7 / 16
                    if y + 1 < h:
                        if x > 0:
                            channel[y + 1, x - 1] += error * 3 / 16
                        channel[y + 1, x] += error * 5 / 16
                        if x + 1 < w:
                            channel[y + 1, x + 1] += error * 1 / 16
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def process_texture(self, input_path: str, output_path: str,
                       blur_amount: float = 0.0,
                       noise_amount: float = 0.0,
                       color_variation: float = 0.0,
                       flood_fill_color: tuple = (255, 255, 255),
                       flood_fill_opacity: float = 0.0,
                       hue_shift: float = 0.0,
                       tint_strength: float = 0.0,
                       enable_surface: bool = False,
                       model_path: str = None,
                       curvature_strength: float = 0.5,
                       edge_highlight: float = 0.3,
                       edge_saturation: float = 0.0,
                       edge_blend: float = 0.7,
                       enable_edge: bool = True,
                       ao_darken: float = 0.5,
                       ao_blend: float = 0.5,
                       enable_ao: bool = True,
                       edge_color: tuple = (255, 136, 0),
                       ao_color: tuple = (50, 30, 20),
                       baked_map_path: str = None,
                       ao_map_path: str = None,
                       pixel_width: int = 16,
                       resample_mode: str = 'nearest',
                       enable_greedy_expand: bool = True,
                       greedy_iterations: int = 5,
                       quantize_method: str = 'bit_depth',
                       bits_per_channel: int = 5,
                       palette_colors: int = 16,
                       dither_mode: str = 'none',
                       bayer_size: int = 4,
                       dither_strength: float = 1.0,
                       is_normal_map: bool = False) -> None:
        """
        Process a texture with all specified effects
        
        Args:
            input_path: Path to input image
            output_path: Path to save output image
            blur_amount: Preprocessing blur radius (0 = no blur)
            noise_amount: Preprocessing noise intensity (0-50)
            color_variation: Preprocessing color variation (0-30)
            pixel_width: Target width in pixels
            resample_mode: 'nearest' or 'bilinear'
            quantize_method: 'none', 'bit_depth', or 'palette'
            bits_per_channel: Bits per channel for bit_depth mode (1-8)
            palette_colors: Number of colors for palette mode (2-256)
            dither_mode: 'none', 'bayer', or 'floyd_steinberg'
            bayer_size: Bayer matrix size ('2x2', '4x4', '8x8')
            dither_strength: Dithering strength (0.0-1.0)
            is_normal_map: Special handling for normal maps (future use)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load image
            image = Image.open(input_path)
            
            # Convert to RGB if needed (preserve alpha for later)
            has_alpha = image.mode == 'RGBA'
            if has_alpha:
                alpha_channel = image.split()[3]
                image = image.convert('RGB')
            else:
                image = image.convert('RGB')
            
            # Phase 1: Surface Effects (apply BEFORE preprocessing so blur affects edges too)
            if enable_surface and baked_map_path:
                print("Applying surface effects from baked maps...")
                image = self.surface_baker.apply_surface_effects_to_texture(
                    image, edge_highlight, edge_saturation, edge_color, baked_map_path, edge_blend, enable_edge,
                    ao_darken, ao_color, ao_map_path, ao_blend, enable_ao
                )
            
            # Phase 2: Preprocessing
            if blur_amount > 0 or noise_amount > 0 or color_variation > 0 or flood_fill_opacity > 0 or hue_shift != 0 or tint_strength > 0:
                image = self.preprocess_image(image, blur_amount, noise_amount, color_variation,
                                             flood_fill_color, flood_fill_opacity, hue_shift, tint_strength)
            
            # Phase 2.5: Greedy expansion BEFORE downsampling (prevents background bleed)
            if enable_greedy_expand:
                image = self.greedy_expand_pixels(image, iterations=greedy_iterations, threshold=15)
            
            # Phase 3: Pixelation
            # Step 1: Downsample
            image = self.downsample_image(image, pixel_width, resample_mode)
            
            # Store downsampled size before upscaling
            downsampled_width = image.width
            downsampled_height = image.height
            
            # Step 2: Apply dithering (before quantization for better results)
            if dither_mode == 'bayer':
                image = self.apply_bayer_dither(image, bayer_size, dither_strength)
            elif dither_mode == 'floyd_steinberg':
                color_levels = 2 ** bits_per_channel if quantize_method == 'bit_depth' else 8
                image = self.apply_floyd_steinberg_dither(image, color_levels)
            
            # Step 3: Color quantization
            if quantize_method == 'bit_depth':
                image = self.quantize_colors_bit_depth(image, bits_per_channel)
            elif quantize_method == 'palette':
                image = self.quantize_colors_palette(image, palette_colors)
            
            # Step 4: Restore alpha channel if it existed
            if has_alpha:
                alpha_resized = alpha_channel.resize(image.size, Image.NEAREST)
                image = image.convert('RGBA')
                image.putalpha(alpha_resized)
            
            # Step 5: Scale back up to original dimensions with nearest neighbor
            # This preserves the pixelated look but at the original texture size
            original_width = Image.open(input_path).width
            original_height = Image.open(input_path).height
            
            # Calculate scale factor to maintain aspect ratio
            scale_x = original_width / downsampled_width
            scale_y = original_height / downsampled_height
            
            # Use the smaller scale to maintain aspect ratio
            scale = min(scale_x, scale_y)
            
            final_width = int(downsampled_width * scale)
            final_height = int(downsampled_height * scale)
            
            # Upscale with nearest neighbor to keep pixels sharp
            image = image.resize((final_width, final_height), Image.NEAREST)
            
            # Save output
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            image.save(output_path)
            
            return True
            
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return False


if __name__ == "__main__":
    # Simple test
    pixelator = TexturePixelator()
    print("TexturePixelator module loaded successfully")
