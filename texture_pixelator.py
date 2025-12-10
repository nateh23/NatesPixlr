"""
Texture Pixelator - Core pixelation and dithering functionality
"""
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List
import os


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
                       pixel_width: int = 64,
                       resample_mode: str = 'nearest',
                       quantize_method: str = 'bit_depth',
                       bits_per_channel: int = 5,
                       palette_colors: int = 32,
                       dither_mode: str = 'none',
                       bayer_size: str = '4x4',
                       dither_strength: float = 0.5,
                       is_normal_map: bool = False) -> bool:
        """
        Process a texture with all specified effects
        
        Args:
            input_path: Path to input image
            output_path: Path to save output image
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
