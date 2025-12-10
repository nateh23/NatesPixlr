"""
Batch Processor - Process multiple textures with consistent settings
"""
import os
from pathlib import Path
from typing import List, Dict
from texture_pixelator import TexturePixelator


class BatchProcessor:
    """Batch process multiple texture files"""
    
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.tga', '.bmp', '.tiff'}
    
    def __init__(self):
        self.pixelator = TexturePixelator()
    
    def find_textures(self, directory: str) -> List[str]:
        """
        Find all supported texture files in directory
        
        Args:
            directory: Path to search
        
        Returns:
            List of texture file paths
        """
        textures = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in self.SUPPORTED_FORMATS:
                    textures.append(os.path.join(root, file))
        return textures
    
    def process_batch(self, input_dir: str, output_dir: str, settings: Dict) -> Dict:
        """
        Process all textures in a directory
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            settings: Dictionary of processing settings
        
        Returns:
            Dictionary with success/failure counts and file lists
        """
        textures = self.find_textures(input_dir)
        
        results = {
            'total': len(textures),
            'success': 0,
            'failed': 0,
            'failed_files': []
        }
        
        for texture_path in textures:
            # Preserve directory structure
            rel_path = os.path.relpath(texture_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            
            # Add suffix to filename if specified
            if settings.get('add_suffix', False):
                base, ext = os.path.splitext(output_path)
                output_path = f"{base}_pixelated{ext}"
            
            # Process texture
            success = self.pixelator.process_texture(
                input_path=texture_path,
                output_path=output_path,
                pixel_width=settings.get('pixel_width', 64),
                resample_mode=settings.get('resample_mode', 'nearest'),
                quantize_method=settings.get('quantize_method', 'bit_depth'),
                bits_per_channel=settings.get('bits_per_channel', 5),
                palette_colors=settings.get('palette_colors', 32),
                dither_mode=settings.get('dither_mode', 'none'),
                bayer_size=settings.get('bayer_size', '4x4'),
                dither_strength=settings.get('dither_strength', 0.5),
                is_normal_map=settings.get('is_normal_map', False)
            )
            
            if success:
                results['success'] += 1
                print(f"✓ Processed: {rel_path}")
            else:
                results['failed'] += 1
                results['failed_files'].append(texture_path)
                print(f"✗ Failed: {rel_path}")
        
        return results


if __name__ == "__main__":
    # Simple test
    processor = BatchProcessor()
    print("BatchProcessor module loaded successfully")
