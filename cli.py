"""
Command-line interface for Texture Pixelator
"""
import argparse
import sys
from texture_pixelator import TexturePixelator
from batch_process import BatchProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Pixelate and dither textures for retro game aesthetics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic pixelation to 64px width
  python cli.py input.png output.png
  
  # PS1 style with dithering
  python cli.py input.png output.png --width 64 --bits 5 --dither bayer --strength 0.4
  
  # Game Boy style (4 colors, heavy dither)
  python cli.py input.png output.png --width 32 --palette 4 --dither bayer --matrix 2x2 --strength 0.7
  
  # Batch process a folder
  python cli.py input_folder/ output_folder/ --batch --width 128 --bits 6
        """
    )
    
    # File/folder arguments
    parser.add_argument('input', help='Input file or folder (use --batch for folders)')
    parser.add_argument('output', help='Output file or folder')
    parser.add_argument('--batch', action='store_true', help='Batch process entire folder')
    
    # Pixelation settings
    parser.add_argument('-w', '--width', type=int, default=64, help='Target pixel width (default: 64)')
    parser.add_argument('-r', '--resample', choices=['nearest', 'bilinear'], default='nearest',
                       help='Resampling mode (default: nearest)')
    
    # Color quantization
    quant_group = parser.add_mutually_exclusive_group()
    quant_group.add_argument('--bits', type=int, metavar='N',
                            help='Bits per channel (1-8), e.g., 5 = 32 colors/channel')
    quant_group.add_argument('--palette', type=int, metavar='N',
                            help='Total palette colors (2-256)')
    
    # Dithering
    parser.add_argument('-d', '--dither', choices=['none', 'bayer', 'floyd_steinberg'], 
                       default='none', help='Dithering mode (default: none)')
    parser.add_argument('-m', '--matrix', choices=['2x2', '4x4', '8x8'], default='4x4',
                       help='Bayer matrix size (default: 4x4)')
    parser.add_argument('-s', '--strength', type=float, default=0.5,
                       help='Dither strength 0.0-1.0 (default: 0.5)')
    
    # Advanced
    parser.add_argument('--suffix', action='store_true',
                       help='Add _pixelated suffix in batch mode')
    
    # Presets
    parser.add_argument('--preset', choices=['ps1', 'n64', 'gameboy', 'snes', 'soft'],
                       help='Use a preset configuration')
    
    args = parser.parse_args()
    
    # Apply presets
    if args.preset:
        if args.preset in ['ps1', 'n64']:
            args.width = 64
            args.bits = 5
            args.dither = 'bayer'
            args.matrix = '4x4'
            args.strength = 0.4
        elif args.preset == 'gameboy':
            args.width = 32
            args.palette = 4
            args.dither = 'bayer'
            args.matrix = '2x2'
            args.strength = 0.7
        elif args.preset == 'snes':
            args.width = 128
            args.bits = 5
            args.dither = 'bayer'
            args.matrix = '4x4'
            args.strength = 0.3
        elif args.preset == 'soft':
            args.width = 128
            args.resample = 'bilinear'
            args.bits = 6
            args.dither = 'bayer'
            args.matrix = '8x8'
            args.strength = 0.2
    
    # Determine quantization method
    if args.palette:
        quantize_method = 'palette'
        bits_per_channel = 5  # default
        palette_colors = args.palette
    elif args.bits:
        quantize_method = 'bit_depth'
        bits_per_channel = args.bits
        palette_colors = 32  # default
    else:
        quantize_method = 'bit_depth'
        bits_per_channel = 5  # default
        palette_colors = 32  # default
    
    # Build settings
    settings = {
        'pixel_width': args.width,
        'resample_mode': args.resample,
        'quantize_method': quantize_method,
        'bits_per_channel': bits_per_channel,
        'palette_colors': palette_colors,
        'dither_mode': args.dither,
        'bayer_size': args.matrix,
        'dither_strength': args.strength,
        'is_normal_map': False,
        'add_suffix': args.suffix
    }
    
    # Process
    if args.batch:
        print(f"Processing batch: {args.input} -> {args.output}")
        print(f"Settings: {args.width}px, {quantize_method}, dither={args.dither}")
        print("-" * 60)
        
        processor = BatchProcessor()
        results = processor.process_batch(args.input, args.output, settings)
        
        print("-" * 60)
        print(f"Complete: {results['success']}/{results['total']} succeeded, {results['failed']} failed")
        
        if results['failed'] > 0:
            print("\nFailed files:")
            for f in results['failed_files']:
                print(f"  - {f}")
            sys.exit(1)
    else:
        print(f"Processing: {args.input} -> {args.output}")
        print(f"Settings: {args.width}px, {quantize_method}, dither={args.dither}")
        
        pixelator = TexturePixelator()
        success = pixelator.process_texture(args.input, args.output, **settings)
        
        if success:
            print("✓ Success!")
        else:
            print("✗ Failed!")
            sys.exit(1)


if __name__ == "__main__":
    main()
