# Texture Pixelator

A free alternative to Pixel8r for creating retro-style pixelated textures with dithering effects. Perfect for use with Blender and GIMP workflows!

## Features

- **Pixelation**: Downsample textures to any resolution with nearest neighbor or bilinear filtering
- **Color Quantization**: Reduce colors using bit depth reduction or adaptive palette quantization
- **Dithering**: Apply Bayer matrix (2x2, 4x4, 8x8) or Floyd-Steinberg dithering
- **UV Seam Fill (GUI)**: Post-upscale push-pull mip pyramid padding + greedy expansion to eliminate white edge seams
- **Batch Processing**: Process entire folders of textures with consistent settings
- **Simple GUI**: Easy parameter tweaking without preview overhead
- **Alpha Channel Support**: Preserves transparency in textures

## Installation

1. Make sure you have Python 3.8+ installed
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install Pillow numpy scipy trimesh
```

For detailed installation instructions (including Windows, macOS, and Linux), see [INSTALL.md](INSTALL.md).

## Usage

### GUI Mode (Recommended)

Launch the GUI application:

```bash
python gui.py
```

#### Single File Processing:
1. Click "Browse" next to "Single File" and select your texture
2. Choose an output location (or let it auto-generate)
3. Adjust parameters:
   - **Pixel Width**: Target resolution (8-512 pixels)
   - **Resample Mode**: Nearest (sharp) or Bilinear (softer)
   - **Color Quantization**: 
     - None: Keep original colors
     - Bit Depth: Reduce to X bits per channel (1-8)
     - Palette: Adaptive palette with X total colors (2-256)
   - **Dithering**:
     - None: No dithering
     - Bayer: Ordered dithering with matrix size
     - Floyd-Steinberg: Error diffusion dithering
   - **Dither Strength**: 0.0-1.0 (only for Bayer)
   - **UV Seam Fill** (optional): push-pull texture padding to eliminate white corners at UV seam edges after upscale
4. Click "Process Single File"

#### Batch Processing:
1. Click "Browse" next to "Batch Folder" and select your input folder
2. Choose an output folder
3. Adjust parameters (same as single file)
4. Enable "Add '_pixelated' suffix" to preserve original filenames
5. Click "Process Batch"

### Command Line Mode

You can also import and use the modules directly in Python:

```python
from texture_pixelator import TexturePixelator

pixelator = TexturePixelator()

pixelator.process_texture(
    input_path="my_texture.png",
    output_path="my_texture_pixelated.png",
    pixel_width=64,
    resample_mode="nearest",
    quantize_method="bit_depth",
    bits_per_channel=5,
    dither_mode="bayer",
    bayer_size="4x4",
    dither_strength=0.5
)
```

## Blender Integration

1. Process your textures using this tool
2. In Blender:
   - Import your 3D model
   - Create materials and add Image Texture nodes
   - Load the pixelated textures
   - **Important**: Set texture interpolation to "Closest" (not Linear) in the Image Texture node
   - For best results, disable mipmaps in texture settings

## Recommended Settings for Different Styles

### PS1/N64 Style
- Pixel Width: 64-128
- Resample: Nearest
- Quantization: Bit Depth, 5 bits per channel
- Dithering: Bayer 4x4, strength 0.3-0.5

### Game Boy / 2-bit Style
- Pixel Width: 32-64
- Resample: Nearest
- Quantization: Palette, 4 colors
- Dithering: Bayer 2x2, strength 0.7

### SNES / 16-bit Style
- Pixel Width: 128-256
- Resample: Nearest
- Quantization: Bit Depth, 5-6 bits per channel
- Dithering: Bayer 4x4 or Floyd-Steinberg

### Soft Pixelation
- Pixel Width: 128-256
- Resample: Bilinear
- Quantization: Bit Depth, 6-7 bits per channel
- Dithering: Bayer 8x8, strength 0.2-0.3

## Tips

- Process textures in order: Downsample → Dither → Quantize (the tool does this automatically)
- Lower dither strength (0.2-0.4) works best with bit depth quantization
- Higher dither strength (0.5-0.8) works better with palette quantization
- Floyd-Steinberg dithering looks great but can't be adjusted with strength
- Keep alpha channels separate if you need precise transparency
- For normal maps (future feature), special handling will preserve directional data

## File Format Support

- Input: PNG, JPG, JPEG, TGA, BMP, TIFF
- Output: PNG (recommended for lossless), JPG, TGA, BMP

## Troubleshooting

**GUI won't launch:**
- Make sure tkinter is installed: `python -m tkinter` (should open a test window)
- On Linux: `sudo apt-get install python3-tk`

**Images look blurry in Blender:**
- Set texture interpolation to "Closest" in the Image Texture node
- Disable automatic mipmap generation

**Colors look wrong:**
- Make sure your input images are in RGB or RGBA mode
- Try using PNG format for both input and output

**Batch processing is slow:**
- This is normal for large images or many files
- Consider processing smaller groups or reducing pixel width

## License

Free to use for personal and commercial projects. No redistribution of the code itself without permission.

## Credits

Inspired by ActionDawg's Pixel8r for Substance Painter. This is an independent, free alternative built for artists who use Blender and GIMP.
