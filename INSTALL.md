# Installation Guide

## Quick Start

If you just want to use the command-line version (no GUI needed):

```bash
# Install dependencies
pip3 install -r requirements.txt

# Process a texture
python cli.py input.png output.png --preset ps1
```

## GUI Installation (Optional)

The GUI requires tkinter, which may not be included with all Python installations on macOS.

### Option 1: Install Python with tkinter via Homebrew (Recommended)

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python with tkinter support
brew install python-tk@3.12

# Use the Homebrew Python
/opt/homebrew/bin/python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Launch GUI
python gui.py
```

### Option 2: Use system Python

macOS system Python usually includes tkinter:

```bash
# Create venv with system Python
/usr/bin/python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Launch GUI
python gui.py
```

### Option 3: Install tkinter for your current Python

If you're using pyenv or another Python manager:

```bash
# For pyenv users
brew install tcl-tk
env \
  PATH="$(brew --prefix tcl-tk)/bin:$PATH" \
  LDFLAGS="-L$(brew --prefix tcl-tk)/lib" \
  CPPFLAGS="-I$(brew --prefix tcl-tk)/include" \
  PKG_CONFIG_PATH="$(brew --prefix tcl-tk)/lib/pkgconfig" \
  pyenv install 3.12.0

pyenv local 3.12.0
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Verify Installation

Test tkinter:
```bash
python -m tkinter
```
This should open a small test window. If it does, you're good to go!

## Using Without GUI

The command-line interface (`cli.py`) works without tkinter and provides all the same functionality:

```bash
# See all options
python cli.py --help

# Quick presets
python cli.py input.png output.png --preset ps1      # PS1/N64 style
python cli.py input.png output.png --preset gameboy  # Game Boy style
python cli.py input.png output.png --preset snes     # SNES style

# Custom settings
python cli.py input.png output.png \
  --width 64 \
  --bits 5 \
  --dither bayer \
  --matrix 4x4 \
  --strength 0.4

# Batch process
python cli.py input_folder/ output_folder/ \
  --batch \
  --preset ps1 \
  --suffix
```

## Troubleshooting

**"command not found: pip" or "command not found: python"**
- Use `pip3` and `python3` instead
- Or install Python: `brew install python`

**GUI won't open**
- Try the CLI version (`cli.py`) instead
- Follow the tkinter installation steps above

**"Permission denied"**
- Use `pip3 install --user -r requirements.txt`
- Or create a virtual environment first

**Virtual environment not activating**
```bash
# Create fresh venv
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
