# Installation Guide

This guide will help you install and run Texture Pixelator on your computer. No programming experience needed!

## What You Need

- Python 3.8 or newer
- About 5-10 minutes to set up

## Installation Steps

### Step 1: Install Python

Download and install Python from [python.org](https://www.python.org/downloads/)

**Important for Windows users:** During installation, make sure to check the box that says **"Add Python to PATH"**. This makes it easier to use Python.

### Step 2: Download Texture Pixelator

Download this project and extract it to a folder on your computer (like `Documents\TexturePixelator`).

### Step 3: Open Command Prompt/Terminal

**Windows:**
1. Press `Windows Key + R`
2. Type `cmd` and press Enter
3. Navigate to your TexturePixelator folder:
   ```powershell
   cd Documents\TexturePixelator
   ```

**macOS:**
1. Press `Command + Space`
2. Type `Terminal` and press Enter
3. Navigate to your TexturePixelator folder:
   ```bash
   cd Documents/TexturePixelator
   ```

### Step 4: Install Required Libraries

Copy and paste this command into your terminal/command prompt and press Enter:

**Windows:**
```powershell
pip install -r requirements.txt
```

**macOS:**
```bash
pip3 install -r requirements.txt
```

This will download and install all the necessary components. It may take a minute or two.

### Step 5: Launch the Application

**Windows:**
```powershell
python gui.py
```

**macOS:**
```bash
python3 gui.py
```

The Texture Pixelator window should open and you're ready to go!

## Easy Launch (Optional)

### Windows

For easier launching, you can use the included batch file:
1. Double-click `launch.bat` in the project folder
2. The application will start automatically

Or create a desktop shortcut:
1. Right-click `launch.bat`
2. Select "Send to" → "Desktop (create shortcut)"
3. Now you can double-click the shortcut to launch the app

### macOS

Create a launch script:
1. Open TextEdit
2. Paste this code (replace with your actual path):
   ```bash
   #!/bin/bash
   cd /Users/YourName/Documents/TexturePixelator
   python3 gui.py
   ```
3. Save as `launch.command` in the project folder
4. Open Terminal and make it executable:
   ```bash
   chmod +x launch.command
   ```
5. Now you can double-click `launch.command` to start the app

## Troubleshooting

### Common Issues

#### Windows

**"'python' is not recognized as an internal or external command"**
- Python isn't installed or wasn't added to PATH during installation
- Solution: Reinstall Python from python.org and check "Add Python to PATH"
- Or try using `py` instead of `python` in the commands

**"No module named 'PIL'" or similar errors**
- The required libraries didn't install correctly
- Solution: Run `pip install -r requirements.txt` again

**Application window won't open**
- Tkinter (the GUI library) might not be installed
- Solution: Reinstall Python and make sure "tcl/tk and IDLE" is checked during installation

#### macOS

**"command not found: python" or "command not found: pip"**
- Use `python3` and `pip3` instead of `python` and `pip`

**Application window won't open (tkinter missing)**
- macOS sometimes doesn't include tkinter by default
- Solution: Install Python with tkinter support using Homebrew:
  ```bash
  # Install Homebrew (if you don't have it)
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  
  # Install Python with tkinter
  brew install python-tk@3.12
  
  # Use Homebrew Python
  pip3 install -r requirements.txt
  python3 gui.py
  ```

**"Permission denied" errors**
- Try adding `--user` to the install command:
  ```bash
  pip3 install --user -r requirements.txt
  ```

#### Both Platforms

**GUI is slow or unresponsive**
- Processing large images takes time - be patient
- Try processing smaller test images first
- Close other applications to free up memory

**Can't find the project folder**
- Make sure you extracted the downloaded files
- Note the full path to where you saved it
- Use `cd` command to navigate to that exact location

## Need More Help?

If you're still having trouble:
1. Make sure you're using Python 3.8 or newer (check with `python --version` or `python3 --version`)
2. Try restarting your computer after installing Python
3. Make sure you're in the correct folder when running commands (use `dir` on Windows or `ls` on macOS to see files)
