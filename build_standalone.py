"""
Build standalone executable for Texture Pixelator using PyInstaller.
Run: python build_standalone.py
Output: dist/TexturePixelator/ (ready to zip and distribute)
"""
import subprocess
import sys
import shutil
from pathlib import Path

def main():
    project_dir = Path(__file__).parent
    
    # Install PyInstaller if not present
    try:
        import PyInstaller
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "TexturePixelator",
        "--windowed",                    # No console window (GUI app)
        "--noconfirm",                   # Overwrite previous build
        # Bundle blender_baker.py as data file
        "--add-data", f"{project_dir / 'blender_baker.py'};.",
        # Hidden imports that PyInstaller may miss
        "--hidden-import", "PIL._tkinter_finder",
        "--hidden-import", "scipy.ndimage",
        "--hidden-import", "trimesh",
        "--hidden-import", "numpy",
        "--hidden-import", "PIL",
        # Collect all trimesh data (it has a lot of submodules)
        "--collect-submodules", "trimesh",
        # Entry point
        str(project_dir / "gui.py"),
    ]
    
    print("Building standalone executable...")
    print(f"Command: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(project_dir))
    
    # Copy a launcher batch file into dist
    dist_dir = project_dir / "dist" / "TexturePixelator"
    launcher = dist_dir / "Launch TexturePixelator.bat"
    launcher.write_text('@echo off\r\ncd /d "%~dp0"\r\nstart "" "TexturePixelator.exe"\r\n')
    
    print()
    print("=" * 60)
    print("BUILD COMPLETE!")
    print(f"Output folder: {dist_dir}")
    print()
    print("To distribute:")
    print(f'  1. Zip the entire "{dist_dir}" folder')
    print("  2. Send the zip to users")
    print('  3. They extract and run "Launch TexturePixelator.bat"')
    print("=" * 60)


if __name__ == "__main__":
    main()
