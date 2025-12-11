"""Test AO baking with Blender"""
import subprocess
import sys
from pathlib import Path

# Paths
obj_path = str(Path.home() / "Desktop" / "swordTest.obj")
output_path = "/tmp/test_ao_sword.png"
blender_path = "/Applications/Blender.app/Contents/MacOS/Blender"
script_path = str(Path(__file__).parent / "blender_baker.py")

print(f"Testing AO bake on: {obj_path}")
print(f"Output will be: {output_path}")
print(f"Using Blender at: {blender_path}")

# Run Blender baker
cmd = [
    blender_path,
    "--background",
    "--python", script_path,
    "--",
    "ao",
    obj_path,
    output_path,
    "1024",
    "128"
]

print(f"\nRunning command:")
print(" ".join(cmd))
print("\n" + "="*60)

result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print("\n" + "="*60)

if result.returncode == 0:
    print(f"\n✓ Success! Check output at: {output_path}")
else:
    print(f"\n✗ Failed with return code: {result.returncode}")
