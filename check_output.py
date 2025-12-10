from PIL import Image
import numpy as np

# Check the curvature map
print("=== Curvature Map ===")
curv_img = Image.open("/Users/nathanhenderson/Desktop/test_curvature_map.png")
curv_array = np.array(curv_img)

print(f"Shape: {curv_array.shape}")
print(f"Min: {curv_array.min()}")
print(f"Max: {curv_array.max()}")
print(f"Mean: {curv_array.mean():.2f}")

# Count pixel value distribution
unique, counts = np.unique(curv_array, return_counts=True)
print(f"\nValue distribution (showing non-128 values):")
for val, count in zip(unique, counts):
    if val != 128:  # 128 is the "gray" middle value
        print(f"  Value {val}: {count} pixels")

# Check the output image
print("\n=== Output Image ===")
output_img = Image.open("/Users/nathanhenderson/Desktop/test_curvature_output.png")
output_array = np.array(output_img)

print(f"Shape: {output_array.shape}")
print(f"R channel - Min: {output_array[:,:,0].min()}, Max: {output_array[:,:,0].max()}, Mean: {output_array[:,:,0].mean():.2f}")
print(f"G channel - Min: {output_array[:,:,1].min()}, Max: {output_array[:,:,1].max()}, Mean: {output_array[:,:,1].mean():.2f}")
print(f"B channel - Min: {output_array[:,:,2].min()}, Max: {output_array[:,:,2].max()}, Mean: {output_array[:,:,2].mean():.2f}")

# Check if there's any variation
print(f"\nAny pixel different from (180, 50, 50)? {np.any((output_array[:,:,0] != 180) | (output_array[:,:,1] != 50) | (output_array[:,:,2] != 50))}")
