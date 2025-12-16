import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# Load the JSON file with labels
print("Loading dataset metadata...")
with open('data/raw/shipsnet.json', 'r') as f:
    dataset = json.load(f)

print(f"Dataset keys: {dataset.keys()}")
print(f"\nTotal images: {len(dataset['data'])}")
print(f"Total labels: {len(dataset['labels'])}")

# Check class distribution
labels = dataset['labels']
label_counts = Counter(labels)
print(f"\nClass Distribution:")
print(f"  No Ship (0): {label_counts[0]} images")
print(f"  Ship (1): {label_counts[1]} images")

# Get image dimensions
print(f"\nImage Properties:")
# Each image is stored as flat array of RGB values
first_image_data = dataset['data'][0]
print(f"  Pixels per image: {len(first_image_data)}")
print(f"  Image size: 80x80 pixels (RGB)")
print(f"  Total values: {len(first_image_data)} = 80 x 80 x 3")

# Visualize sample images
print("\nCreating visualization...")

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle('Sample Images from Dataset', fontsize=16)

# Show 16 ships and 16 non-ships
ship_indices = [i for i, label in enumerate(labels) if label == 1][:16]
no_ship_indices = [i for i, label in enumerate(labels) if label == 0][:16]

for idx, (ax, img_idx) in enumerate(zip(axes.flat, ship_indices + no_ship_indices)):
    # Reshape flat array to 80x80x3 image
    img_data = np.array(dataset['data'][img_idx]).reshape(3, 80, 80)
    img_data = np.transpose(img_data, (1, 2, 0))  # Convert to HWC format
    
    ax.imshow(img_data.astype(np.uint8))
    ax.axis('off')
    
    if idx < 16:
        ax.set_title('Ship', color='green', fontsize=10)
    else:
        ax.set_title('No Ship', color='red', fontsize=10)

plt.tight_layout()
plt.savefig('outputs/visualizations/dataset_samples.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization to: outputs/visualizations/dataset_samples.png")

# Look at actual PNG files
print("\nChecking PNG files...")
png_files = list(Path('data/raw/shipsnet').glob('*.png'))
print(f"  Found {len(png_files)} PNG files")

# Load a sample PNG
sample_png = cv2.imread(str(png_files[0]))
sample_png = cv2.cvtColor(sample_png, cv2.COLOR_BGR2RGB)
print(f"  PNG shape: {sample_png.shape}")
print(f"  PNG dtype: {sample_png.dtype}")

# Show a few PNG samples
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Sample PNG Files', fontsize=16)

for idx, ax in enumerate(axes.flat):
    img = cv2.imread(str(png_files[idx]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.axis('off')
    
    # Filename format: label__date__coordinates.png
    filename = png_files[idx].stem
    label = filename.split('__')[0]
    ax.set_title(f"Label: {label}", fontsize=10)

plt.tight_layout()
plt.savefig('outputs/visualizations/png_samples.png', dpi=150, bbox_inches='tight')
print("Saved PNG samples to: outputs/visualizations/png_samples.png")

print("\nExploration complete!")
print("\nSummary:")
print(f"  - Dataset format: JSON with flat arrays + PNG files")
print(f"  - Image size: 80x80 pixels")
print(f"  - Classes: Binary (Ship=1, No Ship=0)")
print(f"  - Total images: {len(dataset['data'])}")
print(f"  - Balanced: {'Yes' if abs(label_counts[0] - label_counts[1]) < 100 else 'No'}")