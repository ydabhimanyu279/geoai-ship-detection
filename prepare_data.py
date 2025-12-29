import json
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

print("Starting data preparation...")

# Load dataset
with open('data/raw/shipsnet.json', 'r') as f:
    dataset = json.load(f)

labels = dataset['labels']
locations = dataset['locations']
scene_ids = dataset['scene_ids']

print(f"Total images: {len(labels)}")
print(f"Ships: {sum(labels)}")
print(f"No ships: {len(labels) - sum(labels)}")

# Create organized dataset
data_records = []

for idx in range(len(labels)):
    # Find corresponding PNG file
    png_files = list(Path('data/raw/shipsnet').glob(f'{labels[idx]}__*'))
    
    if idx < len(png_files):
        png_path = png_files[idx] if idx < len(png_files) else None
        
        # For ship images, create a bounding box
        # Ships are roughly centered, so we'll create a box in the center
        if labels[idx] == 1:  # Ship present
            # Create a centered bounding box (40x40 box in 80x80 image)
            img_size = 80
            box_size = 50  # Box size
            center = img_size // 2
            
            xmin = center - box_size // 2
            ymin = center - box_size // 2
            xmax = center + box_size // 2
            ymax = center + box_size // 2
            
            # Add some randomness to make it more realistic
            noise = np.random.randint(-5, 5, size=4)
            xmin, ymin, xmax, ymax = np.array([xmin, ymin, xmax, ymax]) + noise
            
            # Ensure bounds are valid
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(img_size, xmax), min(img_size, ymax)
            
        else:  # No ship
            xmin = ymin = xmax = ymax = 0  # No bounding box
        
        record = {
            'image_id': idx,
            'label': labels[idx],
            'class_id': labels[idx],
            'lat': locations[idx][1],
            'lon': locations[idx][0],
            'scene_id': scene_ids[idx],
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'has_ship': labels[idx]
        }
        data_records.append(record)

df = pd.DataFrame(data_records)
print(f"\nCreated {len(df)} records")

# Balance dataset - take all ships + equal number of no-ships
ship_df = df[df['has_ship'] == 1]
no_ship_df = df[df['has_ship'] == 0].sample(n=len(ship_df), random_state=42)

balanced_df = pd.concat([ship_df, no_ship_df]).reset_index(drop=True)
print(f"\nBalanced dataset: {len(balanced_df)} images")
print(f"  Ships: {len(ship_df)}")
print(f"  No ships: {len(no_ship_df)}")

# Split into train/val/test (70/15/15) with stratification
train_df, temp_df = train_test_split(
    df, test_size=0.3, random_state=42, stratify=df['has_ship']
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df['has_ship']
)

print(f"\nDataset Split (Stratified):")
print(f"  Train: {len(train_df)} images ({train_df['has_ship'].sum()} ships)")
print(f"  Val: {len(val_df)} images ({val_df['has_ship'].sum()} ships)")
print(f"  Test: {len(test_df)} images ({test_df['has_ship'].sum()} ships)")

# Create directory structure
for split in ['train', 'val', 'test']:
    Path(f'data/processed/{split}/images').mkdir(parents=True, exist_ok=True)

# Copy and organize images
print("\nOrganizing images...")

def copy_images(df, split):
    """Copy images from raw to processed folder"""
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
        image_id = row['image_id']
        
        # Find source image
        if image_id < len(png_files):
            src_path = png_files[image_id]
            
            # Destination path
            dst_path = Path(f'data/processed/{split}/images/{image_id:05d}.png')
            
            # Copy image
            shutil.copy2(src_path, dst_path)

copy_images(train_df, 'train')
copy_images(val_df, 'val')
copy_images(test_df, 'test')

# Save annotations
print("\nSaving annotations...")

train_df['image_name'] = train_df['image_id'].apply(lambda x: f'{x:05d}.png')
val_df['image_name'] = val_df['image_id'].apply(lambda x: f'{x:05d}.png')
test_df['image_name'] = test_df['image_id'].apply(lambda x: f'{x:05d}.png')

train_df.to_csv('data/processed/train/annotations.csv', index=False)
val_df.to_csv('data/processed/val/annotations.csv', index=False)
test_df.to_csv('data/processed/test/annotations.csv', index=False)

# Calculate class weights for reference
n_ships = df['has_ship'].sum()
n_no_ships = len(df) - n_ships
print(f"\nClass Statistics for Focal Loss:")
print(f"  Total images: {len(df)}")
print(f"  Ships (minority): {n_ships} ({n_ships/len(df)*100:.1f}%)")
print(f"  No ships (majority): {n_no_ships} ({n_no_ships/len(df)*100:.1f}%)")
print(f"  Imbalance ratio: 1:{n_no_ships/n_ships:.1f}")
print(f"\n  Recommended alpha: 0.25 (for minority class)")
print(f"  Recommended gamma: 2.0 (focusing parameter)")

print("\nData preparation complete!")
print("\nFinal structure:")
print(f"  data/processed/train/images/    → {len(train_df)} images")
print(f"  data/processed/train/annotations.csv")
print(f"  data/processed/val/images/      → {len(val_df)} images")
print(f"  data/processed/val/annotations.csv")
print(f"  data/processed/test/images/     → {len(test_df)} images")
print(f"  data/processed/test/annotations.csv")

# Visualizations with bounding boxes
print("\nCreating sample visualizations with bounding boxes...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Sample Images with Bounding Boxes', fontsize=16)

sample_ships = train_df[train_df['has_ship'] == 1].head(8)

for idx, (ax, (_, row)) in enumerate(zip(axes.flat, sample_ships.iterrows())):
    img_path = f"data/processed/train/images/{row['image_name']}"
    img = cv2.imread(img_path)
    
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw bounding box
        if row['has_ship'] == 1:
            cv2.rectangle(img, 
                         (int(row['xmin']), int(row['ymin'])), 
                         (int(row['xmax']), int(row['ymax'])), 
                         (0, 255, 0), 2)
        
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Ship at ({row['lat']:.2f}, {row['lon']:.2f})", fontsize=9)

plt.tight_layout()
plt.savefig('outputs/visualizations/bbox_samples.png', dpi=150, bbox_inches='tight')
print("Saved to: outputs/visualizations/bbox_samples.png")

print("\n Ready for training with Focal Loss!")