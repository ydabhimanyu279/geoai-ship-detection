import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
from pathlib import Path
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ShipDataset(Dataset):
    """Dataset for ship detection"""
    
    def __init__(self, image_dir, annotation_file, transform=None, img_size=80):
        self.image_dir = Path(image_dir)
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform
        self.img_size = img_size
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # Get annotation
        row = self.annotations.iloc[idx]
        
        # Load image
        img_path = self.image_dir / row['image_name']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get labels
        class_label = int(row['class_id'])
        
        # Get bounding box (normalized to [0, 1])
        if row['has_ship'] == 1:
            bbox = np.array([
                row['xmin'] / self.img_size,
                row['ymin'] / self.img_size,
                row['xmax'] / self.img_size,
                row['ymax'] / self.img_size
            ], dtype=np.float32)
        else:
            bbox = np.array([0, 0, 0, 0], dtype=np.float32)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Get location data for later use
        lat = row['lat']
        lon = row['lon']
        
        return {
            'image': image,
            'class_label': torch.tensor(class_label, dtype=torch.long),
            'bbox': torch.tensor(bbox, dtype=torch.float32),
            'lat': lat,
            'lon': lon,
            'image_name': row['image_name']
        }


def get_train_transforms(img_size=80):
    """Training augmentations"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.1),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_val_transforms(img_size=80):
    """Validation transforms (no augmentation)"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])