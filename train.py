import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add paths
sys.path.append('.')
from models.ship_detector import create_model
from utils.dataset import ShipDataset, get_train_transforms, get_val_transforms
from utils.focal_loss import DetectionLoss

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_cls_loss = 0
    total_bbox_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        images = batch['image'].to(device)
        class_labels = batch['class_label'].to(device)
        bbox_targets = batch['bbox'].to(device)
        
        # Forward pass
        class_preds, bbox_preds = model(images)
        
        # Calculate loss
        loss, cls_loss, bbox_loss = criterion(
            class_preds, bbox_preds,
            class_labels, bbox_targets
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_bbox_loss += bbox_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cls': f'{cls_loss.item():.4f}',
            'bbox': f'{bbox_loss.item():.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_bbox_loss = total_bbox_loss / len(dataloader)
    
    return avg_loss, avg_cls_loss, avg_bbox_loss


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    
    total_loss = 0
    total_cls_loss = 0
    total_bbox_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch['image'].to(device)
            class_labels = batch['class_label'].to(device)
            bbox_targets = batch['bbox'].to(device)
            
            # Forward pass
            class_preds, bbox_preds = model(images)
            
            # Calculate loss
            loss, cls_loss, bbox_loss = criterion(
                class_preds, bbox_preds,
                class_labels, bbox_targets
            )
            
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_bbox_loss += bbox_loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(class_preds, 1)
            total += class_labels.size(0)
            correct += (predicted == class_labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_bbox_loss = total_bbox_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, avg_cls_loss, avg_bbox_loss, accuracy


def main():
    # Hyperparameters
    NUM_CLASSES = 2
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.0005
    IMG_SIZE = 80
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    
    # Device
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = ShipDataset(
        image_dir='data/processed/train/images',
        annotation_file='data/processed/train/annotations.csv',
        transform=get_train_transforms(IMG_SIZE)
    )
    
    val_dataset = ShipDataset(
        image_dir='data/processed/val/images',
        annotation_file='data/processed/val/annotations.csv',
        transform=get_val_transforms(IMG_SIZE)
    )
    
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model
    print("\n Building model...")
    model = create_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)
    
    # Loss function (Focal Loss)
    criterion = DetectionLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training loop
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print(f"   Focal Loss: alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA}\n")
    
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_cls, train_bbox = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_cls, val_bbox, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} (cls: {train_cls:.4f}, bbox: {train_bbox:.4f})")
        print(f"  Val Loss:   {val_loss:.4f} (cls: {val_cls:.4f}, bbox: {val_bbox:.4f})")
        print(f"  Val Accuracy: {val_acc:.2f}%")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, 'models/best_model.pth')
            print(f"Best model saved! (val_loss: {val_loss:.4f})")
    
    # Plot training history
    print("\nPlotting training history...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['val_acc'], label='Val Accuracy', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('outputs/visualizations/training_history.png', dpi=150)
    print("Saved to: outputs/visualizations/training_history.png")
    
    print("\nTraining complete!")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Model saved to: models/best_model.pth")


if __name__ == '__main__':
    main()