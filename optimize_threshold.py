import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('.')
from models.ship_detector import create_model
from utils.dataset import get_val_transforms
import cv2

def get_predictions_with_probabilities(model, annotations_csv, images_dir, device):
    """Get all predictions with their probabilities"""
    
    df = pd.read_csv(annotations_csv)
    transform = get_val_transforms(img_size=80)
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        img_path = Path(images_dir) / row['image_name']
        image = cv2.imread(str(img_path))
        
        if image is None:
            continue
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image_rgb)
        image_tensor = transformed['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            class_preds, _ = model(image_tensor)
            probs = torch.softmax(class_preds, dim=1)
            ship_prob = probs[0, 1].item()
        
        results.append({
            'image_name': row['image_name'],
            'ship_probability': ship_prob,
            'true_label': row['has_ship']
        })
    
    return pd.DataFrame(results)

def calculate_metrics_at_threshold(predictions_df, threshold):
    """Calculate precision, recall, f1, accuracy at given threshold"""
    
    predicted = (predictions_df['ship_probability'] >= threshold).astype(int)
    true_labels = predictions_df['true_label']
    
    tp = ((predicted == 1) & (true_labels == 1)).sum()
    fp = ((predicted == 1) & (true_labels == 0)).sum()
    tn = ((predicted == 0) & (true_labels == 0)).sum()
    fn = ((predicted == 0) & (true_labels == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    model = create_model(num_classes=2, pretrained=False)
    checkpoint = torch.load('models/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Get predictions on validation set
    print("Getting predictions on validation set...")
    predictions_df = get_predictions_with_probabilities(
        model, 
        'data/processed/val/annotations.csv',
        'data/processed/val/images',
        device
    )
    
    # Test multiple thresholds
    print("\nTesting different confidence thresholds...\n")
    thresholds = np.arange(0.1, 1.0, 0.05)
    results = []
    
    for threshold in thresholds:
        metrics = calculate_metrics_at_threshold(predictions_df, threshold)
        results.append(metrics)
        
    results_df = pd.DataFrame(results)
    
    # Find optimal threshold (maximize F1 score)
    best_f1_idx = results_df['f1_score'].idxmax()
    best_f1_threshold = results_df.loc[best_f1_idx]
    
    # Find optimal threshold (maximize accuracy)
    best_acc_idx = results_df['accuracy'].idxmax()
    best_acc_threshold = results_df.loc[best_acc_idx]
    
    print("OPTIMAL THRESHOLDS")
    
    
    print(f"\nBest F1-Score: {best_f1_threshold['f1_score']:.4f} at threshold {best_f1_threshold['threshold']:.2f}")
    print(f"   Accuracy:  {best_f1_threshold['accuracy']:.4f}")
    print(f"   Precision: {best_f1_threshold['precision']:.4f}")
    print(f"   Recall:    {best_f1_threshold['recall']:.4f}")
    
    print(f"\nBest Accuracy: {best_acc_threshold['accuracy']:.4f} at threshold {best_acc_threshold['threshold']:.2f}")
    print(f"   F1-Score:  {best_acc_threshold['f1_score']:.4f}")
    print(f"   Precision: {best_acc_threshold['precision']:.4f}")
    print(f"   Recall:    {best_acc_threshold['recall']:.4f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: All metrics vs threshold
    ax = axes[0, 0]
    ax.plot(results_df['threshold'], results_df['accuracy'], label='Accuracy', marker='o')
    ax.plot(results_df['threshold'], results_df['precision'], label='Precision', marker='s')
    ax.plot(results_df['threshold'], results_df['recall'], label='Recall', marker='^')
    ax.plot(results_df['threshold'], results_df['f1_score'], label='F1-Score', marker='d')
    ax.axvline(best_f1_threshold['threshold'], color='red', linestyle='--', alpha=0.5, label='Best F1')
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Metrics vs Confidence Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Precision-Recall curve
    ax = axes[0, 1]
    ax.plot(results_df['recall'], results_df['precision'], marker='o')
    ax.scatter(best_f1_threshold['recall'], best_f1_threshold['precision'], 
               color='red', s=100, zorder=5, label=f'Best F1 (th={best_f1_threshold["threshold"]:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Confusion matrix at best F1
    ax = axes[1, 0]
    cm = [[best_f1_threshold['tn'], best_f1_threshold['fp']],
          [best_f1_threshold['fn'], best_f1_threshold['tp']]]
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Ship', 'Ship'])
    ax.set_yticklabels(['No Ship', 'Ship'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix (threshold={best_f1_threshold["threshold"]:.2f})')
    
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, int(cm[i][j]), ha="center", va="center", color="black", fontsize=20)
    
    # Plot 4: Summary table
    ax = axes[1, 1]
    ax.axis('off')
    
    table_data = [
        ['Metric', 'Best F1', 'Best Acc'],
        ['Threshold', f"{best_f1_threshold['threshold']:.3f}", f"{best_acc_threshold['threshold']:.3f}"],
        ['Accuracy', f"{best_f1_threshold['accuracy']:.3f}", f"{best_acc_threshold['accuracy']:.3f}"],
        ['Precision', f"{best_f1_threshold['precision']:.3f}", f"{best_acc_threshold['precision']:.3f}"],
        ['Recall', f"{best_f1_threshold['recall']:.3f}", f"{best_acc_threshold['recall']:.3f}"],
        ['F1-Score', f"{best_f1_threshold['f1_score']:.3f}", f"{best_acc_threshold['f1_score']:.3f}"],
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig('outputs/visualizations/threshold_optimization.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization to: outputs/visualizations/threshold_optimization.png")
    
    # Save results
    results_df.to_csv('outputs/results/threshold_analysis.csv', index=False)
    print("Saved detailed results to: outputs/results/threshold_analysis.csv")
    
    print(f"\nRECOMMENDATION: Use threshold = {best_f1_threshold['threshold']:.2f} for balanced performance")

if __name__ == '__main__':
    main()