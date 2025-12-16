import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append('.')
from models.ship_detector import create_model
from utils.dataset import get_val_transforms
from utils.geo_utils import detections_to_geodataframe, export_to_shapefile, export_to_geojson
import matplotlib.pyplot as plt

def load_model(model_path, device):
    """Load trained model"""
    print(f"📦 Loading model from {model_path}")
    
    model = create_model(num_classes=2, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"  ✓ Model loaded (epoch {checkpoint['epoch']}, val_acc: {checkpoint['val_acc']:.2f}%)")
    return model

def predict_on_dataset(model, annotations_csv, images_dir, device, confidence_threshold=0.5):
    """Run predictions on dataset"""
    
    # Load annotations (contains lat/lon)
    df = pd.read_csv(annotations_csv)
    print(f"\n🔍 Running predictions on {len(df)} images...")
    
    transform = get_val_transforms(img_size=80)
    
    detections = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Load image
        img_path = Path(images_dir) / row['image_name']
        image = cv2.imread(str(img_path))
        
        if image is None:
            continue
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        transformed = transform(image=image_rgb)
        image_tensor = transformed['image'].unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            class_preds, bbox_preds = model(image_tensor)
            
            # Get class probabilities
            probs = torch.softmax(class_preds, dim=1)
            ship_prob = probs[0, 1].item()  # Probability of ship class
            predicted_class = torch.argmax(probs, dim=1).item()
            
            # Get bounding box (denormalize from [0,1] to pixel coordinates)
            bbox = bbox_preds[0].cpu().numpy() * 80  # Scale to 80x80
        
        # Only save detections above confidence threshold
        if predicted_class == 1 and ship_prob >= confidence_threshold:
            detections.append({
                'image_name': row['image_name'],
                'class': 'ship',
                'class_id': 1,
                'confidence': ship_prob,
                'bbox': bbox.tolist(),
                'lat': row['lat'],
                'lon': row['lon'],
                'true_label': row['has_ship']
            })
    
    return detections, df

def visualize_predictions(detections, images_dir, output_path, num_samples=16):
    """Visualize sample predictions"""
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Sample Ship Detections with Bounding Boxes', fontsize=16)
    
    sample_detections = detections[:num_samples] if len(detections) >= num_samples else detections
    
    for idx, (ax, det) in enumerate(zip(axes.flat, sample_detections)):
        # Load image
        img_path = Path(images_dir) / det['image_name']
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw bounding box
        bbox = det['bbox']
        xmin, ymin, xmax, ymax = map(int, bbox)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # Add confidence text
        conf_text = f"{det['confidence']:.2f}"
        cv2.putText(img, conf_text, (xmin, ymin-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"({det['lat']:.2f}, {det['lon']:.2f})", fontsize=8)
    
    # Hide empty subplots
    for idx in range(len(sample_detections), len(axes.flat)):
        axes.flat[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved visualization to {output_path}")

def calculate_metrics(detections, df):
    """Calculate detection metrics"""
    
    # Get predictions
    predicted_ships = set([d['image_name'] for d in detections])
    
    # Get ground truth
    true_ships = set(df[df['has_ship'] == 1]['image_name'].tolist())
    all_images = set(df['image_name'].tolist())
    true_no_ships = all_images - true_ships
    
    # Calculate metrics
    true_positives = len(predicted_ships & true_ships)
    false_positives = len(predicted_ships & true_no_ships)
    false_negatives = len(true_ships - predicted_ships)
    true_negatives = len(true_no_ships - predicted_ships)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def main():
    # Config
    MODEL_PATH = 'models/best_model.pth'
    TEST_ANNOTATIONS = 'data/processed/test/annotations.csv'
    TEST_IMAGES = 'data/processed/test/images'
    CONFIDENCE_THRESHOLD = 0.45
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load model
    model = load_model(MODEL_PATH, device)
    
    # Run predictions
    detections, df = predict_on_dataset(
        model, TEST_ANNOTATIONS, TEST_IMAGES, 
        device, CONFIDENCE_THRESHOLD
    )
    
    print(f"\n📊 Detection Results:")
    print(f"  Total test images: {len(df)}")
    print(f"  Ships detected: {len(detections)}")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
    
    # Calculate metrics
    metrics = calculate_metrics(detections, df)
    print(f"\n📈 Performance Metrics:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1-Score: {metrics['f1_score']:.3f}")
    print(f"  True Positives: {metrics['true_positives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    
    if len(detections) == 0:
        print("\n⚠️  No ships detected! Try lowering confidence threshold.")
        return
    
    # Visualize predictions
    print("\n🖼️  Creating visualizations...")
    visualize_predictions(
        detections, TEST_IMAGES, 
        'outputs/visualizations/test_predictions.png'
    )
    
    # Convert to GeoDataFrame (THE KEY PART FOR ESRI!)
    print("\n🌍 Converting to geospatial format...")
    gdf = detections_to_geodataframe(detections)
    
    print(f"\n✓ Created GeoDataFrame with {len(gdf)} ship detections")
    print(f"\nSample detections:")
    print(gdf[['class', 'confidence', 'lat', 'lon']].head())
    
    # Export to GIS formats
    print("\n💾 Exporting to GIS formats...")
    export_to_shapefile(gdf, 'outputs/shapefiles/ship_detections.shp')
    export_to_geojson(gdf, 'outputs/shapefiles/ship_detections.geojson')
    
    # Create interactive map
    print("\n🗺️  Creating interactive map...")
    from utils.visualization import create_interactive_map
    create_interactive_map(gdf, 'outputs/visualizations/ship_detections_map.html')
    
    # Save detection results as CSV
    detection_df = pd.DataFrame(detections)
    detection_df.to_csv('outputs/results/detections.csv', index=False)
    print(f"  ✓ Saved CSV to outputs/results/detections.csv")
    
    print("\n✅ Prediction pipeline complete!")
    print("\n📂 Output files:")
    print("  - outputs/shapefiles/ship_detections.shp (Shapefile for ArcGIS)")
    print("  - outputs/shapefiles/ship_detections.geojson (GeoJSON)")
    print("  - outputs/visualizations/ship_detections_map.html (Interactive map)")
    print("  - outputs/visualizations/test_predictions.png (Sample predictions)")
    print("  - outputs/results/detections.csv (All detections)")

if __name__ == '__main__':
    main()