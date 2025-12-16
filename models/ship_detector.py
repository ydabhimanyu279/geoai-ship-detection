import torch
import torch.nn as nn
import torchvision.models as models

class ShipDetector(nn.Module):
    """
    Ship Detection Model
    - ResNet50 backbone for feature extraction
    - Classification head (ship vs no-ship)
    - Bounding box regression head
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        super(ShipDetector, self).__init__()
        
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Use ResNet as feature extractor (remove final FC layer)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Adaptive pooling to get fixed size features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature dimension from ResNet50
        feature_dim = 2048
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Bounding box regression head (outputs 4 coords: xmin, ymin, xmax, ymax)
        self.bbox_regressor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 4),
            nn.Sigmoid()  # Normalize to [0, 1] range
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, 3, H, W) input images
        
        Returns:
            class_logits: (batch_size, num_classes)
            bbox_preds: (batch_size, 4) normalized coordinates
        """
        # Extract features
        features = self.backbone(x)  # (batch_size, 2048, H', W')
        
        # Global pooling
        pooled = self.adaptive_pool(features)  # (batch_size, 2048, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (batch_size, 2048)
        
        # Classification
        class_logits = self.classifier(pooled)
        
        # Bbox regression
        bbox_preds = self.bbox_regressor(pooled)
        
        return class_logits, bbox_preds


def create_model(num_classes=2, pretrained=True):
    """Factory function to create model"""
    model = ShipDetector(num_classes=num_classes, pretrained=pretrained)
    return model


if __name__ == '__main__':
    # Test model
    model = create_model()
    x = torch.randn(2, 3, 80, 80)
    class_out, bbox_out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Class output shape: {class_out.shape}")
    print(f"Bbox output shape: {bbox_out.shape}")