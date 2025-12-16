import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) predicted logits
            targets: (N,) ground truth labels
        """
        # Cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probabilities
        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha if targets[0] == 1 else (1 - self.alpha)
            focal_weight = alpha_t * focal_weight
        
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class DetectionLoss(nn.Module):
    """Combined classification (Focal) + bbox regression loss"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(DetectionLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.bbox_loss = nn.SmoothL1Loss(reduction='mean')
    
    def forward(self, class_preds, bbox_preds, class_targets, bbox_targets):
        """
        Args:
            class_preds: (N, 2) classification logits
            bbox_preds: (N, 4) predicted boxes
            class_targets: (N,) class labels
            bbox_targets: (N, 4) ground truth boxes
        """
        # Classification loss
        cls_loss = self.focal_loss(class_preds, class_targets)
        
        # Bbox loss only for positive samples (ships)
        positive_mask = class_targets > 0
        
        if positive_mask.sum() > 0:
            bbox_loss = self.bbox_loss(
                bbox_preds[positive_mask],
                bbox_targets[positive_mask]
            )
        else:
            bbox_loss = torch.tensor(0.0, device=class_preds.device)
        
        # Total loss
        total_loss = cls_loss + bbox_loss
        
        return total_loss, cls_loss, bbox_loss