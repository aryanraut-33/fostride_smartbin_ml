import math
import sys
import time
import torch
import torchvision
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sklearn.metrics import confusion_matrix

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """
    Standard PyTorch training loop for Mask R-CNN.
    Iterates over the data_loader, computes losses (cls, box, mask, objectness),
    and updates the model weights.
    """
    model.train()
    
    # Simple metric tracking
    total_loss_epoch = 0.0
    num_batches = len(data_loader)
    
    print(f"--- Epoch {epoch} Start ---")
    start_time = time.time()
    
    # Wrap data_loader with tqdm for progress bar
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}", total=num_batches)
    
    for i, (images, targets) in enumerate(pbar):
        # Move to GPU/MPS/CPU
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass: Mask R-CNN returns a dictionary of losses during training
        loss_dict = model(images, targets)
        
        # Total loss is the sum of bounding box, mask, classification, and RPN losses
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        
        total_loss_epoch += loss_value

        if not math.isfinite(loss_value):
            print(f"\nLoss is {loss_value}, stopping training")
            print(loss_dict)
            sys.exit(1)

        # Backward and optimize
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Update progress bar with metrics
        pbar.set_postfix({
            'loss': f"{loss_value:.4f}",
            'cls': f"{loss_dict.get('loss_classifier', torch.tensor(0.0)).item():.4f}", 
            'box': f"{loss_dict.get('loss_box_reg', torch.tensor(0.0)).item():.4f}",
            'mask': f"{loss_dict.get('loss_mask', torch.tensor(0.0)).item():.4f}"
        })
                  
    epoch_time = time.time() - start_time
    avg_loss = total_loss_epoch / num_batches
    print(f"--- Epoch {epoch} End | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.0f}s ---")
    
    return avg_loss

def extract_confusion_matrix_data(preds, targets, iou_threshold=0.5, score_threshold=0.5):
    y_true = []
    y_pred = []
    
    for pred, target in zip(preds, targets):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        pred_labels = pred['labels']
        
        gt_boxes = target['boxes']
        gt_labels = target['labels']
        
        # Filter preds by score
        keep = pred_scores >= score_threshold
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            continue
            
        if len(pred_boxes) == 0:
            for gl in gt_labels:
                y_true.append(gl.item())
                y_pred.append(0)  # 0 is background
            continue
            
        if len(gt_boxes) == 0:
            for pl in pred_labels:
                y_true.append(0)
                y_pred.append(pl.item())
            continue
            
        # Compute IoU matrix
        ious = torchvision.ops.box_iou(gt_boxes, pred_boxes)
        
        # For each GT box, find the best matching prediction
        matched_preds = set()
        for i, gt_l in enumerate(gt_labels):
            best_iou, best_pred_idx = ious[i].max(dim=0)
            if best_iou >= iou_threshold:
                y_true.append(gt_l.item())
                y_pred.append(pred_labels[best_pred_idx].item())
                matched_preds.add(best_pred_idx.item())
            else:
                y_true.append(gt_l.item())
                y_pred.append(0) # Missed / FN
                
        # Any prediction not matched is a false positive
        for j, pl in enumerate(pred_labels):
            if j not in matched_preds:
                y_true.append(0)
                y_pred.append(pl.item())
                
    return y_true, y_pred

def evaluate(model, data_loader, device):
    """
    Evaluates the model using Mean Average Precision (mAP) and computes
    data for a confusion matrix.
    """
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")
    
    all_y_true = []
    all_y_pred = []
    
    print("Evaluating...")
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            outputs = model(images)
            
            preds = []
            for output in outputs:
                preds.append({
                    "boxes": output["boxes"].cpu(),
                    "scores": output["scores"].cpu(),
                    "labels": output["labels"].cpu(),
                })
            
            target_list = []
            for target in targets:
                target_list.append({
                    "boxes": target["boxes"].cpu(),
                    "labels": target["labels"].cpu(),
                })
                
            metric.update(preds, target_list)
            
            # Extract confusion matrix data
            y_true, y_pred = extract_confusion_matrix_data(preds, target_list)
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            
    results = metric.compute()
    
    # Compute confusion matrix
    # Classes are 0=Background, 1=Metal, 2=Paper, 3=Plastic
    cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1, 2, 3])
    
    map_dict = {
        "map": results["map"].item(),
        "map_50": results["map_50"].item(),
        "map_75": results["map_75"].item()
    }
    
    print(f"Evaluation Results -> mAP (IoU=0.50:0.95): {map_dict['map']:.4f}, mAP@50: {map_dict['map_50']:.4f}")
    
    return map_dict, cm
    
