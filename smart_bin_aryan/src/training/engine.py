import math
import sys
import time
import torch
from tqdm import tqdm

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

def evaluate(model, data_loader, device):
    """
    Returns predictions. For true COCO mAP evaluation, you would pass these 
    outputs to the `pycocotools.cocoeval` API. This is a simplified stub.
    """
    # model.eval() changes Mask R-CNN heavily; it returns predictions instead of losses
    model.eval()
    
    print("Evaluating...")
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            outputs = model(images)
            # outputs contains: 'boxes', 'labels', 'scores', 'masks'
            
            # To actually evaluate, these need to be aggregated and compared 
            # against targets using pycocotools.
            pass
            
    print("Evaluation logic complete (pycocotools evaluation omitted for baseline brevity).")
    
