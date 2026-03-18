import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

# Use relative imports so it runs cleanly from the project root
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.coco_dataset import CocoSegmentationDataset
from src.models.mask_rcnn import get_mask_rcnn_model
from src.training.engine import train_one_epoch, evaluate


def collate_fn(batch):
    """
    Custom collate function for object detection datasets.
    Since images and targets (boxes/masks) have different sizes, 
    we need to return them exactly as tuples rather than attempting 
    to stack them into uniform PyTorch tensors.
    """
    return tuple(zip(*batch))


def main():
    print("Initializing M1 Baseline Training Pipeline...")
    
    # 1. Setup paths
    DATA_DIR = PROJECT_ROOT / "data" / "raw"
    TRAIN_DIR = DATA_DIR / "train"
    VALID_DIR = DATA_DIR / "valid"
    
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Training directory {TRAIN_DIR} not found. "
            "Please run `python -m src.data.download_raw` first."
        )

    # Device configuration
device = torch.device("mps") if torch.backends.mps.is_available() else \
         torch.device("cuda") if torch.cuda.is_available() else \
         (_ for _ in ()).throw(RuntimeError("No GPU backend (MPS/CUDA) available"))
             
    print(f"Using device: {device}")

    # 2. Setup DataLoaders
    # Transforms (e.g. converting PIL to Tensor) omitted for brevity in this baseline. 
    # In production, images must be converted to tensors (C, H, W) scaled [0, 1].
    # Here, we use a lambda just to convert PIL -> Tensor
    import torchvision.transforms.functional as F
    
    def get_transform():
        # A minimal transform function expected by CocoSegmentationDataset
        def transform(image, target):
            image = F.to_tensor(image)
            return image, target
        return transform

    print("Loading datasets...")
    dataset_train = CocoSegmentationDataset(TRAIN_DIR, transforms=get_transform())
    dataset_valid = CocoSegmentationDataset(VALID_DIR, transforms=get_transform())
    
    data_loader_train = DataLoader(
        dataset_train, 
        batch_size=2, 
        shuffle=True, 
        num_workers=0, # Set to >0 if not on Windows/Mac to speed up loading
        collate_fn=collate_fn
    )

    data_loader_valid = DataLoader(
        dataset_valid, 
        batch_size=2, 
        shuffle=False, 
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # 3. Initialize Model
    # 4 classes: 0=Background, 1=Metal, 2=Paper, 3=Plastic
    NUM_CLASSES = 4 
    print(f"Building Mask R-CNN model for {NUM_CLASSES} classes...")
    model = get_mask_rcnn_model(num_classes=NUM_CLASSES)
    model.to(device)

    # 4. Initialize Optimizer
    # Fast learning rate for Mask R-CNN baseline
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # Learning rate scheduler (reduces LR by 10x every 3 epochs)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 5. Training Loop
    NUM_EPOCHS = 10
    print(f"Starting training for {NUM_EPOCHS} epoch(s)...")
    
    # Initialize metric tracking and metrics directory
    metrics_dir = PROJECT_ROOT / "models" / "aryan" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    epoch_losses = []
    
    for epoch in range(NUM_EPOCHS):
        avg_loss = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        lr_scheduler.step()
        epoch_losses.append(avg_loss)
        
        # Plot Loss Curve
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(epoch_losses)), epoch_losses, marker='o', linestyle='-', color='b', label='Training Loss')
        plt.title('Training Loss vs Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(metrics_dir / "loss_curve.png")
        plt.close()
        
        # Evaluate model after epoch
        map_dict, cm = evaluate(model, data_loader_valid, device)
        
        # Plot Confusion Matrix
        plt.figure(figsize=(8, 6))
        class_names = ['Background', 'Metal', 'Paper', 'Plastic']
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(metrics_dir / f"epoch_{epoch}_confusion_matrix.png")
        plt.close()
        
    # 6. Save internal checkpoint
    save_path = PROJECT_ROOT / "models" / "best_m1_baseline.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining Complete. Model saved to {save_path}")


if __name__ == "__main__":
    main()
