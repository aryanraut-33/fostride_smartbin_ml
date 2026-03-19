from src.data.coco_dataset import CocoSegmentationDataset

# Verify the dataset only loads 3 classes (metal, paper, plastic)
dataset = CocoSegmentationDataset(root_dir="data/raw/train")

print(f"Total images in dataset: {len(dataset)}")
print(f"Categories: {dataset.categories}")

# Quick check: load the first image and verify labels
img, target = dataset[0]
print(f"\nFirst image - labels: {target['labels'].tolist()}")
print(f"All labels are in [1, 2, 3]: {all(l in {1, 2, 3} for l in target['labels'].tolist())}")
