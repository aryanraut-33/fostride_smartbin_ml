# Fostride Smartbin ML

Welcome to the Fostride Smartbin ML project! This repository contains the machine learning models and data pipelines for the Fostride Smartbin waste classification system. It leverages computer vision techniques to identify and sort waste items efficiently.

---

## 🌎 For General Audience

**What is this project?**
Fostride Smartbin ML uses object detection and image segmentation to automatically classify waste types (like recyclables, compost, and trash). It processes images taken from smart conveyor bins to improve recycling efficiency and reduce manual sorting errors.

**Key Features:**
- Custom waste classification models (COCO segmentation format).
- End-to-end data pipelines integrated with Roboflow.
- Automated dataset preparation and processing.

---

## 🛠 For Collaborators

If you are contributing to this project, follow these guidelines to set up your environment while keeping the repository clean.

### 1. Environment Setup
Clone this repository. We rely on environment variables to manage access to external services (like Roboflow for datasets).

1. Create a `.env` file in the **root** of the repository (`fostride_smartbin_ml/.env`).
2. Add your Roboflow API key:
   ```env
   ROBOFLOW_API_KEY=your_key_here
   ```

### 2. Downloading the Dataset
**Important:** Do NOT commit dataset images or annotations to the repository. The `.gitignore` is pre-configured to ignore dataset contents while preserving the folder layout.

To fetch the raw images and COCO annotations:
```bash
# Navigate to the project module
cd smart_bin_aryan

# Run the dataset download script
python -m src.data.download_raw
```
*(Tip: To quickly test the pipeline with fewer images, use `python -m src.data.download_raw --limit 10`)*

The files will automatically be placed into `smart_bin_aryan/data/raw/`.

### 3. Git Guidelines
- The `smart_bin_aryan/data/` directories contain `.gitkeep` files to maintain the folder structure for all collaborators.
- **Never force-add** dataset images, `.zip` files, or intermediate processed data. The remote repository must always remain clean of large data files.
- Ensure your `.env` file is never committed.

### 4. Training the Baseline Model
We use a PyTorch **Mask R-CNN** implementation for instance segmentation to identify the exact pixel boundaries of waste items.

**Requirements:**
Make sure you have the necessary libraries installed in your Python environment:
```bash
pip install torch torchvision numpy pycocotools Pillow
```

**Running Training:**
To start a small-scale test run of the baseline model, execute:
```bash
cd smart_bin_aryan
python -m models.aryan.m1_baseline
```
The script will automatically initialize the custom COCO dataloaders, build the ResNet50-FPN model, run through the training engine, and save the resultant weights to `models/best_m1_baseline.pth`.
