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

_(Tip: To quickly test the pipeline with fewer images, use `python -m src.data.download_raw --limit 10`)_

The files will automatically be placed into `smart_bin_aryan/data/raw/`.

### 3. Git Guidelines

- The `smart_bin_aryan/data/` directories contain `.gitkeep` files to maintain the folder structure for all collaborators.
- **Never force-add** dataset images, `.zip` files, or intermediate processed data. The remote repository must always remain clean of large data files.
- Ensure your `.env` file is never committed.

### 4. Environment Setup (Virtual Environment)

To avoid dependency conflicts, it is highly recommended to use a Python virtual environment (`venv`).

**For macOS / Linux:**
```bash
# Create a virtual environment named 'env'
python3 -m venv env

# Activate the virtual environment
source env/bin/activate

# Safely install all required libraries
pip install --upgrade pip
pip install -r requirements.txt
```

**For Windows:**
```powershell
# Create a virtual environment named 'env'
python -m venv env

# Activate the virtual environment
.\env\Scripts\activate

# Safely install all required libraries
python -m pip install --upgrade pip
pip install -r requirements.txt
```

*(Note: When you are done working, simply type `deactivate` in your terminal to exit the virtual environment.)*

### 5. Training the Baseline Model

We use a PyTorch **Mask R-CNN** implementation for instance segmentation to identify the exact pixel boundaries of waste items.

**Running Training:**
To start a small-scale test run of the baseline model, execute:

```bash
cd smart_bin_aryan
python -m models.aryan.m1_baseline
```

The script will automatically initialize the custom COCO dataloaders, build the ResNet50-FPN model, run through the training engine, and save the resultant weights to `models/best_m1_baseline.pth`.

### 6. Evaluating the Model

During training, the pipeline automatically evaluates the model at the end of every epoch.

**Evaluation Features:**
- **mAP Scores:** Computes Mean Average Precision (mAP, mAP@50, mAP@75) using `torchmetrics.detection.MeanAveragePrecision`.
- **Confusion Matrix:** Tracks True Positives, False Positives, and False Negatives via IoU matching (>0.5) and plots a 4x4 Confusion Matrix for the classes.
- **Loss Curve:** Plots the training curve across all epochs.

All plots are automatically generated, saved, and updated iteratively in the `smart_bin_aryan/models/aryan/metrics/` directory during execution.

---

## 🤝 For Collaborators: How to Contribute

Instead of modifying the `aryan` baseline model, you will create your own modular model pipeline inside your personal folder. 

### 1. Git Workflow (Branching)
Always work on your own branch. Do not push directly to `main`.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aryanraut-33/fostride_smartbin_ml.git
   cd fostride_smartbin_ml/smart_bin_aryan
   ```
2. **Create and checkout your own branch:**
   ```bash
   # Replace <your-name> with your actual name
   git checkout -b feature/<your-name>-model
   ```
3. **Commit and push your changes:**
   ```bash
   git add .
   git commit -m "Add <your-name> custom model architecture"
   git push origin feature/<your-name>-model
   ```

### 2. Creating Your Model Directory
Your models and scripts must live inside `smart_bin_aryan/models/<your-name>/` to prevent conflicts.

```
smart_bin_aryan/
├── models/
│   ├── aryan/          # Aryan's baseline models (DO NOT EDIT)
│   └── <your-name>/    # Create this folder for your models!
│       ├── my_custom_model.py
│       └── metrics/
```

### 3. IMPORTANT: Dataset Loading
**You must train exclusively on the official loaded dataset.** Do not download data manually.

We provide a custom PyTorch dataset loader that correctly handles the 3 waste classes (Metal, Paper, Plastic) plus the background, and correctly translates bounding boxes and segmentation polygons.

To load the data in your training script, simply import and use:
```python
from src.data.coco_dataset import CocoSegmentationDataset

# Ensure you use a custom collate_fn for detection models
def collate_fn(batch): return tuple(zip(*batch))
```
*(Reference `models/aryan/m1_baseline.py` for a complete example of setting up the DataLoader and Transforms).*

## 🚀 Running on Colab
If you are developing locally but want to train on Colab, you can run your modular script directly!

1. Open a new Google Colab notebook.
2. Ensure GPU is enabled (`Runtime` > `Change runtime type` > `GPU` or `T4 GPU`).
3. Run the following in cell blocks:

```python
# Clone and enter the directory
!git clone https://github.com/aryanraut-33/fostride_smartbin_ml.git
%cd fostride_smartbin_ml/smart_bin_aryan

# Install dependencies
!pip install torch torchvision matplotlib seaborn pycocotools

# Download dataset (Requires ROBOFLOW_API_KEY in colab secrets or pasted directly)
!python -m src.data.download_raw

# Train YOUR model
!python models/<your-name>/my_custom_model.py
```
