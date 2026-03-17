"""
Download raw images + COCO segmentation annotations from Roboflow.
Uses the Roboflow Python SDK to download the dataset.

Usage:
    python -m src.data.download_raw
    python -m src.data.download_raw --limit 10   # download only 10 images (for testing)
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow


# Load API key from the repo root .env file
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # smart_bin_aryan/
REPO_ROOT = PROJECT_ROOT.parent                      # fostride_smartbin_ml/
load_dotenv(REPO_ROOT / ".env")

# Roboflow config
WORKSPACE_ID = "fostride"
PROJECT_ID = "autocrop-conveyer-images"
DATASET_VERSION = 1
EXPORT_FORMAT = "coco-segmentation"

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_dataset(limit: int | None = None):
    """Download dataset from Roboflow with COCO segmentation annotations."""

    # We use the ROBOFLOW_API_KEY environment variable if it exists, otherwise fallback
    api_key = os.getenv("ROBOFLOW_API_KEY") or os.getenv("YOUR_ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError(
            "API key not found in .env file. "
            "Add it as: ROBOFLOW_API_KEY=your_key_here"
        )

    # Connect to Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)

    print(f"Connected to: {WORKSPACE_ID}/{PROJECT_ID}")
    print(f"Downloading version {DATASET_VERSION} in '{EXPORT_FORMAT}' format...")
    print(f"Target directory: {RAW_DATA_DIR}")
    print("This may take a few minutes for large datasets...\n")

    # Download images + COCO segmentation annotations to a temp directory
    version = project.version(DATASET_VERSION)
    dataset = version.download(
        model_format=EXPORT_FORMAT,
        location=str(RAW_DATA_DIR),
        overwrite=True,
    )

    extracted_dir = Path(dataset.location)
    
    # Move files from the Roboflow nested folder (e.g. AutoCrop-Conveyer-Images-1)
    # up to data/raw/
    if extracted_dir.name != RAW_DATA_DIR.name:
        print(f"Moving dataset out of {extracted_dir.name} to {RAW_DATA_DIR.name}...")
        for split in ["train", "valid", "test"]:
            src_split = extracted_dir / split
            if src_split.exists():
                dst_split = RAW_DATA_DIR / split
                # Remove if exists to avoid conflicts
                if dst_split.exists():
                    shutil.rmtree(dst_split)
                shutil.move(str(src_split), str(dst_split))
        
        # Cleanup the nested folder and zip
        shutil.rmtree(extracted_dir, ignore_errors=True)
        zip_file = RAW_DATA_DIR / "roboflow.zip"
        if zip_file.exists():
            zip_file.unlink()

    print(f"\nDataset is ready in: {RAW_DATA_DIR}")

    # Show what was downloaded, applying limits if requested
    for split in ["train", "valid", "test"]:
        split_dir = RAW_DATA_DIR / split
        if split_dir.exists():
            images = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
            annotations = list(split_dir.glob("*.json"))
            
            if limit is not None:
                if limit < len(images):
                    print(f"Trimming {split} split down to {limit} images...")
                    # Keep only limit images
                    images_to_remove = random.sample(images, len(images) - limit)
                    for img in images_to_remove:
                        img.unlink()
                    # Update images count
                    images = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))

            print(f"  {split}/: {len(images)} images, {len(annotations)} annotation file(s)")

    print("\nDone! ✓")


def cleanup_dataset():
    """Delete all downloaded data in data/raw/ after user confirmation."""
    if not RAW_DATA_DIR.exists():
        print(f"Nothing to delete — {RAW_DATA_DIR} does not exist.")
        return

    # Show what will be deleted
    splits = [d for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
    total_files = sum(len(list(s.rglob("*"))) for s in splits)
    print(f"\n⚠️  The following will be permanently deleted:")
    for s in sorted(splits):
        count = len(list(s.rglob("*")))
        print(f"  {s.name}/  ({count} files)")
    print(f"\n  Total: {total_files} files in {RAW_DATA_DIR}\n")

    confirm = input("Are you sure you want to delete? (yes/no): ").strip().lower()
    if confirm in ("yes", "y"):
        for s in splits:
            shutil.rmtree(s)
        # Also remove any leftover files (zips, etc.) but keep the directory itself
        for f in RAW_DATA_DIR.iterdir():
            if f.is_file():
                f.unlink()
        print("✓ Dataset deleted successfully.")
    else:
        print("Aborted — nothing was deleted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download raw images + COCO annotations from Roboflow"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Keep only N images per split for testing (default: all)"
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Delete all downloaded data in data/raw/ (with confirmation)"
    )
    args = parser.parse_args()

    if args.clean:
        cleanup_dataset()
    else:
        download_dataset(limit=args.limit)
