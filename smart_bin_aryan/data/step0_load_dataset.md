# Step 0: Load the Dataset for Training the Model

This guide walks you through downloading the **Fostride Waste Classification** dataset from [Roboflow](https://universe.roboflow.com/fostride/autocrop-conveyer-images) using the Roboflow Python SDK. The dataset contains images with **COCO segmentation annotations** split into `train/`, `valid/`, and `test/` sets.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| Poetry | latest |
| Roboflow API Key | [Get one here](https://app.roboflow.com/settings/api) |

---

## 1 — Clone the Repository

```bash
git clone https://github.com/aryanraut-33/fostride_smartbin_ml.git
cd fostride_smartbin_ml/smart_bin_aryan
```

## 2 — Install Dependencies

```bash
# Create and activate the Poetry environment
make create_environment

# Install all project dependencies (includes roboflow & python-dotenv)
make requirements
```

> [!NOTE]
> If you prefer not to use `make`, you can run `poetry install` directly.

## 3 — Configure Your Roboflow API Key

Create a `.env` file in the **repository root** (`fostride_smartbin_ml/.env`):

```bash
# fostride_smartbin_ml/.env
ROBOFLOW_API_KEY=your_api_key_here
```

> [!CAUTION]
> **Never commit your `.env` file.** It is already listed in `.gitignore`.

### Where to find your API key

1. Sign in to [Roboflow](https://app.roboflow.com/).
2. Go to **Settings → API Keys** (or visit [roboflow.com/settings/api](https://app.roboflow.com/settings/api)).
3. Copy the **Private API Key** and paste it into your `.env` file.

## 4 — Download the Dataset

Run the download script from the `smart_bin_aryan/` directory:

```bash
python -m src.data.download_raw
```

This connects to the Roboflow project **fostride / autocrop-conveyer-images (v1)** and downloads all images and COCO segmentation annotations into `data/raw/`.

### Optional: Download a subset for testing

If you just want a quick sanity check, use the `--limit` flag to keep only *N* images per split:

```bash
python -m src.data.download_raw --limit 50
```

## 5 — Verify the Download

After the script finishes, your `data/raw/` directory should look like this:

```
data/raw/
├── train/
│   ├── <image_001>.jpg
│   ├── <image_002>.jpg
│   ├── ...
│   └── _annotations.coco.json
├── valid/
│   ├── <image_001>.jpg
│   ├── ...
│   └── _annotations.coco.json
└── test/   (if available)
    ├── ...
    └── _annotations.coco.json
```

You can do a quick count to confirm:

```bash
echo "Train images: $(ls data/raw/train/*.jpg 2>/dev/null | wc -l)"
echo "Valid images: $(ls data/raw/valid/*.jpg 2>/dev/null | wc -l)"
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `API key not found in .env file` | Make sure `.env` exists at the **repo root** (`fostride_smartbin_ml/.env`) and contains `ROBOFLOW_API_KEY=...` |
| `ModuleNotFoundError: roboflow` | Run `make requirements` (or `poetry install`) to install dependencies |
| `ModuleNotFoundError: dotenv` | Same as above — `python-dotenv` is included in project dependencies |
| Download is slow / times out | The full dataset is 27 000+ images. Use `--limit 50` first to verify the pipeline, then run the full download on a stable connection |

---

## What Happens Under the Hood

The script `src/data/download_raw.py`:

1. Loads `ROBOFLOW_API_KEY` from `.env` via `python-dotenv`.
2. Connects to the Roboflow workspace **fostride** and project **autocrop-conveyer-images**.
3. Downloads version **1** in **coco-segmentation** format.
4. Extracts and flattens the dataset into `data/raw/{train,valid,test}/`.
5. Cleans up any intermediate zip files.

---

## Next Steps

Once the dataset is downloaded and verified, you're ready to move on to **Step 1 — Data Preprocessing & Exploration**.
