================================================================================
  FOSTRIDE WASTE CLASSIFICATION — PRODUCTION IMPLEMENTATION GUIDE
  Version: 1.0 | Date: March 2026
  Audience: ML Engineering Team
================================================================================

This document outlines a sequential, phase-by-phase engineering plan to take
the waste classification system from a Roboflow dataset to a production
inference service running on AWS EC2.

Assume: team knows basic model training. Focus is on infrastructure and
engineering setup.

--------------------------------------------------------------------------------
PHASE 1 — DATASET HANDLING & VERSIONING
--------------------------------------------------------------------------------

OBJECTIVE:
  Export and organize the dataset locally, establish versioned data management,
  and build a reproducible preprocessing pipeline.

ENGINEERING TASKS:
  1. Export dataset from Roboflow
     - Use Roboflow "Export" (no credits required) → choose COCO JSON or
       YOLOv8 folder format based on model choice.
     - Download via Roboflow Python SDK using API key (no training credit used).
       Example:
         rf = Roboflow(api_key="YOUR_KEY")
         project = rf.workspace("ws").project("waste-cls")
         dataset = project.version(1).download("coco")

  2. Set up local data directory structure
     data/
       raw/          ← original Roboflow export (never modify)
       processed/    ← augmented/resized images ready for training
       splits/
         train/
         val/
         test/

  3. Initialize DVC for dataset versioning
     - `dvc init` in project root
     - `dvc add data/raw` to track the raw dataset
     - Set up a local DVC remote (shared NAS or S3 bucket)
     - Commit `.dvc` files to Git → dataset version is now tied to a Git commit

  4. Build a preprocessing script (not in a notebook)
     - src/data/preprocess.py
     - Normalize images, resize to target resolution (e.g., 224x224 or 640x640)
     - Output processed splits to data/splits/
     - This script must be idempotent and configurable via arguments

TOOLS:
  - Roboflow Python SDK (pip install roboflow)
  - DVC (pip install dvc)
  - Albumentations (augmentation library)
  - OpenCV or PIL for preprocessing

EXPECTED OUTPUTS / ARTIFACTS:
  - data/raw/ directory tracked by DVC
  - data/splits/ (train/val/test) ready for training
  - src/data/preprocess.py
  - .dvc/config with remote configured
  - dvc.lock and data.dvc files committed to Git

--------------------------------------------------------------------------------
PHASE 2 — LOCAL TRAINING PIPELINE
--------------------------------------------------------------------------------

OBJECTIVE:
  Build a production-grade, configurable, reproducible training pipeline that
  runs as a standalone script — not a notebook.

ENGINEERING TASKS:
  1. Establish project structure (use Cookiecutter Data Science as reference)
     fostride-waste-cls/
       configs/           ← Hydra YAML configs
       data/              ← DVC-managed
       models/            ← saved checkpoints
       notebooks/         ← exploration only, not production code
       src/
         data/            ← dataset classes, dataloaders
         models/          ← model definition
         training/        ← training loop, loss, metrics
         utils/           ← helpers
       train.py           ← main entrypoint
       requirements.txt
       Dockerfile
       .dvcignore
       .gitignore

  2. Implement configuration management with Hydra
     - configs/train.yaml (model, optimizer, scheduler, paths)
     - configs/data.yaml (dataset paths, batch size, augmentations)
     - CLI override: `python train.py model.lr=0.001 data.batch_size=32`
     - No hardcoded values anywhere in training code

  3. Implement the training loop as a script (src/training/trainer.py)
     - Model loading, optimizer, loss, metric tracking
     - Checkpointing every N epochs to models/
     - Validation loop with early stopping

  4. Integrate MLflow experiment tracking
     - Auto-log hyperparameters, metrics, and artifacts per run
     - Local MLflow server: `mlflow ui` (no cloud needed)
     - Log: config snapshot, val accuracy curve, final model checkpoint
     - Tag runs with dataset version (DVC commit hash)

  5. Write a DVC pipeline (dvc.yaml)
     - stages: preprocess → train → evaluate
     - `dvc repro` runs the full pipeline reproduciby
     - Enables exact experiment reproduction by any team member

TOOLS:
  - PyTorch + torchvision (or timm for pretrained backbones)
  - Hydra (config management)
  - MLflow (experiment tracking, local)
  - DVC (pipeline + data versioning)
  - pytest (unit tests for data loading, transforms)

EXPECTED OUTPUTS / ARTIFACTS:
  - train.py and src/ fully functional pipeline
  - configs/ directory with all Hydra configs
  - dvc.yaml pipeline definition
  - MLflow runs visible at localhost:5000
  - Trained model checkpoint: models/best.pt
  - requirements.txt pinned

--------------------------------------------------------------------------------
PHASE 3 — INFERENCE PIPELINE
--------------------------------------------------------------------------------

OBJECTIVE:
  Package the trained model into a clean, low-latency inference service that
  accepts image input (from smart bins) and returns a classification result.

ENGINEERING TASKS:
  1. Export trained model to ONNX
     - Reduces inference latency ~1.5–3x on CPU vs PyTorch runtime
     - Command:
         torch.onnx.export(model, dummy_input, "models/waste_cls.onnx",
                           opset_version=17,
                           input_names=["image"],
                           output_names=["logits"])
     - Validate ONNX output against PyTorch output (numerical equivalence check)

  2. Build the inference module (src/inference/predictor.py)
     - Load ONNX model via onnxruntime.InferenceSession at startup
     - preprocess_image(image_bytes) → normalized numpy array
     - predict(image_bytes) → {"class": "Plastic", "confidence": 0.94}
     - Stateless, thread-safe design

  3. Build a FastAPI inference server (src/api/server.py)
     Endpoints:
       POST /predict     → accepts multipart image, returns JSON prediction
       GET  /health      → liveness check (for load balancer / monitoring)
       GET  /model-info  → returns model version and classes
     - Load model ONCE at application startup (not per request)
     - Add request logging middleware
     - Return structured JSON: {class, confidence, latency_ms}

  4. Validate latency and correctness
     - Benchmark with locust or a simple Python script: target <200ms p95
     - Test with sample bin images from each class (Plastic, Paper, Metal, Other)
     - Write integration tests for the /predict endpoint

  5. Preprocessing contract
     - Document exact preprocessing (resize, normalize constants) that MUST
       match between training and inference.
     - Store this as a config or metadata alongside the ONNX model file.

TOOLS:
  - ONNX + onnxruntime (CPU inference)
  - FastAPI + Uvicorn
  - Pillow / OpenCV (image preprocessing)
  - pytest + httpx (API integration tests)

EXPECTED OUTPUTS / ARTIFACTS:
  - models/waste_cls.onnx (exported model)
  - src/inference/predictor.py
  - src/api/server.py
  - Dockerfile (inference image)
  - Test suite passing for /predict endpoint
  - Benchmark results documented

--------------------------------------------------------------------------------
PHASE 4 — DEPLOYMENT INFRASTRUCTURE (AWS EC2)
--------------------------------------------------------------------------------

OBJECTIVE:
  Containerize the inference service, deploy to EC2, establish CI/CD for
  updates, and add basic monitoring.

ENGINEERING TASKS:
  1. Write the production Dockerfile
     - Base: python:3.11-slim (small, fast)
     - Multi-stage build: builder stage installs deps, final stage is minimal
     - Copy only src/, models/waste_cls.onnx, configs/inference.yaml
     - Expose port 8000, run with Uvicorn (not Flask dev server)
     - Run as non-root user (security baseline)
     - Health check instruction in Dockerfile
     Example CMD:
       CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0",
            "--port", "8000", "--workers", "2"]

  2. Build and push Docker image to a registry
     - Use AWS ECR (Elastic Container Registry) — free tier is sufficient
     - Tag with Git commit SHA: fostride/waste-cls:<git-sha>
     - Push via: `docker push <ecr-url>/waste-cls:<sha>`

  3. Provision EC2 instance
     - Recommended: t3.small (2 vCPU, 2GB RAM) for CPU-only ONNX inference
     - AMI: Amazon Linux 2023 or Ubuntu 22.04
     - Security Group: allow inbound TCP 8000 (inference), 22 (SSH)
     - Attach an IAM role with ECR pull permissions

  4. Deploy container to EC2
     - SSH to EC2, install Docker
     - Pull latest image from ECR and run:
         docker run -d -p 8000:8000 --name waste-cls \
           --restart=always \
           <ecr-url>/waste-cls:<sha>

  5. Set up GitHub Actions CI/CD (simple SSH deploy)
     Pipeline triggered on push to `main`:
       Step 1: Run tests (pytest)
       Step 2: Build Docker image
       Step 3: Push to ECR
       Step 4: SSH into EC2, pull new image, restart container
     - Store EC2 SSH key and ECR credentials as GitHub Secrets
     - Zero-downtime: start new container before stopping old one

  6. Basic monitoring setup
     - Logging: configure Uvicorn → write logs to stdout → captured by Docker
     - Use `docker logs -f waste-cls` or ship to CloudWatch Logs (optional)
     - Add /health endpoint to EC2 instance health checks
     - Set up a Cron job or simple watchdog to alert if /health returns non-200
     - Track: prediction distribution per class over time (log to CSV or DB)
     - Alert threshold: if "Other" class exceeds 40% sustain → may indicate
       model drift or bin camera misalignment

  7. Model update workflow
     When a new model is trained:
       a. Export new ONNX → models/waste_cls_v2.onnx
       b. Run benchmark + integration tests locally
       c. Push to ECR with new tag
       d. Deploy to EC2 via CI/CD pipeline (or manual docker pull + restart)
       e. Update model-info endpoint → teams can verify which version is live

TOOLS:
  - Docker
  - AWS EC2 + ECR
  - GitHub Actions
  - Uvicorn (ASGI server)
  - AWS CloudWatch Logs (optional, for centralized logging)
  - Evidently AI (open source, for drift detection — run as a batch job)

EXPECTED OUTPUTS / ARTIFACTS:
  - Dockerfile (production-ready)
  - .github/workflows/deploy.yml (CI/CD pipeline)
  - ECR repository with versioned images
  - Running EC2 instance serving /predict at a public IP
  - /health endpoint returning 200 OK
  - Deployment runbook (simple .md doc): how to deploy a new model version

--------------------------------------------------------------------------------
SUMMARY — DELIVERABLES PER PHASE
--------------------------------------------------------------------------------

Phase 1:  data/ (DVC-tracked), preprocess.py, .dvc files committed
Phase 2:  train.py, configs/, dvc.yaml, MLflow UI with tracked runs,
          models/best.pt
Phase 3:  models/waste_cls.onnx, FastAPI server, integration tests passing,
          benchmark results
Phase 4:  Dockerfile, GitHub Actions deploy pipeline, live EC2 endpoint,
          deployment runbook

--------------------------------------------------------------------------------
TECHNOLOGY STACK SUMMARY
--------------------------------------------------------------------------------

Category              Tool/Library
-----------           ----------------
Data Versioning       DVC
Config Management     Hydra
Experiment Tracking   MLflow (local)
Model Export          ONNX + onnxruntime
Inference Server      FastAPI + Uvicorn
Containerization      Docker
Image Registry        AWS ECR
Compute               AWS EC2 (t3.small)
CI/CD                 GitHub Actions
Monitoring            Docker logs + Evidently AI (batch drift detection)
Testing               pytest + httpx

================================================================================
END OF DOCUMENT
================================================================================
