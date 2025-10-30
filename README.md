# MRPC Paraphrase Detection with Docker 🐳

This project trains a DistilBERT model for paraphrase detection on the MRPC (Microsoft Research Paraphrase Corpus) dataset using PyTorch Lightning and Docker.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Educational-green.svg)](LICENSE)

## 📊 Project Overview

- **Task**: Binary classification (paraphrase detection)
- **Dataset**: MRPC from GLUE benchmark (3,668 training / 408 validation samples)
- **Model**: DistilBERT (distilbert-base-uncased, 67M parameters)
- **Framework**: PyTorch Lightning
- **Experiment Tracking**: Weights & Biases (W&B)
- **Deployment**: Docker + GitHub Codespaces

## 🎯 Best Results

The model was trained with hyperparameters optimized in Project 1:

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 84.80% |
| **F1 Score** | 89.46% |
| **Validation Loss** | 0.4108 |

### Best Hyperparameters:
```
Learning Rate: 2.8e-5
Warmup Steps: 150
Scheduler: linear
Batch Size: 16
Weight Decay: 0.01
Epochs: 3
Random Seed: 42
```

## 🚀 Quick Start

### Prerequisites

- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Git installed
- 8GB+ RAM recommended

### Option 1: Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/oyannaise/hyperparameter-tuning.git
cd hyperparameter-tuning

# Build and run training
docker-compose up --build
```

That's it! The training will start automatically with the best hyperparameters.

### Option 2: Using Docker directly

```bash
# Clone the repository
git clone https://github.com/oyannaise/hyperparameter-tuning.git
cd hyperparameter-tuning

# Build the Docker image
docker build -t mrpc-training:latest .

# Run training
docker run -v $(pwd)/docker-checkpoints:/app/checkpoints mrpc-training:latest
```

## 📁 Project Structure

```
hyperparameter-tuning/
├── src/
│   ├── __init__.py
│   ├── data.py          # Data loading and preprocessing
│   ├── model.py         # DistilBERT model definition
│   └── train.py         # Training logic
├── main.py              # Entry point with CLI arguments
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker image definition
├── docker-compose.yml   # Docker Compose configuration
├── run_training.sh      # Training script with best hyperparameters
├── .dockerignore        # Files to exclude from Docker image
├── .gitignore           # Files to exclude from Git
├── .devcontainer/       # VS Code dev container config
│   └── devcontainer.json
├── DOCKER_README.md     # Detailed Docker documentation
├── CODESPACES_GUIDE.md  # GitHub Codespaces deployment guide
└── README.md            # This file
```

## 🔧 Manual Training (Without Docker)

If you prefer to run training directly on your machine:

### 1. Install Dependencies

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run Training

```bash
python main.py \
  --checkpoint_dir ./checkpoints \
  --lr 2.8e-5 \
  --warmup_steps 150 \
  --scheduler linear \
  --batch_size 16 \
  --weight_decay 0.01 \
  --epochs 3 \
  --project_name mrpc-training \
  --run_name my-training-run
```

### Available Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--checkpoint_dir` | Yes | - | Directory to save model checkpoints |
| `--lr` | Yes | - | Learning rate (e.g., 2.8e-5) |
| `--warmup_steps` | No | 150 | Number of warmup steps |
| `--scheduler` | No | linear | LR scheduler (linear/cosine) |
| `--batch_size` | No | 16 | Training batch size |
| `--weight_decay` | No | 0.01 | Weight decay for optimizer |
| `--epochs` | No | 3 | Number of training epochs |
| `--project_name` | No | mrpc-docker-training | W&B project name |
| `--run_name` | No | auto-generated | W&B run name |

## 📊 Experiment Tracking

This project uses **Weights & Biases (W&B)** for experiment tracking.

### Offline Mode (Default)

By default, W&B runs in offline mode. Logs are saved to `./wandb/` directory.

To view offline logs:
```bash
wandb sync ./wandb/offline-run-XXXXXX
```

### Online Mode

To enable online logging:

1. Get your API key from https://wandb.ai/authorize
2. Set environment variable:
   ```bash
   export WANDB_API_KEY=your_api_key_here
   ```
3. Or modify `docker-compose.yml`:
   ```yaml
   environment:
     - WANDB_API_KEY=your_api_key_here
     # Remove or comment out WANDB_MODE=offline
   ```

## 🌐 Running on GitHub Codespaces

This project includes a dev container configuration for easy deployment on GitHub Codespaces.

### Steps:

1. **Fork or navigate to this repository** on GitHub
2. Click the **"Code"** button → **"Codespaces"** tab
3. Click **"Create codespace on main"**
4. Wait for the Codespace to initialize (~2-3 minutes)
5. Once ready, in the Codespace terminal:
   ```bash
   docker-compose up --build
   ```

### What Codespaces Provides:
- ✅ Pre-configured Python 3.11 environment
- ✅ Docker-in-Docker support
- ✅ All VS Code extensions installed
- ✅ Persistent storage for checkpoints
- ✅ Free tier: 60 hours/month for 2-core machine

### Performance Comparison:

| Platform | Accuracy | F1 Score | Training Time | Notes |
|----------|----------|----------|---------------|-------|
| Local Docker (Mac M1) | 84.80% | 89.46% | ~15 min | CPU only |
| GitHub Codespaces | 84.80%* | 89.46%* | ~20-25 min* | 2-core VM, CPU only |
| Docker Playground | 84.80%* | 89.46%* | ~25-30 min* | Shared resources |

*Expected values - should be identical due to fixed random seed (42)

### Troubleshooting Codespaces:

**Issue: "Docker daemon not running"**
```bash
# Wait a few seconds for Docker to initialize, then retry
docker ps
```

**Issue: Out of memory**
```bash
# Edit run_training.sh and reduce batch size:
--batch_size 8  # instead of 16
```

## 🐳 Docker Details

### Image Information

- **Base Image**: `python:3.11-slim`
- **Final Size**: ~2GB (including all dependencies)
- **Platform**: linux/amd64 (compatible with most systems)
- **Build Time**: ~3-5 minutes (depending on internet speed)

### What the Docker Container Does

1. ✅ Sets up Python 3.11 environment
2. ✅ Installs all dependencies from `requirements.txt`
3. ✅ Downloads DistilBERT model from Hugging Face (~268MB)
4. ✅ Downloads MRPC dataset from GLUE (~1MB)
5. ✅ Trains the model for 3 epochs with best hyperparameters
6. ✅ Saves checkpoints to `./docker-checkpoints/` (mounted volume)
7. ✅ Logs metrics with W&B (offline by default)
8. ✅ Displays final results in terminal

### Viewing Results

After training completes:

```bash
# View full container logs
docker logs mrpc-training

# View last 50 lines (includes final metrics)
docker logs mrpc-training | tail -50

# Check saved checkpoints
ls -lh docker-checkpoints/

# View W&B offline logs
ls -lh wandb/

# Sync W&B logs to cloud (if you have API key)
wandb sync ./wandb/offline-run-*
```

### Docker Files Explained

| File | Purpose |
|------|---------|
| `Dockerfile` | Defines the Docker image (OS, Python, dependencies) |
| `docker-compose.yml` | Simplifies running Docker with preset configurations |
| `.dockerignore` | Excludes unnecessary files from the image (reduces size) |
| `run_training.sh` | Shell script that runs training with best hyperparameters |

### Customizing Docker Training

To modify hyperparameters, edit `run_training.sh`:

```bash
python main.py \
  --checkpoint_dir /app/checkpoints \
  --lr 2.8e-5 \              # <-- Change learning rate
  --warmup_steps 150 \       # <-- Change warmup
  --scheduler linear \       # <-- Change scheduler (linear/cosine)
  --batch_size 16 \          # <-- Change batch size
  --weight_decay 0.01 \      # <-- Change weight decay
  --epochs 3 \               # <-- Change number of epochs
  --project_name mrpc-docker-training \
  --run_name docker-best-params
```

Then rebuild and run:
```bash
docker-compose up --build
```

## ⚠️ Common Issues & Solutions

### Issue 1: Out of Memory

**Solution**: Reduce batch size in `run_training.sh`:
```bash
--batch_size 8  # or even 4
```

### Issue 2: Docker Daemon Not Running

**Error**: `Cannot connect to the Docker daemon`

**Solution**: Start Docker Desktop application

### Issue 3: Permission Denied on Checkpoints

**Solution**:
```bash
mkdir -p docker-checkpoints
chmod 755 docker-checkpoints
```

### Issue 4: Slow Training on CPU

**Expected**: Training on CPU takes 15-30 minutes (vs. 3-5 minutes on GPU)

**Solution**: This is normal. Be patient or use a GPU-enabled cloud service.

## 🧪 Testing

To verify everything works:

```bash
# Quick test with 1 epoch
python main.py \
  --checkpoint_dir ./test-checkpoints \
  --lr 2.8e-5 \
  --epochs 1
```

Expected output:
- Dataset downloaded
- Model training starts
- Validation metrics shown after each epoch
- Checkpoints saved

## 🌐 Alternative: Docker Playground

If you don't want to use Codespaces, you can test on Docker Playground:

### Steps:

1. **Go to**: https://labs.play-with-docker.com/
2. **Click "Start"** and then **"Add New Instance"**
3. **Clone the repository**:
   ```bash
   git clone https://github.com/oyannaise/hyperparameter-tuning.git
   cd hyperparameter-tuning
   ```
4. **Run training**:
   ```bash
   docker build -t mrpc-training:latest .
   docker run -v $(pwd)/docker-checkpoints:/app/checkpoints mrpc-training:latest
   ```

### Docker Playground Notes:
- ⏱️ Sessions last 4 hours
- 💻 Shared resources (may be slower)
- 🆓 Completely free
- 🚀 No account required

## 📈 Cross-Platform Performance

| Platform | Accuracy | F1 Score | Loss | Training Time | CPU |
|----------|----------|----------|------|---------------|-----|
| Local Docker (Mac M1) | 84.80% | 89.46% | 0.4108 | ~15 min | Apple Silicon |
| GitHub Codespaces | 84.80%* | 89.46%* | 0.4108* | ~20-25 min* | Intel Xeon |
| Docker Playground | 84.80%* | 89.46%* | 0.4108* | ~25-30 min* | Shared |

*Expected values - results should be identical (±0.001) due to:
- Fixed random seed (42)
- Deterministic algorithms
- Same PyTorch/CUDA versions
- Identical hyperparameters

**Only training time varies** based on CPU performance.

## 📝 Requirements

```
torch>=2.0.0
lightning>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
evaluate>=0.4.0
scikit-learn>=1.0.0
wandb>=0.15.0
```

## 🤝 Contributing

This is a course project, but feel free to:
- Report issues
- Suggest improvements
- Fork and experiment

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- **Dataset**: MRPC from the GLUE benchmark
- **Model**: DistilBERT by Hugging Face
- **Framework**: PyTorch Lightning
- **Experiment Tracking**: Weights & Biases

## � Experiment Tracking Details

This project uses **Weights & Biases (W&B)** to track experiments.

### Default: Offline Mode

By default, W&B runs in **offline mode** (no internet required). Logs are saved locally to `./wandb/` directory.

**Advantages:**
- ✅ No API key needed
- ✅ Works without internet
- ✅ Full privacy (data stays local)
- ✅ Can sync later

**View offline logs:**
```bash
# List all runs
ls -lh wandb/

# View run summary
cat wandb/offline-run-*/files/wandb-summary.json

# Sync to W&B cloud later (requires API key)
wandb sync ./wandb/offline-run-*
```

### Enable Online Mode

To enable live experiment tracking on wandb.ai:

**1. Get your API key**: https://wandb.ai/authorize

**2. Set it in docker-compose.yml**:
```yaml
environment:
  - WANDB_API_KEY=your_api_key_here
  # Comment out or remove: - WANDB_MODE=offline
```

**3. Or export as environment variable**:
```bash
export WANDB_API_KEY=your_api_key_here
docker-compose up --build
```

**4. Or pass directly to Python**:
```bash
python main.py \
  --checkpoint_dir ./checkpoints \
  --lr 2.8e-5 \
  --project_name your-project-name \
  --run_name your-run-name
```

### What W&B Tracks

- 📊 Training loss per step
- 📈 Validation metrics (accuracy, F1 score, loss)
- ⚙️ All hyperparameters
- 💻 System info (CPU, RAM, Python version)
- ⏱️ Training time per epoch
- 📁 Model checkpoints (optional)

### W&B Dashboard

When online mode is enabled, you can:
- Compare multiple runs
- Visualize metrics in real-time
- Share results with collaborators
- Download metrics as CSV

Access at: `https://wandb.ai/YOUR_USERNAME/mrpc-docker-training`

## 👤 Author

- GitHub: [@oyannaise](https://github.com/oyannaise)
- Repository: [hyperparameter-tuning](https://github.com/oyannaise/hyperparameter-tuning)

---

**Last Updated**: October 2025

**Status**: ✅ Ready for deployment
