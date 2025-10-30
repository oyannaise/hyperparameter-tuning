# MRPC Paraphrase Detection with Docker 🐳

This project trains a DistilBERT model for paraphrase detection on the MRPC (Microsoft Research Paraphrase Corpus) dataset using PyTorch Lightning and Docker.

## 📊 Project Overview

- **Task**: Binary classification (paraphrase detection)
- **Dataset**: MRPC from GLUE benchmark
- **Model**: DistilBERT (distilbert-base-uncased)
- **Framework**: PyTorch Lightning
- **Experiment Tracking**: Weights & Biases (W&B)
- **Deployment**: Docker

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

1. Fork or clone this repository to your GitHub account
2. Click the **"Code"** button → **"Codespaces"** tab
3. Click **"Create codespace on main"**
4. Once the Codespace is ready, run:
   ```bash
   docker-compose up --build
   ```

For detailed instructions, see [CODESPACES_GUIDE.md](CODESPACES_GUIDE.md)

## 🐳 Docker Details

### Image Information

- **Base Image**: `python:3.11-slim`
- **Size**: ~2GB (including all dependencies)
- **Platform**: linux/amd64 (compatible with most systems)

### What the Docker Container Does

1. Downloads DistilBERT model from Hugging Face
2. Downloads MRPC dataset from GLUE
3. Trains the model for 3 epochs
4. Saves checkpoints to `./docker-checkpoints/`
5. Logs metrics with W&B (offline by default)

### Viewing Results

After training completes:

```bash
# View container logs
docker logs mrpc-training

# Check saved checkpoints
ls -lh docker-checkpoints/

# View W&B logs
ls -lh wandb/
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

## 📈 Performance Comparison

| Platform | Accuracy | F1 Score | Training Time |
|----------|----------|----------|---------------|
| Local Docker (Mac M1) | 84.80% | 89.46% | ~15 min |
| GitHub Codespaces | TBD | TBD | TBD |
| Docker Playground | TBD | TBD | TBD |

*Results should be identical (±0.1%) due to fixed random seed (42)*

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

## 📚 Additional Documentation

- [DOCKER_README.md](DOCKER_README.md) - Detailed Docker usage and troubleshooting
- [CODESPACES_GUIDE.md](CODESPACES_GUIDE.md) - GitHub Codespaces deployment guide

## 👤 Author

- GitHub: [@oyannaise](https://github.com/oyannaise)
- Repository: [hyperparameter-tuning](https://github.com/oyannaise/hyperparameter-tuning)

---

**Last Updated**: October 2025

**Status**: ✅ Ready for deployment
