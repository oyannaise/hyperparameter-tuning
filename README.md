# Hyperparameter Tuning Project

This project provides a dockerized training pipeline for fine-tuning transformer models with configurable hyperparameters. It uses PyTorch Lightning, Weights & Biases for experiment tracking, and includes the best hyperparameters found during hyperparameter optimization. This project was carried out as part of a university project for the Machine Learning Operations module at HSLU in HS25.

## Project Overview

- **Task**: Fine-tuning DistilBERT on GLUE MRPC task
- **Model**: `distilbert-base-uncased`
- **Best Hyperparameters**:
  - Learning Rate: `2.8e-5`
  - Batch Size: `16`
  - Warmup Steps: `150`
  - Weight Decay: `0.01`
  - Epochs: `3`

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (for containerized training)
- Weights & Biases account (for experiment tracking)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/oyannaise/hyperparameter-tuning
   cd hyperparametertuning
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Login to Weights & Biases**
   ```bash
   wandb login
   ```
   Enter your API key when prompted.

4. **Run training with default hyperparameters**
   ```bash
   python main.py
   ```

5. **Run training with custom hyperparameters**
   ```bash
   python main.py --lr 1e-3 --train_batch_size 32 --epochs 5 --checkpoint_dir models
   ```

## Docker Usage

### Build the Docker Image

```bash
docker build -t glue-training .
```

### Run Training in Docker

**Basic run with best hyperparameters:**
```bash
docker run --rm -e WANDB_API_KEY=your_api_key_here glue-training
```

**With custom hyperparameters:**
```bash
docker run --rm -e WANDB_API_KEY=your_api_key_here glue-training \
  python main.py --lr 1e-3 --train_batch_size 32 --epochs 5
```

**Save checkpoints to host machine:**
```bash
docker run --rm \
  -e WANDB_API_KEY=your_api_key_here \
  -v $(pwd)/models:/app/models \
  glue-training
```

### Run on GitHub Codespaces

1. Open this repository in GitHub Codespaces
2. Build the Docker image:
   ```bash
   docker build -t glue-training .
   ```
3. Run training:
   ```bash
   docker run --rm -e WANDB_API_KEY=$WANDB_API_KEY glue-training
   ```


## Project Structure

```
hyperparametertuning/
├── data_module.py          # PyTorch Lightning DataModule for GLUE
├── model.py                # PyTorch Lightning model definition
├── main.py                 # Main training script with CLI
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
├── .dockerignore          # Files to exclude from Docker build
├── .gitignore             # Files to exclude from Git
└── README.md              # This file
```

## Experiment Tracking

All training runs are automatically logged to Weights & Biases. You can monitor:
- Training/validation loss
- Accuracy and F1 score
- Learning rate schedule
- System metrics

**Note**:

- Set `WANDB_API_KEY` as environment variable for logging
- Ensure sufficient disk space for model downloads (~500MB)

