# GLUE Hyperparameter Tuning Project

This project provides a dockerized training pipeline for fine-tuning transformer models on GLUE tasks with configurable hyperparameters. It uses PyTorch Lightning, Weights & Biases for experiment tracking, and includes the best hyperparameters found during hyperparameter optimization.

## 📋 Project Overview

- **Task**: Fine-tuning DistilBERT on GLUE MRPC task
- **Model**: `distilbert-base-uncased`
- **Best Hyperparameters**:
  - Learning Rate: `2.8e-5`
  - Batch Size: `16`
  - Warmup Steps: `150`
  - Weight Decay: `0.01`
  - Epochs: `3`

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker (for containerized training)
- Weights & Biases account (for experiment tracking)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
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

## 🐳 Docker Usage

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

### Run on Docker Playground

1. Go to [Docker Playground](https://labs.play-with-docker.com/)
2. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd hyperparametertuning
   ```
3. Build and run:
   ```bash
   docker build -t glue-training .
   docker run --rm -e WANDB_API_KEY=your_api_key_here glue-training
   ```

## 📊 Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_name` | str | `distilbert-base-uncased` | Pretrained model name |
| `--task_name` | str | `mrpc` | GLUE task (mrpc, sst2, cola, etc.) |
| `--lr` | float | `2.8e-5` | Learning rate |
| `--train_batch_size` | int | `16` | Training batch size |
| `--eval_batch_size` | int | `32` | Evaluation batch size |
| `--epochs` | int | `3` | Number of training epochs |
| `--warmup_steps` | int | `150` | Number of warmup steps |
| `--weight_decay` | float | `0.01` | Weight decay |
| `--max_seq_length` | int | `128` | Maximum sequence length |
| `--checkpoint_dir` | str | `models` | Directory to save checkpoints |
| `--run_name` | str | auto-generated | Name for W&B run |
| `--wandb_project` | str | `mrpc-training` | W&B project name |
| `--seed` | int | `42` | Random seed |
| `--accelerator` | str | `auto` | Accelerator type (auto, cpu, gpu) |

## 📁 Project Structure

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

## 🔬 Experiment Tracking

All training runs are automatically logged to Weights & Biases. You can monitor:
- Training/validation loss
- Accuracy and F1 score
- Learning rate schedule
- System metrics

View your experiments at: https://wandb.ai/your-username/mrpc-training

## 🎯 Example Commands

**Train with higher learning rate:**
```bash
python main.py --lr 5e-5 --run_name high-lr-experiment
```

**Train on different GLUE task:**
```bash
python main.py --task_name sst2 --run_name sst2-baseline
```

**Train with larger batch size and gradient accumulation:**
```bash
python main.py --train_batch_size 8 --gradient_accumulation_steps 4
```

**CPU-only training:**
```bash
python main.py --accelerator cpu
```

## 📝 Notes

### Task 3 Observations

When running the Docker image across different platforms:

- **Local Machine (CPU)**: Training takes significantly longer without GPU support (~30-45 minutes for 3 epochs)
- **GitHub Codespaces**: Similar performance to local CPU execution
- **Docker Playground**: May have resource limitations affecting training time

**Performance Consistency**: 
- Results should be reproducible across platforms due to fixed random seed (42)
- Minor differences may occur due to:
  - Different CPU architectures
  - Floating-point precision variations
  - Dataset download/caching behavior

**Required Adaptations**:
- Set `WANDB_API_KEY` as environment variable for logging
- Ensure sufficient disk space for model downloads (~500MB)
- Consider reducing batch size if memory issues occur

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- GLUE benchmark: https://gluebenchmark.com/
- Hugging Face Transformers: https://huggingface.co/transformers/
- PyTorch Lightning: https://lightning.ai/
- Weights & Biases: https://wandb.ai/
