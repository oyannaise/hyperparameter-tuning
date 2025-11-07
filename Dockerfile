# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY data_module.py .
COPY model.py .
COPY main.py .

# Create directory for model checkpoints
RUN mkdir -p /app/models

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command: Run training with best hyperparameters from Project 1
CMD ["python", "main.py", \
     "--lr", "2.8e-5", \
     "--train_batch_size", "16", \
     "--eval_batch_size", "32", \
     "--warmup_steps", "150", \
     "--weight_decay", "0.01", \
     "--epochs", "3", \
     "--checkpoint_dir", "models", \
     "--run_name", "docker-run-best-params"]
