# Docker Training Setup

This Docker setup allows you to run a single training run with the best hyperparameters from Project 1.

## Best Hyperparameters (from Project 1)

Based on `models/lightning_logs/version_0/hparams.yaml`:
- **Learning Rate**: 2.8e-5
- **Warmup Steps**: 150
- **Scheduler**: linear
- **Batch Size**: 16
- **Weight Decay**: 0.01
- **Epochs**: 3

## Prerequisites

- Docker installed on your machine
- (Optional) Docker Compose installed

## Option 1: Using Docker Compose (Recommended)

### Build and run the training:

```bash
docker-compose up --build
```

This will:
1. Build the Docker image
2. Run training with the best hyperparameters
3. Save checkpoints to `./docker-checkpoints/` directory

### To run again without rebuilding:

```bash
docker-compose up
```

### To stop and remove the container:

```bash
docker-compose down
```

## Option 2: Using Docker directly

### 1. Build the Docker image:

```bash
docker build -t mrpc-training:latest .
```

### 2. Run the training:

```bash
docker run -v $(pwd)/docker-checkpoints:/app/checkpoints mrpc-training:latest
```

Or with W&B logging (if you have an API key):

```bash
docker run \
  -e WANDB_API_KEY=your_api_key_here \
  -v $(pwd)/docker-checkpoints:/app/checkpoints \
  mrpc-training:latest
```

## Weights & Biases (W&B) Configuration

By default, W&B is set to offline mode in the docker-compose.yml file:

```yaml
environment:
  - WANDB_MODE=offline
```

To enable online logging:
1. Get your W&B API key from https://wandb.ai/authorize
2. Uncomment and set the `WANDB_API_KEY` in `docker-compose.yml`, OR
3. Pass it when running Docker directly (see Option 2 above)

## Output

- **Checkpoints**: Saved in `./docker-checkpoints/` directory
- **Logs**: Displayed in the terminal during training
- **W&B Logs**: Available in the `wandb/` directory (offline mode) or on W&B website (online mode)

## Notes

- Training will run on CPU since most local machines don't have GPU support in Docker
- This will take longer than GPU training on Colab
- You can modify hyperparameters by editing `run_training.sh`

## Troubleshooting

### Permission Issues
If you encounter permission issues with the checkpoints directory:

```bash
mkdir -p docker-checkpoints
chmod 755 docker-checkpoints
```

### Out of Memory
If you run out of memory, reduce the batch size in `run_training.sh`:

```bash
--batch_size 8  # or even 4
```
