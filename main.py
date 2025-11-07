"""Main training script for GLUE tasks with hyperparameter support"""

import argparse
import os
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import wandb

from data_module import GLUEDataModule
from model import GLUETransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GLUE model with configurable hyperparameters")
    
    # Model and task settings
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", 
                        help="Pretrained model name or path")
    parser.add_argument("--task_name", type=str, default="mrpc", 
                        help="GLUE task name (e.g., mrpc, sst2, cola)")
    
    # Training hyperparameters
    parser.add_argument("--lr", "--learning_rate", type=float, default=2.8e-5, 
                        dest="learning_rate", help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=16, 
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, 
                        help="Evaluation batch size")
    parser.add_argument("--epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=150, 
                        help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, 
                        help="Weight decay")
    parser.add_argument("--max_seq_length", type=int, default=128, 
                        help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                        help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, 
                        help="Maximum gradient norm for clipping")
    
    # Checkpoint and logging
    parser.add_argument("--checkpoint_dir", type=str, default="models", 
                        help="Directory to save model checkpoints")
    parser.add_argument("--run_name", type=str, default=None, 
                        help="Name for this training run (for W&B logging)")
    parser.add_argument("--wandb_project", type=str, default="mrpc-training", 
                        help="Weights & Biases project name")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    # Other settings
    parser.add_argument("--accelerator", type=str, default="auto", 
                        help="Accelerator type (auto, cpu, gpu, tpu)")
    parser.add_argument("--devices", type=int, default=1, 
                        help="Number of devices to use")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed for reproducibility
    L.seed_everything(args.seed)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Generate run name if not provided
    if args.run_name is None:
        args.run_name = f"run-bs{args.train_batch_size}-lr{args.learning_rate}-w{args.warmup_steps}"
    
    # Initialize W&B
    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config={
            "learning_rate": args.learning_rate,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "epochs": args.epochs,
            "model": args.model_name,
            "task_name": args.task_name,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_seq_length": args.max_seq_length,
            "max_grad_norm": args.max_grad_norm,
        }
    )
    
    # Create W&B logger
    logger = WandbLogger(project=args.wandb_project, name=args.run_name)
    
    # Initialize data module
    print(f"\n{'='*60}")
    print(f"Initializing data module for task: {args.task_name}")
    print(f"{'='*60}\n")
    
    dm = GLUEDataModule(
        model_name_or_path=args.model_name,
        task_name=args.task_name,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        max_seq_length=args.max_seq_length,
    )
    dm.setup("fit")
    
    # Initialize model
    print(f"\n{'='*60}")
    print(f"Initializing model: {args.model_name}")
    print(f"{'='*60}\n")
    
    model = GLUETransformer(
        model_name_or_path=args.model_name,
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )
    
    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=logger,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=args.max_grad_norm,
        default_root_dir=args.checkpoint_dir,
        enable_checkpointing=True,
    )
    
    # Train
    print(f"\n{'='*60}")
    print(f"Starting training: {args.run_name}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Batch Size: {args.train_batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Warmup Steps: {args.warmup_steps}")
    print(f"{'='*60}\n")
    
    trainer.fit(model, datamodule=dm)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Run completed: {args.run_name}")
    print(f"Validation Accuracy: {trainer.callback_metrics.get('accuracy', 'N/A')}")
    print(f"Validation F1: {trainer.callback_metrics.get('f1', 'N/A')}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"{'='*60}\n")
    
    wandb.finish()


if __name__ == "__main__":
    main()
