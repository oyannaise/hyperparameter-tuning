"""
Main entry point for training
"""
import argparse
from src.train import train_model


def main():
    parser = argparse.ArgumentParser(
        description='Train DistilBERT on MRPC paraphrase detection'
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        required=True,
        help='Directory to save model checkpoints'
    )
    parser.add_argument(
        '--lr',
        type=float,
        required=True,
        help='Learning rate (e.g., 2.8e-5)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=150,
        help='Number of warmup steps (default: 150)'
    )
    parser.add_argument(
        '--scheduler',
        type=str,
        default='linear',
        choices=['linear', 'cosine'],
        help='Learning rate scheduler type (default: linear)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Training batch size (default: 16)'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help='Weight decay (default: 0.01)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs (default: 3)'
    )
    parser.add_argument(
        '--project_name',
        type=str,
        default='mrpc-docker-training',
        help='W&B project name'
    )
    parser.add_argument(
        '--run_name',
        type=str,
        default=None,
        help='W&B run name (auto-generated if not provided)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("MRPC Paraphrase Detection Training")
    print("="*60)
    print(f"Configuration:")
    print(f"  Learning Rate:    {args.lr}")
    print(f"  Warmup Steps:     {args.warmup_steps}")
    print(f"  Scheduler:        {args.scheduler}")
    print(f"  Batch Size:       {args.batch_size}")
    print(f"  Weight Decay:     {args.weight_decay}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Checkpoint Dir:   {args.checkpoint_dir}")
    print(f"  W&B Project:      {args.project_name}")
    print("="*60 + "\n")
    
    # Train model
    metrics = train_model(
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        scheduler_type=args.scheduler,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        project_name=args.project_name,
        run_name=args.run_name,
    )
    
    print("\n" + "="*60)
    print("Training Completed!")
    print("="*60)
    print(f"Final Validation Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
    print(f"Final F1 Score:            {metrics.get('f1', 'N/A'):.4f}")
    print(f"Final Validation Loss:     {metrics.get('val_loss', 'N/A'):.4f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
