"""
Training logic
"""
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from .data import GLUEDataModule
from .model import GLUETransformer


def train_model(
    learning_rate,
    warmup_steps,
    scheduler_type,
    checkpoint_dir,
    batch_size=16,
    weight_decay=0.01,
    epochs=3,
    project_name="mrpc-docker-training",  # Neuer Parameter
    run_name=None,  # Neuer Parameter
):
    """Train the model with given hyperparameters"""
    
    # Set seed for reproducibility
    L.seed_everything(42)
    
    # Create W&B logger
    if run_name is None:
        run_name = f"lr{learning_rate}_warmup{warmup_steps}_{scheduler_type}"
    
    wandb_logger = WandbLogger(
        project=project_name,
        name=run_name,
        config={
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "scheduler_type": scheduler_type,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "epochs": epochs,
        }
    )
    
    # Create data module
    dm = GLUEDataModule(
        model_name_or_path="distilbert-base-uncased",
        task_name="mrpc",
        train_batch_size=batch_size,
        eval_batch_size=32,
        max_seq_length=128,
    )
    dm.setup("fit")
    
    # Create model
    model = GLUETransformer(
        model_name_or_path="distilbert-base-uncased",
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        train_batch_size=batch_size,
        eval_batch_size=32,
        scheduler_type=scheduler_type,
    )
    
    # Create trainer with W&B logger
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        default_root_dir=checkpoint_dir,
        enable_progress_bar=True,
        logger=wandb_logger,  # <-- W&B Logger hinzugefügt!
    )
    
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"W&B Run: {run_name}")
    print(f"{'='*60}\n")
    
    # Train!
    trainer.fit(model, datamodule=dm)
    
    # Finish W&B run
    wandb_logger.experiment.finish()
    
    # Return metrics
    return trainer.callback_metrics