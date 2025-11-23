#!/usr/bin/env python3
"""
Training script for coTherapist model.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import TrainingArguments, Trainer
from peft import PeftModel

from coTherapist.models.therapist_model import CoTherapistModel
from coTherapist.utils.config_loader import load_config
from coTherapist.utils.data_loader import (
    load_training_data,
    prepare_dataset_for_training,
    create_sample_training_data
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(
    config_path: str = None,
    data_path: str = None,
    output_dir: str = None,
    create_sample_data: bool = False
):
    """
    Train the coTherapist model.
    
    Args:
        config_path: Path to configuration file
        data_path: Path to training data
        output_dir: Output directory for model
        create_sample_data: Whether to create sample data for demo
    """
    # Load configuration
    config = load_config(config_path)
    
    if output_dir:
        config['output']['model_output_dir'] = output_dir
    
    if data_path:
        config['data']['training_data_path'] = data_path
    
    logger.info("="*70)
    logger.info("coTherapist Model Training")
    logger.info("="*70)
    
    # Create sample data if requested
    if create_sample_data:
        sample_path = config['data']['training_data_path']
        if not sample_path.endswith('.json'):
            sample_path = os.path.join(sample_path, 'sample_conversations.json')
        
        logger.info(f"Creating sample training data at {sample_path}")
        create_sample_training_data(sample_path, num_samples=100)
        config['data']['training_data_path'] = sample_path
    
    # Initialize model
    logger.info("\n1. Initializing model...")
    model_trainer = CoTherapistModel(config)
    model_trainer.load_model()
    model_trainer.setup_lora()
    
    # Load training data
    logger.info("\n2. Loading training data...")
    data_path = config['data']['training_data_path']
    
    if not os.path.exists(data_path):
        logger.error(f"Training data not found at {data_path}")
        logger.info("Run with --create-sample-data to create demo data")
        return
    
    dataset = load_training_data(data_path, format='json')
    
    # Prepare dataset
    logger.info("\n3. Preparing dataset...")
    processed_dataset = prepare_dataset_for_training(
        dataset,
        model_trainer.tokenizer,
        max_length=config['model']['max_length']
    )
    
    # Split dataset
    train_test = processed_dataset.train_test_split(
        test_size=config['data']['validation_split']
    )
    train_dataset = train_test['train']
    eval_dataset = train_test['test']
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(eval_dataset)}")
    
    # Set up training arguments
    logger.info("\n4. Setting up training configuration...")
    
    training_args = TrainingArguments(
        output_dir=config['output']['checkpoint_dir'],
        num_train_epochs=config['fine_tuning']['num_epochs'],
        per_device_train_batch_size=config['fine_tuning']['batch_size'],
        per_device_eval_batch_size=config['fine_tuning']['batch_size'],
        gradient_accumulation_steps=config['fine_tuning']['gradient_accumulation_steps'],
        learning_rate=config['fine_tuning']['learning_rate'],
        warmup_steps=config['fine_tuning']['warmup_steps'],
        max_grad_norm=config['fine_tuning']['max_grad_norm'],
        weight_decay=config['fine_tuning']['weight_decay'],
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        report_to="none",  # Change to "wandb" if using W&B
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    logger.info("\n5. Initializing trainer...")
    
    trainer = Trainer(
        model=model_trainer.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train
    logger.info("\n6. Starting training...")
    logger.info("This may take a while depending on your hardware...\n")
    
    trainer.train()
    
    # Save model
    logger.info("\n7. Saving fine-tuned model...")
    output_path = config['output']['model_output_dir']
    os.makedirs(output_path, exist_ok=True)
    
    model_trainer.save_model(output_path)
    
    logger.info(f"\nModel saved to: {output_path}")
    logger.info("\n" + "="*70)
    logger.info("Training complete!")
    logger.info("="*70)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train coTherapist model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to training data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for model"
    )
    parser.add_argument(
        "--create-sample-data",
        action="store_true",
        help="Create sample training data for demonstration"
    )
    
    args = parser.parse_args()
    
    train_model(
        config_path=args.config,
        data_path=args.data,
        output_dir=args.output,
        create_sample_data=args.create_sample_data
    )


if __name__ == "__main__":
    main()
