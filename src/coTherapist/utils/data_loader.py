"""Data loading utilities for training and evaluation."""

import os
import json
from typing import List, Dict, Any, Optional
from datasets import Dataset, load_dataset
import logging

logger = logging.getLogger(__name__)


def load_training_data(data_path: str, format: str = 'json') -> Dataset:
    """
    Load training data for fine-tuning.
    
    Args:
        data_path: Path to training data
        format: Data format ('json', 'jsonl', 'csv', 'huggingface')
        
    Returns:
        HuggingFace Dataset object
    """
    logger.info(f"Loading training data from {data_path}")
    
    if format == 'huggingface':
        # Load from HuggingFace hub
        dataset = load_dataset(data_path)
    elif format == 'json':
        with open(data_path, 'r') as f:
            data = json.load(f)
        dataset = Dataset.from_list(data)
    elif format == 'jsonl':
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        dataset = Dataset.from_list(data)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Loaded {len(dataset)} examples")
    return dataset


def format_conversation_for_training(
    conversations: List[Dict[str, str]],
    tokenizer
) -> List[str]:
    """
    Format conversations for LLaMA instruction tuning.
    
    Args:
        conversations: List of conversation dicts with 'user' and 'assistant' keys
        tokenizer: Tokenizer to use for formatting
        
    Returns:
        List of formatted strings
    """
    formatted = []
    
    for conv in conversations:
        # LLaMA 3.2 Instruct format
        text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a compassionate mental health support assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

{conv['user']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{conv['assistant']}<|eot_id|>"""
        
        formatted.append(text)
    
    return formatted


def prepare_dataset_for_training(
    dataset: Dataset,
    tokenizer,
    max_length: int = 2048
) -> Dataset:
    """
    Prepare dataset for training by tokenizing and formatting.
    
    Args:
        dataset: Raw dataset
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Processed dataset ready for training
    """
    def tokenize_function(examples):
        # Format conversations
        if 'user' in examples and 'assistant' in examples:
            texts = []
            for user, assistant in zip(examples['user'], examples['assistant']):
                text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a compassionate mental health support assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{assistant}<|eot_id|>"""
                texts.append(text)
        elif 'text' in examples:
            texts = examples['text']
        else:
            raise ValueError("Dataset must have 'user'/'assistant' or 'text' fields")
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        
        # Add labels (same as input_ids for causal LM)
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
    
    processed = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    return processed


def create_sample_training_data(output_path: str, num_samples: int = 100):
    """
    Create sample therapeutic conversation data for demonstration.
    
    Args:
        output_path: Where to save the sample data
        num_samples: Number of sample conversations to generate
    """
    samples = [
        {
            "user": "I've been feeling really anxious lately and I don't know what to do.",
            "assistant": "I hear that you're feeling anxious, and I want you to know that what you're experiencing is valid. Anxiety can be overwhelming. Can you tell me a bit more about when you notice these feelings most? Understanding the patterns can help us explore some coping strategies together."
        },
        {
            "user": "I feel like nobody understands what I'm going through.",
            "assistant": "Feeling misunderstood can be incredibly isolating and painful. Your feelings matter, and it's important that you feel heard. Would you like to share more about what's been happening? Sometimes just talking about it can help, and together we can explore ways to help you feel more connected and supported."
        },
        {
            "user": "I'm having trouble sleeping because I can't stop thinking about work.",
            "assistant": "It sounds like work-related thoughts are really interfering with your rest, which must be exhausting. This is a common challenge, and there are strategies that can help. Have you tried any relaxation techniques before bed, like deep breathing or progressive muscle relaxation? We can explore some evidence-based approaches to help quiet your mind at night."
        },
        {
            "user": "I had a panic attack today and it scared me.",
            "assistant": "I'm sorry you experienced a panic attack - those can be really frightening. It's important to know that while panic attacks feel overwhelming, they are not dangerous and they do pass. You're taking a positive step by talking about it. Can you tell me what you were doing when it happened? Understanding the context can help us develop strategies to manage and reduce these episodes."
        },
        {
            "user": "I don't see the point in anything anymore.",
            "assistant": "I'm really concerned about what you're sharing, and I'm glad you reached out. These feelings of hopelessness are serious and you don't have to face them alone. I strongly encourage you to speak with a mental health professional who can provide the support you need. In the meantime, if you're having thoughts of harming yourself, please contact the National Suicide Prevention Lifeline at 988 or 1-800-273-8255. They're available 24/7 and can provide immediate support. Would you be willing to reach out to them or a local mental health service?"
        }
    ]
    
    # Replicate samples to reach desired count
    full_dataset = []
    while len(full_dataset) < num_samples:
        full_dataset.extend(samples)
    full_dataset = full_dataset[:num_samples]
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(full_dataset, f, indent=2)
    
    logger.info(f"Created {num_samples} sample conversations at {output_path}")
