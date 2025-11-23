#!/usr/bin/env python3
"""
Evaluation script for coTherapist model using COTHERF framework.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coTherapist.pipeline import CoTherapistPipeline
from coTherapist.utils.data_loader import load_training_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(
    config_path: str = None,
    test_data_path: str = None,
    output_path: str = None
):
    """
    Evaluate the coTherapist model using COTHERF.
    
    Args:
        config_path: Path to configuration file
        test_data_path: Path to test data
        output_path: Path to save results
    """
    logger.info("="*70)
    logger.info("coTherapist Model Evaluation - COTHERF Framework")
    logger.info("="*70)
    
    # Initialize pipeline
    logger.info("\n1. Initializing pipeline...")
    pipeline = CoTherapistPipeline(config_path)
    pipeline.setup(load_knowledge_base=False)
    
    # Load test data
    logger.info("\n2. Loading test data...")
    if test_data_path and os.path.exists(test_data_path):
        dataset = load_training_data(test_data_path, format='json')
        test_data = [{'user': ex['user'], 'assistant': ex.get('assistant', '')} 
                     for ex in dataset]
    else:
        logger.info("No test data provided, using sample queries...")
        test_data = [
            {'user': "I'm feeling really anxious about my job interview tomorrow."},
            {'user': "I can't stop thinking about a mistake I made last week."},
            {'user': "I feel like my friends don't really care about me."},
            {'user': "I'm having trouble sleeping and it's affecting my work."},
            {'user': "I think I might be depressed but I'm not sure."}
        ]
    
    logger.info(f"Evaluating {len(test_data)} examples")
    
    # Evaluate
    logger.info("\n3. Running evaluation...")
    results = pipeline.evaluate_dataset(test_data)
    
    # Display results
    logger.info("\n" + "="*70)
    logger.info("COTHERF Evaluation Results")
    logger.info("="*70)
    
    eval_stats = results['evaluation']
    logger.info("\nAverage Scores:")
    for metric, stats in eval_stats.items():
        logger.info(f"  {metric.upper()}:")
        logger.info(f"    Mean: {stats['mean']:.3f}")
        logger.info(f"    Std:  {stats['std']:.3f}")
        logger.info(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    
    if results['traits']:
        logger.info("\n" + "-"*70)
        logger.info("Psychometric Trait Analysis")
        logger.info("-"*70)
        
        trait_stats = results['traits']
        logger.info("\nAverage Personality Traits:")
        for trait, stats in trait_stats.items():
            logger.info(f"  {trait.upper()}:")
            logger.info(f"    Mean: {stats['mean']:.3f}")
            logger.info(f"    Std:  {stats['std']:.3f}")
    
    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved to: {output_path}")
    
    logger.info("\n" + "="*70)
    logger.info("Evaluation complete!")
    logger.info("="*70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate coTherapist model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to test data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation_results.json",
        help="Path to save evaluation results"
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        config_path=args.config,
        test_data_path=args.test_data,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
