"""
Basic usage examples for coTherapist.

This script demonstrates the key features of coTherapist:
1. Basic response generation
2. Response with detailed evaluation
3. RAG integration
4. Safety mechanisms
5. Batch evaluation
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coTherapist.pipeline import CoTherapistPipeline


def example_1_basic_response():
    """Example 1: Generate a basic therapeutic response."""
    print("\n" + "="*70)
    print("Example 1: Basic Therapeutic Response")
    print("="*70 + "\n")
    
    # Initialize pipeline
    pipeline = CoTherapistPipeline()
    pipeline.setup(load_model=True, load_knowledge_base=False)
    
    # User input
    user_input = "I've been feeling really anxious about my new job."
    
    # Generate response
    result = pipeline.generate_response(user_input)
    
    print(f"User: {user_input}")
    print(f"\ncoTherapist: {result['response']}")
    print(f"\nSafety Check: {'✓ Safe' if result['safe'] else '✗ Unsafe'}")


def example_2_detailed_analysis():
    """Example 2: Response with detailed COTHERF evaluation and traits."""
    print("\n" + "="*70)
    print("Example 2: Detailed Analysis with COTHERF")
    print("="*70 + "\n")
    
    pipeline = CoTherapistPipeline()
    pipeline.setup(load_model=True, load_knowledge_base=False)
    
    user_input = "I can't stop thinking about a mistake I made at work last week."
    
    # Generate with detailed analysis
    result = pipeline.generate_response(user_input, return_details=True)
    
    print(f"User: {user_input}")
    print(f"\ncoTherapist: {result['response']}")
    
    # Show COTHERF scores
    if 'details' in result and result['details']['evaluation']:
        print("\n--- COTHERF Evaluation Scores ---")
        for metric, score in result['details']['evaluation'].items():
            print(f"  {metric.capitalize()}: {score:.3f}")
    
    # Show personality traits
    if 'details' in result and result['details']['traits']:
        print("\n--- Personality Trait Scores ---")
        for trait, score in result['details']['traits'].items():
            print(f"  {trait}: {score:.3f}")
    
    # Show therapeutic profile
    if 'details' in result and result['details']['therapeutic_profile']:
        print("\n--- Therapeutic Profile ---")
        print(f"  {result['details']['therapeutic_profile']}")


def example_3_crisis_detection():
    """Example 3: Safety mechanism and crisis detection."""
    print("\n" + "="*70)
    print("Example 3: Crisis Detection and Safety")
    print("="*70 + "\n")
    
    pipeline = CoTherapistPipeline()
    pipeline.setup(load_model=True, load_knowledge_base=False)
    
    # Input with crisis indicators
    user_input = "I don't want to live anymore."
    
    result = pipeline.generate_response(user_input, return_details=True)
    
    print(f"User: {user_input}")
    print(f"\ncoTherapist: {result['response']}")
    print(f"\nCrisis Detected: {'✓ Yes' if result.get('crisis_detected') else '✗ No'}")
    
    if result.get('crisis_detected'):
        print("\n⚠️  Emergency resources were provided in the response.")


def example_4_batch_evaluation():
    """Example 4: Batch evaluation on multiple examples."""
    print("\n" + "="*70)
    print("Example 4: Batch Evaluation")
    print("="*70 + "\n")
    
    pipeline = CoTherapistPipeline()
    pipeline.setup(load_model=True, load_knowledge_base=False)
    
    # Test examples
    test_cases = [
        {"user": "I'm feeling stressed about school."},
        {"user": "I had a fight with my best friend."},
        {"user": "I'm worried about my health."},
    ]
    
    print("Evaluating multiple examples...\n")
    results = pipeline.evaluate_dataset(test_cases)
    
    print("--- Aggregate COTHERF Scores ---")
    for metric, stats in results['evaluation'].items():
        print(f"{metric.upper()}: Mean={stats['mean']:.3f}, Std={stats['std']:.3f}")
    
    if results['traits']:
        print("\n--- Aggregate Personality Traits ---")
        for trait, stats in results['traits'].items():
            print(f"{trait.upper()}: Mean={stats['mean']:.3f}, Std={stats['std']:.3f}")


def example_5_custom_configuration():
    """Example 5: Using custom configuration."""
    print("\n" + "="*70)
    print("Example 5: Custom Configuration")
    print("="*70 + "\n")
    
    # Load with custom config (if available)
    pipeline = CoTherapistPipeline()
    
    # Modify config on the fly
    pipeline.config['model']['temperature'] = 0.5  # More deterministic
    pipeline.config['agentic']['enabled'] = False   # Disable reasoning for speed
    
    pipeline.setup(load_model=True, load_knowledge_base=False)
    
    user_input = "What are some ways to manage stress?"
    result = pipeline.generate_response(user_input)
    
    print(f"User: {user_input}")
    print(f"\ncoTherapist: {result['response']}")
    print("\n(Generated with temperature=0.5, no agentic reasoning)")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("coTherapist - Usage Examples")
    print("="*70)
    print("\nThese examples demonstrate key features of coTherapist.")
    print("Note: First-time setup will download the base model (takes time).\n")
    
    examples = [
        ("Basic Response", example_1_basic_response),
        ("Detailed Analysis", example_2_detailed_analysis),
        ("Crisis Detection", example_3_crisis_detection),
        ("Batch Evaluation", example_4_batch_evaluation),
        ("Custom Configuration", example_5_custom_configuration),
    ]
    
    print("Available examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRun specific example: python basic_usage.py <number>")
    print("Run all examples: python basic_usage.py all\n")
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "all":
            for name, func in examples:
                try:
                    func()
                except Exception as e:
                    print(f"\n⚠️  Example '{name}' failed: {e}")
        else:
            try:
                idx = int(sys.argv[1]) - 1
                if 0 <= idx < len(examples):
                    examples[idx][1]()
                else:
                    print(f"Invalid example number. Choose 1-{len(examples)}")
            except ValueError:
                print("Invalid argument. Use a number or 'all'")
    else:
        print("Please specify an example to run.")


if __name__ == "__main__":
    main()
