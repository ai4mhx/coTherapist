"""Unit tests for COTHERF evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from coTherapist.evaluation.cotherf import COTHERFEvaluator
from coTherapist.utils.config_loader import load_config


def test_empathy_evaluation():
    """Test empathy scoring."""
    config = load_config()
    evaluator = COTHERFEvaluator(config)
    
    # High empathy response
    high_empathy = "I understand that you're feeling anxious. That sounds really difficult."
    score = evaluator._evaluate_empathy("I'm anxious", high_empathy)
    assert score > 0.6, f"Expected high empathy score, got {score}"
    
    # Low empathy response
    low_empathy = "Just do it. Stop complaining."
    score = evaluator._evaluate_empathy("I'm anxious", low_empathy)
    assert score < 0.5, f"Expected low empathy score, got {score}"


def test_safety_evaluation():
    """Test safety scoring."""
    config = load_config()
    evaluator = COTHERFEvaluator(config)
    
    # Safe response
    safe_response = "Let's explore some coping strategies together."
    score = evaluator._evaluate_safety(safe_response, None)
    assert score > 0.8, f"Expected high safety score, got {score}"
    
    # Unsafe response (diagnosis)
    unsafe_response = "You definitely have depression. You should take medication."
    score = evaluator._evaluate_safety(unsafe_response, None)
    assert score < 0.8, f"Expected lower safety score, got {score}"


def test_full_evaluation():
    """Test full COTHERF evaluation."""
    config = load_config()
    evaluator = COTHERFEvaluator(config)
    
    query = "I'm feeling stressed about work."
    response = "I hear that work stress is affecting you. That must be challenging. Have you considered some relaxation techniques like deep breathing?"
    
    score = evaluator.evaluate_response(query, response)
    
    # Check all scores are in valid range
    assert 0 <= score.empathy <= 1
    assert 0 <= score.relevance <= 1
    assert 0 <= score.informativeness <= 1
    assert 0 <= score.safety <= 1
    assert 0 <= score.therapeutic_alliance <= 1
    assert 0 <= score.clinical_accuracy <= 1
    assert 0 <= score.overall <= 1


def test_batch_evaluation():
    """Test batch evaluation."""
    config = load_config()
    evaluator = COTHERFEvaluator(config)
    
    evaluations = [
        {
            'query': "I'm anxious",
            'response': "I understand you're feeling anxious.",
            'context': None
        },
        {
            'query': "I'm sad",
            'response': "It's okay to feel sad sometimes.",
            'context': None
        }
    ]
    
    stats = evaluator.batch_evaluate(evaluations)
    
    # Check statistics exist for all metrics
    for metric in ['empathy', 'relevance', 'informativeness', 'safety', 
                   'therapeutic_alliance', 'clinical_accuracy', 'overall']:
        assert metric in stats
        assert 'mean' in stats[metric]
        assert 'std' in stats[metric]


if __name__ == "__main__":
    print("Running COTHERF evaluation tests...")
    
    test_empathy_evaluation()
    print("✓ Empathy evaluation test passed")
    
    test_safety_evaluation()
    print("✓ Safety evaluation test passed")
    
    test_full_evaluation()
    print("✓ Full evaluation test passed")
    
    test_batch_evaluation()
    print("✓ Batch evaluation test passed")
    
    print("\nAll tests passed!")
