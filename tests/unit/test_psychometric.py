"""Unit tests for psychometric analysis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from coTherapist.psychometric.traits_analyzer import TraitsAnalyzer
from coTherapist.utils.config_loader import load_config


def test_agreeableness_scoring():
    """Test agreeableness trait scoring."""
    config = load_config()
    analyzer = TraitsAnalyzer(config)
    
    # High agreeableness response
    high_agree = "I understand and support you. Let's work together to help you."
    scores = analyzer.analyze_response(high_agree)
    assert scores.agreeableness > 0.6, f"Expected high agreeableness, got {scores.agreeableness}"
    
    # Low agreeableness response
    low_agree = "I disagree. You're wrong about this."
    scores = analyzer.analyze_response(low_agree)
    assert scores.agreeableness < 0.6, f"Expected lower agreeableness, got {scores.agreeableness}"


def test_conscientiousness_scoring():
    """Test conscientiousness trait scoring."""
    config = load_config()
    analyzer = TraitsAnalyzer(config)
    
    # High conscientiousness response
    high_consc = "Let's create a structured plan with clear goals. We'll organize steps carefully."
    scores = analyzer.analyze_response(high_consc)
    assert scores.conscientiousness > 0.6, f"Expected high conscientiousness, got {scores.conscientiousness}"


def test_emotional_stability_scoring():
    """Test emotional stability trait scoring."""
    config = load_config()
    analyzer = TraitsAnalyzer(config)
    
    # High emotional stability response
    high_stable = "Let's stay calm and manage this situation together. You can cope with this."
    scores = analyzer.analyze_response(high_stable)
    assert scores.emotional_stability > 0.5, f"Expected moderate-high stability, got {scores.emotional_stability}"


def test_therapeutic_profile():
    """Test therapeutic profile generation."""
    config = load_config()
    analyzer = TraitsAnalyzer(config)
    
    # Therapeutic response
    response = "I understand and support you. Let's create a careful plan to manage this calmly."
    scores = analyzer.analyze_response(response)
    profile = analyzer.therapeutic_profile(scores)
    
    assert isinstance(profile, str)
    assert len(profile) > 0


def test_trait_interpretation():
    """Test trait score interpretation."""
    config = load_config()
    analyzer = TraitsAnalyzer(config)
    
    # Create sample scores
    response = "I understand and care about your situation."
    scores = analyzer.analyze_response(response)
    interpretations = analyzer.interpret_scores(scores)
    
    # Check interpretations exist
    for trait in ['Agreeableness', 'Conscientiousness', 'Emotional_Stability', 
                  'Openness', 'Extraversion']:
        assert trait in interpretations
        assert interpretations[trait] in ['Low', 'Moderately Low', 'Moderate', 
                                          'Moderately High', 'High']


def test_batch_analysis():
    """Test batch trait analysis."""
    config = load_config()
    analyzer = TraitsAnalyzer(config)
    
    responses = [
        "I understand and support you.",
        "Let's plan this carefully and organize our approach.",
        "Stay calm and manage this situation."
    ]
    
    stats = analyzer.batch_analyze(responses)
    
    # Check statistics exist
    for trait in ['agreeableness', 'conscientiousness', 'emotional_stability',
                  'openness', 'extraversion']:
        assert trait in stats
        assert 'mean' in stats[trait]


if __name__ == "__main__":
    print("Running psychometric analysis tests...")
    
    test_agreeableness_scoring()
    print("✓ Agreeableness scoring test passed")
    
    test_conscientiousness_scoring()
    print("✓ Conscientiousness scoring test passed")
    
    test_emotional_stability_scoring()
    print("✓ Emotional stability scoring test passed")
    
    test_therapeutic_profile()
    print("✓ Therapeutic profile test passed")
    
    test_trait_interpretation()
    print("✓ Trait interpretation test passed")
    
    test_batch_analysis()
    print("✓ Batch analysis test passed")
    
    print("\nAll tests passed!")
