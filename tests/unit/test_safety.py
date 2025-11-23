"""Unit tests for safety filter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from coTherapist.safety.safety_filter import SafetyFilter
from coTherapist.utils.config_loader import load_config


def test_crisis_detection():
    """Test crisis keyword detection."""
    config = load_config()
    safety = SafetyFilter(config)
    
    # Test suicide-related inputs
    is_safe, msg, details = safety.check_input("I want to kill myself")
    assert not is_safe
    assert details['crisis_detected']
    assert 'suicide' in details.get('crisis_type', '').lower() or 'crisis' in details.get('crisis_type', '').lower()
    
    # Test safe input
    is_safe, msg, details = safety.check_input("I'm feeling a bit sad today")
    assert is_safe


def test_safe_input():
    """Test that safe inputs pass through."""
    config = load_config()
    safety = SafetyFilter(config)
    
    safe_messages = [
        "I'm feeling anxious about my presentation.",
        "Can you help me manage stress?",
        "I had a disagreement with my friend.",
    ]
    
    for msg in safe_messages:
        is_safe, warning, details = safety.check_input(msg)
        assert is_safe, f"False positive for: {msg}"


def test_emergency_resources():
    """Test emergency resources generation."""
    config = load_config()
    safety = SafetyFilter(config)
    
    resources = safety.get_emergency_resources()
    assert "988" in resources or "1-800-273-8255" in resources
    assert "crisis" in resources.lower() or "suicide" in resources.lower()


def test_safety_prefix():
    """Test safety prefix addition."""
    config = load_config()
    safety = SafetyFilter(config)
    
    # Response with advice should get prefix
    response = "You should try meditation daily."
    prefixed = safety.add_safety_prefix(response)
    assert "assistant" in prefixed.lower() or "not a licensed" in prefixed.lower() or "AI" in prefixed


if __name__ == "__main__":
    print("Running safety filter tests...")
    test_crisis_detection()
    print("✓ Crisis detection test passed")
    
    test_safe_input()
    print("✓ Safe input test passed")
    
    test_emergency_resources()
    print("✓ Emergency resources test passed")
    
    test_safety_prefix()
    print("✓ Safety prefix test passed")
    
    print("\nAll tests passed!")
