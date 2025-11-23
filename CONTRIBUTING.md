# Contributing to coTherapist

Thank you for your interest in contributing to coTherapist! This project aims to advance AI-assisted mental healthcare through research and ethical development.

## Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/coTherapist.git
   cd coTherapist
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

4. **Run Tests**
   ```bash
   # Run unit tests
   python tests/unit/test_safety.py
   python tests/unit/test_evaluation.py
   python tests/unit/test_psychometric.py
   ```

## Areas for Contribution

### 1. Training Data
- **Therapeutic Conversations**: Collect or curate ethical training data
- **Diverse Scenarios**: Various mental health concerns and contexts
- **Quality Annotations**: COTHERF scores, trait labels
- **Format**: JSON with user-assistant pairs
- **Ethical Review**: Ensure privacy and consent

### 2. Knowledge Base
- **Evidence-Based Content**: Peer-reviewed therapeutic techniques
- **Coping Strategies**: Practical, validated interventions
- **Mental Health Education**: Accurate, accessible information
- **Crisis Resources**: Updated helpline and resource information
- **Format**: Text files with proper citations

### 3. Evaluation Metrics
- **COTHERF Extensions**: New therapeutic quality dimensions
- **Automated Metrics**: ML-based evaluation approaches
- **Human Evaluation**: Expert therapist assessment protocols
- **Benchmarking**: Standardized test sets

### 4. Safety Improvements
- **Crisis Detection**: Enhanced pattern recognition
- **Toxicity Filtering**: Better harmful content detection
- **Bias Detection**: Identify and mitigate biases
- **Privacy Protection**: Enhanced data handling

### 5. Model Improvements
- **Fine-tuning Strategies**: Better training approaches
- **Prompt Engineering**: Improved system prompts
- **Multi-turn Context**: Conversation history management
- **Personalization**: User-adaptive responses

### 6. Features
- **Multi-language Support**: Translation and localization
- **Voice Interface**: Speech-to-text/text-to-speech
- **Accessibility**: Screen reader compatibility, etc.
- **Integration APIs**: Teletherapy platform connections

### 7. Documentation
- **Tutorials**: Step-by-step guides
- **Use Cases**: Real-world examples
- **Research Papers**: Academic publications
- **Blog Posts**: Educational content

## Contribution Guidelines

### Code Quality

1. **Follow PEP 8**: Python style guide
2. **Add Docstrings**: Document functions and classes
3. **Type Hints**: Use type annotations where applicable
4. **Comments**: Explain complex logic
5. **Tests**: Add tests for new features

### Git Workflow

1. **Create Branch**: `git checkout -b feature/your-feature`
2. **Make Changes**: Implement your contribution
3. **Test**: Ensure all tests pass
4. **Commit**: Clear, descriptive commit messages
5. **Push**: `git push origin feature/your-feature`
6. **Pull Request**: Submit PR with description

### Commit Messages

Format: `[type] Brief description`

Types:
- `[feat]` New feature
- `[fix]` Bug fix
- `[docs]` Documentation
- `[test]` Tests
- `[refactor]` Code restructuring
- `[safety]` Safety improvements

Example:
```
[feat] Add multi-language support for Spanish
[fix] Correct crisis detection false positives
[safety] Improve toxicity filtering threshold
```

### Pull Request Process

1. **Describe Changes**: What and why
2. **Link Issues**: Reference related issues
3. **Test Results**: Show tests passing
4. **Documentation**: Update relevant docs
5. **Review**: Address reviewer feedback

## Ethical Guidelines

### Mandatory Principles

1. **Do No Harm**: Prioritize user safety
2. **Privacy**: Never expose personal information
3. **Transparency**: Clear about AI limitations
4. **Professional Boundaries**: No diagnosis/treatment
5. **Crisis Protocol**: Immediate referral to professionals
6. **Informed Consent**: Users know they're talking to AI
7. **Accessibility**: Ensure equitable access
8. **Cultural Sensitivity**: Respect diverse backgrounds

### Research Ethics

- **IRB Approval**: Required for human subject research
- **Data Ethics**: Proper consent and anonymization
- **Bias Testing**: Regular audits for fairness
- **Responsible Disclosure**: Report safety issues

## Testing Requirements

### Unit Tests

Test individual components:
```python
def test_safety_filter():
    """Test crisis detection."""
    safety = SafetyFilter(config)
    is_safe, msg, details = safety.check_input("crisis text")
    assert not is_safe
    assert details['crisis_detected']
```

### Integration Tests

Test component interactions:
```python
def test_full_pipeline():
    """Test complete response generation."""
    pipeline = CoTherapistPipeline()
    pipeline.setup()
    result = pipeline.generate_response("test input")
    assert 'response' in result
```

### Evaluation Tests

Test quality metrics:
```python
def test_cotherf_scores():
    """Test COTHERF evaluation."""
    evaluator = COTHERFEvaluator(config)
    score = evaluator.evaluate_response(query, response)
    assert 0 <= score.empathy <= 1
```

## Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Functions have docstrings
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] Safety mechanisms intact
- [ ] No hardcoded secrets
- [ ] Ethical guidelines followed
- [ ] Performance considerations addressed

## Questions?

- **Issues**: [GitHub Issues](https://github.com/ai4mhx/coTherapist/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ai4mhx/coTherapist/discussions)
- **Security**: Report via private disclosure

## Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes
- Academic papers (for significant contributions)

Thank you for contributing to ethical AI in mental healthcare! 💚
