# Quick Start Guide

Get started with coTherapist in 5 minutes!

## 1. Installation

```bash
# Clone repository
git clone https://github.com/ai4mhx/coTherapist.git
cd coTherapist

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. First Run - Interactive Chat

```bash
# Start interactive chat (model will be downloaded automatically on first run)
python scripts/chat.py
```

**Note**: First run will download LLaMA 3.2-1B-Instruct (~2GB). This takes a few minutes.

## 3. Simple Python Usage

```python
from coTherapist.pipeline import CoTherapistPipeline

# Initialize pipeline
pipeline = CoTherapistPipeline()
pipeline.setup()

# Generate response
result = pipeline.generate_response("I'm feeling anxious about my presentation tomorrow.")
print(result['response'])
```

## 4. Training Your Own Model

### Option A: Use Sample Data

```bash
# Creates sample therapeutic conversations and trains
python scripts/train.py --create-sample-data --output models/my_model
```

### Option B: Use Your Own Data

Prepare data as JSON:
```json
[
  {
    "user": "I'm feeling stressed.",
    "assistant": "I hear that you're feeling stressed. That's completely valid..."
  }
]
```

Train:
```bash
python scripts/train.py --data data/training/my_data.json --output models/my_model
```

## 5. Evaluation

```bash
# Evaluate with COTHERF framework
python scripts/evaluate.py --test-data data/test.json
```

Sample output:
```
COTHERF Evaluation Results
==================================================
  EMPATHY: Mean: 0.823 (High emotional understanding)
  SAFETY: Mean: 0.950 (Excellent crisis handling)
  ...
```

## 6. Advanced Usage - Detailed Analysis

```python
from coTherapist.pipeline import CoTherapistPipeline

pipeline = CoTherapistPipeline()
pipeline.setup()

# Get detailed evaluation and trait analysis
result = pipeline.generate_response(
    "I can't sleep because I'm worried about work.",
    return_details=True
)

print(f"Response: {result['response']}\n")

# COTHERF Scores
print("COTHERF Evaluation:")
for metric, score in result['details']['evaluation'].items():
    print(f"  {metric}: {score:.3f}")

# Personality Traits
print("\nPersonality Traits:")
for trait, score in result['details']['traits'].items():
    print(f"  {trait}: {score:.3f}")

# Therapeutic Profile
print(f"\nProfile: {result['details']['therapeutic_profile']}")
```

## 7. Configuration

Customize behavior via `config/default_config.yaml`:

```yaml
model:
  temperature: 0.7  # Lower = more deterministic (0.5-0.9)
  
agentic:
  enabled: true     # Enable multi-step reasoning
  
retrieval:
  enabled: true     # Enable RAG
  top_k: 3         # Number of context chunks to retrieve
  
safety:
  crisis_detection: true  # Enable crisis detection
```

Or programmatically:

```python
pipeline = CoTherapistPipeline()
pipeline.config['model']['temperature'] = 0.5
pipeline.config['agentic']['enabled'] = False  # Faster responses
pipeline.setup()
```

## 8. Key Features Demo

### Crisis Detection

```python
result = pipeline.generate_response("I don't want to live anymore.")
print(result['response'])  # Will include emergency resources
print(f"Crisis Detected: {result['crisis_detected']}")
```

### RAG Integration

```python
# Add knowledge base documents
pipeline.rag_system.add_documents([
    "Cognitive Behavioral Therapy (CBT) is an evidence-based approach...",
    "Mindfulness meditation has been shown to reduce anxiety..."
])

# Responses will use this knowledge
result = pipeline.generate_response("What is CBT?")
```

### Batch Evaluation

```python
test_cases = [
    {"user": "I'm stressed."},
    {"user": "I had a panic attack."},
    {"user": "I feel lonely."}
]

results = pipeline.evaluate_dataset(test_cases)
print(f"Average Empathy: {results['evaluation']['empathy']['mean']:.3f}")
```

## 9. Common Use Cases

### Use Case 1: Mental Health Chatbot

```python
# Interactive support bot
pipeline = CoTherapistPipeline()
pipeline.setup()

conversation_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit']:
        break
    
    result = pipeline.generate_response(user_input)
    print(f"Bot: {result['response']}\n")
    
    if result['crisis_detected']:
        print("⚠️  Crisis resources provided")
```

### Use Case 2: Training Data Validation

```python
# Validate quality of training responses
evaluator = COTHERFEvaluator(config)

with open('training_data.json') as f:
    data = json.load(f)

for item in data:
    score = evaluator.evaluate_response(item['user'], item['assistant'])
    if score.empathy < 0.5 or score.safety < 0.8:
        print(f"Low quality response: {item['user']}")
```

### Use Case 3: Research on AI Personality

```python
# Analyze AI personality traits
analyzer = TraitsAnalyzer(config)

responses = [
    # ... your model responses
]

traits = analyzer.batch_analyze(responses)
print(f"Model Agreeableness: {traits['agreeableness']['mean']:.3f}")
print(f"Model Conscientiousness: {traits['conscientiousness']['mean']:.3f}")
```

## 10. Troubleshooting

### Out of Memory

```python
# Use CPU instead of GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Or reduce batch size in config
pipeline.config['fine_tuning']['batch_size'] = 2
```

### Slow Inference

```python
# Disable optional features
pipeline.config['agentic']['enabled'] = False  # Disable reasoning
pipeline.config['retrieval']['enabled'] = False  # Disable RAG
```

### Model Download Issues

```bash
# Manually download model
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct

# Or use alternative mirror
export HF_ENDPOINT=https://hf-mirror.com
```

## Next Steps

- 📚 Read [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
- 🚀 See [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment
- 🤝 Check [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- 💡 Explore [examples/basic_usage.py](examples/basic_usage.py) for more examples

## Getting Help

- **Documentation**: Check README.md and other .md files
- **Issues**: [GitHub Issues](https://github.com/ai4mhx/coTherapist/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ai4mhx/coTherapist/discussions)

## Important Reminders

⚠️ **This is a research tool, not a replacement for professional mental healthcare**

- Always include disclaimers when deploying
- Ensure proper crisis response mechanisms
- Have human oversight for production use
- Follow ethical guidelines

🆘 **Crisis Resources**:
- National Suicide Prevention Lifeline: 988 or 1-800-273-8255
- Crisis Text Line: Text HOME to 741741

---

Happy building! 💚 Remember: Mental health matters, and technology should support, not replace, human care.
