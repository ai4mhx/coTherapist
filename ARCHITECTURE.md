# coTherapist Architecture

## System Overview

coTherapist is a comprehensive mental healthcare AI system that combines multiple advanced techniques to provide empathetic, safe, and therapeutically sound responses.

## Core Components

### 1. CoTherapistModel (`src/coTherapist/models/therapist_model.py`)

**Purpose**: Fine-tuned LLaMA 3.2-1B-Instruct for therapeutic conversations

**Key Features**:
- Parameter-efficient fine-tuning with LoRA (Low-Rank Adaptation)
- 4-bit quantization using bitsandbytes for memory efficiency
- Specialized prompt formatting for therapeutic contexts
- Maintains small model size while achieving expert-like behavior

**Technical Details**:
- Base Model: meta-llama/Llama-3.2-1B-Instruct
- LoRA Configuration: r=16, alpha=32, dropout=0.05
- Target Modules: Query, Key, Value projections + MLP layers
- Quantization: 4-bit NF4 with nested quantization

### 2. RAGSystem (`src/coTherapist/retrieval/rag_system.py`)

**Purpose**: Retrieval Augmented Generation for knowledge enhancement

**Key Features**:
- Semantic search over therapeutic knowledge base
- Context-aware response augmentation
- Efficient vector storage with FAISS
- Configurable similarity thresholds

**Technical Details**:
- Embedding Model: sentence-transformers/all-MiniLM-L6-v2
- Vector Store: FAISS (Facebook AI Similarity Search)
- Chunk Size: 512 tokens with 50-token overlap
- Top-K Retrieval: Configurable (default: 3)

**Workflow**:
1. User query → Embed with sentence-transformer
2. Search FAISS index for similar chunks
3. Filter by similarity threshold
4. Return top-k relevant contexts
5. Integrate contexts into model prompt

### 3. ReasoningAgent (`src/coTherapist/agentic/reasoning_agent.py`)

**Purpose**: Multi-step reasoning and decision-making

**Key Features**:
- Chain-of-thought reasoning
- Situation analysis
- Therapeutic needs assessment
- Self-critique and refinement
- Response reflection

**Reasoning Pipeline**:
```
User Query
    ↓
Situation Analysis (CoT)
    ↓
Needs Assessment (rule-based + model)
    ↓
Initial Response Generation
    ↓
Self-Critique
    ↓
Refinement (if needed)
    ↓
Reflection
    ↓
Final Response
```

### 4. SafetyFilter (`src/coTherapist/safety/safety_filter.py`)

**Purpose**: Content safety and crisis detection

**Key Features**:
- Crisis keyword detection (suicide, self-harm)
- Pattern-based crisis identification
- Toxicity scoring with Detoxify
- Emergency resource provision
- Medical advice boundary maintenance

**Safety Checks**:
- **Input Safety**: Crisis detection, toxicity
- **Output Safety**: Inappropriate advice, toxicity
- **Crisis Response**: Immediate emergency resources

**Crisis Keywords**:
- Suicide indicators: "kill myself", "end my life", "suicide"
- Self-harm indicators: "hurt myself", "self-harm", "cutting"

### 5. COTHERFEvaluator (`src/coTherapist/evaluation/cotherf.py`)

**Purpose**: Comprehensive Therapeutic Evaluation and Rating Framework

**Six Evaluation Dimensions**:

1. **Empathy** (25% weight)
   - Emotional understanding
   - Validation phrases
   - Reflective listening
   - Non-judgmental language

2. **Relevance** (15% weight)
   - Topic alignment
   - Concern addressing
   - Contextual appropriateness

3. **Informativeness** (15% weight)
   - Content richness
   - Actionable suggestions
   - Educational value

4. **Safety** (25% weight)
   - Crisis handling
   - Boundary maintenance
   - Absence of harmful content

5. **Therapeutic Alliance** (10% weight)
   - Warmth and rapport
   - Collaborative language
   - Trust building

6. **Clinical Accuracy** (10% weight)
   - Evidence-based content
   - Theoretical soundness
   - Professional boundaries

**Scoring**: Each dimension scored 0-1, weighted average for overall score

### 6. TraitsAnalyzer (`src/coTherapist/psychometric/traits_analyzer.py`)

**Purpose**: Personality trait analysis (Big Five)

**Analyzed Traits**:

1. **Agreeableness**
   - Compassion, cooperation
   - Indicators: "understand", "support", "help", "care"

2. **Conscientiousness**
   - Organization, responsibility
   - Indicators: "plan", "structure", "goal", "careful"

3. **Emotional Stability**
   - Calm, resilience
   - Indicators: "calm", "manage", "cope", "balance"

4. **Openness**
   - Creativity, curiosity
   - Indicators: "explore", "curious", "perspective", "alternative"

5. **Extraversion**
   - Social engagement
   - Indicators: "connect", "share", "interact", "together"

**Scoring Method**:
- Linguistic indicator counting
- Positive vs negative indicators
- Normalized to 0-1 scale
- Sigmoid-like transformation

## Integration: CoTherapistPipeline

The `CoTherapistPipeline` class integrates all components into a cohesive system.

### Processing Flow

```
1. User Input
   ↓
2. Safety Check (Input)
   - Crisis detection
   - Toxicity check
   ↓
3. RAG Retrieval (if enabled)
   - Query embedding
   - Relevant context retrieval
   ↓
4. Agentic Reasoning (if enabled)
   - Multi-step analysis
   - Needs assessment
   - Response generation
   - Self-critique
   ↓
5. Model Generation
   - Prompt formatting
   - LLM inference
   - Response extraction
   ↓
6. Safety Check (Output)
   - Inappropriate content
   - Medical advice flags
   ↓
7. Evaluation (if requested)
   - COTHERF scoring
   - Trait analysis
   ↓
8. Final Response
```

## Training Pipeline

### Fine-Tuning Process

1. **Data Preparation**
   - Load conversation pairs (user-assistant)
   - Format with LLaMA chat template
   - Tokenize with padding/truncation

2. **Model Configuration**
   - Load base LLaMA model with 4-bit quantization
   - Apply LoRA adapters
   - Prepare for k-bit training

3. **Training**
   - HuggingFace Trainer
   - Gradient accumulation
   - Mixed precision (FP16)
   - Checkpoint saving

4. **Evaluation**
   - Validation split evaluation
   - COTHERF metrics
   - Best model selection

### Training Script Usage

```bash
# With sample data
python scripts/train.py --create-sample-data

# With custom data
python scripts/train.py --data path/to/data.json --output models/my_model
```

## Data Formats

### Training Data

```json
[
  {
    "user": "User's message",
    "assistant": "Therapeutic response"
  }
]
```

### Knowledge Base

Text files in `data/knowledge_base/`:
- Therapeutic techniques
- Mental health information
- Evidence-based practices
- Coping strategies

### Evaluation Results

```json
{
  "evaluation": {
    "empathy": {"mean": 0.82, "std": 0.05},
    "safety": {"mean": 0.95, "std": 0.03},
    ...
  },
  "traits": {
    "agreeableness": {"mean": 0.85, "std": 0.04},
    ...
  }
}
```

## Configuration

Centralized in `config/default_config.yaml`:

- **model**: Generation parameters (temperature, top-p, etc.)
- **fine_tuning**: LoRA config, training hyperparameters
- **retrieval**: RAG settings, embedding model
- **agentic**: Reasoning configuration
- **safety**: Crisis keywords, toxicity thresholds
- **evaluation**: COTHERF metrics to compute
- **psychometric**: Traits to analyze

## Performance Considerations

### Memory Optimization
- 4-bit quantization reduces memory by ~75%
- LoRA reduces trainable parameters by ~95%
- Gradient checkpointing for training
- Efficient FAISS indexing

### Speed Optimization
- Optional RAG (can disable for faster inference)
- Optional agentic reasoning (can disable)
- Batch processing support
- GPU acceleration when available

### Quality Optimization
- Chain-of-thought reasoning
- Self-critique mechanism
- Context augmentation
- Safety filtering

## Extending the System

### Adding New Evaluation Metrics

Edit `src/coTherapist/evaluation/cotherf.py`:
```python
def _evaluate_new_metric(self, query, response):
    # Your metric logic
    return score

# Add to evaluate_response method
```

### Adding Knowledge Base Content

Add .txt files to `data/knowledge_base/`:
```python
pipeline.rag_system.load_knowledge_base_from_files("data/knowledge_base")
```

### Custom Trait Analysis

Edit `src/coTherapist/psychometric/traits_analyzer.py`:
```python
self.new_trait_indicators = {
    'positive': [...],
    'negative': [...]
}
```

## Research Applications

1. **Domain-Specific LLM Studies**
   - Small model capabilities
   - Fine-tuning effectiveness
   - Transfer learning in healthcare

2. **Therapeutic AI Research**
   - Empathy modeling
   - Safety mechanisms
   - Quality evaluation

3. **Psychometric Analysis**
   - AI personality traits
   - Behavioral consistency
   - Expert-like qualities

## Ethical Considerations

- **Not a Replacement**: Clearly positioned as support tool
- **Crisis Handling**: Immediate referral to professionals
- **Boundary Maintenance**: No diagnosis or prescriptions
- **Transparency**: Users know they're interacting with AI
- **Privacy**: No storage of personal conversations
- **Safety First**: Multiple layers of content filtering

## Future Enhancements

- Multi-turn conversation context
- Multi-language support
- Voice interface integration
- Specialized therapeutic modalities (CBT, DBT)
- Integration with teletherapy platforms
- Real-time learning from therapist feedback
- Enhanced knowledge base with research literature
- Improved crisis detection with ML models
