"""
coTherapist: A Mental Healthcare AI Copilot

This package provides a domain-specific fine-tuned LLaMA model with:
- Fine-tuning on therapeutic conversations
- Retrieval augmentation for knowledge enhancement
- Agentic reasoning for complex decision-making
- Safety and empathy mechanisms
- COTHERF evaluation framework
- Psychometric analysis
"""

__version__ = "0.1.0"
__author__ = "ai4mhx"

from .models.therapist_model import CoTherapistModel
from .retrieval.rag_system import RAGSystem
from .agentic.reasoning_agent import ReasoningAgent
from .safety.safety_filter import SafetyFilter
from .evaluation.cotherf import COTHERFEvaluator
from .psychometric.traits_analyzer import TraitsAnalyzer

__all__ = [
    "CoTherapistModel",
    "RAGSystem",
    "ReasoningAgent",
    "SafetyFilter",
    "COTHERFEvaluator",
    "TraitsAnalyzer",
]
