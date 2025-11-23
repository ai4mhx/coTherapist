"""
Psychometric Traits Analysis for coTherapist.

Analyzes personality traits in model responses, particularly:
- Agreeableness
- Conscientiousness
- Emotional Stability
- Openness
- Extraversion
"""

import logging
from typing import Dict, Any, List
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TraitScores:
    """Container for Big Five personality trait scores."""
    agreeableness: float
    conscientiousness: float
    emotional_stability: float
    openness: float
    extraversion: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'Agreeableness': self.agreeableness,
            'Conscientiousness': self.conscientiousness,
            'Emotional_Stability': self.emotional_stability,
            'Openness': self.openness,
            'Extraversion': self.extraversion
        }


class TraitsAnalyzer:
    """
    Psychometric personality traits analyzer for AI responses.
    
    Analyzes responses to determine Big Five personality traits,
    with focus on therapeutic qualities like Agreeableness and Conscientiousness.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize traits analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config['psychometric']
        self.enabled = self.config['enabled']
        self.traits = self.config['traits']
        
        # Define trait indicators
        self._setup_trait_indicators()
        
        logger.info("Initialized TraitsAnalyzer")
        
    def _setup_trait_indicators(self):
        """Set up linguistic indicators for each trait."""
        
        # Agreeableness: Compassionate, cooperative, trusting
        self.agreeableness_indicators = {
            'positive': [
                'understand', 'agree', 'together', 'support', 'help',
                'care', 'kindness', 'compassion', 'empathy', 'warm',
                'appreciate', 'thank', 'sorry', 'apologize', 'valid'
            ],
            'negative': [
                'disagree', 'wrong', 'shouldn\'t', 'must not', 'refuse',
                'reject', 'oppose', 'conflict', 'argue'
            ]
        }
        
        # Conscientiousness: Organized, responsible, goal-directed
        self.conscientiousness_indicators = {
            'positive': [
                'plan', 'organize', 'structure', 'goal', 'step', 'strategy',
                'careful', 'consider', 'think about', 'prepare', 'practice',
                'consistent', 'regular', 'routine', 'schedule', 'systematic'
            ],
            'negative': [
                'random', 'whenever', 'spontaneous', 'impulsive', 'careless'
            ]
        }
        
        # Emotional Stability: Calm, even-tempered, resilient
        self.emotional_stability_indicators = {
            'positive': [
                'calm', 'stable', 'peace', 'balance', 'manage', 'cope',
                'resilient', 'handle', 'steady', 'composed', 'grounded',
                'center', 'breathe', 'relax', 'regulate'
            ],
            'negative': [
                'panic', 'anxious', 'overwhelm', 'crisis', 'disaster',
                'terrible', 'awful', 'catastrophe'
            ]
        }
        
        # Openness: Creative, curious, open to new ideas
        self.openness_indicators = {
            'positive': [
                'explore', 'curious', 'wonder', 'imagine', 'creative',
                'new', 'different', 'perspective', 'possibility', 'alternative',
                'consider', 'think about', 'reflect', 'insight', 'learn'
            ],
            'negative': [
                'always', 'never', 'only way', 'must', 'rigid', 'fixed'
            ]
        }
        
        # Extraversion: Sociable, energetic, talkative
        self.extraversion_indicators = {
            'positive': [
                'together', 'social', 'connect', 'reach out', 'talk to',
                'share', 'express', 'communicate', 'engage', 'interact',
                'group', 'people', 'friends', 'others'
            ],
            'negative': [
                'alone', 'isolate', 'withdraw', 'avoid', 'private', 'quiet'
            ]
        }
    
    def analyze_response(self, response: str) -> TraitScores:
        """
        Analyze a response for personality trait indicators.
        
        Args:
            response: Text to analyze
            
        Returns:
            TraitScores object with normalized scores (0-1)
        """
        if not self.enabled:
            return TraitScores(0.5, 0.5, 0.5, 0.5, 0.5)
        
        response_lower = response.lower()
        
        scores = {}
        
        # Analyze each trait
        if 'Agreeableness' in self.traits:
            scores['agreeableness'] = self._score_trait(
                response_lower,
                self.agreeableness_indicators
            )
        
        if 'Conscientiousness' in self.traits:
            scores['conscientiousness'] = self._score_trait(
                response_lower,
                self.conscientiousness_indicators
            )
        
        if 'Emotional_Stability' in self.traits:
            scores['emotional_stability'] = self._score_trait(
                response_lower,
                self.emotional_stability_indicators
            )
        
        if 'Openness' in self.traits:
            scores['openness'] = self._score_trait(
                response_lower,
                self.openness_indicators
            )
        
        if 'Extraversion' in self.traits:
            scores['extraversion'] = self._score_trait(
                response_lower,
                self.extraversion_indicators
            )
        
        return TraitScores(
            agreeableness=scores.get('agreeableness', 0.5),
            conscientiousness=scores.get('conscientiousness', 0.5),
            emotional_stability=scores.get('emotional_stability', 0.5),
            openness=scores.get('openness', 0.5),
            extraversion=scores.get('extraversion', 0.5)
        )
    
    def _score_trait(
        self,
        text: str,
        indicators: Dict[str, List[str]]
    ) -> float:
        """
        Score a single trait based on linguistic indicators.
        
        Args:
            text: Text to analyze (already lowercased)
            indicators: Dictionary with 'positive' and 'negative' indicator lists
            
        Returns:
            Normalized score (0-1), where 0.5 is neutral
        """
        positive_count = sum(
            1 for indicator in indicators['positive']
            if indicator in text
        )
        
        negative_count = sum(
            1 for indicator in indicators['negative']
            if indicator in text
        )
        
        # Calculate raw score
        raw_score = positive_count - negative_count
        
        # Normalize to 0-1 range
        # Using sigmoid-like transformation
        max_expected = 10  # Maximum expected count
        normalized = 0.5 + (raw_score / (2 * max_expected))
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, normalized))
    
    def batch_analyze(
        self,
        responses: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze multiple responses and return aggregate statistics.
        
        Args:
            responses: List of response texts
            
        Returns:
            Dictionary with mean scores and statistics
        """
        all_scores = [self.analyze_response(resp) for resp in responses]
        
        traits = ['agreeableness', 'conscientiousness', 'emotional_stability',
                 'openness', 'extraversion']
        
        stats = {}
        for trait in traits:
            if trait.replace('_', ' ').title().replace(' ', '_') in self.traits:
                values = [getattr(score, trait) for score in all_scores]
                stats[trait] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return stats
    
    def interpret_scores(self, scores: TraitScores) -> Dict[str, str]:
        """
        Provide interpretations of trait scores.
        
        Args:
            scores: TraitScores object
            
        Returns:
            Dictionary mapping traits to interpretations
        """
        interpretations = {}
        
        def interpret_value(value: float) -> str:
            if value >= 0.7:
                return "High"
            elif value >= 0.55:
                return "Moderately High"
            elif value >= 0.45:
                return "Moderate"
            elif value >= 0.3:
                return "Moderately Low"
            else:
                return "Low"
        
        score_dict = scores.to_dict()
        for trait, value in score_dict.items():
            interpretations[trait] = interpret_value(value)
        
        return interpretations
    
    def therapeutic_profile(self, scores: TraitScores) -> str:
        """
        Generate a therapeutic behavior profile based on trait scores.
        
        Args:
            scores: TraitScores object
            
        Returns:
            Text description of therapeutic characteristics
        """
        profile = []
        
        # Agreeableness
        if scores.agreeableness >= 0.6:
            profile.append("Demonstrates high empathy and supportiveness characteristic of effective therapeutic communication.")
        
        # Conscientiousness
        if scores.conscientiousness >= 0.6:
            profile.append("Shows structured and organized approach, providing clear guidance and actionable strategies.")
        
        # Emotional Stability
        if scores.emotional_stability >= 0.6:
            profile.append("Maintains calm and balanced tone, modeling emotional regulation.")
        
        # Openness
        if scores.openness >= 0.5:
            profile.append("Encourages exploration of different perspectives and possibilities.")
        
        # Combined therapeutic qualities
        if scores.agreeableness >= 0.6 and scores.conscientiousness >= 0.6:
            profile.append("Exhibits expert-like therapeutic behavior combining compassion with professional structure.")
        
        if not profile:
            profile.append("Displays moderate therapeutic characteristics.")
        
        return " ".join(profile)
