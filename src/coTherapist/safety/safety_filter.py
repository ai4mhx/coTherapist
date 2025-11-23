"""
Safety filtering system for coTherapist.
"""

import logging
from typing import Dict, Any, Tuple, Optional
import re

logger = logging.getLogger(__name__)


class SafetyFilter:
    """
    Safety system for detecting harmful content and crisis situations.
    
    Features:
    - Crisis keyword detection
    - Toxicity filtering
    - Inappropriate content blocking
    - Emergency response triggering
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the safety filter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config['safety']
        self.enabled = self.config['enabled']
        self.toxicity_threshold = self.config['toxicity_threshold']
        self.crisis_keywords = self.config['crisis_keywords']
        self.emergency_response = self.config['emergency_response']
        self.block_harmful = self.config['block_harmful_content']
        self.crisis_detection = self.config['crisis_detection']
        
        # Try to load detoxify model if available
        self.toxicity_model = None
        if self.enabled:
            try:
                from detoxify import Detoxify
                self.toxicity_model = Detoxify('original')
                logger.info("Loaded toxicity detection model")
            except ImportError:
                logger.warning("Detoxify not available, using rule-based filtering only")
        
        logger.info("Initialized SafetyFilter")
        
    def check_input(self, text: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Check user input for safety concerns.
        
        Args:
            text: User input text
            
        Returns:
            Tuple of (is_safe, warning_message, details)
        """
        if not self.enabled:
            return True, None, {}
        
        details = {}
        
        # Check for crisis indicators
        if self.crisis_detection:
            is_crisis, crisis_type = self._detect_crisis(text)
            details['crisis_detected'] = is_crisis
            details['crisis_type'] = crisis_type
            
            if is_crisis:
                logger.warning(f"Crisis detected: {crisis_type}")
                return False, self.emergency_response, details
        
        # Check for toxicity
        if self.toxicity_model:
            toxicity_score = self._check_toxicity(text)
            details['toxicity_score'] = toxicity_score
            
            if toxicity_score > self.toxicity_threshold:
                logger.warning(f"High toxicity detected: {toxicity_score:.2f}")
                return False, "I'm here to provide support in a respectful way. Let's keep our conversation constructive.", details
        
        return True, None, details
    
    def check_output(self, text: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Check model output for safety concerns.
        
        Args:
            text: Model-generated text
            
        Returns:
            Tuple of (is_safe, replacement_message, details)
        """
        if not self.enabled:
            return True, None, {}
        
        details = {}
        
        # Check for inappropriate medical advice
        medical_red_flags = [
            'diagnose', 'diagnosis', 'you have', 'you are',
            'prescribe', 'medication dosage', 'stop taking'
        ]
        
        text_lower = text.lower()
        for flag in medical_red_flags:
            if flag in text_lower:
                details['medical_advice_flag'] = flag
                logger.warning(f"Potential inappropriate medical advice detected: {flag}")
                # Don't block, but log for review
        
        # Check output toxicity
        if self.toxicity_model:
            toxicity_score = self._check_toxicity(text)
            details['toxicity_score'] = toxicity_score
            
            if toxicity_score > self.toxicity_threshold:
                logger.warning(f"Model generated toxic content: {toxicity_score:.2f}")
                return False, "I apologize, but I need to rephrase my response. How can I better support you?", details
        
        return True, None, details
    
    def _detect_crisis(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Detect crisis situations in text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (is_crisis, crisis_type)
        """
        text_lower = text.lower()
        
        # Patterns for different crisis types
        suicide_patterns = [
            r'\b(kill|end|take)\s+(my|own)\s+life\b',
            r'\b(commit|committing)\s+suicide\b',
            r'\bsuicid(e|al)\b',
            r'\bwant\s+to\s+die\b',
            r'\bdon\'?t\s+want\s+to\s+live\b',
        ]
        
        self_harm_patterns = [
            r'\b(cut|cutting|hurt|harm)\s+(myself|my)\b',
            r'\bself[\s-]harm\b',
            r'\b(cutting|burning|hitting)\s+myself\b',
        ]
        
        # Check for suicide risk
        for pattern in suicide_patterns:
            if re.search(pattern, text_lower):
                return True, "suicide_risk"
        
        # Check for self-harm
        for pattern in self_harm_patterns:
            if re.search(pattern, text_lower):
                return True, "self_harm_risk"
        
        # Check simple keyword matches
        for keyword in self.crisis_keywords:
            if keyword.lower() in text_lower:
                return True, "crisis_keyword_match"
        
        return False, None
    
    def _check_toxicity(self, text: str) -> float:
        """
        Check text toxicity using the detoxify model.
        
        Args:
            text: Input text
            
        Returns:
            Toxicity score (0-1)
        """
        if not self.toxicity_model:
            return 0.0
        
        try:
            results = self.toxicity_model.predict(text)
            # Return the maximum toxicity score across all categories
            return max(results.values())
        except Exception as e:
            logger.error(f"Error checking toxicity: {e}")
            return 0.0
    
    def add_safety_prefix(self, response: str) -> str:
        """
        Add safety disclaimers to responses when appropriate.
        
        Args:
            response: Model response
            
        Returns:
            Response with safety prefix if needed
        """
        # Check if response is giving advice
        advice_indicators = ['should', 'must', 'need to', 'have to', 'recommend']
        if any(indicator in response.lower() for indicator in advice_indicators):
            prefix = "Please note: I'm an AI assistant, not a licensed therapist. "
            if not response.startswith(prefix):
                return prefix + response
        
        return response
    
    def get_emergency_resources(self) -> str:
        """
        Get emergency resource information.
        
        Returns:
            Formatted emergency resources string
        """
        resources = """
🆘 EMERGENCY RESOURCES:

If you're in immediate danger:
• Call 911 (US) or your local emergency number

Crisis Support:
• National Suicide Prevention Lifeline: 988 or 1-800-273-8255
• Crisis Text Line: Text HOME to 741741
• International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/

Remember: You don't have to face this alone. Professional help is available 24/7.
"""
        return resources.strip()
