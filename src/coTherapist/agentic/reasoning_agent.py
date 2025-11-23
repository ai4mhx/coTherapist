"""
Agentic reasoning system for coTherapist.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ReasoningAgent:
    """
    Agentic reasoning system that enables multi-step reasoning and reflection.
    
    Features:
    - Chain-of-thought reasoning
    - Self-reflection and critique
    - Multi-step problem solving
    - Context-aware decision making
    """
    
    def __init__(self, config: Dict[str, Any], model=None):
        """
        Initialize the reasoning agent.
        
        Args:
            config: Configuration dictionary
            model: CoTherapistModel instance for generating responses
        """
        self.config = config['agentic']
        self.model = model
        self.max_reasoning_steps = self.config['max_reasoning_steps']
        self.reflection_enabled = self.config['reflection_enabled']
        self.self_critique_enabled = self.config['self_critique_enabled']
        self.chain_of_thought = self.config['chain_of_thought']
        
        logger.info("Initialized ReasoningAgent")
        
    def set_model(self, model):
        """Set the language model for reasoning."""
        self.model = model
        
    def reason(self, query: str, context: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform multi-step reasoning on a query.
        
        Args:
            query: User's query or problem
            context: Optional retrieved context
            
        Returns:
            Dictionary containing reasoning steps and final response
        """
        if not self.config['enabled']:
            # If agentic reasoning is disabled, just return a simple response
            if self.model:
                response = self.model.generate_response(query, context)
                return {
                    'steps': [],
                    'final_response': response,
                    'reasoning_used': False
                }
            return {
                'steps': [],
                'final_response': "",
                'reasoning_used': False
            }
        
        logger.info("Starting agentic reasoning process")
        
        reasoning_steps = []
        current_query = query
        
        # Step 1: Analyze the situation
        if self.chain_of_thought:
            analysis = self._analyze_situation(current_query, context)
            reasoning_steps.append({
                'step': 'situation_analysis',
                'content': analysis
            })
            logger.debug(f"Situation analysis: {analysis[:100]}...")
        
        # Step 2: Identify key emotional and therapeutic needs
        needs_assessment = self._assess_therapeutic_needs(current_query)
        reasoning_steps.append({
            'step': 'needs_assessment',
            'content': needs_assessment
        })
        logger.debug(f"Needs assessment: {needs_assessment[:100]}...")
        
        # Step 3: Generate initial response
        initial_response = self._generate_therapeutic_response(
            current_query, 
            context, 
            needs_assessment
        )
        reasoning_steps.append({
            'step': 'initial_response',
            'content': initial_response
        })
        
        # Step 4: Self-critique and refinement
        if self.self_critique_enabled:
            critique = self._self_critique(initial_response, needs_assessment)
            reasoning_steps.append({
                'step': 'self_critique',
                'content': critique
            })
            logger.debug(f"Self-critique: {critique[:100]}...")
            
            # Refine response based on critique
            if "needs improvement" in critique.lower() or "consider" in critique.lower():
                refined_response = self._refine_response(
                    initial_response,
                    critique,
                    needs_assessment
                )
                reasoning_steps.append({
                    'step': 'refined_response',
                    'content': refined_response
                })
                final_response = refined_response
            else:
                final_response = initial_response
        else:
            final_response = initial_response
        
        # Step 5: Reflection (if enabled)
        if self.reflection_enabled:
            reflection = self._reflect_on_response(final_response, query)
            reasoning_steps.append({
                'step': 'reflection',
                'content': reflection
            })
            logger.debug(f"Reflection: {reflection[:100]}...")
        
        logger.info(f"Completed reasoning with {len(reasoning_steps)} steps")
        
        return {
            'steps': reasoning_steps,
            'final_response': final_response,
            'reasoning_used': True
        }
    
    def _analyze_situation(self, query: str, context: Optional[List[str]]) -> str:
        """Analyze the user's situation using chain-of-thought."""
        analysis_prompt = f"""Analyze this therapeutic situation step by step:

User's message: {query}

Consider:
1. What emotions might the person be experiencing?
2. What are the underlying concerns or needs?
3. Are there any crisis indicators?
4. What would be most helpful right now?

Provide a brief analysis:"""
        
        if self.model:
            return self.model.generate_response(analysis_prompt, context, max_new_tokens=200)
        return "Unable to analyze - model not available"
    
    def _assess_therapeutic_needs(self, query: str) -> str:
        """Assess what the user needs therapeutically."""
        # Rule-based assessment (can be enhanced with model)
        query_lower = query.lower()
        
        needs = []
        
        # Emotional validation
        emotion_words = ['sad', 'angry', 'anxious', 'worried', 'scared', 'frustrated', 'hopeless']
        if any(word in query_lower for word in emotion_words):
            needs.append("emotional_validation")
        
        # Information seeking
        if '?' in query or any(word in query_lower for word in ['how', 'what', 'why', 'when', 'where']):
            needs.append("information")
        
        # Coping strategies
        if any(word in query_lower for word in ['help', 'cope', 'manage', 'deal with']):
            needs.append("coping_strategies")
        
        # Crisis support
        crisis_indicators = ['suicide', 'kill', 'harm', 'hurt myself', 'end it']
        if any(indicator in query_lower for indicator in crisis_indicators):
            needs.append("crisis_support")
        
        # Support and empathy
        if len(needs) == 0 or any(word in query_lower for word in ['feel', 'feeling', 'felt']):
            needs.append("empathy_and_support")
        
        return ", ".join(needs) if needs else "general_support"
    
    def _generate_therapeutic_response(
        self,
        query: str,
        context: Optional[List[str]],
        needs: str
    ) -> str:
        """Generate a therapeutic response based on assessed needs."""
        enhanced_prompt = f"""User's message: {query}

Therapeutic needs identified: {needs}

Provide a compassionate, empathetic response that addresses these needs."""
        
        if self.model:
            return self.model.generate_response(enhanced_prompt, context)
        return "I'm here to support you."
    
    def _self_critique(self, response: str, needs: str) -> str:
        """Critique the generated response for quality and appropriateness."""
        critique_prompt = f"""Evaluate this therapeutic response:

Response: {response}

Needs to address: {needs}

Critique checklist:
- Is it empathetic and validating?
- Does it address the identified needs?
- Is it safe and appropriate?
- Is it actionable if needed?
- Does it maintain boundaries?

Provide brief critique:"""
        
        if self.model:
            return self.model.generate_response(critique_prompt, max_new_tokens=150)
        return "Response appears appropriate."
    
    def _refine_response(self, initial: str, critique: str, needs: str) -> str:
        """Refine the response based on critique."""
        refinement_prompt = f"""Original response: {initial}

Critique: {critique}

Needs: {needs}

Provide an improved response that addresses the critique:"""
        
        if self.model:
            return self.model.generate_response(refinement_prompt, max_new_tokens=300)
        return initial
    
    def _reflect_on_response(self, response: str, original_query: str) -> str:
        """Reflect on the quality of the final response."""
        reflection = []
        
        # Check for empathy indicators
        empathy_words = ['understand', 'hear', 'feel', 'valid', 'makes sense']
        if any(word in response.lower() for word in empathy_words):
            reflection.append("Response shows empathy")
        
        # Check for actionable content
        if any(word in response.lower() for word in ['try', 'consider', 'might', 'could', 'suggest']):
            reflection.append("Provides actionable suggestions")
        
        # Check for appropriate length
        if 50 < len(response.split()) < 200:
            reflection.append("Appropriate length")
        elif len(response.split()) <= 50:
            reflection.append("May be too brief")
        else:
            reflection.append("May be too lengthy")
        
        return "; ".join(reflection) if reflection else "Response generated"
