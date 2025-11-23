"""
Main coTherapist Pipeline integrating all components.
"""

import logging
from typing import Dict, Any, Optional, List
from .models.therapist_model import CoTherapistModel
from .retrieval.rag_system import RAGSystem
from .agentic.reasoning_agent import ReasoningAgent
from .safety.safety_filter import SafetyFilter
from .evaluation.cotherf import COTHERFEvaluator
from .psychometric.traits_analyzer import TraitsAnalyzer
from .utils.config_loader import load_config

logger = logging.getLogger(__name__)


class CoTherapistPipeline:
    """
    Complete coTherapist pipeline integrating:
    - LLaMA 3.2-1B fine-tuned model
    - Retrieval augmentation (RAG)
    - Agentic reasoning
    - Safety filtering
    - COTHERF evaluation
    - Psychometric analysis
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the complete pipeline.
        
        Args:
            config_path: Path to configuration file (uses default if None)
        """
        logger.info("Initializing coTherapist Pipeline")
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize components
        self.model = CoTherapistModel(self.config)
        self.rag_system = RAGSystem(self.config)
        self.safety_filter = SafetyFilter(self.config)
        self.evaluator = COTHERFEvaluator(self.config)
        self.traits_analyzer = TraitsAnalyzer(self.config)
        
        # Initialize reasoning agent (will set model after loading)
        self.reasoning_agent = ReasoningAgent(self.config)
        
        self.is_model_loaded = False
        
        logger.info("Pipeline initialized successfully")
    
    def setup(self, load_model: bool = True, load_knowledge_base: bool = True):
        """
        Set up the pipeline by loading models and knowledge base.
        
        Args:
            load_model: Whether to load the language model
            load_knowledge_base: Whether to load the knowledge base for RAG
        """
        logger.info("Setting up pipeline components...")
        
        if load_model:
            logger.info("Loading language model...")
            self.model.load_model()
            self.model.setup_lora()
            self.reasoning_agent.set_model(self.model)
            self.is_model_loaded = True
            logger.info("Model loaded and configured")
        
        if load_knowledge_base and self.config['retrieval']['enabled']:
            kb_path = self.config['retrieval']['knowledge_base_path']
            logger.info(f"Loading knowledge base from {kb_path}...")
            self.rag_system.load_knowledge_base_from_files(kb_path)
            logger.info("Knowledge base loaded")
        
        logger.info("Pipeline setup complete")
    
    def generate_response(
        self,
        user_input: str,
        use_rag: bool = True,
        use_reasoning: bool = True,
        return_details: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a therapeutic response with full pipeline integration.
        
        Args:
            user_input: User's message
            use_rag: Whether to use retrieval augmentation
            use_reasoning: Whether to use agentic reasoning
            return_details: Whether to return detailed information
            
        Returns:
            Dictionary containing response and optional details
        """
        if not self.is_model_loaded:
            raise RuntimeError("Model not loaded. Call setup() first.")
        
        logger.info("Processing user input through pipeline")
        
        # Step 1: Safety check on input
        input_safe, safety_warning, safety_details = self.safety_filter.check_input(user_input)
        
        if not input_safe:
            logger.warning("Safety filter triggered on input")
            return {
                'response': safety_warning,
                'safe': False,
                'crisis_detected': True,
                'details': safety_details if return_details else None
            }
        
        # Step 2: Retrieve relevant context (if RAG enabled)
        context = None
        if use_rag and self.config['retrieval']['enabled']:
            context = self.rag_system.retrieve(user_input)
            logger.info(f"Retrieved {len(context)} context chunks")
        
        # Step 3: Generate response with or without reasoning
        if use_reasoning and self.config['agentic']['enabled']:
            reasoning_result = self.reasoning_agent.reason(user_input, context)
            response = reasoning_result['final_response']
            reasoning_steps = reasoning_result['steps']
        else:
            response = self.model.generate_response(user_input, context)
            reasoning_steps = []
        
        # Step 4: Safety check on output
        output_safe, replacement, output_safety = self.safety_filter.check_output(response)
        
        if not output_safe:
            logger.warning("Safety filter triggered on output")
            response = replacement
        
        # Step 5: Add safety prefix if needed
        response = self.safety_filter.add_safety_prefix(response)
        
        # Step 6: Evaluate response (if requested)
        evaluation = None
        if return_details:
            eval_context = {
                'safety_details': {**safety_details, **output_safety},
                'reasoning_steps': reasoning_steps
            }
            evaluation = self.evaluator.evaluate_response(
                user_input,
                response,
                eval_context
            )
        
        # Step 7: Analyze personality traits (if requested)
        traits = None
        if return_details:
            traits = self.traits_analyzer.analyze_response(response)
        
        result = {
            'response': response,
            'safe': input_safe and output_safe,
            'crisis_detected': safety_details.get('crisis_detected', False)
        }
        
        if return_details:
            result['details'] = {
                'context_used': context,
                'reasoning_steps': reasoning_steps,
                'safety_checks': {**safety_details, **output_safety},
                'evaluation': evaluation.to_dict() if evaluation else None,
                'traits': traits.to_dict() if traits else None,
                'therapeutic_profile': self.traits_analyzer.therapeutic_profile(traits) if traits else None
            }
        
        logger.info("Response generated successfully")
        return result
    
    def chat(self, max_turns: int = 10):
        """
        Interactive chat interface for testing.
        
        Args:
            max_turns: Maximum number of conversation turns
        """
        if not self.is_model_loaded:
            print("Setting up pipeline...")
            self.setup()
        
        print("\n" + "="*70)
        print("coTherapist - Mental Healthcare AI Copilot")
        print("="*70)
        print("\nType 'quit' or 'exit' to end the conversation.")
        print("Type 'details' to see detailed analysis of responses.")
        print("\n")
        
        show_details = False
        
        for turn in range(max_turns):
            # Get user input
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("\nThank you for using coTherapist. Take care!")
                break
            
            if user_input.lower() == 'details':
                show_details = not show_details
                print(f"\nDetailed analysis: {'ON' if show_details else 'OFF'}\n")
                continue
            
            if not user_input:
                continue
            
            # Generate response
            result = self.generate_response(
                user_input,
                return_details=show_details
            )
            
            # Display response
            print(f"\ncoTherapist: {result['response']}\n")
            
            # Display details if requested
            if show_details and 'details' in result:
                details = result['details']
                
                if details.get('evaluation'):
                    print("\n--- COTHERF Evaluation ---")
                    for metric, score in details['evaluation'].items():
                        print(f"  {metric}: {score:.2f}")
                
                if details.get('traits'):
                    print("\n--- Personality Traits ---")
                    for trait, score in details['traits'].items():
                        print(f"  {trait}: {score:.2f}")
                
                if details.get('therapeutic_profile'):
                    print(f"\n--- Therapeutic Profile ---")
                    print(f"  {details['therapeutic_profile']}")
                
                print()
    
    def evaluate_dataset(self, test_data: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Evaluate the model on a test dataset.
        
        Args:
            test_data: List of dicts with 'user' and optionally 'assistant' keys
            
        Returns:
            Evaluation statistics
        """
        if not self.is_model_loaded:
            raise RuntimeError("Model not loaded. Call setup() first.")
        
        logger.info(f"Evaluating on {len(test_data)} examples")
        
        evaluations = []
        all_traits = []
        
        for example in test_data:
            result = self.generate_response(
                example['user'],
                return_details=True
            )
            
            evaluations.append({
                'query': example['user'],
                'response': result['response'],
                'context': result['details']
            })
            
            if result['details']['traits']:
                all_traits.append(result['details']['traits'])
        
        # Compute statistics
        eval_stats = self.evaluator.batch_evaluate(evaluations)
        
        # Compute trait statistics
        trait_stats = None
        if all_traits:
            responses = [result['response'] for result in 
                        [self.generate_response(ex['user']) for ex in test_data]]
            trait_stats = self.traits_analyzer.batch_analyze(responses)
        
        return {
            'evaluation': eval_stats,
            'traits': trait_stats,
            'num_examples': len(test_data)
        }
