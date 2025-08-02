"""
LLM Integration for Pest Management Chatbot
==========================================

Integration with LM Studio and fine-tuning capabilities for agricultural domain.
"""

import os
import logging
import json
import requests
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class LMStudioIntegration:
    """Integration with LM Studio for local LLM inference."""
    
    def __init__(self, api_url: str = None):
        """Initialize LM Studio integration."""
        self.api_url = api_url or os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234/v1')
        self.model_name = "llama-2-7b-chat"  # Default model name
        self.system_prompt = self._create_agriculture_system_prompt()
        
    def _create_agriculture_system_prompt(self) -> str:
        """Create system prompt specialized for organic farming."""
        return """You are an expert organic farm pest management assistant. Your role is to:

1. Help farmers identify pests from descriptions or images
2. Provide ONLY organic, OMRI-approved treatment recommendations
3. Follow Integrated Pest Management (IPM) principles
4. Consider crop safety, beneficial insects, and environmental impact
5. Provide practical, actionable advice for small to medium farms

Guidelines:
- Always prioritize organic solutions
- Consider beneficial insects and pollinators
- Suggest monitoring and prevention strategies
- Provide timing recommendations for treatments
- Explain the reasoning behind recommendations
- Ask clarifying questions when needed

Available pest types: ants, bees, beetles, caterpillars, earthworms, earwigs, grasshoppers, moths, slugs, snails, wasps, weevils

Keep responses practical, farmer-friendly, and focused on sustainable agriculture."""

    def generate_response_with_progress(self, user_message: str, pest_context: Dict[str, Any] = None, progress_callback=None) -> str:
        """Generate response with optional progress callback for UI feedback."""
        import time
        import threading
        
        if progress_callback:
            # Start progress indicator in separate thread
            stop_progress = threading.Event()
            progress_thread = threading.Thread(
                target=self._show_progress, 
                args=(stop_progress, progress_callback)
            )
            progress_thread.start()
        
        try:
            result = self.generate_response(user_message, pest_context)
            return result
        finally:
            if progress_callback:
                stop_progress.set()
                progress_thread.join()
    
    def _show_progress(self, stop_event, callback):
        """Show progress dots while waiting for LLM response."""
        import time
        dots = 0
        while not stop_event.is_set():
            message = "🤖 Thinking" + "." * (dots % 4)
            callback(message)
            dots += 1
            time.sleep(0.5)

    def generate_response(self, user_message: str, pest_context: Dict[str, Any] = None) -> str:
        """Generate response using LM Studio API."""
        try:
            logger.info("Generating response with LM Studio (this may take up to 1 minute)...")
            
            # Prepare conversation context
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            # Add pest context if available
            if pest_context:
                context_message = self._format_pest_context(pest_context)
                messages.append({"role": "system", "content": context_message})
            
            messages.append({"role": "user", "content": user_message})
            
            # Make API call to LM Studio with extended timeout for local inference
            response = requests.post(
                f"{self.api_url}/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 500,
                    "stream": False
                },
                timeout=120  # Extended to 2 minutes for local LLM inference
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"LM Studio API error: {response.status_code}")
                return self._fallback_response(user_message)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Connection to LM Studio failed: {e}")
            return self._fallback_response(user_message)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._fallback_response(user_message)

    def _format_pest_context(self, pest_context: Dict[str, Any]) -> str:
        """Format pest detection context for LLM."""
        context_parts = []
        
        if pest_context.get('pest_type'):
            context_parts.append(f"Detected pest: {pest_context['pest_type']}")
        
        if pest_context.get('confidence'):
            context_parts.append(f"Detection confidence: {pest_context['confidence']:.1%}")
        
        if pest_context.get('uncertainty'):
            context_parts.append(f"Uncertainty: {pest_context['uncertainty']:.1%}")
        
        if pest_context.get('affected_crops'):
            context_parts.append(f"Commonly affects: {', '.join(pest_context['affected_crops'])}")
        
        if pest_context.get('harm_level'):
            context_parts.append(f"Threat level: {pest_context['harm_level']}")
        
        return "PEST DETECTION CONTEXT:\n" + "\n".join(context_parts)

    def _fallback_response(self, user_message: str) -> str:
        """Provide fallback response when LLM is unavailable."""
        return """I'm currently having trouble connecting to the AI assistant. However, I can still help you with basic pest management guidance. 

For immediate assistance:
1. Check the pest identification results if you uploaded an image
2. Look for organic treatment recommendations in the system
3. Consider these general organic approaches: beneficial insects, neem oil, diatomaceous earth, companion planting

Would you like me to provide specific guidance based on a particular pest type?"""

class FineTuningDataGenerator:
    """Generate training data for fine-tuning on agricultural domain."""
    
    def __init__(self):
        """Initialize fine-tuning data generator."""
        self.training_examples = []
        
    def generate_training_data(self) -> List[Dict[str, Any]]:
        """Generate training examples for agricultural pest management."""
        
        examples = [
            {
                "instruction": "A farmer found caterpillars eating their tomato leaves. What organic treatment do you recommend?",
                "input": "Pest: caterpillars, Crop: tomatoes, Damage: leaf feeding",
                "output": "For caterpillars on tomatoes, I recommend: 1) Hand-picking in early morning/evening when they're most active. 2) Bt (Bacillus thuringiensis) spray - organic and specifically targets caterpillars. 3) Row covers during vulnerable growth stages. 4) Encourage beneficial predators like birds with habitat. 5) Plant companion herbs like basil and marigolds as deterrents. Apply Bt in evening when caterpillars are feeding, and reapply after rain."
            },
            {
                "instruction": "How can I prevent slug damage in my lettuce crop organically?",
                "input": "Pest prevention: slugs, Crop: lettuce",
                "output": "Organic slug prevention for lettuce: 1) Improve drainage - slugs thrive in moist conditions. 2) Use copper tape around beds as a barrier. 3) Diatomaceous earth around plants (reapply after rain). 4) Beer traps placed away from crops to lure them out. 5) Encourage natural predators like ground beetles and birds. 6) Morning cultivation to expose eggs to drying. 7) Plant timing - avoid peak slug seasons when possible."
            },
            {
                "instruction": "I see beneficial bees in my garden but also some pest insects. How do I protect the bees while managing pests?",
                "input": "Concern: protecting beneficial bees while managing pests",
                "output": "Protecting bees while managing pests: 1) Never spray during bloom period when bees are foraging. 2) Use targeted applications in evening when bees are less active. 3) Choose bee-safe organic products like neem oil, spinosad, or Bt. 4) Use physical barriers like row covers when possible. 5) Spot-treat rather than broadcast spraying. 6) Plant diverse flowering plants to support bee populations. 7) Create bee habitat away from treatment areas. Always read labels for bee safety warnings."
            }
        ]
        
        return examples
    
    def save_training_data(self, output_path: str):
        """Save training data in format suitable for fine-tuning."""
        training_data = self.generate_training_data()
        
        # Format for different fine-tuning frameworks
        formats = {
            'alpaca': self._format_alpaca(training_data),
            'chat': self._format_chat(training_data),
            'jsonl': self._format_jsonl(training_data)
        }
        
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        for format_name, data in formats.items():
            output_file = output_dir / f"pest_management_training_{format_name}.json"
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        logger.info(f"Training data saved to {output_dir}")
    
    def _format_alpaca(self, examples: List[Dict]) -> List[Dict]:
        """Format for Alpaca-style fine-tuning."""
        return examples
    
    def _format_chat(self, examples: List[Dict]) -> List[Dict]:
        """Format for chat-based fine-tuning."""
        formatted = []
        for example in examples:
            formatted.append({
                "messages": [
                    {"role": "system", "content": "You are an expert organic farm pest management assistant."},
                    {"role": "user", "content": f"{example['instruction']} Context: {example['input']}"},
                    {"role": "assistant", "content": example['output']}
                ]
            })
        return formatted
    
    def _format_jsonl(self, examples: List[Dict]) -> str:
        """Format as JSONL for streaming training."""
        lines = []
        for example in examples:
            lines.append(json.dumps(example))
        return '\n'.join(lines)

# Usage example
if __name__ == "__main__":
    # Test LM Studio integration
    llm = LMStudioIntegration()
    
    # Test response generation
    pest_context = {
        'pest_type': 'caterpillars',
        'confidence': 0.89,
        'affected_crops': ['tomatoes', 'cabbage', 'broccoli'],
        'harm_level': 'medium_to_high'
    }
    
    response = llm.generate_response(
        "How do I treat caterpillars on my tomato plants organically?",
        pest_context
    )
    print("LLM Response:", response)
    
    # Generate fine-tuning data
    data_generator = FineTuningDataGenerator()
    data_generator.save_training_data("training_data")
