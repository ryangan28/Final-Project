"""
Organic Farm Pest Management AI System
Main Application Entry Point

This system provides offline-first pest identification and organic treatment recommendations
for farmers using computer vision and conversational AI.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pest_management.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Import core components for test patching
try:
    from treatments.recommendation_engine import TreatmentEngine
    from vision.pest_detector import PestDetector
except ImportError:
    TreatmentEngine = None
    PestDetector = None

class PestManagementSystem:
    """Main system orchestrator for the pest management AI."""
    
    def __init__(self):
        """Initialize the system components."""
        logger.info("Initializing Organic Farm Pest Management AI System")
        
        # Import modules with error handling
        try:
            from vision.pest_detector import PestDetector
            self.pest_detector = PestDetector()
        except ImportError as e:
            logger.warning(f"Vision module not available: {e}")
            self.pest_detector = None
            
        try:
            from treatments.recommendation_engine import TreatmentEngine
            self.treatment_engine = TreatmentEngine()
        except ImportError as e:
            logger.warning(f"Treatment module not available: {e}")
            self.treatment_engine = None
            
        try:
            from conversation.chat_interface import ChatInterface
            self.chat_interface = ChatInterface()
        except ImportError as e:
            logger.warning(f"Chat module not available: {e}")
            self.chat_interface = None
            
        try:
            from edge.model_optimizer import ModelOptimizer
            self.model_optimizer = ModelOptimizer()
        except ImportError as e:
            logger.warning(f"Edge optimization module not available: {e}")
            self.model_optimizer = None
        
        logger.info("System initialization complete")
    
    def identify_pest(self, image_path):
        """
        Identify pest from image and provide treatment recommendations.
        
        Args:
            image_path (str): Path to the pest image
            
        Returns:
            dict: Pest identification results and treatment recommendations
        """
        try:
            if not self.pest_detector:
                return {
                    'pest_identified': False,
                    'error': 'Pest detection module not available',
                    'success': False
                }
            
            # Detect pest in image
            pest_results = self.pest_detector.identify_pest(image_path)
            
            if pest_results['confidence'] > 0.7:
                # Get treatment category for treatment engine
                treatment_category = pest_results.get('treatment_category', pest_results['pest_type'].title())
                
                # Get treatment recommendations
                if self.treatment_engine:
                    treatments = self.treatment_engine.get_treatments(
                        treatment_category,
                        pest_results['severity']
                    )
                else:
                    treatments = {'message': 'Treatment engine not available'}
                
                return {
                    'pest_identified': True,
                    'pest_type': pest_results['pest_type'],
                    'confidence': pest_results['confidence'],
                    'severity': pest_results['severity'],
                    'treatments': treatments,
                    'success': True
                }
            else:
                return {
                    'pest_identified': False,
                    'confidence': pest_results['confidence'],
                    'message': 'Pest identification uncertain. Please try with a clearer image.',
                    'success': False
                }
                
        except Exception as e:
            logger.error(f"Error in pest identification: {str(e)}")
            return {
                'pest_identified': False,
                'error': str(e),
                'success': False
            }
    
    def chat_with_system(self, message, context=None):
        """
        Interact with the conversational AI interface.
        
        Args:
            message (str): User message
            context (dict): Optional context from pest identification
            
        Returns:
            str: AI response
        """
        if not self.chat_interface:
            return "Chat interface not available. Please check system setup."
            
        return self.chat_interface.process_message(message, context)
    
    def optimize_for_edge(self):
        """Optimize models for edge deployment."""
        if not self.model_optimizer:
            logger.warning("Model optimizer not available")
            return {'status': 'not_available'}
            
        logger.info("Optimizing models for edge deployment")
        results = self.model_optimizer.optimize_all_models()
        logger.info("Edge optimization complete")
        return results

def main():
    """Main entry point for the application."""
    print("🌱 Organic Farm Pest Management AI System")
    print("=" * 50)
    
    # Initialize the system
    system = PestManagementSystem()
    
    # Try to create and run the web interface
    try:
        from mobile.app_interface import create_app
        app = create_app(system)
        
        print("\n🚀 Starting web interface...")
        print("📱 Access the system at: http://localhost:8501")
        print("💡 Upload pest images for identification and treatment recommendations")
        
        app.run()
        
    except ImportError as e:
        logger.error(f"Web interface not available: {e}")
        print("\n❌ Web interface not available")
        print("💡 Running in console mode...")
        
        # Console mode interaction
        print("\n🌱 Console Mode - Organic Farm Pest Management AI")
        print("Ask me questions about organic pest management:")
        
        while True:
            try:
                user_input = input("\n👨‍🌾 You: ")
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    break
                
                response = system.chat_with_system(user_input)
                print(f"🤖 AI: {response}")
                
            except KeyboardInterrupt:
                break
        
        print("\n👋 Thank you for using the Organic Farm Pest Management AI System!")

if __name__ == "__main__":
    main()
