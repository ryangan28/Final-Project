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
import os
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pest_management.log'),
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
    
    def __init__(self, selected_model=None):
        """Initialize the system components."""
        logger.info("Initializing Organic Farm Pest Management AI System")
        
        # Import unified pest detector with all capabilities
        try:
            from vision.pest_detector import UnifiedPestDetector
            self.pest_detector = UnifiedPestDetector(selected_model=selected_model)
            logger.info("Unified pest detector loaded with all available backends")
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
    
    def get_available_models(self):
        """Get available models for selection."""
        if not self.pest_detector:
            return []
        return self.pest_detector.get_available_models()
    
    def switch_model(self, selected_model):
        """Switch to a different model."""
        if not self.pest_detector:
            return False
        return self.pest_detector.switch_model(selected_model)
    
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
            
            # Detect pest in image using unified detector
            pest_results = self.pest_detector.detect_pest(image_path)
            
            # Check if detection was successful
            if pest_results.get('success', False) and pest_results.get('confidence', 0) > 0.4:  # Lowered threshold
                # Extract metadata if available
                metadata = pest_results.get('metadata', {})
                
                # Get treatment category for treatment engine
                treatment_category = pest_results.get('pest_type', 'Unknown').title()
                
                # Get treatment recommendations
                if self.treatment_engine:
                    treatments = self.treatment_engine.get_treatments(
                        treatment_category,
                        'medium'  # Default severity
                    )
                else:
                    treatments = {'message': 'Treatment engine not available'}
                
                return {
                    'pest_identified': True,
                    'pest_type': pest_results.get('pest_type', 'Unknown'),
                    'confidence': pest_results.get('confidence', 0.0),
                    'severity': 'medium',  # Default for now
                    'uncertainty': pest_results.get('uncertainty', 0.0),
                    'scientific_name': metadata.get('scientific_name', 'Unknown'),
                    'common_names': metadata.get('common_names', []),
                    'affected_crops': metadata.get('affected_crops', []),
                    'is_beneficial': metadata.get('damage_type') == 'beneficial',
                    'detection_method': pest_results.get('method', 'unknown'),
                    'recommendation': f"Detected {pest_results.get('pest_type', 'pest')} with {pest_results.get('confidence', 0)*100:.1f}% confidence",
                    'treatments': treatments,
                    'success': True
                }
            else:
                # Use the detector's own error message or reason if available
                message = pest_results.get('reason', pest_results.get('error', 'Pest identification uncertain. Please try with a clearer image.'))
                
                return {
                    'pest_identified': False,
                    'confidence': pest_results.get('confidence', 0.0),
                    'uncertainty': pest_results.get('uncertainty', 1.0),
                    'message': message,
                    'detection_method': pest_results.get('method', 'unknown'),
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
        
        # Set pest context if provided
        if context:
            self.chat_interface.set_pest_context(context)
            
        return self.chat_interface.process_message(message)
    
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
    
    # Check if running in Streamlit context
    try:
        import streamlit as st
        # Running in Streamlit - use session state for persistence
        
        # 1) Cache the system so it survives reruns
        if "pest_system" not in st.session_state:
            st.session_state.pest_system = PestManagementSystem()
        
        system = st.session_state.pest_system
        
        # 2) If user has already picked a model, make sure the detector still uses it
        if "selected_model" in st.session_state:
            system.switch_model(st.session_state.selected_model)
        
        # Create and run the web interface with persistent system
        from mobile.app_interface import create_app
        app = create_app(system)
        
        print("\n🚀 Starting web interface...")
        print("📱 Access the system at: http://localhost:8501")
        print("💡 Upload pest images for identification and treatment recommendations")
        
        app.run()
        
    except ImportError:
        # Not in Streamlit context - create system normally for console mode
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
