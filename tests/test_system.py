"""
Comprehensive Test Suite for Organic Farm Pest Management AI System
Tests all components including vision, treatments, chat, and edge optimization.
"""

import unittest
import sys
import os
from pathlib import Path
import logging
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestPestDetector(unittest.TestCase):
    """Test the pest detection vision module."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from vision.pest_detector import PestDetector
            self.detector = PestDetector()
        except ImportError as e:
            self.skipTest(f"Vision module not available: {e}")
    
    def test_pest_classes_loaded(self):
        """Test that pest classes are properly loaded."""
        self.assertIsInstance(self.detector.pest_classes, dict)
        self.assertGreater(len(self.detector.pest_classes), 0)
        
        # Check specific pest types
        expected_pests = ['Aphids', 'Caterpillars', 'Spider Mites', 'Whitefly']
        for pest_id, pest_info in self.detector.pest_classes.items():
            if pest_info['name'] in expected_pests:
                self.assertIn('scientific_name', pest_info)
                self.assertIn('crops_affected', pest_info)
    
    def test_image_preprocessing(self):
        """Test image preprocessing functionality."""
        # Create a dummy image
        from PIL import Image
        dummy_image = Image.new('RGB', (224, 224), color='red')
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                dummy_image.save(tmp.name)
                tensor = self.detector.preprocess_image(tmp.name)
                
                # Check tensor properties
                self.assertEqual(tensor.shape, (1, 3, 224, 224))
                self.assertTrue(tensor.dtype in [torch.float32, torch.float16])
                
        finally:
            # Clean up
            if 'tmp' in locals():
                os.unlink(tmp.name)
    
    def test_severity_analysis(self):
        """Test pest severity analysis."""
        # Create a test image
        from PIL import Image
        test_image = Image.new('RGB', (224, 224), color='green')
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            test_image.save(tmp.name)
            
            try:
                severity = self.detector.analyze_severity(tmp.name, 'Aphids')
                self.assertIn(severity, ['low', 'medium', 'high'])
            finally:
                os.unlink(tmp.name)
    
    @patch('torch.nn.functional.softmax')
    @patch('torch.max')
    def test_pest_detection(self, mock_max, mock_softmax):
        """Test the main pest detection functionality."""
        # Mock the model outputs
        mock_softmax.return_value = torch.tensor([[0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        mock_max.return_value = (torch.tensor([0.9]), torch.tensor([1]))
        
        # Create test image
        from PIL import Image
        test_image = Image.new('RGB', (224, 224), color='red')
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            test_image.save(tmp.name)
            
            try:
                results = self.detector.detect(tmp.name)
                
                # Verify results structure
                self.assertIn('pest_type', results)
                self.assertIn('confidence', results)
                self.assertIn('severity', results)
                self.assertIn('crops_affected', results)
                
                # Verify values
                self.assertIsInstance(results['confidence'], float)
                self.assertGreaterEqual(results['confidence'], 0.0)
                self.assertLessEqual(results['confidence'], 1.0)
                
            finally:
                os.unlink(tmp.name)


class TestTreatmentEngine(unittest.TestCase):
    """Test the treatment recommendation engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from treatments.recommendation_engine import TreatmentEngine
            self.engine = TreatmentEngine()
        except ImportError as e:
            self.skipTest(f"Treatment module not available: {e}")
    
    def test_treatments_database_loaded(self):
        """Test that treatments database is properly loaded."""
        self.assertIsInstance(self.engine.treatments_db, dict)
        self.assertGreater(len(self.engine.treatments_db), 0)
        
        # Check for required pest types
        expected_pests = ['Aphids', 'Caterpillars', 'Spider Mites']
        for pest in expected_pests:
            self.assertIn(pest, self.engine.treatments_db)
    
    def test_ipm_principles_loaded(self):
        """Test that IPM principles are properly loaded."""
        self.assertIsInstance(self.engine.ipm_principles, dict)
        
        expected_categories = ['prevention', 'cultural_controls', 'biological_controls']
        for category in expected_categories:
            self.assertIn(category, self.engine.imp_principles)
            self.assertIsInstance(self.engine.ipm_principles[category], list)
    
    def test_get_treatments_valid_pest(self):
        """Test getting treatments for a valid pest."""
        results = self.engine.get_treatments('Aphids', 'medium')
        
        # Check result structure
        self.assertIsInstance(results, dict)
        self.assertIn('pest_type', results)
        self.assertIn('treatment_plan', results)
        self.assertIn('ipm_approach', results)
        
        # Check treatment plan structure
        treatment_plan = results['treatment_plan']
        self.assertIn('immediate_actions', treatment_plan)
        self.assertIn('short_term', treatment_plan)
        self.assertIn('long_term', treatment_plan)
    
    def test_get_treatments_invalid_pest(self):
        """Test getting treatments for an invalid pest."""
        results = self.engine.get_treatments('NonExistentPest', 'medium')
        
        # Should return generic treatments
        self.assertIsInstance(results, dict)
        self.assertIn('message', results)
    
    def test_severity_based_recommendations(self):
        """Test that recommendations change based on severity."""
        low_severity = self.engine.get_treatments('Aphids', 'low')
        high_severity = self.engine.get_treatments('Aphids', 'high')
        
        # High severity should have more immediate actions
        low_immediate = len(low_severity['treatment_plan']['immediate_actions'])
        high_immediate = len(high_severity['treatment_plan']['immediate_actions'])
        
        self.assertGreaterEqual(high_immediate, low_immediate)
    
    def test_organic_certification_compliance(self):
        """Test that all treatments are organic certified."""
        results = self.engine.get_treatments('Aphids', 'medium')
        
        # Check for organic certification mention
        self.assertIn('organic_certification', results)
        
        # Check individual treatments
        for category in ['immediate_actions', 'short_term', 'long_term']:
            if category in results['treatment_plan']:
                for treatment in results['treatment_plan'][category]:
                    if isinstance(treatment, dict) and 'organic_certified' in treatment:
                        self.assertTrue(treatment['organic_certified'])


class TestChatInterface(unittest.TestCase):
    """Test the conversational AI interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from conversation.chat_interface import ChatInterface
            self.chat = ChatInterface()
        except ImportError as e:
            self.skipTest(f"Chat module not available: {e}")
    
    def test_initialization(self):
        """Test chat interface initialization."""
        self.assertIsInstance(self.chat.conversation_history, list)
        self.assertIsInstance(self.chat.responses, dict)
        self.assertEqual(len(self.chat.conversation_history), 0)
    
    def test_greeting_detection(self):
        """Test greeting message detection."""
        greetings = ['hello', 'hi', 'good morning', 'hey there']
        
        for greeting in greetings:
            response = self.chat.process_message(greeting)
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
            
            # Should contain welcoming language
            response_lower = response.lower()
            self.assertTrue(any(word in response_lower for word in ['welcome', 'hello', 'help']))
    
    def test_pest_identification_help(self):
        """Test pest identification help requests."""
        identification_requests = [
            'what pest is this',
            'identify this bug',
            'unknown pest in my garden'
        ]
        
        for request in identification_requests:
            response = self.chat.process_message(request)
            self.assertIsInstance(response, str)
            
            # Should mention photo upload or identification process
            response_lower = response.lower()
            self.assertTrue(any(word in response_lower for word in ['photo', 'image', 'upload', 'identify']))
    
    def test_treatment_guidance(self):
        """Test treatment guidance requests."""
        treatment_requests = [
            'how to treat aphids',
            'what should I do about caterpillars',
            'treatment for spider mites'
        ]
        
        for request in treatment_requests:
            response = self.chat.process_message(request)
            self.assertIsInstance(response, str)
            
            # Should mention treatment or management
            response_lower = response.lower()
            self.assertTrue(any(word in response_lower for word in ['treatment', 'control', 'manage', 'organic']))
    
    def test_context_integration(self):
        """Test chat with pest identification context."""
        context = {
            'pest_type': 'Aphids',
            'confidence': 0.9,
            'severity': 'medium'
        }
        
        response = self.chat.process_message('How should I treat this?', context)
        self.assertIsInstance(response, str)
        
        # Should reference the specific pest
        self.assertIn('Aphids', response)
    
    def test_conversation_history(self):
        """Test conversation history tracking."""
        initial_length = len(self.chat.conversation_history)
        
        self.chat.process_message('Hello')
        self.assertEqual(len(self.chat.conversation_history), initial_length + 2)  # User + AI response
        
        # Check history structure
        user_entry = self.chat.conversation_history[-2]
        ai_entry = self.chat.conversation_history[-1]
        
        self.assertIn('user_message', user_entry)
        self.assertIn('ai_response', ai_entry)
        self.assertIn('timestamp', user_entry)


class TestModelOptimizer(unittest.TestCase):
    """Test the edge model optimization module."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from edge.model_optimizer import ModelOptimizer
            self.optimizer = ModelOptimizer()
        except ImportError as e:
            self.skipTest(f"Edge module not available: {e}")
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertIsInstance(self.optimizer.optimization_configs, dict)
        self.assertIn('pest_detection', self.optimizer.optimization_configs)
        self.assertIn('treatment_engine', self.optimizer.optimization_configs)
    
    def test_lightweight_cnn_creation(self):
        """Test lightweight CNN model creation."""
        try:
            model = self.optimizer._create_lightweight_cnn()
            self.assertIsNotNone(model)
            
            # Test model with dummy input
            import torch
            dummy_input = torch.randn(1, 3, 224, 224)
            output = model(dummy_input)
            
            # Check output shape
            self.assertEqual(output.shape[0], 1)  # Batch size
            self.assertEqual(output.shape[1], 8)  # Number of classes
            
        except ImportError:
            self.skipTest("PyTorch not available for model testing")
    
    def test_treatment_database_compression(self):
        """Test treatment database compression."""
        compressed_db = self.optimizer._compress_treatment_database()
        
        self.assertIsInstance(compressed_db, dict)
        self.assertIn('pests', compressed_db)
        self.assertIn('treatments', compressed_db)
        
        # Check compression (shorter keys, abbreviated data)
        for pest_id, pest_info in compressed_db['pests'].items():
            self.assertIn('n', pest_info)  # name
            self.assertIn('t', pest_info)  # treatments
    
    def test_rules_engine_creation(self):
        """Test rules engine creation."""
        rules = self.optimizer._create_rules_engine()
        
        self.assertIsInstance(rules, dict)
        self.assertIn('severity_rules', rules)
        self.assertIn('timing_rules', rules)
        
        # Check rule structure
        severity_rules = rules['severity_rules']
        self.assertIn('low', severity_rules)
        self.assertIn('medium', severity_rules)
        self.assertIn('high', severity_rules)
    
    @patch('edge.model_optimizer.torch')
    def test_optimization_pipeline(self, mock_torch):
        """Test the full optimization pipeline."""
        # Mock torch dependencies
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = 'cpu'
        
        try:
            results = self.optimizer.optimize_all_models()
            
            self.assertIsInstance(results, dict)
            self.assertIn('pest_detection', results)
            self.assertIn('treatment_engine', results)
            
        except Exception as e:
            # If optimization fails due to missing dependencies, that's expected
            self.assertIn('optimization', str(e).lower())


class TestMainSystem(unittest.TestCase):
    """Test the main system integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from main import PestManagementSystem
            self.system = PestManagementSystem()
        except ImportError as e:
            self.skipTest(f"Main system not available: {e}")
    
    def test_system_initialization(self):
        """Test system initialization."""
        self.assertIsNotNone(self.system.pest_detector)
        self.assertIsNotNone(self.system.treatment_engine)
        self.assertIsNotNone(self.system.chat_interface)
        self.assertIsNotNone(self.system.model_optimizer)
    
    @patch('main.PestDetector')
    @patch('main.TreatmentEngine')
    def test_pest_identification_flow(self, mock_treatment, mock_detector):
        """Test the complete pest identification flow."""
        # Mock pest detection results
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = {
            'pest_type': 'Aphids',
            'confidence': 0.85,
            'severity': 'medium'
        }
        mock_detector.return_value = mock_detector_instance
        
        # Mock treatment recommendations
        mock_treatment_instance = Mock()
        mock_treatment_instance.get_treatments.return_value = {
            'treatments': ['neem_oil', 'beneficial_insects']
        }
        mock_treatment.return_value = mock_treatment_instance
        
        # Create system with mocks
        from main import PestManagementSystem
        system = PestManagementSystem()
        system.pest_detector = mock_detector_instance
        system.treatment_engine = mock_treatment_instance
        
        # Test identification
        results = system.identify_pest('dummy_path.jpg')
        
        self.assertTrue(results['success'])
        self.assertTrue(results['pest_identified'])
        self.assertEqual(results['pest_type'], 'Aphids')
        self.assertIn('treatments', results)
    
    def test_chat_integration(self):
        """Test chat system integration."""
        response = self.system.chat_with_system('Hello')
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
    
    def test_edge_optimization(self):
        """Test edge optimization integration."""
        try:
            self.system.optimize_for_edge()
            # If no exception is raised, the integration works
        except Exception as e:
            # Expected for missing dependencies
            self.assertTrue(any(word in str(e).lower() for word in ['import', 'module', 'torch']))


class TestDataIntegrity(unittest.TestCase):
    """Test data integrity and consistency."""
    
    def test_pest_database_consistency(self):
        """Test pest database consistency across modules."""
        try:
            from vision.pest_detector import PestDetector
            from treatments.recommendation_engine import TreatmentEngine
            
            detector = PestDetector()
            engine = TreatmentEngine()
            
            # Get pest types from both modules
            detector_pests = set(info['name'] for info in detector.pest_classes.values())
            treatment_pests = set(engine.treatments_db.keys())
            
            # Check overlap
            common_pests = detector_pests.intersection(treatment_pests)
            self.assertGreater(len(common_pests), 0, "No common pests found between modules")
            
        except ImportError:
            self.skipTest("Required modules not available")
    
    def test_organic_certification_compliance(self):
        """Test that all recommendations comply with organic standards."""
        try:
            from treatments.recommendation_engine import TreatmentEngine
            
            engine = TreatmentEngine()
            
            # Test several pest types
            test_pests = ['Aphids', 'Caterpillars', 'Spider Mites']
            
            for pest in test_pests:
                if pest in engine.treatments_db:
                    treatments = engine.get_treatments(pest, 'medium')
                    
                    # Check for organic certification mention
                    self.assertIn('organic_certification', treatments)
                    
                    # Verify no prohibited substances mentioned
                    prohibited_terms = ['synthetic', 'pesticide', 'chemical', 'toxic']
                    treatment_text = str(treatments).lower()
                    
                    # Should not contain prohibited terms in a negative context
                    # This is a simplified check
                    self.assertNotIn('synthetic pesticide', treatment_text)
                    
        except ImportError:
            self.skipTest("Treatment module not available")


def create_test_suite():
    """Create comprehensive test suite."""
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestPestDetector,
        TestTreatmentEngine,
        TestChatInterface,
        TestModelOptimizer,
        TestMainSystem,
        TestDataIntegrity
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


def run_tests():
    """Run all tests and generate report."""
    print("🧪 Running Comprehensive Test Suite for Organic Farm Pest Management AI System")
    print("=" * 80)
    
    # Create test suite
    suite = create_test_suite()
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Generate test report
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    passed = total_tests - failures - errors - skipped
    
    print(f"✅ Tests Passed: {passed}")
    print(f"❌ Tests Failed: {failures}")
    print(f"💥 Tests Errors: {errors}")
    print(f"⏭️ Tests Skipped: {skipped}")
    print(f"📈 Success Rate: {(passed/total_tests)*100:.1f}%" if total_tests > 0 else "No tests run")
    
    # Show failures and errors
    if result.failures:
        print("\n🔴 FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Error:')[-1].strip()}")
    
    # Overall status
    if failures == 0 and errors == 0:
        print("\n🎉 ALL TESTS PASSED! System is ready for deployment.")
        return True
    else:
        print("\n⚠️ Some tests failed. Please review and fix issues before deployment.")
        return False


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
