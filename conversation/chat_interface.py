"""
Conversational AI Interface for Farmer Interaction
Provides natural language interaction for pest management guidance.
"""

import logging
import re
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ChatInterface:
    """Conversational AI interface for farmer interaction."""
    
    def __init__(self):
        """Initialize the chat interface."""
        self.conversation_history = []
        self.farmer_profile = {}
        self.context = {}
        
        # Predefined responses for common scenarios
        self.responses = self._load_response_templates()
        
    def _load_response_templates(self):
        """Load response templates for common farmer questions."""
        return {
            'greeting': [
                "Hello! I'm your organic farm pest management assistant. How can I help you today?",
                "Welcome to the organic pest management system! What pest issue can I help you with?",
                "Hi there! Ready to tackle some pest problems organically? What's happening in your fields?"
            ],
            'pest_identification_help': [
                "I can help identify pests from photos. Please upload a clear image of the pest or damage you're seeing.",
                "For best results, take photos in good lighting showing the pest and any damage clearly.",
                "Multiple angles can help - try to capture the pest, any eggs, and the damage pattern."
            ],
            'treatment_explanation': [
                "All my recommendations are organic-certified and safe for your certification.",
                "I follow Integrated Pest Management (IPM) principles for sustainable control.",
                "These treatments are designed to work with nature, not against it."
            ],
            'urgency_assessment': [
                "Based on the severity, here's what I recommend doing first:",
                "Time is important with pest management. Let's prioritize your actions:",
                "I'll help you tackle this step by step, starting with the most urgent measures:"
            ],
            'encouragement': [
                "Organic pest management takes patience, but you're on the right track!",
                "Every organic farmer faces these challenges - you're not alone in this.",
                "Your commitment to organic farming makes a real difference for the environment."
            ]
        }
    
    def process_message(self, user_message, context=None):
        """
        Process user message and generate appropriate response.
        
        Args:
            user_message (str): Message from the user
            context (dict): Optional context from pest identification
            
        Returns:
            str: AI response to the user
        """
        try:
            logger.info(f"Processing message: {user_message[:50]}...")
            
            # Update context if provided
            if context:
                self.context.update(context)
            
            # Store conversation
            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'user_message': user_message,
                'context': context
            })
            
            # Determine response type
            response = self._generate_response(user_message)
            
            # Store AI response
            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'ai_response': response
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return "I'm sorry, I encountered an error. Could you please try again?"
    
    def _generate_response(self, message):
        """Generate contextual response based on user message."""
        message_lower = message.lower()
        
        # Greeting detection
        if any(word in message_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            return self._get_greeting_response()
        
        # Pest identification requests
        if any(word in message_lower for word in ['identify', 'what is', 'what pest', 'unknown pest']):
            return self._get_identification_help()
        
        # Treatment questions
        if any(word in message_lower for word in ['how to treat', 'treatment', 'control', 'get rid of']):
            return self._get_treatment_guidance(message)
        
        # Urgency questions
        if any(word in message_lower for word in ['urgent', 'emergency', 'spreading fast', 'quickly']):
            return self._get_urgency_response()
        
        # Organic certification questions
        if any(word in message_lower for word in ['organic', 'certified', 'omri', 'allowed']):
            return self._get_organic_guidance()
        
        # Timing questions
        if any(word in message_lower for word in ['when', 'timing', 'best time', 'schedule']):
            return self._get_timing_guidance()
        
        # Prevention questions
        if any(word in message_lower for word in ['prevent', 'prevention', 'avoid', 'stop']):
            return self._get_prevention_advice()
        
        # Cost/budget questions
        if any(word in message_lower for word in ['cost', 'expensive', 'cheap', 'budget', 'affordable']):
            return self._get_cost_guidance()
        
        # Effectiveness questions
        if any(word in message_lower for word in ['work', 'effective', 'success', 'results']):
            return self._get_effectiveness_info()
        
        # Default response with context
        return self._get_contextual_response(message)
    
    def _get_greeting_response(self):
        """Generate greeting response."""
        import random
        base_greeting = random.choice(self.responses['greeting'])
        
        if self.context.get('pest_type'):
            return f"{base_greeting} I see you've identified {self.context['pest_type']} in your crops. How can I help you manage this pest?"
        
        return base_greeting
    
    def _get_identification_help(self):
        """Provide pest identification help."""
        import random
        response = random.choice(self.responses['pest_identification_help'])
        
        tips = [
            "📸 Photo tips:",
            "• Take clear, well-lit photos",
            "• Show the pest and damage",
            "• Include a size reference if possible",
            "• Multiple angles are helpful"
        ]
        
        return f"{response}\n\n" + "\n".join(tips)
    
    def _get_treatment_guidance(self, message):
        """Provide treatment guidance based on context."""
        if not self.context.get('pest_type'):
            return ("I'd be happy to help with treatment options! First, let me identify the pest. "
                   "Please upload a photo of the pest or damage you're seeing.")
        
        pest_type = self.context['pest_type']
        severity = self.context.get('severity', 'medium')
        
        response = f"For {pest_type} control, I recommend an Integrated Pest Management approach:\n\n"
        
        if severity == 'low':
            response += "🟢 **Low severity** - Focus on prevention and monitoring:\n"
            response += "• Cultural controls (companion planting, habitat modification)\n"
            response += "• Regular monitoring\n"
            response += "• Beneficial insect conservation\n"
        elif severity == 'medium':
            response += "🟡 **Medium severity** - Active management needed:\n"
            response += "• Biological controls (beneficial insects, organic sprays)\n"
            response += "• Mechanical controls (traps, barriers)\n"
            response += "• Enhanced monitoring\n"
        else:
            response += "🔴 **High severity** - Immediate action required:\n"
            response += "• Immediate mechanical controls\n"
            response += "• Biological treatments\n"
            response += "• Emergency organic interventions\n"
        
        response += "\n💚 All recommendations are organic-certified and safe for your certification status."
        
        return response
    
    def _get_urgency_response(self):
        """Handle urgent pest situations."""
        import random
        base_response = random.choice(self.responses['urgency_assessment'])
        
        if self.context.get('severity') == 'high':
            return (f"{base_response}\n\n"
                   "🚨 **IMMEDIATE ACTIONS:**\n"
                   "1. Physical removal of visible pests\n"
                   "2. Apply organic contact treatments\n"
                   "3. Isolate affected areas if possible\n"
                   "4. Document the situation with photos\n\n"
                   "⏰ Act within 24-48 hours for best results!")
        
        return (f"{base_response}\n\n"
               "📋 **QUICK ACTION PLAN:**\n"
               "1. Confirm pest identification\n"
               "2. Assess spread and severity\n"
               "3. Start with safest, fastest treatments\n"
               "4. Monitor response closely")
    
    def _get_organic_guidance(self):
        """Provide organic certification guidance."""
        import random
        base_response = random.choice(self.responses['treatment_explanation'])
        
        return (f"{base_response}\n\n"
               "✅ **ORGANIC COMPLIANCE:**\n"
               "• All treatments are OMRI-approved or naturally acceptable\n"
               "• No synthetic pesticides or prohibited substances\n"
               "• Methods support biodiversity and soil health\n"
               "• Integrated approach minimizes environmental impact\n\n"
               "📋 Always check with your certifier if you have specific concerns!")
    
    def _get_timing_guidance(self):
        """Provide timing guidance for treatments."""
        if self.context.get('pest_type'):
            pest_type = self.context['pest_type']
            timing_map = {
                'Aphids': 'Early morning applications, weekly monitoring during growing season',
                'Caterpillars': 'Target young larvae, evening applications for biological controls',
                'Spider Mites': 'Increase frequency during hot, dry weather',
                'Whitefly': 'Early morning when adults are less active',
                'Thrips': 'Evening applications, avoid windy conditions'
            }
            
            specific_timing = timing_map.get(pest_type, 'Follow general IPM timing principles')
            
            return (f"⏰ **TIMING FOR {pest_type.upper()}:**\n"
                   f"{specific_timing}\n\n"
                   "🌅 **GENERAL TIMING TIPS:**\n"
                   "• Early morning: Best for most applications\n"
                   "• Avoid midday heat and direct sun\n"
                   "• Check weather - no rain for 24 hours\n"
                   "• Monitor every 2-3 days during active treatment")
        
        return ("⏰ **GENERAL TREATMENT TIMING:**\n"
               "• Early morning (6-9 AM) for most treatments\n"
               "• Evening (6-8 PM) for beneficial insect releases\n"
               "• Avoid windy or rainy conditions\n"
               "• Monitor every 2-3 days for effectiveness")
    
    def _get_prevention_advice(self):
        """Provide prevention advice."""
        return ("🛡️ **PREVENTION IS THE BEST MEDICINE:**\n\n"
               "🌱 **CULTURAL PRACTICES:**\n"
               "• Crop rotation (2-3 year cycles)\n"
               "• Companion planting\n"
               "• Proper plant spacing\n"
               "• Soil health management\n\n"
               "🔍 **MONITORING:**\n"
               "• Weekly field inspections\n"
               "• Sticky traps for early detection\n"
               "• Weather monitoring\n"
               "• Record keeping\n\n"
               "🐛 **HABITAT MANAGEMENT:**\n"
               "• Encourage beneficial insects\n"
               "• Maintain biodiversity\n"
               "• Remove pest breeding sites\n"
               "• Clean cultivation practices")
    
    def _get_cost_guidance(self):
        """Provide cost-effective treatment guidance."""
        return ("💰 **COST-EFFECTIVE ORGANIC PEST MANAGEMENT:**\n\n"
               "💚 **LOW-COST OPTIONS:**\n"
               "• Hand picking and physical removal\n"
               "• Companion planting\n"
               "• Water sprays and mechanical controls\n"
               "• Homemade organic sprays (soap, neem)\n\n"
               "💛 **MEDIUM-COST INVESTMENTS:**\n"
               "• Beneficial insect releases\n"
               "• Row covers and barriers\n"
               "• Organic approved sprays\n"
               "• Trap crops\n\n"
               "📈 **LONG-TERM SAVINGS:**\n"
               "• Prevention reduces treatment costs\n"
               "• Beneficial insects provide ongoing control\n"
               "• Healthy soil reduces pest pressure\n"
               "• IPM approach optimizes resource use")
    
    def _get_effectiveness_info(self):
        """Provide effectiveness information."""
        if self.context.get('pest_type'):
            return ("📊 **TREATMENT EFFECTIVENESS:**\n\n"
                   f"For {self.context['pest_type']}:\n"
                   "• Biological controls: 70-90% effective with time\n"
                   "• Cultural controls: 60-80% effective for prevention\n"
                   "• Mechanical controls: 50-70% effective immediately\n\n"
                   "⏰ **TIMELINE:**\n"
                   "• Immediate results: Mechanical controls\n"
                   "• 1-2 weeks: Contact treatments\n"
                   "• 2-4 weeks: Biological controls\n"
                   "• Full season: Cultural/preventive measures\n\n"
                   "🔄 **SUCCESS FACTORS:**\n"
                   "• Early intervention\n"
                   "• Consistent application\n"
                   "• Multiple control methods\n"
                   "• Regular monitoring")
        
        return ("📊 **ORGANIC TREATMENT EFFECTIVENESS:**\n\n"
               "🎯 **SUCCESS RATES:**\n"
               "• IPM approach: 80-95% effective\n"
               "• Single method: 40-60% effective\n"
               "• Prevention focus: 90%+ effective\n\n"
               "📈 **FACTORS FOR SUCCESS:**\n"
               "• Early detection and intervention\n"
               "• Multiple control strategies\n"
               "• Consistent monitoring\n"
               "• Proper timing of treatments\n"
               "• Healthy ecosystem maintenance")
    
    def _get_contextual_response(self, message):
        """Generate response based on current context."""
        if self.context.get('pest_type'):
            pest_type = self.context['pest_type']
            return (f"I understand you're dealing with {pest_type}. "
                   f"Could you be more specific about what aspect you'd like help with? "
                   f"I can provide information about:\n\n"
                   f"• Treatment options\n"
                   f"• Application timing\n"
                   f"• Prevention strategies\n"
                   f"• Organic compliance\n"
                   f"• Cost considerations\n"
                   f"• Effectiveness expectations")
        
        return ("I'm here to help with organic pest management! I can assist with:\n\n"
               "🔍 **Pest identification** from photos\n"
               "💚 **Organic treatment** recommendations\n"
               "⏰ **Timing guidance** for applications\n"
               "🛡️ **Prevention strategies**\n"
               "📊 **Effectiveness** information\n"
               "💰 **Cost-effective** solutions\n\n"
               "What specific help do you need today?")
