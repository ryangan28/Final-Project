"""
Mobile/Web Application Interface
Creates user-friendly interface for farmers to interact with the pest management system.
"""

import streamlit as st
import logging
from pathlib import Path
import json
from datetime import datetime
import io
import tempfile
import os
from PIL import Image
import base64

logger = logging.getLogger(__name__)

# Constants
EDGE_OPTIMIZATION_LABEL = "Edge Optimization"
CHAT_ASSISTANT_PAGE = "Chat Assistant"
TREATMENT_LIBRARY_PAGE = "Treatment Library"


class AppStyles:
    """Handles application styling and CSS."""
    
    @staticmethod
    def apply_custom_css():
        """Apply custom CSS for better styling."""
        st.markdown("""
            <style>
            .main-header {
                font-size: 2.5rem;
                color: #2E7D32;
                text-align: center;
                margin-bottom: 2rem;
            }
            .section-header {
                font-size: 1.5rem;
                color: #388E3C;
                margin-top: 2rem;
                margin-bottom: 1rem;
            }
            .success-box {
                background-color: #E8F5E8;
                color: #2E7D32;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid #4CAF50;
                margin: 1rem 0;
            }
            .warning-box {
                background-color: #FFF3E0;
                color: #E65100;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid #FF9800;
                margin: 1rem 0;
            }
            .error-box {
                background-color: #FFEBEE;
                color: #C62828;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid #F44336;
                margin: 1rem 0;
            }
            .info-card {
                background-color: #F3E5F5;
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 1rem 0;
            }
            </style>
        """, unsafe_allow_html=True)


class NavigationHandler:
    """Handles navigation and page routing."""
    
    @staticmethod
    def get_pages():
        return [
            "Pest Identification", 
            CHAT_ASSISTANT_PAGE,
            TREATMENT_LIBRARY_PAGE
        ]
    
    @staticmethod
    def handle_navigation_triggers():
        """Handle navigation triggers and return the appropriate page."""
        # Handle programmatic navigation with priority over radio selection
        if st.session_state.get('navigate_to_chat', False):
            st.session_state.navigate_to_chat = False
            return CHAT_ASSISTANT_PAGE
            
        if st.session_state.get('navigate_to_library', False):
            st.session_state.navigate_to_library = False
            return TREATMENT_LIBRARY_PAGE
        
        # Return the page from the radio selector
        return st.session_state.get('page_radio', 'Pest Identification')


class ImageHandler:
    """Handles image processing and demo image management."""
    
    @staticmethod
    def get_demo_images():
        """Get sample images from the datasets for demo purposes."""
        try:
            dataset_path = Path("datasets")
            demo_images = []
            
            if not dataset_path.exists():
                return demo_images
                
            pest_directories = [
                'ants', 'bees', 'beetle', 'caterpillars', 'earthworms', 
                'earwig', 'grasshopper', 'moth', 'slug', 'snail', 'wasp', 'weevil'
            ]
            
            for pest_dir in pest_directories:
                pest_path = dataset_path / pest_dir
                if pest_path.exists():
                    image_files = list(pest_path.glob('*.jpg')) + list(pest_path.glob('*.png'))
                    if image_files:
                        demo_images.append((pest_dir, str(image_files[0])))
            
            return demo_images
            
        except Exception as e:
            logger.error(f"Error getting demo images: {e}")
            return []
    
    @staticmethod
    def display_demo_images(demo_images):
        """Display demo images in a grid layout."""
        if not demo_images:
            st.info("📁 Add datasets folder with pest images for photo examples")
            return
            
        st.markdown("**Good Photo Examples:**")
        selected_categories = demo_images[:6]
        
        for i in range(0, len(selected_categories), 2):
            row_col1, row_col2 = st.columns(2)
            
            ImageHandler._display_single_demo_image(selected_categories, i, row_col1)
            if i + 1 < len(selected_categories):
                ImageHandler._display_single_demo_image(selected_categories, i + 1, row_col2)
    
    @staticmethod
    def _display_single_demo_image(categories, index, column):
        """Display a single demo image."""
        pest_type, image_path = categories[index]
        if not Path(image_path).exists():
            return
            
        with column:
            try:
                demo_image = Image.open(image_path)
                st.image(demo_image, caption=f"✅ {pest_type.title()}", use_container_width=True)
            except Exception:
                st.caption(f"✅ {pest_type.title()} (sample)")
    
    @staticmethod
    def process_uploaded_image(uploaded_file, system):
        """Process uploaded image and return results."""
        try:
            original_filename = getattr(uploaded_file, 'name', 'uploaded_image.jpg')
            temp_path = ImageHandler._create_temp_file(original_filename)
            
            image = Image.open(uploaded_file)
            image.save(temp_path)
            
            results = system.identify_pest(temp_path)
            ImageHandler._cleanup_temp_file(temp_path)
            
            return results, None
            
        except Exception as e:
            return None, str(e)
    
    @staticmethod
    def _create_temp_file(original_filename):
        """Create a secure temporary file path."""
        temp_dir = tempfile.gettempdir()
        safe_filename = "".join(c for c in original_filename if c.isalnum() or c in ('_', '-', '.')).lower()
        return os.path.join(temp_dir, f"pest_upload_{safe_filename}")
    
    @staticmethod
    def _cleanup_temp_file(temp_path):
        """Clean up temporary file."""
        try:
            os.unlink(temp_path)
        except OSError:
            pass


class SessionStateManager:
    """Manages session state for the application."""
    
    @staticmethod
    def store_identification_results(results):
        """Store identification results in session state."""
        st.session_state.last_identification = results
        
        if 'recent_identifications' not in st.session_state:
            st.session_state.recent_identifications = []
        st.session_state.recent_identifications.append(results)
    
    @staticmethod
    def init_chat_history():
        """Initialize chat history if not exists."""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
    
    @staticmethod
    def setup_chat_navigation(results, system):
        """Setup navigation to chat with pest context."""
        SessionStateManager.init_chat_history()
        
        pest_name = results['pest_type']
        treatment_question = f"I have identified {pest_name} in my crops. What are the best organic treatment options available? Please provide specific recommendations."
        
        st.session_state.chat_history.extend([
            ("System", f"🔄 Switching to Chat Assistant for {pest_name} treatment discussion..."),
            ("User", treatment_question)
        ])
        
        try:
            ai_response = system.chat_with_system(treatment_question, results)
            st.session_state.chat_history.append(("Assistant", ai_response))
        except Exception as e:
            error_msg = f"I apologize, but I encountered an error getting treatment recommendations: {str(e)}"
            st.session_state.chat_history.append(("Assistant", error_msg))


class PestIdentificationPage:
    """Handles the pest identification page."""
    
    def __init__(self, system):
        self.system = system
    
    def display(self):
        st.markdown('<h2 class="section-header">🔍 Pest Identification</h2>', unsafe_allow_html=True)
        
        # Model selection section
        self._display_model_selection()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            self._display_upload_section()
        
        with col2:
            self._display_results()
        
        # Photography Tips section below both columns
        self._display_photography_tips()
    
    def _display_model_selection(self):
        """Display model selection dropdown."""
        st.markdown("### 🤖 Model Selection")
        
        # Get available models
        available_models = self.system.get_available_models()
        
        if not available_models:
            st.warning("No models available. Please check your model files.")
            return
        
        # Initialize selected model if not set
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = available_models[0]['id']
            # Set the initial model in the system
            self.system.switch_model(st.session_state.selected_model)
        
        # Create model options
        model_options = []
        model_ids = []
        for model in available_models:
            display_name = f"{model['name']} ({model['type']})"
            model_options.append(display_name)
            model_ids.append(model['id'])
        
        # Find current index
        current_index = 0
        if st.session_state.selected_model in model_ids:
            current_index = model_ids.index(st.session_state.selected_model)
        
        # Model selection dropdown
        selected_option = st.selectbox(
            "Choose the model for pest detection:",
            options=model_options,
            index=current_index,
            help="Different models may have different accuracy and speed characteristics"
        )
        
        # Check if selection changed
        selected_index = model_options.index(selected_option)
        new_model_id = model_ids[selected_index]
        
        if new_model_id != st.session_state.selected_model:
            with st.spinner(f"Switching to {selected_option}..."):
                st.session_state.selected_model = new_model_id
                
                if self.system.switch_model(new_model_id):
                    st.success(f"✅ Switched to {selected_option}")
                    # Force rerun to update UI immediately
                    st.rerun()
                else:
                    st.error("❌ Failed to switch model")
                    # Revert session state on failure
                    st.session_state.selected_model = model_ids[current_index]
        
        # Display current model info
        current_model = next((m for m in available_models if m['id'] == st.session_state.selected_model), None)
        if current_model:
            st.info(f"🎯 Active model: **{current_model['name']}** ({current_model['type']})")
            if current_model.get('description'):
                st.caption(current_model['description'])
        
        st.markdown("---")
    
    def _display_upload_section(self):
        st.markdown("### 📸 Upload Your Own Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of the pest or damage for identification"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔬 Analyze Image", type="primary"):
                self._process_image_analysis(uploaded_file)
    
    def _process_image_analysis(self, uploaded_file):
        with st.spinner("Analyzing image for pest identification..."):
            results, error = ImageHandler.process_uploaded_image(uploaded_file, self.system)
            
            if results:
                SessionStateManager.store_identification_results(results)
                st.success("✅ Analysis complete!")
                st.rerun()
            else:
                st.error(f"❌ Error during analysis: {error}")
    
    def _display_photography_tips(self):
        st.markdown("### 📷 Photography Tips")
        st.markdown("For best identification results, follow these photography guidelines:")
        
        demo_images = ImageHandler.get_demo_images()
        ImageHandler.display_demo_images(demo_images)
        
        st.info("""
        **Tips for Clear Photos:**
        - 📱 Hold device steady
        - 🔍 Get close to the pest
        - ☀️ Use good lighting (natural light preferred)
        - 🎯 Focus on the pest clearly
        - 📐 Include size reference if possible
        - Take close-up shots of the pest
        - Include any visible damage
        - Keep the image in focus
        - Try multiple angles if possible
        """)
    
    def _display_results(self):
        st.markdown("### 📊 Identification Results")
        
        results = st.session_state.get('last_identification')
        if not results:
            st.info("Upload an image to begin pest identification.")
            return
        
        if results.get('success'):
            self._display_successful_results(results)
        else:
            self._display_failed_results(results)
    
    def _display_successful_results(self, results):
        # Display identification results in a single container
        detection_method = results.get('detection_method', 'unknown')
        st.markdown(f"""
        <div class="success-box">
            <p><strong>🐛 Pest Identified:</strong> {results['pest_type']}</p>
            <p><strong>🎯 Confidence:</strong> {results['confidence']:.1%}</p>
            <p><strong>⚠️ Severity:</strong> {results['severity'].title()}</p>
            <p><strong>🤖 Model Used:</strong> {detection_method}</p>
        </div>
        """, unsafe_allow_html=True)
        
        self._display_severity_indicator(results['severity'])
        self._display_treatment_summary(results)
        self._display_action_buttons(results)
    
    def _display_severity_indicator(self, severity):
        severity_config = {
            'low': ("🟢 Low Severity - Monitor and prevent", st.success),
            'medium': ("🟡 Medium Severity - Active treatment needed", st.warning),
            'high': ("🔴 High Severity - Immediate action required", st.error)
        }
        
        message, display_func = severity_config.get(severity, ("Unknown severity", st.info))
        display_func(message)
    
    def _display_treatment_summary(self, results):
        treatments = results.get('treatments', {})
        if not treatments:
            return
            
        st.markdown("### 🌱 Treatment Plan")
        
        # Display immediate actions
        immediate_actions = treatments.get('treatment_plan', {}).get('immediate_actions', [])
        if immediate_actions:
            st.markdown("**⚡ Immediate Actions:**")
            for action in immediate_actions[:2]:
                method = action.get('method', 'N/A')
                details = action.get('details', 'N/A')
                st.markdown(f"• {method}: {details}")
        
        # Display IPM approach
        ipm_approach = treatments.get('ipm_approach', {}).get('approach')
        if ipm_approach:
            st.info(f"**IPM Approach:** {ipm_approach}")
    
    def _display_action_buttons(self, results):
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("💬 Chat About Treatment"):
                st.session_state.navigate_to_chat = True
                st.session_state.last_identification = results
                SessionStateManager.setup_chat_navigation(results, self.system)
                st.success("🔄 Switching to Chat Assistant...")
                st.rerun()
        
        with col_b:
            if st.button("📚 View Treatment Library"):
                st.session_state.navigate_to_library = True
                st.session_state.library_pest_context = results.get('pest_type')
                st.success("🔄 Switching to Treatment Library...")
                st.rerun()
    
    def _display_failed_results(self, results):
        error_message = f"Message: {results['message']}" if 'message' in results else ""
        detection_method = results.get('detection_method', 'unknown')
        st.markdown(f"""
        <div class="error-box">
            <p><strong>❌ Identification Failed</strong></p>
            {f"<p>{error_message}</p>" if error_message else ""}
            <p><strong>🤖 Model Used:</strong> {detection_method}</p>
        </div>
        """, unsafe_allow_html=True)


class ChatPage:
    """Handles the chat assistant page."""
    
    def __init__(self, system):
        self.system = system
    
    def display(self):
        st.markdown('<h2 class="section-header">💬 AI Chat Assistant</h2>', unsafe_allow_html=True)
        
        SessionStateManager.init_chat_history()
        self._ensure_welcome_message()
        
        self._display_chat_history()
        self._handle_user_input()
        self._display_quick_actions()
    
    def _ensure_welcome_message(self):
        if not st.session_state.chat_history:
            welcome_msg = self.system.chat_with_system("Hello")
            st.session_state.chat_history.append(("Assistant", welcome_msg))
    
    def _display_chat_history(self):
        chat_container = st.container()
        
        with chat_container:
            for role, message in st.session_state.chat_history:
                icon = "👨‍🌾" if role == "User" else "🤖"
                st.markdown(f"**{icon} {role}:** {message}")
                st.markdown("---")
    
    def _handle_user_input(self):
        user_input = st.chat_input("Ask me anything about organic pest management...")
        
        if user_input:
            st.session_state.chat_history.append(("User", user_input))
            
            context = st.session_state.get('last_identification')
            ai_response = self._get_ai_response(user_input, context)
            st.session_state.chat_history.append(("Assistant", ai_response))
            
            st.rerun()
    
    def _get_ai_response(self, user_input, context):
        try:
            return self.system.chat_with_system(user_input, context)
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def _display_quick_actions(self):
        st.markdown("### 🚀 Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        quick_actions = [
            ("🔍 Pest Identification Help", "I need help identifying a pest in my crops. What should I do?"),
            ("🌱 Organic Treatment Options", "What organic treatment options do you recommend?"),
            ("⏰ Treatment Timing", "When is the best time to apply pest treatments?")
        ]
        
        for i, (button_text, message) in enumerate(quick_actions):
            col = [col1, col2, col3][i]
            with col:
                if st.button(button_text):
                    self._handle_quick_action(message)
    
    def _handle_quick_action(self, message):
        st.session_state.chat_history.append(("User", message))
        response = self.system.chat_with_system(message)
        st.session_state.chat_history.append(("Assistant", response))
        st.rerun()


class TreatmentLibraryPage:
    """Handles the treatment library page."""
    
    def __init__(self, system):
        self.system = system
    
    def display(self):
        st.markdown('<h2 class="section-header">📚 Organic Treatment Library</h2>', unsafe_allow_html=True)
        
        # Check if we have pest context from identification
        pest_context = st.session_state.get('library_pest_context')
        if pest_context:
            st.info(f"🎯 Showing treatments for recently identified pest: **{pest_context}**")
            # Clear the context after showing it once
            if st.button("🔄 Browse All Pests"):
                st.session_state.library_pest_context = None
                st.rerun()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_category, selected_pest = self._display_selection_panel(pest_context)
        
        with col2:
            self._display_treatment_details(selected_category, selected_pest)
    
    def _display_selection_panel(self, pest_context=None):
        st.markdown("### 🎯 Treatment Categories")
        
        categories = {
            "🐛 Biological Controls": "biological",
            "🌾 Cultural Controls": "cultural", 
            "🔧 Mechanical Controls": "mechanical",
            "🛡️ Preventive Measures": "preventive"
        }
        
        selected_category = st.radio("Select Category:", list(categories.keys()))
        
        st.markdown("### 🐛 Select Pest Type")
        pest_types = [
            "Aphids", "Caterpillars", "Spider Mites", "Whitefly",
            "Thrips", "Colorado Potato Beetle", "Cucumber Beetle", "Flea Beetle"
        ]
        
        # If we have pest context, try to find and pre-select it
        default_index = 0
        if pest_context:
            # Create a mapping from detected pest names to library entries
            pest_mapping = {
                'ants': 'Aphids',  # Closest match in available types
                'bees': 'Aphids',  # Beneficial insects, but show organic-friendly options
                'beetle': 'Colorado Potato Beetle',
                'caterpillars': 'Caterpillars',
                'catterpillar': 'Caterpillars',
                'earthworms': 'Aphids',  # Beneficial, show general organic options
                'earwig': 'Flea Beetle',  # Similar pest category
                'grasshopper': 'Caterpillars',  # Similar damage patterns
                'moth': 'Caterpillars',  # Moth larvae are caterpillars
                'slug': 'Aphids',  # Show general organic pest control
                'snail': 'Aphids',  # Show general organic pest control
                'wasp': 'Aphids',  # Often beneficial, show organic-friendly options
                'weevil': 'Cucumber Beetle'  # Similar beetle family
            }
            
            # Try direct mapping first
            pest_lower = pest_context.lower()
            if pest_lower in pest_mapping:
                target_pest = pest_mapping[pest_lower]
                try:
                    default_index = pest_types.index(target_pest)
                except ValueError:
                    pass
            else:
                # Try to find the pest in our list (case-insensitive matching)
                for i, pest in enumerate(pest_types):
                    if pest.lower() in pest_lower or pest_lower in pest.lower():
                        default_index = i
                        break
        
        selected_pest = st.selectbox("Pest Type:", pest_types, index=default_index)
        
        return selected_category, selected_pest
    
    def _display_treatment_details(self, selected_category, selected_pest):
        # Check if this is from pest identification context
        pest_context = st.session_state.get('library_pest_context')
        if pest_context and pest_context.lower() != selected_pest.lower():
            st.markdown(f"### {selected_category} for {selected_pest}")
            st.info(f"💡 **Note**: You identified **{pest_context}**, but we're showing treatments for **{selected_pest}** as the closest match in our library.")
        else:
            st.markdown(f"### {selected_category} for {selected_pest}")
        
        try:
            treatments = self.system.treatment_engine.get_treatments(selected_pest)
            
            if treatments and 'treatment_plan' in treatments:
                # If we have context from identification, show a quick summary first
                if pest_context:
                    self._display_identification_context_summary(treatments, pest_context)
                
                self._display_treatment_by_category(selected_category, treatments)
                self._display_additional_info(treatments)
                st.success("✅ All treatments are organic-certified and OMRI-approved")
            
        except Exception as e:
            st.error(f"Error loading treatment information: {str(e)}")
    
    def _display_identification_context_summary(self, treatments, pest_context):
        """Display a summary box when coming from pest identification."""
        st.markdown("### 🎯 Quick Treatment Summary")
        st.caption(f"Based on your identification of: {pest_context}")
        
        # Get immediate actions
        immediate_actions = treatments.get('treatment_plan', {}).get('immediate_actions', [])
        if immediate_actions:
            st.markdown("**⚡ Immediate Actions Recommended:**")
            for action in immediate_actions[:3]:  # Show top 3
                method = action.get('method', 'N/A')
                details = action.get('details', 'N/A')
                st.markdown(f"• **{method}**: {details}")
        
        # Get IPM approach
        ipm_approach = treatments.get('imp_approach', {}).get('approach')
        if ipm_approach:
            st.info(f"**🔄 IPM Strategy**: {ipm_approach}")
        
        st.markdown("---")  # Separator
    
    def _display_treatment_by_category(self, selected_category, treatments):
        category_mapping = {
            "🐛 Biological Controls": ["immediate_actions", "short_term"],
            "🌾 Cultural Controls": ["long_term"],
            "🔧 Mechanical Controls": ["immediate_actions"],
            "🛡️ Preventive Measures": ["prevention_tips"]
        }
        
        treatment_plan = treatments['treatment_plan']
        relevant_sections = category_mapping.get(selected_category, [])
        
        for section in relevant_sections:
            if section == "prevention_tips" and section in treatments:
                st.markdown("**Prevention Strategies:**")
                for tip in treatments[section]:
                    st.markdown(f"• {tip}")
            elif section in treatment_plan:
                section_title = section.replace('_', ' ').title()
                self._display_treatment_section(f"⚡ {section_title}", treatment_plan[section])
    
    def _display_treatment_section(self, title, treatments):
        if not treatments:
            return
            
        st.markdown(f"**{title}:**")
        for treatment in treatments:
            method = treatment.get('method', 'N/A')
            details = treatment.get('details', 'N/A')
            effectiveness = treatment.get('effectiveness', 'N/A')
            cost = treatment.get('cost', 'N/A')
            
            with st.expander(f"🌱 {method}"):
                st.markdown(f"**Details:** {details}")
                st.markdown(f"**Effectiveness:** {effectiveness}")
                st.markdown(f"**Cost:** {cost}")
                st.markdown("**Organic Certified:** ✅")
    
    def _display_additional_info(self, treatments):
        if 'ipm_approach' in treatments:
            st.markdown("### 🔄 IPM Approach")
            approach = treatments['ipm_approach'].get('approach', 'Integrated management approach')
            st.info(approach)


class SidebarManager:
    """Manages the sidebar content."""
    
    @staticmethod
    def create_sidebar():
        st.sidebar.title("🧭 Navigation")
        
        pages = NavigationHandler.get_pages()
        
        # Handle programmatic navigation by forcing radio button selection
        if st.session_state.get('navigate_to_chat', False):
            # Clear the flag and set the radio button state
            st.session_state.navigate_to_chat = False
            # Set the radio button state directly
            st.session_state.page_radio = CHAT_ASSISTANT_PAGE
            
        elif st.session_state.get('navigate_to_library', False):
            # Clear the flag and set the radio button state
            st.session_state.navigate_to_library = False
            # Set the radio button state directly
            st.session_state.page_radio = TREATMENT_LIBRARY_PAGE
        
        # Create the radio button without manual index to avoid warning
        # Let Streamlit handle the state automatically
        st.sidebar.radio(
            "Select Page", 
            pages, 
            key="page_radio"
        )
    
    @staticmethod
    def _display_quick_stats():
        st.sidebar.markdown("### 📊 Quick Stats")
        st.sidebar.metric("Pests in Database", "8 Common Types")
        st.sidebar.metric("Treatment Methods", "50+ Organic Options")
        st.sidebar.metric("System Status", "✅ Online")
    
    @staticmethod
    def _display_emergency_info():
        st.sidebar.markdown("### 🚨 Emergency")
        st.sidebar.info("""
        **Severe Infestations:**
        Contact your local agricultural extension office for immediate assistance.
        
        **System Issues:**
        Check the System Status page for troubleshooting.
        """)


class StreamlitApp:
    """Main Streamlit application class."""
    
    def __init__(self, system):
        self.system = system
        self.setup_page_config()
        AppStyles.apply_custom_css()
        
        # Initialize page handlers
        self.pages = {
            'Pest Identification': PestIdentificationPage(system),
            CHAT_ASSISTANT_PAGE: ChatPage(system),
            TREATMENT_LIBRARY_PAGE: TreatmentLibraryPage(system)
        }
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Organic Farm Pest Management AI",
            page_icon="🌱",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def run(self):
        """Run the Streamlit application."""
        st.markdown('<h1 class="main-header">🌱 Organic Farm Pest Management AI System</h1>', 
                   unsafe_allow_html=True)
        
        SidebarManager.create_sidebar()
        
        current_page = NavigationHandler.handle_navigation_triggers()
        page_handler = self.pages.get(current_page)
        
        if page_handler:
            page_handler.display()
        else:
            st.error(f"Page '{current_page}' not found")


def create_app(pest_system):
    """
    Create Streamlit web application for the pest management system.
    
    Args:
        pest_system: The main PestManagementSystem instance
        
    Returns:
        StreamlitApp: Configured Streamlit application
    """
    return StreamlitApp(pest_system)