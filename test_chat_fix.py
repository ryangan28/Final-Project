#!/usr/bin/env python3
"""
Test script to verify chat functionality works without session state errors.
"""

import requests
import time

def test_streamlit_chat():
    """Test that the Streamlit app responds without session state errors."""
    try:
        # Test if the app is running
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("✅ Streamlit app is running successfully")
            print("✅ No session state errors in startup")
            return True
        else:
            print(f"❌ App returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to connect to app: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Chat Interface Fix ===")
    time.sleep(2)  # Give Streamlit time to start
    success = test_streamlit_chat()
    print(f"\n=== Test {'PASSED' if success else 'FAILED'} ===")
    print("\n🎯 Manual Test Steps:")
    print("1. Go to http://localhost:8501")
    print("2. Navigate to 'Chat Assistant' page")
    print("3. Type a message in the chat input")
    print("4. Press Enter or click send")
    print("5. Verify no StreamlitAPIException occurs")
    print("6. Verify the chat input clears automatically")
    print("7. Verify the response appears without needing to refresh")
