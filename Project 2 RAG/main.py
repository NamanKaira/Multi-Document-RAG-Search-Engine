"""
Entry point for the Hybrid RAG Search Engine.

Usage:
    python main.py

This will launch the Streamlit application.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Streamlit application."""
    app_path = Path(__file__).parent / "app" / "streamlit_app.py"
    
    print("🚀 Starting Hybrid RAG Search Engine...")
    print(f"📱 Launching Streamlit app from: {app_path}")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            check=True
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
