#!/usr/bin/env python3
"""Quick validation script for the FastAPI app."""
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from yield_arbitrage.main import create_app

if __name__ == "__main__":
    try:
        app = create_app()
        print("✅ FastAPI app created successfully")
        print(f"✅ App title: {app.title}")
        print(f"✅ App version: {app.version}")
        print("✅ Subtask 1.2 validation passed!")
    except Exception as e:
        print(f"❌ Error creating app: {e}")
        sys.exit(1)