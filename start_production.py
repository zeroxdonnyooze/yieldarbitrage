#!/usr/bin/env python3
"""Production startup script for Railway deployment."""

import os
import sys
import uvicorn
from pathlib import Path

# Add the src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def main():
    """Start the production server."""
    # Import after path is set
    from yield_arbitrage.config.settings import settings
    
    # Ensure we're in production mode
    os.environ.setdefault("ENVIRONMENT", "production")
    
    # Configure uvicorn for production
    uvicorn.run(
        "yield_arbitrage.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        workers=int(os.environ.get("WEB_CONCURRENCY", 1)),
        log_level="info",
        access_log=True,
        proxy_headers=True,
        forwarded_allow_ips="*",
        reload=False
    )

if __name__ == "__main__":
    main()