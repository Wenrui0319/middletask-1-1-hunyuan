#!/usr/bin/env python
"""
Convenient script to run the SAM segmentation server.
"""

import os
import sys
from pathlib import Path

# Add backend to Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

# Change to backend directory
os.chdir(backend_dir)

# Run the server
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting SAM Segmentation Demo Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("💡 Press Ctrl+C to stop the server\n")
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(backend_dir), str(backend_dir.parent / "src")],
        log_level="info"
    )