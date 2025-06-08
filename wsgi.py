import sys
from pathlib import Path
from waitress import serve
from app import app

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

application = app

if __name__ == "__main__":
    serve(application, host='0.0.0.0', port=5000)
