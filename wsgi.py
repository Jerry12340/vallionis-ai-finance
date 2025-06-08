import sys
from pathlib import Path
from waitress import serve
from app import app
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

application = app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
