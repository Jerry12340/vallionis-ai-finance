#!/bin/bash
# Install Ollama and download quantized models for ARM64
# Run this after setup-vm.sh

set -e

echo "🤖 Installing Ollama for ARM64..."

# Install Ollama
echo "📥 Downloading and installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
echo "🚀 Starting Ollama service..."
sudo systemctl enable ollama
sudo systemctl start ollama

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to be ready..."
sleep 10

# Download recommended models (Q4 quantized for ARM efficiency)
echo "📦 Downloading AI models..."

# Llama 3.1 8B Instruct (excellent for financial advice)
echo "🦙 Downloading Llama 3.1 8B Instruct (Q4)..."
ollama pull llama3.1:8b-instruct-q4_0

# Mistral 7B Instruct (fast and efficient)
echo "🌟 Downloading Mistral 7B Instruct (Q4)..."
ollama pull mistral:7b-instruct-q4_0

# Phi-3 Mini (ultra-lightweight backup model)
echo "🔬 Downloading Phi-3 Mini (Q4)..."
ollama pull phi3:mini-4k-instruct-q4_0

# Test the installation
echo "🧪 Testing Ollama installation..."
if ollama list | grep -q "llama3.1"; then
    echo "✅ Ollama installation successful!"
    echo "📊 Available models:"
    ollama list
else
    echo "❌ Ollama installation failed!"
    exit 1
fi

# Configure Ollama for external access (if needed)
echo "🔧 Configuring Ollama..."
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf << EOF
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_ORIGINS=*"
EOF

# Restart Ollama with new configuration
sudo systemctl daemon-reload
sudo systemctl restart ollama

# Create a simple test script
tee ~/test-ollama.py << EOF
#!/usr/bin/env python3
import requests
import json

def test_ollama():
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "llama3.1:8b-instruct-q4_0",
        "prompt": "What is compound interest in simple terms?",
        "stream": False
    }
    
    try:
        response = requests.post(url, json=data, timeout=60)
        if response.status_code == 200:
            result = response.json()
            print("✅ Ollama is working!")
            print("Response:", result.get('response', '')[:200] + "...")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

if __name__ == "__main__":
    test_ollama()
EOF

chmod +x ~/test-ollama.py

echo "✅ Ollama setup complete!"
echo "🎯 Default model: llama3.1:8b-instruct-q4_0"
echo "🔗 API endpoint: http://localhost:11434"
echo "🧪 Test with: python3 ~/test-ollama.py"
echo ""
echo "💾 Model storage usage:"
du -sh /mnt/models/* 2>/dev/null || echo "Models stored in: ~/.ollama/models"
