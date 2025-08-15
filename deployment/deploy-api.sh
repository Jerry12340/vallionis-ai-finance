#!/bin/bash
# Deploy the Vallionis AI Finance Coach API
# Run this after setup-database.sh

set -e

echo "🚀 Deploying Vallionis AI Finance Coach API..."

# Create Python virtual environment
echo "🐍 Setting up Python environment..."
cd ~/vallionis-ai
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip

# Create requirements file for the API
tee requirements.txt << EOF
fastapi==0.104.1
uvicorn[standard]==0.24.0
httpx==0.25.2
psycopg2-binary==2.9.9
pgvector==0.2.4
sentence-transformers==2.2.2
numpy==1.24.3
pydantic==2.5.0
python-multipart==0.0.6
jinja2==3.1.2
python-dotenv==1.0.0
prometheus-client==0.19.0
EOF

pip install -r requirements.txt

# Copy the API file
echo "📋 Setting up API application..."
cp /home/ubuntu/vallionis-ai-finance-website/deployment/ai_coach_api.py ./api.py

# Create environment configuration
echo "🔧 Creating environment configuration..."
tee .env << EOF
# Database Configuration
DB_PASSWORD=secure_password_change_me
DB_HOST=localhost
DB_PORT=5432
DB_NAME=vallionis_ai
DB_USER=vallionis

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=llama3.1:8b-instruct-q4_0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Security (change these!)
SECRET_KEY=your-secret-key-change-me
CORS_ORIGINS=*
EOF

# Create systemd service for the API
echo "⚙️ Creating systemd service..."
sudo tee /etc/systemd/system/vallionis-ai-api.service << EOF
[Unit]
Description=Vallionis AI Finance Coach API
After=network.target postgresql.service ollama.service
Requires=postgresql.service ollama.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/vallionis-ai
Environment=PATH=/home/ubuntu/vallionis-ai/venv/bin
ExecStart=/home/ubuntu/vallionis-ai/venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=10

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=vallionis-ai-api

# Security
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/home/ubuntu/vallionis-ai

[Install]
WantedBy=multi-user.target
EOF

# Create embedding generation script
echo "🔍 Creating embedding generation script..."
tee generate_embeddings.py << EOF
#!/usr/bin/env python3
"""
Generate embeddings for existing documents in the knowledge base
"""
import sys
import os
sys.path.append('/home/ubuntu/vallionis-ai')

from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np
from db_config import get_db_connection

def generate_embeddings():
    """Generate embeddings for all documents without embeddings"""
    print("🔍 Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("📊 Connecting to database...")
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Get documents without embeddings
    cur.execute("""
        SELECT d.id, d.title, d.content 
        FROM documents d 
        LEFT JOIN embeddings e ON d.id = e.document_id 
        WHERE e.document_id IS NULL
    """)
    
    documents = cur.fetchall()
    print(f"📚 Found {len(documents)} documents to process")
    
    for doc_id, title, content in documents:
        print(f"Processing: {title}")
        
        # Split content into chunks (simple sentence splitting)
        sentences = content.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < 500:  # Keep chunks under 500 chars
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Generate embeddings for each chunk
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 10:  # Skip very short chunks
                embedding = model.encode(chunk)
                
                cur.execute("""
                    INSERT INTO embeddings (document_id, chunk_text, embedding, chunk_index)
                    VALUES (%s, %s, %s, %s)
                """, (doc_id, chunk, embedding.tolist(), i))
    
    conn.commit()
    cur.close()
    conn.close()
    
    print("✅ Embedding generation complete!")

if __name__ == "__main__":
    generate_embeddings()
EOF

chmod +x generate_embeddings.py

# Create nginx configuration for reverse proxy
echo "🌐 Configuring Nginx reverse proxy..."
sudo tee /etc/nginx/sites-available/vallionis-ai << EOF
server {
    listen 80;
    server_name _;  # Replace with your domain
    
    # API endpoints
    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
        
        # Streaming support
        proxy_buffering off;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
    
    # Health check
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
    
    # Static files (if serving frontend)
    location / {
        root /var/www/html;
        try_files \$uri \$uri/ /index.html;
    }
}
EOF

# Enable the site
sudo ln -sf /etc/nginx/sites-available/vallionis-ai /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test nginx configuration
sudo nginx -t

# Create startup script
echo "🔄 Creating startup script..."
tee start_services.sh << EOF
#!/bin/bash
# Start all Vallionis AI services

echo "🚀 Starting Vallionis AI Finance Coach services..."

# Start PostgreSQL
sudo systemctl start postgresql
echo "✅ PostgreSQL started"

# Start Ollama
sudo systemctl start ollama
echo "✅ Ollama started"

# Wait for services to be ready
sleep 5

# Start API
sudo systemctl start vallionis-ai-api
echo "✅ API started"

# Start Nginx
sudo systemctl start nginx
echo "✅ Nginx started"

# Enable services for auto-start
sudo systemctl enable postgresql ollama vallionis-ai-api nginx

echo "🎉 All services started successfully!"
echo "🔗 API available at: http://localhost/api/health"
echo "📊 Check status with: sudo systemctl status vallionis-ai-api"
EOF

chmod +x start_services.sh

# Create monitoring script
tee monitor_services.sh << EOF
#!/bin/bash
# Monitor Vallionis AI services

echo "📊 Vallionis AI Finance Coach - Service Status"
echo "=============================================="

services=("postgresql" "ollama" "vallionis-ai-api" "nginx")

for service in "\${services[@]}"; do
    if systemctl is-active --quiet \$service; then
        echo "✅ \$service: RUNNING"
    else
        echo "❌ \$service: STOPPED"
    fi
done

echo ""
echo "🔗 Quick Tests:"
echo "Health Check: curl -s http://localhost/api/health | jq"
echo "Ollama Status: curl -s http://localhost:11434/api/tags"
echo ""
echo "📋 Logs:"
echo "API Logs: sudo journalctl -u vallionis-ai-api -f"
echo "Nginx Logs: sudo tail -f /var/log/nginx/access.log"
EOF

chmod +x monitor_services.sh

# Populate sample data and generate embeddings
echo "📚 Populating knowledge base..."
python3 populate_knowledge_base.py

echo "🔍 Generating embeddings..."
python3 generate_embeddings.py

# Start services
echo "🚀 Starting services..."
sudo systemctl daemon-reload
./start_services.sh

# Test the deployment
echo "🧪 Testing deployment..."
sleep 10

if curl -s http://localhost/api/health > /dev/null; then
    echo "✅ API deployment successful!"
    echo "🔗 Health check: http://localhost/api/health"
    echo "💬 Chat endpoint: http://localhost/api/chat"
    echo "🔍 Retrieve endpoint: http://localhost/api/retrieve"
else
    echo "❌ API deployment failed!"
    echo "📋 Check logs: sudo journalctl -u vallionis-ai-api -n 50"
fi

echo ""
echo "🎉 Deployment complete!"
echo "📊 Monitor services: ./monitor_services.sh"
echo "🔧 Manage API: sudo systemctl {start|stop|restart} vallionis-ai-api"
echo ""
echo "Next steps:"
echo "1. Test the API endpoints"
echo "2. Configure your domain and SSL certificates"
echo "3. Set up monitoring and backups"
echo "4. Customize the knowledge base with your content"
EOF
