#!/bin/bash
# Quick deployment commands for OCI Always Free tier

# 1. Upload AI Coach API to your OCI instance
# scp deployment/ai_coach_api.py ubuntu@YOUR_OCI_IP:/home/ubuntu/

# 2. Install dependencies on OCI
sudo apt update
pip3 install fastapi uvicorn httpx psycopg2-binary pgvector sentence-transformers numpy

# 3. Create systemd service for AI Coach API
sudo tee /etc/systemd/system/ai-coach.service > /dev/null <<EOF
[Unit]
Description=Vallionis AI Coach API
After=network.target ollama.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu
ExecStart=/home/ubuntu/.local/bin/uvicorn ai_coach_api:app --host 0.0.0.0 --port 8001
Restart=always
RestartSec=3
Environment=OLLAMA_BASE_URL=http://localhost:11434
Environment=ENABLE_DB=true
Environment=DATABASE_URL=postgresql://ai_coach:your_password@localhost/ai_coach_db

[Install]
WantedBy=multi-user.target
EOF

# 4. Enable and start the service
sudo systemctl enable ai-coach
sudo systemctl start ai-coach
sudo systemctl status ai-coach

# 5. Configure nginx to proxy AI requests
sudo tee /etc/nginx/sites-available/ai-coach > /dev/null <<EOF
server {
    listen 80;
    server_name YOUR_DOMAIN_OR_IP;
    
    location /api/ {
        proxy_pass http://localhost:8001/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
    
    location / {
        return 200 "Vallionis AI Coach API is running";
        add_header Content-Type text/plain;
    }
}
EOF

# 6. Enable nginx site
sudo ln -s /etc/nginx/sites-available/ai-coach /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

echo "AI Coach API deployed! Test with:"
echo "curl http://YOUR_IP/api/health"
