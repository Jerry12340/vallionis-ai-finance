# Render + Oracle Cloud Always Free Deployment Guide

## Architecture
- **Render**: Flask web application (free tier)
- **Oracle Cloud Always Free**: AI Coach API with Ollama + PostgreSQL + pgvector

## Step 1: Deploy AI Coach API to Oracle Cloud

### 1.1 Create OCI VM Instance
```bash
# Use the existing setup scripts from your deployment folder
# On your OCI Always Free Ampere A1 instance (4 OCPUs, 24GB RAM):

# 1. Run the setup scripts
chmod +x deployment/*.sh
./deployment/setup-vm.sh
./deployment/install-ollama.sh
./deployment/setup-database.sh
./deployment/deploy-api.sh
./deployment/setup-ssl.sh
```

### 1.2 Configure AI Coach API for Production
Update your OCI instance with the production AI Coach API:

```bash
# Copy ai_coach_api.py to your OCI instance
scp deployment/ai_coach_api.py ubuntu@your-oci-ip:/home/ubuntu/ai_coach_api.py

# Create systemd service
sudo tee /etc/systemd/system/ai-coach.service > /dev/null <<EOF
[Unit]
Description=Vallionis AI Coach API
After=network.target

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

# Enable and start the service
sudo systemctl enable ai-coach
sudo systemctl start ai-coach
```

### 1.3 Configure Nginx for AI Coach API
```bash
# Add to your nginx configuration
sudo tee /etc/nginx/sites-available/ai-coach > /dev/null <<EOF
server {
    listen 80;
    server_name your-domain.com;
    
    location /api/ {
        proxy_pass http://localhost:8001/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
EOF

# Enable the site
sudo ln -s /etc/nginx/sites-available/ai-coach /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Step 2: Deploy Flask App to Render

### 2.1 Prepare for Render Deployment
Create/update these files in your project root:

**render.yaml** (already exists - update if needed):
```yaml
services:
  - type: web
    name: vallionis-ai-finance
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn wsgi:app
    envVars:
      - key: FLASK_ENV
        value: production
      - key: AI_SERVICE_URL
        value: https://your-oci-domain.com/api/chat
      - key: AI_HEALTH_URL
        value: https://your-oci-domain.com/api/health
      - key: DATABASE_URL
        fromDatabase:
          name: vallionis-db
          property: connectionString
```

### 2.2 Environment Variables for Render
Set these environment variables in your Render dashboard:

```bash
# AI Coach Configuration
AI_SERVICE_URL=https://your-oci-domain.com/api/chat
AI_HEALTH_URL=https://your-oci-domain.com/api/health
AI_COACH_TIMEOUT=60
AI_COACH_ENABLED=true

# Your existing environment variables
FLASK_SECRET_KEY=your-secret-key
STRIPE_PUBLISHABLE_KEY=your-stripe-key
STRIPE_SECRET_KEY=your-stripe-secret
# ... other existing vars
```

## Step 3: Configure Domain and SSL

### 3.1 Set up Domain for OCI
```bash
# On your OCI instance, get SSL certificate
sudo certbot --nginx -d your-domain.com

# Update nginx configuration for HTTPS
sudo tee /etc/nginx/sites-available/ai-coach > /dev/null <<EOF
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    location /api/ {
        proxy_pass http://localhost:8001/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://\$server_name\$request_uri;
}
EOF
```

## Step 4: Test the Integration

### 4.1 Test OCI AI Service
```bash
# Test health endpoint
curl https://your-oci-domain.com/api/health

# Test chat endpoint
curl -X POST https://your-oci-domain.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is alpha in finance?", "model": "mistral:instruct"}'
```

### 4.2 Deploy to Render
```bash
# Push your code to GitHub
git add .
git commit -m "Configure for Render + OCI deployment"
git push origin main

# Connect your GitHub repo to Render and deploy
```

## Cost Breakdown
- **Render**: $0/month (free tier)
- **Oracle Cloud Always Free**: $0/month (4 OCPUs, 24GB RAM, 200GB storage)
- **Domain**: ~$10-15/year (optional, can use IP)
- **Total**: $0/month + optional domain cost

## Benefits of This Setup
- ✅ Zero monthly costs
- ✅ Powerful AI with 7-8B parameter models
- ✅ RAG capabilities with PostgreSQL + pgvector
- ✅ Scalable architecture
- ✅ Professional deployment with SSL
- ✅ Separation of concerns (web app vs AI service)
