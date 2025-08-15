#!/bin/bash
# Quick Start Script for Vallionis AI Finance Coach on OCI
# This script orchestrates the entire deployment process

set -e

echo "🚀 Vallionis AI Finance Coach - Quick Start Deployment"
echo "====================================================="
echo ""

# Check if running on OCI VM
if ! grep -q "Oracle" /sys/class/dmi/id/sys_vendor 2>/dev/null; then
    echo "⚠️  Warning: This doesn't appear to be an Oracle Cloud VM"
    echo "This script is optimized for OCI Always Free tier"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Get user configuration
echo "📋 Configuration Setup"
echo "====================="

read -p "Enter your domain name (or press Enter for localhost): " DOMAIN_NAME
DOMAIN_NAME=${DOMAIN_NAME:-localhost}

if [ "$DOMAIN_NAME" != "localhost" ]; then
    read -p "Enter your email for SSL certificates: " EMAIL
    EMAIL=${EMAIL:-admin@$DOMAIN_NAME}
fi

read -p "Enter database password (or press Enter for default): " DB_PASSWORD
DB_PASSWORD=${DB_PASSWORD:-secure_password_change_me}

echo ""
echo "🔧 Configuration Summary:"
echo "Domain: $DOMAIN_NAME"
echo "Email: ${EMAIL:-N/A}"
echo "Database Password: [HIDDEN]"
echo ""

read -p "Proceed with installation? (Y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    exit 1
fi

# Export configuration for scripts
export DOMAIN_NAME
export EMAIL
export DB_PASSWORD

echo ""
echo "🎯 Starting deployment process..."
echo "This will take approximately 15-30 minutes depending on your connection speed."
echo ""

# Step 1: VM Setup
echo "📦 Step 1/5: Setting up VM and dependencies..."
chmod +x setup-vm.sh
./setup-vm.sh

# Step 2: Install Ollama and Models
echo "🤖 Step 2/5: Installing Ollama and AI models..."
chmod +x install-ollama.sh
./install-ollama.sh

# Step 3: Setup Database
echo "🗄️ Step 3/5: Setting up PostgreSQL with pgvector..."
chmod +x setup-database.sh
# Update database password in script
sed -i "s/secure_password_change_me/$DB_PASSWORD/g" setup-database.sh
./setup-database.sh

# Step 4: Deploy API
echo "🚀 Step 4/5: Deploying FastAPI application..."
chmod +x deploy-api.sh
# Update database password in API deployment
sed -i "s/secure_password_change_me/$DB_PASSWORD/g" deploy-api.sh
./deploy-api.sh

# Step 5: Setup SSL (if domain provided)
if [ "$DOMAIN_NAME" != "localhost" ]; then
    echo "🔒 Step 5/5: Setting up SSL certificates..."
    chmod +x setup-ssl.sh
    ./setup-ssl.sh
else
    echo "⚠️ Step 5/5: Skipping SSL setup (localhost deployment)"
fi

# Setup backup system
echo "💾 Setting up automated backups..."
chmod +x backup-system.sh
# Schedule daily backups at 2 AM
(crontab -l 2>/dev/null; echo "0 2 * * * /home/ubuntu/vallionis-ai/backup-system.sh") | crontab -

# Final verification
echo ""
echo "🧪 Running final verification tests..."

# Test services
services=("postgresql" "ollama" "vallionis-ai-api" "nginx")
all_running=true

for service in "${services[@]}"; do
    if systemctl is-active --quiet $service; then
        echo "✅ $service: RUNNING"
    else
        echo "❌ $service: STOPPED"
        all_running=false
    fi
done

# Test API endpoints
echo ""
echo "🔗 Testing API endpoints..."

if [ "$DOMAIN_NAME" = "localhost" ]; then
    BASE_URL="http://localhost"
else
    BASE_URL="https://$DOMAIN_NAME"
fi

# Health check
if curl -s "$BASE_URL/api/health" > /dev/null; then
    echo "✅ Health check: PASSED"
else
    echo "❌ Health check: FAILED"
    all_running=false
fi

# Test chat endpoint
chat_response=$(curl -s -X POST "$BASE_URL/api/chat" \
    -H "Content-Type: application/json" \
    -d '{"message":"What is compound interest?"}' | jq -r '.response' 2>/dev/null)

if [ "$chat_response" != "null" ] && [ -n "$chat_response" ]; then
    echo "✅ Chat endpoint: PASSED"
else
    echo "❌ Chat endpoint: FAILED"
    all_running=false
fi

echo ""
if [ "$all_running" = true ]; then
    echo "🎉 DEPLOYMENT SUCCESSFUL!"
    echo "========================="
    echo ""
    echo "🔗 Your AI Finance Coach is ready at:"
    echo "   Health Check: $BASE_URL/api/health"
    echo "   Chat API: $BASE_URL/api/chat"
    echo "   Retrieve API: $BASE_URL/api/retrieve"
    echo "   Coach API: $BASE_URL/api/coach"
    echo ""
    echo "📊 System Information:"
    echo "   Models: $(ollama list | grep -v NAME | wc -l) AI models installed"
    echo "   Database: PostgreSQL with pgvector ready"
    echo "   Storage: $(df -h /mnt/models | tail -1 | awk '{print $4}') free space"
    echo ""
    echo "🛠️ Management Commands:"
    echo "   Monitor: ./monitor_services.sh"
    echo "   Backup: ./backup-system.sh"
    echo "   Logs: sudo journalctl -u vallionis-ai-api -f"
    echo ""
    echo "📚 Next Steps:"
    echo "1. Customize the knowledge base with your financial content"
    echo "2. Integrate with your existing Flask application (see integration_guide.md)"
    echo "3. Set up monitoring and alerting"
    echo "4. Configure your domain's DNS to point to this server"
    echo ""
    echo "💰 Monthly Cost: $0.00 (Oracle Always Free tier)"
else
    echo "❌ DEPLOYMENT ISSUES DETECTED"
    echo "============================="
    echo ""
    echo "Some services are not running properly. Check the logs:"
    echo "sudo journalctl -u vallionis-ai-api -n 50"
    echo "sudo journalctl -u ollama -n 50"
    echo ""
    echo "You can also run individual setup scripts to debug:"
    echo "./setup-vm.sh"
    echo "./install-ollama.sh" 
    echo "./setup-database.sh"
    echo "./deploy-api.sh"
fi

echo ""
echo "📋 Deployment log saved to: ~/vallionis-ai/deployment.log"

# Save deployment info
tee ~/vallionis-ai/deployment.log << EOF
Vallionis AI Finance Coach - Deployment Summary
==============================================
Date: $(date)
Domain: $DOMAIN_NAME
Base URL: $BASE_URL

Services Status:
$(systemctl is-active postgresql ollama vallionis-ai-api nginx)

System Resources:
$(free -h)
$(df -h)

Models Installed:
$(ollama list)

Configuration Files:
- API: ~/vallionis-ai/api.py
- Database: ~/vallionis-ai/db_config.py
- Nginx: /etc/nginx/sites-available/vallionis-ai
- Service: /etc/systemd/system/vallionis-ai-api.service

Backup Schedule:
$(crontab -l | grep backup)

Next Steps:
1. Integrate with Flask app using integration_guide.md
2. Customize knowledge base
3. Set up monitoring
4. Configure DNS (if using custom domain)
EOF
