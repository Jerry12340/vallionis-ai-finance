#!/bin/bash
# Initial VM Setup for Vallionis AI Finance Coach
# Run this script first after SSH into your OCI VM

set -e

echo "🚀 Setting up Vallionis AI Finance Coach on OCI..."

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential packages
echo "🔧 Installing essential packages..."
sudo apt install -y \
    curl \
    wget \
    git \
    htop \
    nginx \
    python3 \
    python3-pip \
    python3-venv \
    postgresql \
    postgresql-contrib \
    postgresql-server-dev-all \
    build-essential \
    pkg-config \
    libssl-dev \
    libffi-dev \
    unzip

# Install Docker (for easier service management)
echo "🐳 Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
rm get-docker.sh

# Create directories
echo "📁 Creating application directories..."
mkdir -p ~/vallionis-ai
mkdir -p ~/vallionis-ai/models
mkdir -p ~/vallionis-ai/data
mkdir -p ~/vallionis-ai/logs
mkdir -p ~/vallionis-ai/backups

# Mount block volume for models
echo "💾 Setting up block volume for models..."
# Note: Replace /dev/sdb with your actual block device
sudo mkfs.ext4 /dev/sdb
sudo mkdir -p /mnt/models
sudo mount /dev/sdb /mnt/models
sudo chown ubuntu:ubuntu /mnt/models

# Add to fstab for persistent mounting
echo "/dev/sdb /mnt/models ext4 defaults 0 2" | sudo tee -a /etc/fstab

# Create symlink to models directory
ln -sf /mnt/models ~/vallionis-ai/models

# Configure firewall
echo "🔥 Configuring firewall..."
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443
sudo ufw --force enable

# Set up log rotation
echo "📝 Configuring log rotation..."
sudo tee /etc/logrotate.d/vallionis-ai << EOF
/home/ubuntu/vallionis-ai/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 ubuntu ubuntu
}
EOF

# Install Node.js (for potential frontend builds)
echo "📦 Installing Node.js..."
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

echo "✅ VM setup complete!"
echo "Next steps:"
echo "1. Run install-ollama.sh to set up the LLM"
echo "2. Run setup-database.sh to configure PostgreSQL"
echo "3. Run deploy-api.sh to deploy the FastAPI application"
