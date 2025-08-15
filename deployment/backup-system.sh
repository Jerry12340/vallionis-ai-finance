#!/bin/bash
# Automated backup system for Vallionis AI Finance Coach
# Backs up database, models, and configurations to OCI Object Storage

set -e

# Configuration
BACKUP_DIR="/home/ubuntu/vallionis-ai/backups"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# OCI Object Storage configuration (set these in your environment)
OCI_BUCKET_NAME="${OCI_BUCKET_NAME:-vallionis-ai-backups}"
OCI_NAMESPACE="${OCI_NAMESPACE:-your-namespace}"
OCI_REGION="${OCI_REGION:-us-ashburn-1}"

echo "🔄 Starting Vallionis AI backup process..."

# Create backup directory
mkdir -p "$BACKUP_DIR"

# 1. Database backup
echo "🗄️ Backing up PostgreSQL database..."
pg_dump -h localhost -U vallionis -d vallionis_ai > "$BACKUP_DIR/database_$DATE.sql"
gzip "$BACKUP_DIR/database_$DATE.sql"

# 2. Configuration backup
echo "⚙️ Backing up configurations..."
tar -czf "$BACKUP_DIR/config_$DATE.tar.gz" \
    /home/ubuntu/vallionis-ai/.env \
    /home/ubuntu/vallionis-ai/requirements.txt \
    /etc/systemd/system/vallionis-ai-api.service \
    /etc/nginx/sites-available/vallionis-ai

# 3. Application code backup
echo "💾 Backing up application code..."
tar -czf "$BACKUP_DIR/app_$DATE.tar.gz" \
    /home/ubuntu/vallionis-ai/*.py \
    /home/ubuntu/vallionis-ai/*.sh

# 4. Model metadata backup (models are large, backup metadata only)
echo "🤖 Backing up model metadata..."
ollama list > "$BACKUP_DIR/models_$DATE.txt"

# 5. Upload to OCI Object Storage (if configured)
if command -v oci &> /dev/null && [ -n "$OCI_BUCKET_NAME" ]; then
    echo "☁️ Uploading backups to OCI Object Storage..."
    
    for file in "$BACKUP_DIR"/*_$DATE.*; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            oci os object put \
                --bucket-name "$OCI_BUCKET_NAME" \
                --namespace "$OCI_NAMESPACE" \
                --file "$file" \
                --name "backups/$filename" \
                --force
            echo "✅ Uploaded: $filename"
        fi
    done
else
    echo "⚠️ OCI CLI not configured or bucket name not set. Skipping cloud upload."
    echo "Local backups stored in: $BACKUP_DIR"
fi

# 6. Cleanup old local backups
echo "🧹 Cleaning up old backups (keeping last $RETENTION_DAYS days)..."
find "$BACKUP_DIR" -name "*_*.sql.gz" -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR" -name "*_*.tar.gz" -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR" -name "*_*.txt" -mtime +$RETENTION_DAYS -delete

# 7. Generate backup report
echo "📊 Generating backup report..."
tee "$BACKUP_DIR/backup_report_$DATE.txt" << EOF
Vallionis AI Finance Coach - Backup Report
==========================================
Date: $(date)
Backup ID: $DATE

Files Created:
- Database: database_$DATE.sql.gz
- Configuration: config_$DATE.tar.gz  
- Application: app_$DATE.tar.gz
- Models: models_$DATE.txt

Backup Location: $BACKUP_DIR
Cloud Storage: ${OCI_BUCKET_NAME:-Not configured}

System Status:
- PostgreSQL: $(systemctl is-active postgresql)
- Ollama: $(systemctl is-active ollama)
- API: $(systemctl is-active vallionis-ai-api)
- Nginx: $(systemctl is-active nginx)

Disk Usage:
$(df -h /home/ubuntu/vallionis-ai)

Model Storage:
$(du -sh /mnt/models/* 2>/dev/null || echo "No models found")
EOF

echo "✅ Backup process completed successfully!"
echo "📋 Report saved: $BACKUP_DIR/backup_report_$DATE.txt"
echo "💾 Backup files:"
ls -lh "$BACKUP_DIR"/*_$DATE.*
