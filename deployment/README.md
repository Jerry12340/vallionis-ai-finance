# Self-Hosted AI Finance Coach on Oracle Cloud (Always Free)

This guide will help you deploy a complete AI finance coaching system on Oracle Cloud Infrastructure's Always Free tier, achieving zero SaaS bills while maintaining full control over your data and models.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    OCI Always Free VM                       │
│                  (Ampere A1 - ARM64)                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Ollama    │  │  FastAPI    │  │  PostgreSQL +      │ │
│  │   (LLM)     │  │  Gateway    │  │  pgvector (RAG)    │ │
│  │             │  │             │  │                     │ │
│  │ 7-8B Q4     │  │ /chat       │  │ Finance Knowledge   │ │
│  │ Model       │  │ /retrieve   │  │ Base                │ │
│  │             │  │ /coach      │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                 Block Volume (Models)                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  OCI Object Storage │
                    │  (Static Assets &   │
                    │   Nightly Backups)  │
                    └─────────────────────┘
```

## Prerequisites

1. **Oracle Cloud Account** with Always Free tier access
2. **SSH Key Pair** for VM access
3. **Domain Name** (optional, for HTTPS)

## Step 1: Provision OCI Infrastructure

### 1.1 Create Ampere A1 VM

1. Log into OCI Console
2. Navigate to **Compute > Instances**
3. Click **Create Instance**
4. Configure:
   - **Name**: `vallionis-ai-coach`
   - **Image**: Ubuntu 22.04 (ARM64)
   - **Shape**: VM.Standard.A1.Flex (Always Free)
   - **CPU**: 4 OCPUs (max free tier)
   - **Memory**: 24 GB (max free tier)
   - **Boot Volume**: 47 GB (max free tier)
   - **SSH Keys**: Upload your public key

### 1.2 Create Block Volume for Models

1. Navigate to **Storage > Block Volumes**
2. Click **Create Block Volume**
3. Configure:
   - **Name**: `model-storage`
   - **Size**: 50 GB (Always Free)
   - **Availability Domain**: Same as VM

### 1.3 Configure Security List

1. Navigate to **Networking > Virtual Cloud Networks**
2. Select your VCN > Security Lists
3. Add Ingress Rules:
   - **Port 22**: SSH access
   - **Port 80**: HTTP
   - **Port 443**: HTTPS
   - **Port 11434**: Ollama API (internal)
   - **Port 8000**: FastAPI (internal)

## Step 2: Initial VM Setup

SSH into your VM and run the setup script:

```bash
ssh ubuntu@<your-vm-ip>
```

## Step 3: Install Dependencies

See the installation scripts in this directory:
- `setup-vm.sh` - Initial VM configuration
- `install-ollama.sh` - Ollama installation and model setup
- `setup-database.sh` - PostgreSQL + pgvector setup
- `deploy-api.sh` - FastAPI application deployment

## Step 4: Deploy Your Application

1. Clone your repository to the VM
2. Run the deployment scripts in order
3. Configure environment variables
4. Start all services

## Step 5: Configure Monitoring & Backups

- Set up automated backups to OCI Object Storage
- Configure log rotation
- Set up basic monitoring

## Cost Breakdown (Always Free Tier)

- **Compute**: VM.Standard.A1.Flex (4 OCPU, 24GB RAM) - FREE
- **Storage**: 200 GB Block + Boot volumes - FREE
- **Network**: 10 TB outbound data transfer/month - FREE
- **Object Storage**: 20 GB - FREE

**Total Monthly Cost: $0.00**

## Performance Expectations

With the Always Free tier resources:
- **Model**: 7-8B parameters, Q4 quantization
- **Response Time**: 2-5 seconds for typical queries
- **Concurrent Users**: 5-10 simultaneous conversations
- **Storage**: Sufficient for 10K+ financial documents

## Security Considerations

- All data remains on your infrastructure
- No external API calls for LLM inference
- Encrypted storage and transmission
- Regular security updates via automation

## Next Steps

1. Follow the setup scripts in order
2. Customize the financial knowledge base
3. Configure your domain and SSL certificates
4. Set up monitoring and alerting
