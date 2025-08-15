# Render + Oracle Cloud Deployment Guide

This guide shows how to deploy your Vallionis AI Finance Coach using **Render for your Flask app** and **Oracle Cloud for the AI backend**.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        RENDER                               │
│                  (Flask Web App)                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Flask     │  │  Templates  │  │    Static Files     │ │
│  │   Routes    │  │             │  │                     │ │
│  │             │  │ ai_coach.   │  │ CSS/JS for AI       │ │
│  │ /ai-coach   │  │ html        │  │ chat interface      │ │
│  │ /api/ai/*   │  │             │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │ HTTPS API Calls
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 ORACLE CLOUD (Always Free)                 │
│                    AI Backend Services                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Ollama    │  │  FastAPI    │  │  PostgreSQL +      │ │
│  │   (LLM)     │  │  Gateway    │  │  pgvector (RAG)    │ │
│  │             │  │             │  │                     │ │
│  │ 7-8B Q4     │  │ /chat       │  │ Finance Knowledge   │ │
│  │ Models      │  │ /retrieve   │  │ Base                │ │
│  │             │  │ /coach      │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Step-by-Step Deployment

### Step 1: Deploy AI Backend to Oracle Cloud

1. **Provision OCI VM** (Ampere A1, Always Free tier)
2. **SSH into your VM** and run:
   ```bash
   # Clone your repository
   git clone https://github.com/your-username/vallionis-ai-finance-website.git
   cd vallionis-ai-finance-website/deployment
   
   # Make scripts executable
   chmod +x *.sh
   
   # Run full deployment
   ./quick-start.sh
   ```

3. **Configure your domain** (optional but recommended):
   - Point your domain/subdomain to your OCI VM IP
   - Run SSL setup: `./setup-ssl.sh`

### Step 2: Update Your Flask App for Render

1. **Add the integration code** to your `app.py`:

```python
# Add this import at the top
from render_integration import *

# The routes are already defined in render_integration.py
```

2. **Update your requirements.txt**:
```txt
# Add to your existing requirements.txt
requests>=2.31.0
```

3. **Add environment variables** in Render dashboard:
```env
AI_COACH_API_URL=https://your-oci-domain.com/api
AI_COACH_ENABLED=true
AI_COACH_TIMEOUT=60
```

### Step 3: Update Navigation

Add AI Coach links to your navigation template:

```html
<!-- Add to your navigation -->
<li class="nav-item">
    <a class="nav-link" href="{{ url_for('ai_coach_page') }}">
        <i class="fas fa-robot"></i> AI Coach
    </a>
</li>
```

### Step 4: Deploy to Render

1. **Push your changes** to your GitHub repository
2. **Render will automatically deploy** your updated Flask app
3. **Test the integration** by visiting `/ai-coach` on your Render app

## 🔧 Configuration

### Environment Variables (Render)

Set these in your Render service environment:

| Variable | Value | Description |
|----------|-------|-------------|
| `AI_COACH_API_URL` | `https://your-domain.com/api` | Your OCI backend URL |
| `AI_COACH_ENABLED` | `true` | Enable/disable AI features |
| `AI_COACH_TIMEOUT` | `60` | API timeout in seconds |

### CORS Configuration (OCI Backend)

Update your FastAPI app to allow Render domain:

```python
# In your ai_coach_api.py on OCI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-render-app.onrender.com",  # Your Render domain
        "https://your-custom-domain.com"         # Your custom domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 💰 Cost Breakdown

### Render Costs
- **Free Tier**: 750 hours/month (sufficient for personal use)
- **Starter Plan**: $7/month (for production use)

### Oracle Cloud Costs
- **Always Free Tier**: $0/month
  - VM.Standard.A1.Flex (4 OCPU, 24GB RAM)
  - 200GB total storage
  - 10TB outbound data transfer

### Total Monthly Cost
- **Development**: $0/month (Render Free + OCI Free)
- **Production**: $7/month (Render Starter + OCI Free)

## 🚀 Benefits of This Architecture

### ✅ Advantages
- **Cost-effective**: Minimal monthly costs
- **Scalable**: Render handles web traffic, OCI handles AI
- **Secure**: AI processing on your own infrastructure
- **Fast deployment**: Use existing Render setup
- **Data privacy**: Sensitive AI data stays on your servers

### ⚠️ Considerations
- **Network latency**: API calls between Render and OCI
- **Complexity**: Managing two separate deployments
- **Monitoring**: Need to monitor both services

## 🔍 Testing Your Deployment

### Test AI Backend (OCI)
```bash
# Health check
curl https://your-oci-domain.com/api/health

# Chat test
curl -X POST https://your-oci-domain.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is compound interest?"}'
```

### Test Render Integration
1. Visit `https://your-render-app.onrender.com/ai-coach`
2. Check connection status (should show "Connected")
3. Send a test message
4. Verify response from AI backend

## 🛠️ Troubleshooting

### Common Issues

**1. Connection Timeout**
- Check OCI firewall rules (ports 80, 443)
- Verify SSL certificates
- Test direct API access

**2. CORS Errors**
- Update CORS origins in FastAPI app
- Restart OCI services: `sudo systemctl restart vallionis-ai-api`

**3. AI Service Offline**
- Check OCI VM status
- Restart services: `./start_services.sh`
- Check logs: `sudo journalctl -u vallionis-ai-api -f`

### Monitoring Commands

**On OCI VM:**
```bash
# Service status
./monitor_services.sh

# View logs
sudo journalctl -u vallionis-ai-api -f

# Resource usage
htop
df -h
```

**On Render:**
- Check Render dashboard for deployment logs
- Monitor response times in browser dev tools

## 🔄 Maintenance

### Regular Tasks
1. **Weekly**: Check OCI VM health and disk space
2. **Monthly**: Review API usage and performance
3. **Quarterly**: Update AI models and dependencies

### Backup Strategy
- **OCI**: Automated daily backups to Object Storage
- **Render**: Code backed up in Git repository
- **Database**: Daily PostgreSQL dumps

## 🎯 Next Steps

1. **Deploy OCI backend** using the provided scripts
2. **Update your Flask app** with integration code
3. **Configure environment variables** in Render
4. **Test the full integration**
5. **Monitor performance** and optimize as needed

This hybrid approach gives you the best of both worlds: reliable web hosting on Render and cost-effective AI services on Oracle Cloud!
