# Integration Guide: Vallionis AI Finance Coach with Your Flask App

This guide shows how to integrate the self-hosted AI finance coach with your existing Flask application.

## Architecture Integration

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Flask App                           │
│              (Existing Vallionis Website)                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Routes    │  │  Templates  │  │    Static Files     │ │
│  │             │  │             │  │                     │ │
│  │ /dashboard  │  │ chat.html   │  │ chat.js             │ │
│  │ /chat       │  │ coach.html  │  │ coach.css           │ │
│  │ /coach      │  │             │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │ HTTP Requests
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                OCI Self-Hosted AI Coach                     │
│                    (FastAPI Backend)                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Ollama    │  │  FastAPI    │  │  PostgreSQL +      │ │
│  │   (LLM)     │  │  Gateway    │  │  pgvector (RAG)    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Integration Options

### Option 1: Direct API Integration (Recommended)
Your Flask app makes HTTP requests to the FastAPI backend running on OCI.

### Option 2: Hybrid Deployment
Run both Flask and FastAPI on the same OCI VM with Nginx routing.

### Option 3: Microservices Architecture
Keep Flask app on current hosting, use OCI only for AI services.

## Implementation Steps

### Step 1: Add AI Chat Routes to Your Flask App

Add these routes to your existing `app.py`:

```python
import requests
import json
from flask import render_template, request, jsonify, stream_template

# Configuration
AI_COACH_API_URL = "https://your-oci-domain.com/api"  # Update with your OCI domain
AI_COACH_API_KEY = "your-api-key"  # Optional, for authentication

@app.route('/chat')
@login_required
def chat_page():
    """AI Finance Coach chat interface"""
    return render_template('chat.html', user=current_user)

@app.route('/api/chat', methods=['POST'])
@login_required
def chat_api():
    """Proxy chat requests to OCI AI backend"""
    try:
        data = request.get_json()
        
        # Add user context from your existing user system
        data['user_id'] = str(current_user.id)
        data['context'] = {
            'user_name': current_user.username,
            'user_level': getattr(current_user, 'experience_level', 'beginner'),
            'preferences': getattr(current_user, 'preferences', {})
        }
        
        # Forward request to OCI AI backend
        response = requests.post(
            f"{AI_COACH_API_URL}/chat",
            json=data,
            timeout=60,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': 'AI service unavailable'}), 503
            
    except Exception as e:
        app.logger.error(f"Chat API error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/coach')
@login_required
def coach_page():
    """Personalized financial coaching page"""
    return render_template('coach.html', user=current_user)

@app.route('/api/coach', methods=['POST'])
@login_required
def coach_api():
    """Specialized coaching endpoint"""
    try:
        data = request.get_json()
        data['user_id'] = str(current_user.id)
        
        response = requests.post(
            f"{AI_COACH_API_URL}/coach",
            json=data,
            timeout=60
        )
        
        return jsonify(response.json())
        
    except Exception as e:
        app.logger.error(f"Coach API error: {e}")
        return jsonify({'error': 'Coaching service unavailable'}), 500
```

### Step 2: Create Frontend Templates

Create `templates/chat.html`:

```html
{% extends "base.html" %}

{% block title %}AI Finance Coach - Chat{% endblock %}

{% block content %}
<div class="container-fluid">
    <div class="row">
        <div class="col-md-8 mx-auto">
            <div class="card">
                <div class="card-header">
                    <h4><i class="fas fa-robot"></i> AI Finance Coach</h4>
                </div>
                <div class="card-body">
                    <div id="chat-container" class="chat-container mb-3">
                        <div class="chat-message system">
                            <strong>Vallionis AI:</strong> Hello {{ user.username }}! I'm your personal AI finance coach. How can I help you today?
                        </div>
                    </div>
                    
                    <div class="input-group">
                        <input type="text" id="chat-input" class="form-control" 
                               placeholder="Ask me anything about finance..." 
                               onkeypress="handleKeyPress(event)">
                        <button class="btn btn-primary" onclick="sendMessage()">
                            <i class="fas fa-paper-plane"></i> Send
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<style>
.chat-container {
    height: 400px;
    overflow-y: auto;
    border: 1px solid #ddd;
    padding: 15px;
    background-color: #f8f9fa;
}

.chat-message {
    margin-bottom: 15px;
    padding: 10px;
    border-radius: 8px;
}

.chat-message.user {
    background-color: #007bff;
    color: white;
    margin-left: 20%;
}

.chat-message.system {
    background-color: white;
    border: 1px solid #ddd;
    margin-right: 20%;
}

.typing-indicator {
    font-style: italic;
    color: #666;
}
</style>

<script>
async function sendMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Add user message to chat
    addMessageToChat(message, 'user');
    input.value = '';
    
    // Show typing indicator
    const typingDiv = addMessageToChat('AI is thinking...', 'system typing-indicator');
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });
        
        const data = await response.json();
        
        // Remove typing indicator
        typingDiv.remove();
        
        if (data.response) {
            addMessageToChat(data.response, 'system');
            
            // Show sources if available
            if (data.sources && data.sources.length > 0) {
                const sourcesHtml = '<div class="mt-2"><small><strong>Sources:</strong> ' + 
                    data.sources.map(s => s.title).join(', ') + '</small></div>';
                document.querySelector('.chat-message:last-child').innerHTML += sourcesHtml;
            }
        } else {
            addMessageToChat('Sorry, I encountered an error. Please try again.', 'system');
        }
        
    } catch (error) {
        typingDiv.remove();
        addMessageToChat('Connection error. Please check your internet connection.', 'system');
    }
}

function addMessageToChat(message, type) {
    const container = document.getElementById('chat-container');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${type}`;
    
    if (type === 'user') {
        messageDiv.innerHTML = `<strong>You:</strong> ${message}`;
    } else {
        messageDiv.innerHTML = `<strong>Vallionis AI:</strong> ${message}`;
    }
    
    container.appendChild(messageDiv);
    container.scrollTop = container.scrollHeight;
    
    return messageDiv;
}

function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}
</script>
{% endblock %}
```

### Step 3: Update Navigation

Add AI Coach links to your navigation template:

```html
<li class="nav-item">
    <a class="nav-link" href="{{ url_for('chat_page') }}">
        <i class="fas fa-robot"></i> AI Coach
    </a>
</li>
<li class="nav-item">
    <a class="nav-link" href="{{ url_for('coach_page') }}">
        <i class="fas fa-graduation-cap"></i> Personal Coach
    </a>
</li>
```

### Step 4: Environment Configuration

Add to your `.env` file:

```env
# AI Coach Configuration
AI_COACH_API_URL=https://your-oci-domain.com/api
AI_COACH_ENABLED=true
AI_COACH_TIMEOUT=60
```

### Step 5: User Profile Integration

Extend your User model to include AI coaching preferences:

```python
# Add to your User model
class User(UserMixin, db.Model):
    # ... existing fields ...
    
    # AI Coach preferences
    ai_coaching_enabled = db.Column(db.Boolean, default=True)
    risk_tolerance = db.Column(db.String(50))  # 'conservative', 'moderate', 'aggressive'
    investment_goals = db.Column(db.Text)  # JSON string
    experience_level = db.Column(db.String(50), default='beginner')
    preferred_topics = db.Column(db.Text)  # JSON string
```

## Deployment Strategies

### Strategy 1: Keep Flask App on Current Host
- Minimal changes to existing deployment
- OCI handles only AI services
- Cross-origin requests need CORS configuration

### Strategy 2: Move Everything to OCI
- Maximum cost savings (everything on free tier)
- Single point of management
- May require more powerful VM configuration

### Strategy 3: Hybrid Approach
- Critical services on reliable hosting
- AI services on OCI free tier
- Best of both worlds

## Security Considerations

1. **API Authentication**: Add API keys or JWT tokens
2. **Rate Limiting**: Prevent abuse of AI services
3. **Data Privacy**: Ensure user data stays secure
4. **HTTPS**: Always use encrypted connections
5. **Input Validation**: Sanitize all user inputs

## Monitoring and Analytics

Track AI coach usage in your existing analytics:

```python
@app.route('/api/chat', methods=['POST'])
@login_required
def chat_api():
    # ... existing code ...
    
    # Log AI coach usage
    log_user_activity(
        user_id=current_user.id,
        action='ai_chat',
        details={'query_length': len(data['message'])}
    )
```

## Cost Optimization

- Use OCI Always Free tier: **$0/month**
- Monitor usage to stay within limits
- Implement caching for common queries
- Use model quantization for efficiency

## Next Steps

1. Deploy the OCI infrastructure using the provided scripts
2. Test the AI coach API endpoints
3. Integrate with your Flask application
4. Customize the knowledge base with your content
5. Monitor performance and user engagement
