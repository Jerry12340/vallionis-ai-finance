"""
Render Integration for Vallionis AI Finance Coach
Add these routes to your existing Flask app for AI coach functionality
"""
import os
import requests
import json
from flask import request, jsonify, render_template
from functools import wraps
import logging

# Configuration - Add these to your Render environment variables
AI_COACH_API_URL = os.getenv('AI_COACH_API_URL', 'https://your-oci-domain.com/api')
AI_COACH_TIMEOUT = int(os.getenv('AI_COACH_TIMEOUT', '180'))
AI_COACH_ENABLED = os.getenv('AI_COACH_ENABLED', 'true').lower() == 'true'

logger = logging.getLogger(__name__)

def ai_coach_required(f):
    """Decorator to check if AI coach is enabled"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not AI_COACH_ENABLED:
            return jsonify({'error': 'AI Coach service is currently disabled'}), 503
        return f(*args, **kwargs)
    return decorated_function

# Add these routes to your existing Flask app (app.py)

@app.route('/ai-coach')
@login_required
def ai_coach_page():
    """AI Finance Coach main page"""
    return render_template('ai_coach.html', user=current_user)

@app.route('/api/ai/chat', methods=['POST'])
@login_required
@ai_coach_required
def ai_chat():
    """Proxy chat requests to OCI AI backend"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        # Add user context from your existing user system
        enhanced_request = {
            'message': data['message'],
            'model': data.get('model', 'llama3.1:8b-instruct-q4_0'),
            'user_id': str(current_user.id),
            'context': {
                'user_name': current_user.username,
                'user_email': current_user.email,
                'timestamp': data.get('timestamp'),
                # Add any other user context from your existing system
            }
        }
        
        # Forward request to OCI AI backend
        response = requests.post(
            f"{AI_COACH_API_URL}/chat",
            json=enhanced_request,
            timeout=AI_COACH_TIMEOUT,
            headers={
                'Content-Type': 'application/json',
                'User-Agent': 'Vallionis-Render-App/1.0'
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Log the interaction in your existing system
            try:
                # Add to your existing logging/analytics
                logger.info(f"AI Chat - User: {current_user.id}, Response time: {result.get('response_time_ms', 0)}ms")
            except Exception as e:
                logger.error(f"Logging error: {e}")
            
            return jsonify(result)
        else:
            logger.error(f"AI backend error: {response.status_code} - {response.text}")
            return jsonify({'error': 'AI service temporarily unavailable'}), 503
            
    except requests.exceptions.Timeout:
        return jsonify({'error': 'AI service timeout. Please try again.'}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({'error': 'Cannot connect to AI service'}), 503
    except Exception as e:
        logger.error(f"AI Chat error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/ai/coach', methods=['POST'])
@login_required
@ai_coach_required
def ai_coach():
    """Specialized coaching endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data or 'coaching_type' not in data:
            return jsonify({'error': 'Query and coaching_type are required'}), 400
        
        enhanced_request = {
            'user_id': str(current_user.id),
            'query': data['query'],
            'coaching_type': data['coaching_type'],
            'user_data': {
                'username': current_user.username,
                'email': current_user.email,
                # Add any existing user profile data you have
                **data.get('user_data', {})
            }
        }
        
        response = requests.post(
            f"{AI_COACH_API_URL}/coach",
            json=enhanced_request,
            timeout=AI_COACH_TIMEOUT
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': 'Coaching service unavailable'}), 503
            
    except Exception as e:
        logger.error(f"AI Coach error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/ai/health')
def ai_health():
    """Check AI backend health"""
    try:
        if not AI_COACH_ENABLED:
            return jsonify({'status': 'disabled', 'message': 'AI Coach is disabled'})
        
        response = requests.get(
            f"{AI_COACH_API_URL}/health",
            timeout=10
        )
        
        if response.status_code == 200:
            backend_health = response.json()
            return jsonify({
                'status': 'healthy',
                'backend': backend_health,
                'render_integration': 'active'
            })
        else:
            return jsonify({'status': 'backend_error', 'code': response.status_code}), 503
            
    except Exception as e:
        return jsonify({'status': 'connection_error', 'error': str(e)}), 503

# Error handlers for AI coach
@app.errorhandler(503)
def ai_service_unavailable(error):
    """Handle AI service unavailable errors"""
    if request.path.startswith('/api/ai/'):
        return jsonify({
            'error': 'AI Coach service is temporarily unavailable',
            'suggestion': 'Please try again in a few minutes'
        }), 503
    return error

@app.errorhandler(504)
def ai_service_timeout(error):
    """Handle AI service timeout errors"""
    if request.path.startswith('/api/ai/'):
        return jsonify({
            'error': 'AI Coach is taking longer than usual to respond',
            'suggestion': 'Please try a shorter question or try again later'
        }), 504
    return error
