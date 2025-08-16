#!/usr/bin/env python3
"""
Vallionis AI Finance Coach API
FastAPI gateway for LLM, RAG, and coaching functionality
"""
import os
import json
import time
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Vallionis AI Finance Coach",
    description="Self-hosted AI finance coaching with RAG capabilities",
    version="1.0.0"
)

# CORS middleware for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:8b-instruct-q4_0"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight sentence transformer

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'vallionis_ai',
    'user': 'vallionis',
    'password': os.getenv('DB_PASSWORD', 'secure_password_change_me')
}

# Initialize embedding model
try:
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Embedding model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    embedding_model = None

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = DEFAULT_MODEL
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    model: str
    response_time_ms: int
    sources: Optional[List[Dict[str, Any]]] = None

class RetrieveRequest(BaseModel):
    query: str
    limit: Optional[int] = 5
    document_types: Optional[List[str]] = None

class CoachRequest(BaseModel):
    user_id: str
    query: str
    coaching_type: str  # 'risk_assessment', 'portfolio_review', 'learning_path'
    user_data: Optional[Dict[str, Any]] = None

# Database utilities
def get_db_connection():
    """Get database connection with pgvector support"""
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    return conn

def get_embedding(text: str) -> List[float]:
    """Generate embedding for text using sentence transformer"""
    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Embedding model not available")
    
    embedding = embedding_model.encode(text)
    return embedding.tolist()

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Ollama connection
        async with httpx.AsyncClient() as client:
            ollama_response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            ollama_healthy = ollama_response.status_code == 200
        
        # Test database connection
        try:
            conn = get_db_connection()
            conn.close()
            db_healthy = True
        except:
            db_healthy = False
        
        return {
            "status": "healthy" if ollama_healthy and db_healthy else "degraded",
            "ollama": "up" if ollama_healthy else "down",
            "database": "up" if db_healthy else "down",
            "embedding_model": "loaded" if embedding_model else "not_loaded",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with RAG integration"""
    start_time = time.time()
    
    try:
        # Get relevant context from knowledge base
        relevant_docs = await retrieve_relevant_documents(request.message, limit=3)
        
        # Build enhanced prompt with context
        context_text = ""
        sources = []
        
        if relevant_docs:
            context_text = "\n\nRelevant financial knowledge:\n"
            for doc in relevant_docs:
                context_text += f"- {doc['title']}: {doc['content'][:200]}...\n"
                sources.append({
                    "title": doc['title'],
                    "type": doc['document_type'],
                    "relevance_score": doc['similarity']
                })
        
        # Enhanced system prompt for financial coaching
        system_prompt = """You are Vallionis, an expert AI financial coach. You provide personalized, actionable financial advice based on established financial principles. 

Key guidelines:
- Always prioritize the user's financial safety and long-term wealth building
- Explain complex concepts in simple terms
- Provide specific, actionable recommendations when appropriate
- Acknowledge when professional financial advice is needed
- Use the provided knowledge base context to enhance your responses
- Be encouraging and supportive while being realistic about risks

Remember: You are not a licensed financial advisor. Always recommend consulting with qualified professionals for major financial decisions."""

        full_prompt = f"{system_prompt}\n\nUser question: {request.message}{context_text}\n\nProvide a helpful, personalized response:"
        
        # Call Ollama API
        async with httpx.AsyncClient(timeout=60.0) as client:
            ollama_request = {
                "model": request.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 1000
                }
            }
            
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=ollama_request
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="Ollama API error")
            
            result = response.json()
            ai_response = result.get('response', '')
        
        response_time = int((time.time() - start_time) * 1000)
        
        # Log the interaction
        await log_user_query(request.message, ai_response, request.model, response_time)
        
        return ChatResponse(
            response=ai_response,
            model=request.model,
            response_time_ms=response_time,
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint for real-time responses"""
    try:
        # Get relevant context (same as above)
        relevant_docs = await retrieve_relevant_documents(request.message, limit=3)
        
        context_text = ""
        if relevant_docs:
            context_text = "\n\nRelevant financial knowledge:\n"
            for doc in relevant_docs:
                context_text += f"- {doc['title']}: {doc['content'][:200]}...\n"
        
        system_prompt = """You are Vallionis, an expert AI financial coach. Provide helpful, personalized financial guidance."""
        full_prompt = f"{system_prompt}\n\nUser question: {request.message}{context_text}\n\nResponse:"
        
        async def generate_stream():
            async with httpx.AsyncClient(timeout=120.0) as client:
                ollama_request = {
                    "model": request.model,
                    "prompt": full_prompt,
                    "stream": True,
                    "options": {"temperature": 0.7}
                }
                
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json=ollama_request
                ) as response:
                    async for chunk in response.aiter_lines():
                        if chunk:
                            try:
                                data = json.loads(chunk)
                                if 'response' in data:
                                    yield f"data: {json.dumps({'content': data['response']})}\n\n"
                                if data.get('done', False):
                                    yield f"data: {json.dumps({'done': True})}\n\n"
                                    break
                            except json.JSONDecodeError:
                                continue
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve")
async def retrieve(request: RetrieveRequest):
    """Retrieve relevant documents from knowledge base"""
    try:
        docs = await retrieve_relevant_documents(
            request.query,
            limit=request.limit,
            document_types=request.document_types
        )
        
        return {
            "query": request.query,
            "results": docs,
            "count": len(docs)
        }
        
    except Exception as e:
        logger.error(f"Retrieve error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/coach")
async def coach(request: CoachRequest):
    """Specialized coaching endpoint with personalized recommendations"""
    try:
        # Get user profile if available
        user_profile = await get_user_profile(request.user_id)
        
        # Generate coaching response based on type
        if request.coaching_type == "risk_assessment":
            response = await generate_risk_assessment(request.query, user_profile, request.user_data)
        elif request.coaching_type == "portfolio_review":
            response = await generate_portfolio_review(request.query, user_profile, request.user_data)
        elif request.coaching_type == "learning_path":
            response = await generate_learning_path(request.query, user_profile)
        else:
            raise HTTPException(status_code=400, detail="Invalid coaching type")
        
        return response
        
    except Exception as e:
        logger.error(f"Coaching error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions

async def retrieve_relevant_documents(query: str, limit: int = 5, document_types: List[str] = None) -> List[Dict]:
    """Retrieve relevant documents using vector similarity search"""
    try:
        query_embedding = get_embedding(query)
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Build query with optional document type filtering
        where_clause = ""
        params = [query_embedding, limit]
        
        if document_types:
            placeholders = ','.join(['%s'] * len(document_types))
            where_clause = f"WHERE d.document_type IN ({placeholders})"
            params.extend(document_types)
            params[-1], params[-2] = params[-2], params[-1]  # Swap limit and types
        
        query_sql = f"""
        SELECT d.title, d.content, d.document_type, d.source_url,
               e.embedding <-> %s::vector as similarity
        FROM documents d
        JOIN embeddings e ON d.id = e.document_id
        {where_clause}
        ORDER BY similarity
        LIMIT %s
        """
        
        cur.execute(query_sql, params)
        results = cur.fetchall()
        
        docs = []
        for row in results:
            docs.append({
                'title': row[0],
                'content': row[1],
                'document_type': row[2],
                'source_url': row[3],
                'similarity': float(row[4])
            })
        
        cur.close()
        conn.close()
        
        return docs
        
    except Exception as e:
        logger.error(f"Document retrieval error: {e}")
        return []

async def log_user_query(query: str, response: str, model: str, response_time: int):
    """Log user interactions for analytics"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO user_queries (query_text, response_text, model_used, response_time_ms)
            VALUES (%s, %s, %s, %s)
        """, (query, response, model, response_time))
        
        conn.commit()
        cur.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Logging error: {e}")

async def get_user_profile(user_id: str) -> Dict:
    """Get user profile for personalized coaching"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT risk_tolerance, time_horizon, investment_goals, 
                   current_knowledge_level, preferred_topics
            FROM user_profiles WHERE user_id = %s
        """, (user_id,))
        
        result = cur.fetchone()
        cur.close()
        conn.close()
        
        if result:
            return {
                'risk_tolerance': result[0],
                'time_horizon': result[1],
                'investment_goals': result[2],
                'knowledge_level': result[3],
                'preferred_topics': result[4]
            }
        
        return {}
        
    except Exception as e:
        logger.error(f"Profile retrieval error: {e}")
        return {}

async def generate_risk_assessment(query: str, profile: Dict, user_data: Dict) -> Dict:
    """Generate personalized risk assessment"""
    # Implementation for risk assessment coaching
    return {"type": "risk_assessment", "recommendations": [], "next_steps": []}

async def generate_portfolio_review(query: str, profile: Dict, user_data: Dict) -> Dict:
    """Generate portfolio review and recommendations"""
    # Implementation for portfolio review coaching
    return {"type": "portfolio_review", "analysis": {}, "recommendations": []}

async def generate_learning_path(query: str, profile: Dict) -> Dict:
    """Generate personalized learning path"""
    # Implementation for learning path generation
    return {"type": "learning_path", "modules": [], "estimated_time": ""}

if __name__ == "__main__":
    logger.info("Starting Uvicorn server...")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
