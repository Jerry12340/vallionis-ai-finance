#!/bin/bash
# Setup PostgreSQL with pgvector for RAG functionality
# Run this after install-ollama.sh

set -e

echo "🗄️ Setting up PostgreSQL with pgvector for RAG..."

# Configure PostgreSQL
echo "🔧 Configuring PostgreSQL..."
sudo -u postgres psql << EOF
CREATE USER vallionis WITH PASSWORD 'secure_password_change_me';
CREATE DATABASE vallionis_ai OWNER vallionis;
GRANT ALL PRIVILEGES ON DATABASE vallionis_ai TO vallionis;
\q
EOF

# Install pgvector extension
echo "🔍 Installing pgvector extension..."
cd /tmp
git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# Enable pgvector in database
echo "✨ Enabling pgvector extension..."
sudo -u postgres psql -d vallionis_ai << EOF
CREATE EXTENSION vector;
\q
EOF

# Create tables for financial knowledge base
echo "📋 Creating database schema..."
sudo -u postgres psql -d vallionis_ai << EOF
-- Documents table for storing financial content
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    document_type VARCHAR(100), -- 'lesson', 'glossary', 'company_profile', 'regulation'
    source_url VARCHAR(1000),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Embeddings table for vector search
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    embedding vector(1536), -- OpenAI embedding size, adjust if using different model
    chunk_index INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for vector similarity search
CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- User queries table for analytics
CREATE TABLE user_queries (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    response_text TEXT,
    model_used VARCHAR(100),
    response_time_ms INTEGER,
    user_rating INTEGER CHECK (user_rating >= 1 AND user_rating <= 5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Financial profiles table for personalized coaching
CREATE TABLE user_profiles (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) UNIQUE, -- Can link to your main app's user system
    risk_tolerance VARCHAR(50), -- 'conservative', 'moderate', 'aggressive'
    time_horizon VARCHAR(50), -- 'short', 'medium', 'long'
    investment_goals TEXT[],
    current_knowledge_level VARCHAR(50), -- 'beginner', 'intermediate', 'advanced'
    preferred_topics TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Grant permissions to vallionis user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO vallionis;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO vallionis;

\q
EOF

# Create database configuration file
echo "📝 Creating database configuration..."
tee ~/vallionis-ai/db_config.py << EOF
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import psycopg2
from pgvector.psycopg2 import register_vector

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'vallionis_ai',
    'user': 'vallionis',
    'password': os.getenv('DB_PASSWORD', 'secure_password_change_me')
}

def get_db_connection():
    """Get a direct psycopg2 connection for vector operations"""
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    return conn

def get_sqlalchemy_engine():
    """Get SQLAlchemy engine for ORM operations"""
    db_url = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    return create_engine(db_url)

def test_connection():
    """Test database connection and pgvector functionality"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Test basic connection
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        print(f"✅ PostgreSQL connected: {version}")
        
        # Test pgvector
        cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        if cur.fetchone():
            print("✅ pgvector extension is installed")
        else:
            print("❌ pgvector extension not found")
            return False
        
        # Test vector operations
        cur.execute("SELECT '[1,2,3]'::vector <-> '[4,5,6]'::vector;")
        distance = cur.fetchone()[0]
        print(f"✅ Vector operations working, test distance: {distance}")
        
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()
EOF

# Install Python dependencies for database operations
echo "📦 Installing Python database dependencies..."
pip3 install --user psycopg2-binary sqlalchemy pgvector

# Test the database setup
echo "🧪 Testing database setup..."
cd ~/vallionis-ai
python3 db_config.py

# Create sample data insertion script
tee ~/vallionis-ai/populate_knowledge_base.py << EOF
#!/usr/bin/env python3
"""
Script to populate the knowledge base with financial content
"""
import psycopg2
from db_config import get_db_connection
import json

def insert_sample_data():
    """Insert sample financial knowledge"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Sample financial lessons
    lessons = [
        {
            'title': 'Compound Interest Fundamentals',
            'content': '''Compound interest is the interest calculated on the initial principal and the accumulated interest from previous periods. It's often called "interest on interest" and can significantly boost investment returns over time. The formula is A = P(1 + r/n)^(nt), where A is the final amount, P is principal, r is annual interest rate, n is compounding frequency, and t is time in years.''',
            'document_type': 'lesson'
        },
        {
            'title': 'Risk Tolerance Assessment',
            'content': '''Risk tolerance is your ability and willingness to lose some or all of your original investment in exchange for greater potential returns. Factors include age, income stability, investment timeline, and emotional comfort with volatility. Conservative investors prefer stable, lower-return investments, while aggressive investors accept higher volatility for potentially greater returns.''',
            'document_type': 'lesson'
        },
        {
            'title': 'Dollar-Cost Averaging',
            'content': '''Dollar-cost averaging (DCA) is an investment strategy where you invest a fixed amount regularly, regardless of market conditions. This approach reduces the impact of volatility by purchasing more shares when prices are low and fewer when prices are high. It's particularly effective for long-term investors and helps remove emotion from investment decisions.''',
            'document_type': 'lesson'
        }
    ]
    
    # Sample glossary terms
    glossary = [
        {
            'title': 'Asset Allocation',
            'content': '''The process of dividing investments among different asset categories, such as stocks, bonds, and cash. The goal is to balance risk and reward according to an individual\'s goals, risk tolerance, and investment horizon.''',
            'document_type': 'glossary'
        },
        {
            'title': 'Diversification',
            'content': '''A risk management strategy that mixes a wide variety of investments within a portfolio. The rationale is that a portfolio constructed of different kinds of assets will, on average, yield higher long-term returns and lower the risk of any individual holding.''',
            'document_type': 'glossary'
        }
    ]
    
    all_content = lessons + glossary
    
    for item in all_content:
        cur.execute("""
            INSERT INTO documents (title, content, document_type)
            VALUES (%s, %s, %s)
        """, (item['title'], item['content'], item['document_type']))
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"✅ Inserted {len(all_content)} documents into knowledge base")

if __name__ == "__main__":
    insert_sample_data()
EOF

chmod +x ~/vallionis-ai/populate_knowledge_base.py

echo "✅ Database setup complete!"
echo "🗄️ PostgreSQL with pgvector is ready"
echo "📊 Database: vallionis_ai"
echo "👤 User: vallionis"
echo "🔑 Password: secure_password_change_me (CHANGE THIS!)"
echo "🧪 Test with: cd ~/vallionis-ai && python3 db_config.py"
echo "📚 Populate sample data: cd ~/vallionis-ai && python3 populate_knowledge_base.py"
