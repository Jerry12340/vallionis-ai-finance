from datetime import datetime
from app import db

class MacroIndicator(db.Model):
    """Stores macroeconomic indicators with their values and metadata."""
    __tablename__ = 'macro_indicators'
    
    id = db.Column(db.Integer, primary_key=True)
    indicator_name = db.Column(db.String(100), nullable=False)  # e.g., 'inflation', 'gdp_growth', 'unemployment_rate'
    country = db.Column(db.String(50), nullable=False, default='US')  # ISO country code
    frequency = db.Column(db.String(20), nullable=False)  # 'daily', 'monthly', 'quarterly', 'annual'
    date = db.Column(db.Date, nullable=False)  # The date this data point refers to
    value = db.Column(db.Float, nullable=False)  # The actual value
    unit = db.Column(db.String(20))  # e.g., '%', 'USD', etc.
    source = db.Column(db.String(100))  # Source of the data (e.g., 'FRED', 'World Bank')
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # For efficient querying
    __table_args__ = (
        db.Index('idx_macro_indicator', 'indicator_name', 'country', 'date', unique=True),
    )
    
    def to_dict(self):
        """Convert the model to a dictionary for JSON serialization."""
        return {
            'id': self.id,
            'indicator_name': self.indicator_name,
            'country': self.country,
            'frequency': self.frequency,
            'date': self.date.isoformat() if self.date else None,
            'value': self.value,
            'unit': self.unit,
            'source': self.source,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }

class MacroAnalysis(db.Model):
    """Stores AI-generated analysis of macroeconomic indicators."""
    __tablename__ = 'macro_analysis'
    
    id = db.Column(db.Integer, primary_key=True)
    analysis_type = db.Column(db.String(50), nullable=False)  # e.g., 'monthly_summary', 'inflation_forecast'
    analysis_date = db.Column(db.Date, nullable=False, default=datetime.utcnow().date)
    content = db.Column(db.Text, nullable=False)  # The actual analysis text
    indicators_covered = db.Column(db.ARRAY(db.String(100)))  # List of indicator names this analysis covers
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert the model to a dictionary for JSON serialization."""
        return {
            'id': self.id,
            'analysis_type': self.analysis_type,
            'analysis_date': self.analysis_date.isoformat() if self.analysis_date else None,
            'content': self.content,
            'indicators_covered': self.indicators_covered or [],
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
