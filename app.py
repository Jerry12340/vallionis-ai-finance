from flask import Flask, render_template, request, session, Response, abort, redirect, url_for, jsonify, \
    send_from_directory, make_response
import time
import pandas as pd
import yfinance as yf
import numpy as np
import finnhub
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import requests
from dotenv import load_dotenv
import warnings
from sklearn.impute import SimpleImputer
from io import StringIO
from flask_wtf import FlaskForm, CSRFProtect
from wtforms import StringField, SelectField, IntegerField, BooleanField, PasswordField, SubmitField
from wtforms.validators import DataRequired, NumberRange, Email, EqualTo
import stripe
from jinja2 import Environment
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask import flash
from sqlalchemy import event
from sqlalchemy.orm.attributes import get_history
import traceback
from apscheduler.schedulers.background import BackgroundScheduler
import logging
from datetime import datetime, timezone, timedelta
from flask_migrate import Migrate
from wtforms.validators import ValidationError
from authlib.integrations.flask_client import OAuth
from authlib.common.security import generate_token
import sklearn
from packaging import version
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from psycopg2 import OperationalError as Psycopg2OpError
from flask import send_file
import requests
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadTimeSignature
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import socket

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables
load_dotenv('.env')

# Feature flags for controlling updates
FEATURE_FLAGS = {
    'new_design': os.getenv('NEW_DESIGN', 'false').lower() == 'true',
    'beta_features': os.getenv('BETA_FEATURES', 'false').lower() == 'true',
    'maintenance_mode': os.getenv('MAINTENANCE_MODE', 'false').lower() == 'true'
}

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

app.secret_key = os.getenv('SECRET_KEY')
if not app.secret_key:
    raise ValueError("No SECRET_KEY set for Flask application")

# Database URL handling
db_url = os.environ.get('DATABASE_URL')
if not db_url:
    raise ValueError("DATABASE_URL environment variable is not set")

# Fix common URL issues
if db_url.startswith('postgres://'):
    db_url = db_url.replace('postgres://', 'postgresql://', 1)

# Configure SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 3600,
    'connect_args': {
        'sslmode': 'require',
    }
}
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions (order matters)
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Verify database connection
try:
    with app.app_context():
        db.engine.connect()
    print("✅ Database connection verified")
except Exception as e:
    print(f"❌ Failed to connect to database: {e}")
    raise

# Security configurations (ONLY ONE INSTANCE)
app.config.update(
    SESSION_COOKIE_SECURE=os.getenv('FLASK_ENV') == 'production',
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),
    PREFERRED_URL_SCHEME='https' if os.getenv('FLASK_ENV') == 'production' else 'http',
    FLASK_ENV=os.getenv('FLASK_ENV', 'development'),
    DEBUG=os.getenv('DEBUG', 'False').lower() == 'true'
)

# Configure logging (ONLY ONE INSTANCE)
logging.basicConfig(
    level=logging.DEBUG if app.config['DEBUG'] else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize OAuth after db
oauth = OAuth(app)
redirect_uri = os.environ.get("GOOGLE_REDIRECT_URI")
google = oauth.register(
    name='google',
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    access_token_url='https://oauth2.googleapis.com/token',
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    client_kwargs={
        'scope': 'openid email profile',
        'token_endpoint_auth_method': 'client_secret_post'
    },
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    jwks_uri='https://www.googleapis.com/oauth2/v3/certs',
    issuer='https://accounts.google.com'
)


def initialize_database(retries=5, delay=20):
    for i in range(retries):
        try:
            print("✅ Database initialized successfully")
            return
        except (OperationalError, Psycopg2OpError) as e:
            print(f"❌ Attempt {i + 1} failed: {e}")
            if i < retries - 1:
                print(f"⏳ Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("🚨 Failed to connect to database after retries. Exiting.")
                raise


initialize_database()


def categorize_stock_risk(symbol, beta, dividend_yield, pe_ratio, industry):
    """
    Categorize stocks by risk profile to help with style-specific filtering
    """
    # Conservative stocks (low risk, stable, dividend-paying)
    conservative_stocks = {
        'LLY', 'BRK-B', 'JNJ', 'PG', 'KO', 'PEP', 'MMM', 'SO', 'DUK', 'CVX', 'O', 
        'T', 'PFE', 'ABT', 'WMT', 'COST', 'MCD', 'GIS', 'ELV', 'BMY'
    }
    
    # Aggressive stocks (high growth, high volatility)
    aggressive_stocks = {
        'NVDA', 'HOOD', 'COIN', 'AMD', 'MU'
    }
    
    # Check explicit categorization first
    if symbol in conservative_stocks:
        return 'conservative'
    elif symbol in aggressive_stocks:
        return 'aggressive'
    
    # Categorize based on metrics
    if (beta <= 0.8 and dividend_yield >= 2.0 and pe_ratio <= 25):
        return 'conservative'
    elif (beta >= 1.3 and dividend_yield <= 1.0 and pe_ratio >= 30):
        return 'aggressive'
    elif industry in ['Technology', 'Semiconductors', 'Software'] and beta >= 1.2:
        return 'aggressive'
    elif industry in ['Consumer Defensive', 'Utilities'] and beta <= 1.0:
        return 'conservative'
    
    return 'moderate'


def fetch_valid_tickers(tickers, premium):
    import time
    rows = []
    skipped_tickers = []

    # Define expected columns with defaults
    base_template = {
        'symbol': None,
        'industry': 'Unknown',
        'trailing_pe': None,
        'forward_pe': None,
        'beta': None,
        'dividend_yield': None,
        'debt_to_equity': None,
        'earnings_growth': None,
        'ps_ratio': None,
        'pb_ratio': None,
        'roe': None,
        'next_5y_eps_growth': np.nan,
        'next_year_eps_growth': np.nan,
        'peg_ratio': np.nan
    }

    for sym in tickers:
        try:
            data = get_industry_pe_beta(sym) or {}  # Ensure we get a dict
            # Merge with base template to ensure all keys exist
            merged_data = {**base_template, **data}
            merged_data['symbol'] = sym  # Ensure symbol is always set
            rows.append(merged_data)
        except Exception as e:
            print(f"Error fetching {sym}: {e}")
            # Add fallback entry with symbol only
            rows.append({**base_template, 'symbol': sym})
            skipped_tickers.append(sym)
        if premium:
            time.sleep(1.5)
        else:
            time.sleep(3)

    # Create DataFrame with guaranteed columns
    df = pd.DataFrame(rows, columns=list(base_template.keys()))

    # Fill remaining missing values
    df['industry'] = df['industry'].fillna('Unknown')
    return df


def get_one_hot_encoder():
    ohe_kwargs = {"handle_unknown": "ignore"}
    if version.parse(sklearn.__version__) >= version.parse("1.2"):
        ohe_kwargs["sparse_output"] = False
    else:
        ohe_kwargs["sparse"] = False
    return OneHotEncoder(**ohe_kwargs)


# Email configuration
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True').lower() == 'true'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER', os.getenv('MAIL_USERNAME'))

# Initialize Flask-Mail
mail = Mail(app)

# Token serializer for password reset
token_serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])



# Security configurations
app.config.update(
    SESSION_COOKIE_SECURE=os.getenv('FLASK_ENV') == 'production',
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),
    PREFERRED_URL_SCHEME='https' if os.getenv('FLASK_ENV') == 'production' else 'http',
    FLASK_ENV=os.getenv('FLASK_ENV', 'development'),
    DEBUG=os.getenv('DEBUG', 'False').lower() == 'true'
)



# Configure logging
logging.basicConfig(
    level=logging.DEBUG if app.config['DEBUG'] else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================
# Extensions Initialization
# =============================================
csrf = CSRFProtect(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = None
login_manager.session_protection = "strong"


class UserPreferenceHistory(db.Model):
    """Tracks historical changes to user preferences"""
    __tablename__ = 'user_preference_history'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Preference fields
    investing_style = db.Column(db.String(20))
    time_horizon = db.Column(db.Integer)
    risk_tolerance = db.Column(db.String(20))
    sector_focus = db.Column(db.String(20))
    dividend_preference = db.Column(db.Boolean)

    # Relationship (using back_populates)
    user = db.relationship('User', back_populates='preference_history')


class CurrentUserPreferences(db.Model):
    """Stores the current preferences for quick access"""
    __tablename__ = 'current_user_preferences'

    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), primary_key=True)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Preference fields
    investing_style = db.Column(db.String(20))
    time_horizon = db.Column(db.Integer)
    risk_tolerance = db.Column(db.String(20))
    sector_focus = db.Column(db.String(20))
    dividend_preference = db.Column(db.Boolean)

    # Relationship (using back_populates)
    user = db.relationship('User', back_populates='current_preferences')


class User(db.Model, UserMixin):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    reset_token = db.Column(db.String(100), nullable=True)
    reset_token_expiry = db.Column(db.DateTime, nullable=True)
    premium = db.Column(db.Boolean, default=False)
    subscription_expires = db.Column(db.DateTime)
    stripe_customer_id = db.Column(db.String(50))
    subscription_type = db.Column(db.String(20), default=None)
    subscription_status = db.Column(db.String(20), default=None)

    # Relationships (using back_populates)
    current_preferences = db.relationship(
        'CurrentUserPreferences',
        back_populates='user',
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True
    )

    preference_history = db.relationship(
        'UserPreferenceHistory',
        back_populates='user',
        lazy='dynamic',
        cascade="all, delete-orphan",
        passive_deletes=True
    )

    def get_subscription_status(self):
        if self.premium:
            if self.subscription_type == 'lifetime':
                return True
            elif self.subscription_status in ['trialing', 'active']:
                if self.subscription_expires and self.subscription_expires > datetime.utcnow():
                    return True
        return False

    def update_preferences(self, investing_style, time_horizon, risk_tolerance=None,
                         sector_focus=None, dividend_preference=None):
        """Update user preferences with history tracking"""
        try:
            # Create historical record
            history_entry = UserPreferenceHistory(
                user_id=self.id,
                investing_style=investing_style,
                time_horizon=time_horizon,
                risk_tolerance=risk_tolerance,
                sector_focus=sector_focus,
                dividend_preference=dividend_preference
            )
            db.session.add(history_entry)

            # Update or create current preferences
            if not self.current_preferences:
                self.current_preferences = CurrentUserPreferences(
                    user_id=self.id,
                    investing_style=investing_style,
                    time_horizon=time_horizon,
                    risk_tolerance=risk_tolerance,
                    sector_focus=sector_focus,
                    dividend_preference=dividend_preference
                )
            else:
                self.current_preferences.investing_style = investing_style
                self.current_preferences.time_horizon = time_horizon
                self.current_preferences.risk_tolerance = risk_tolerance
                self.current_preferences.sector_focus = sector_focus
                self.current_preferences.dividend_preference = dividend_preference
                self.current_preferences.last_updated = datetime.utcnow()

            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating preferences for user {self.id}: {str(e)}")
            return False
    
    def generate_reset_token(self):
        """Generate a password reset token for the user"""
        token = token_serializer.dumps(self.email, salt='password-reset')
        self.reset_token = token
        self.reset_token_expiry = datetime.utcnow() + timedelta(hours=1)  # Token expires in 1 hour
        db.session.commit()
        return token
    
    @staticmethod
    def verify_reset_token(token, max_age=3600):  # 1 hour in seconds
        """Verify a password reset token and return the user"""
        try:
            email = token_serializer.loads(token, salt='password-reset', max_age=max_age)
            user = User.query.filter_by(email=email).first()
            if user and user.reset_token == token and user.reset_token_expiry and user.reset_token_expiry > datetime.utcnow():
                return user
        except (SignatureExpired, BadTimeSignature):
            pass
        return None
    
    def clear_reset_token(self):
        """Clear the password reset token"""
        self.reset_token = None
        self.reset_token_expiry = None
        db.session.commit()


class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')


class RecommendationForm(FlaskForm):
    investing_style = SelectField(
        'Investment Style',
        choices=[
            ('conservative', 'Conservative'),
            ('moderate', 'Moderate'),
            ('aggressive', 'Aggressive')
        ],
        validators=[DataRequired()]
    )
    time_horizon = IntegerField(  # Corrected field name
        'Time Horizon (years, min: 3, max: 10)',
        validators=[DataRequired(), NumberRange(min=3, max=10)]
    )
    stocks_amount = IntegerField(
        'Number of Stocks (min: 5, max: 20)',
        validators=[DataRequired(), NumberRange(min=5, max=20)]
    )
    premium = BooleanField('Premium Features')
    sector_focus = SelectField(
        'Sector Focus (Premium)',
        choices=[
            ('all', 'All Sectors'),
            ('tech', 'Technology'),
            ('healthcare', 'Healthcare'),
            ('finance', 'Financial Services'),
            ('consumer', 'Consumer Goods'),
            ('energy', 'Energy')
        ],
        default='all'
    )
    risk_tolerance = SelectField(
        'Risk Tolerance (Premium)',
        choices=[
            ('low', 'Low Risk'),
            ('medium', 'Medium Risk'),
            ('high', 'High Risk')
        ],
        default='medium'
    )
    dividend_preference = BooleanField(
        'Prioritize Dividend Stocks (Premium)'
    )


# =============================================
# External Services
# =============================================
# Initialize Finnhub client
finnhub_api_key = os.getenv('FINNHUB_API_KEY')
if not finnhub_api_key:
    logger.error("FINNHUB_API_KEY not found in environment variables")
    finnhub_client = None
else:
    finnhub_client = finnhub.Client(api_key=finnhub_api_key)

stripe.api_key = os.getenv('STRIPE_LIVE_SECRET_KEY')
STRIPE_PUBLISHABLE_KEY = os.getenv('STRIPE_LIVE_PUBLISHABLE_KEY')

STRIPE_GBP_MONTHLY_PRICE_ID = os.getenv('STRIPE_GBP_MONTHLY_PRICE_ID')
STRIPE_GBP_LIFETIME_PRICE_ID = os.getenv('STRIPE_GBP_LIFETIME_PRICE_ID')

STRIPE_EUR_MONTHLY_PRICE_ID = os.getenv('STRIPE_EUR_MONTHLY_PRICE_ID')
STRIPE_EUR_LIFETIME_PRICE_ID = os.getenv('STRIPE_EUR_LIFETIME_PRICE_ID')

STRIPE_USD_MONTHLY_PRICE_ID = os.getenv('STRIPE_USD_MONTHLY_PRICE_ID')
STRIPE_USD_LIFETIME_PRICE_ID = os.getenv('STRIPE_USD_LIFETIME_PRICE_ID')

STRIPE_WEBHOOK_SECRET = os.getenv('STRIPE_LIVE_WEBHOOK_SECRET')


# =============================================
# User Loader
# =============================================
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Forms
class ChangePasswordForm(FlaskForm):
    old_password = PasswordField('Current Password', validators=[DataRequired()])
    new_password = PasswordField('New Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm New Password',
                                     validators=[DataRequired(), EqualTo('new_password')])
    submit = SubmitField('Change Password')


class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')


# Track customer ID changes
def track_customer_id_changes(mapper, connection, target):
    hist = get_history(target, 'stripe_customer_id')
    if hist.has_changes():
        current = target.stripe_customer_id
        previous = hist.deleted[0] if hist.deleted else None
        logger.warning(f"Customer ID changed for user {target.id}: {previous} -> {current}")


event.listen(User, 'before_update', track_customer_id_changes)


class RegistrationForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField(
        'Confirm Password',
        validators=[DataRequired(), EqualTo('password')]
    )
    submit = SubmitField('Sign Up')

    def validate_password(self, field):
        if len(field.data) < 8:
            raise ValidationError('Password must be at least 8 characters long')

    def validate_email(self, field):
        if User.query.filter_by(email=field.data).first():
            raise ValidationError('Email already registered')


class ForgotPasswordForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    submit = SubmitField('Send Reset Link')

    def validate_email(self, field):
        user = User.query.filter_by(email=field.data).first()
        if not user:
            raise ValidationError('No account found with that email address.')


class ResetPasswordForm(FlaskForm):
    password = PasswordField('New Password', validators=[DataRequired()])
    confirm_password = PasswordField(
        'Confirm New Password',
        validators=[DataRequired(), EqualTo('password')]
    )
    submit = SubmitField('Reset Password')

    def validate_password(self, field):
        if len(field.data) < 8:
            raise ValidationError('Password must be at least 8 characters long')


# Login manager
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# Helper functions
def get_industry_pe_beta(symbol):
    """
    Fetch fundamental data for a stock with robust None handling and multiple EPS growth sources
    Args:
        symbol (str): Stock ticker symbol
    Returns:
        dict: Dictionary containing financial metrics with safe rounding
    """
    # Initialize with default values
    result = {
        'symbol': symbol,
        'industry': 'Unknown',
        'trailing_pe': None,
        'forward_pe': None,
        'beta': None,
        'dividend_yield': None,
        'debt_to_equity': None,
        'ps_ratio': None,
        'pb_ratio': None,
        'roe': None,
        'next_5y_eps_growth': np.nan,
        'next_year_eps_growth': np.nan,
        'peg_ratio': np.nan,
        'market_cap': None
    }

    try:
        # Safe value retrieval with None checks
        def safe_get(data, key, default=None, round_digits=2):
            val = data.get(key, default)
            if val is None or isinstance(val, (str, bool)):
                return default
            try:
                if round_digits is None:
                    return float(val)
                return round(float(val), round_digits)
            except (TypeError, ValueError):
                return default

        # Get market cap first
        yf_data = yf.Ticker(symbol).info
        result['market_cap'] = safe_get(yf_data, 'marketCap', round_digits=None)

        # Define growth caps based on company size
        GROWTH_CAPS = {
            'mega': 0.25,  # >$200B
            'large': 0.35,  # $10B-$200B
            'small': 0.50  # <$10B
        }
        company_size = 'mega' if (result['market_cap'] or 0) > 200e9 else \
            'large' if (result['market_cap'] or 0) > 10e9 else 'small'
        max_growth = GROWTH_CAPS[company_size]

        # Get Finnhub data if available
        finnhub_metrics = {}
        if finnhub_client:
            try:
                profile = finnhub_client.company_profile2(symbol=symbol)
                result['industry'] = profile.get('finnhubIndustry', 'Unknown')
                finnhub_metrics = finnhub_client.company_basic_financials(symbol=symbol, metric='all').get('metric', {})
            except Exception as e:
                logger.debug(f"Finnhub data fetch failed for {symbol}: {e}")

        # Get EPS growth estimates from multiple sources
        def get_eps_growth_estimates():
            sources = {
                'yf_5y': None,
                'yf_next': None,
                'finnhub_5y': None,
                'historical': None,
                'alpha_vantage_5y': None,
                'alpha_vantage_next': None
            }

            # Yahoo Finance analyst estimates
            try:
                yf_estimates = yf.Ticker(symbol).analyst_price_target
                if not yf_estimates.empty:
                    if 'growth' in yf_estimates.columns:
                        sources['yf_5y'] = safe_get(yf_estimates, 'growth', round_digits=None) / 100
            except Exception as e:
                logger.debug(f"YF analyst estimates failed for {symbol}: {e}")

            # Finnhub 5Y growth
            sources['finnhub_5y'] = safe_get(finnhub_metrics, '5YAvgEPSGrowth', round_digits=None)

            # Historical EPS growth (5 years)
            try:
                hist = yf.Ticker(symbol).history(period="5y")
                if 'EPS' in hist.columns and len(hist['EPS']) >= 2:
                    eps_growth = (hist['EPS'].iloc[-1] / hist['EPS'].iloc[0]) ** (1 / 5) - 1
                    sources['historical'] = eps_growth
            except Exception as e:
                logger.debug(f"Historical EPS calc failed for {symbol}: {e}")

            # Next year growth from Yahoo
            sources['yf_next'] = safe_get(yf_data, 'earningsGrowth', round_digits=None)

            # Filter out None values and calculate medians
            valid_5y = [v for v in
                        [sources['yf_5y'], sources['finnhub_5y'], sources['alpha_vantage_5y'], sources['historical']] if
                        v is not None]
            median_5y = np.median(valid_5y) if valid_5y else None
            next_year = sources['yf_next'] if sources['yf_next'] is not None else sources['alpha_vantage_next'] if \
            sources['alpha_vantage_next'] is not None else (median_5y * 1.2 if median_5y else None)

            # Apply reasonable caps
            if median_5y:
                median_5y = min(median_5y, max_growth)
            if next_year:
                next_year = min(next_year, max_growth * 1.5)

            return median_5y or min(0.15, max_growth), next_year or min(0.18, max_growth * 1.5)

        result['next_5y_eps_growth'], result['next_year_eps_growth'] = get_eps_growth_estimates()

        # Calculate PEG ratio safely
        forward_pe = safe_get(finnhub_metrics, 'forwardPE') or safe_get(yf_data, 'forwardPE')

        # Populate remaining metrics with safe rounding
        result.update({
            'trailing_pe': safe_get(yf_data, 'trailingPE'),
            'forward_pe': forward_pe,
            'beta': safe_get(yf_data, 'beta'),
            'dividend_yield': safe_get(yf_data, 'dividendYield', 0),
            'debt_to_equity': safe_get(yf_data, 'debtToEquity'),
            'ps_ratio': safe_get(yf_data, 'priceToSalesTrailing12Months'),
            'pb_ratio': safe_get(yf_data, 'priceToBook'),
            'roe': safe_get(yf_data, 'returnOnEquity'),
        })

    except Exception as e:
        logger.error(f"Error processing {symbol}: {str(e)}", exc_info=True)

    return result


def build_training_set(df, years):
    records = []
    end = pd.Timestamp.today()

    for _, row in df.iterrows():
        sym = row['symbol']
        try:
            # Get maximum available history
            hist = yf.Ticker(sym).history(period="30y")
            if len(hist) < 2:
                continue

            # Calculate returns
            start_price = hist['Close'].iloc[0]
            end_price = hist['Close'].iloc[-1]
            total_return = (end_price - start_price) / start_price

            # Calculate annualized return
            days = (hist.index[-1] - hist.index[0]).days
            years_held = max(days / 365.25, 1)  # At least 1 year
            annual_return = ((1 + total_return) ** (1 / years_held) - 1) * 100

            # Create record
            record = row.to_dict()
            record['annual_return'] = annual_return
            records.append(record)

        except Exception as e:
            logger.error(f"Error processing {sym}: {str(e)}")
            continue

    if not records:
        logger.warning("No valid records found with return data")
        return pd.DataFrame()

    result_df = pd.DataFrame(records)

    # Ensure annual_return exists before dropping NA
    if 'annual_return' not in result_df.columns:
        logger.error("No annual_return column created")
        return pd.DataFrame()

    return result_df.dropna(subset=['annual_return']).copy()


def train_rank(
        df, years, top_n, min_ann_return=10, max_pe=40, max_ann_return=25,
        investing_style=None, risk_free_rate=4
):
    # Early exit for invalid input
    if df.empty or 'annual_return' not in df.columns:
        print("Empty dataframe or missing annual_return column")
        return pd.DataFrame()

    # Updated features including growth estimates
    required_columns = [
        'symbol', 'annual_return', 'trailing_pe', 'forward_pe', 'beta',
        'dividend_yield', 'debt_to_equity', 'earnings_growth', 'ps_ratio',
        'pb_ratio', 'roe', 'industry', 'next_5y_eps_growth',
        'next_year_eps_growth', 'peg_ratio'
    ]

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return pd.DataFrame()

    try:
        # Style-specific filtering to exclude inappropriate stocks
        if investing_style == 'aggressive':
            # For aggressive portfolios, exclude conservative stocks
            conservative_stocks = ['LLY', 'BRK-B', 'JNJ', 'PG', 'KO', 'PEP', 'MMM', 'SO', 'DUK', 'CVX', 'O', 'T', 'PFE', 'ABT', 'WMT', 'COST', 'MCD']
            df = df[~df['symbol'].isin(conservative_stocks)].copy()
            
            # Also filter out stocks with very low beta (too conservative)
            df = df[df['beta'].fillna(0) >= 0.8].copy()
            
            # Filter out stocks with very high dividend yields (typically conservative)
            df = df[df['dividend_yield'].fillna(0) <= 3.0].copy()
            
        elif investing_style == 'conservative':
            # For conservative portfolios, prefer stable, dividend-paying stocks
            # Boost scores for stocks with good dividend yields and low beta
            df['dividend_bonus'] = np.where(df['dividend_yield'].fillna(0) >= 2.0, 2.0, 0)
            df['stability_bonus'] = np.where(df['beta'].fillna(1.0) <= 1.0, 1.5, 0)
            
        elif investing_style == 'moderate':
            # For moderate portfolios, balance between growth and stability
            # Slight preference for moderate beta and dividend yields
            df['moderate_bonus'] = np.where(
                (df['beta'].fillna(1.0).between(0.8, 1.5)) & 
                (df['dividend_yield'].fillna(0).between(1.0, 4.0)), 
                1.0, 0
            )

        # Prepare features
        numeric_features = [
            'trailing_pe', 'forward_pe', 'beta', 'dividend_yield',
            'debt_to_equity', 'earnings_growth', 'ps_ratio', 'pb_ratio',
            'roe', 'next_5y_eps_growth', 'next_year_eps_growth', 'peg_ratio'
        ]
        categorical_features = ['industry']

        # Fill missing values
        df[numeric_features] = df[numeric_features].fillna(0)
        df[categorical_features] = df[categorical_features].fillna('Unknown')

        y = df['annual_return'].values

        # Preprocessing and model pipeline
        preprocessor = ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                ('encoder', get_one_hot_encoder())
            ]), categorical_features)
        ])

        model = Pipeline([
            ('pre', preprocessor),
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5))
        ])

        model.fit(df[numeric_features + categorical_features], y)
        df['predicted_ann_return'] = model.predict(df[numeric_features + categorical_features])

        # Apply style-specific adjustments (FIXED: now properly reflects risk tolerance)
        style_adjustments = {
            'conservative': 0.85,  # Conservative estimates (lower returns expected)
            'moderate': 0.9,       # No adjustment
            'aggressive': 1.0     # More aggressive estimates (higher returns expected)
        }
        if investing_style in style_adjustments:
            df['predicted_ann_return'] *= style_adjustments[investing_style]

        # Apply style-specific bonuses
        if investing_style == 'conservative':
            df['predicted_ann_return'] += df.get('dividend_bonus', 0) + df.get('stability_bonus', 0)
        elif investing_style == 'moderate':
            df['predicted_ann_return'] += df.get('moderate_bonus', 0)

        # Apply return caps and filters
        df['predicted_ann_return'] = np.where(
            df['predicted_ann_return'] > 25,
            df['predicted_ann_return'] * 0.87,  # Cap very high returns
            df['predicted_ann_return']
        )

        # Progressive capping for different return levels
        df['predicted_ann_return'] = np.where(
            df['predicted_ann_return'] > 20,
            df['predicted_ann_return'] * 0.87,
            df['predicted_ann_return']
        )

        df['predicted_ann_return'] = np.where(
            df['predicted_ann_return'] > 15,
            df['predicted_ann_return'] * 0.9,
            df['predicted_ann_return']
        )

        # Penalty for high P/E ratios
        df['predicted_ann_return'] = df['predicted_ann_return'] - 0.25 * np.maximum(df['forward_pe'] - 15, 0)

        # Calculate total return over the time horizon
        df['predicted_total_return'] = ((1 + df['predicted_ann_return'] / 100) ** years - 1) * 100

        # Apply filters
        df = df[df['predicted_ann_return'] >= min_ann_return]
        df = df[df['trailing_pe'] <= max_pe]
        if max_ann_return is not None:
            df['predicted_ann_return'] = np.minimum(df['predicted_ann_return'], max_ann_return)

        return df.nlargest(top_n, 'predicted_ann_return').copy()

    except Exception as e:
        print(f"Error in train_rank: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame()


def process_request(
        investing_style,
        time_horizon,
        stocks_amount,
        premium,
        sector_focus='all',
        risk_tolerance='medium',
        dividend_preference=False
):
    try:
        if current_user.is_authenticated:
            current_user.update_preferences(
                investing_style=investing_style,
                time_horizon=time_horizon,
                risk_tolerance=risk_tolerance if premium else None,
                sector_focus=sector_focus if premium else None,
                dividend_preference=dividend_preference if premium else None
            )

        if premium and not current_user.get_subscription_status():
            premium = False
            flash('Premium features disabled - subscription required', 'warning')

        original_stocks = stocks_amount
        if not premium:
            stocks_amount += 3

        allocation_recommendations = {
            'conservative': {
                'sp500': 30,  # Reduced from 25 to make room for gold
                'bonds': 10,
                'btc': 0,
                'gold': 10,  # New gold allocation
                'stocks': 50,
                'notes': ['10% allocation to gold (e.g., IAU, GLD, physical gold)']
            },
            'moderate': {
                'sp500': 10,
                'bonds': 5,
                'btc': 2,
                'gold': 5,  # No gold for moderate
                'stocks': 78,
                'notes': ['BTC allocation is optional']
            },
            'aggressive': {
                'sp500': 5,
                'bonds': 0,
                'btc': 5,
                'gold': 0,  # No gold for aggressive
                'stocks': 90,
                'notes': ['BTC allocation is optional']
            }
        }
        allocation = allocation_recommendations[investing_style]

        # Define ticker lists
        main_tickers = [
            'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'AVGO', 'CRM',
            'WMT', 'JPM', 'V', 'MA', 'XOM', 'COST', 'ORCL', 'HD',
            'BAC', 'ADP', 'PEP', 'AXP', 'MS', 'ISRG', 'NOW', 'GS',
            'PGR', 'QCOM', 'ADBE', 'TJX', 'BSX', 'AMD',
            'CAT', 'BLK', 'TXN', 'BA', 'MMC', 'PANW', 'LMT', 'AMAT', 'SO', 'BMY', 'ELV',
            'DASH', 'ASML', 'REGN', 'HOOD', 'GIS', 'DUK', 'PFE', 'TSLA', 'MU', 'COIN'
        ]

        backup_tickers = {
            'conservative': [
                'JNJ', 'PG', 'KO', 'PEP', 'MMM', 'SO', 'DUK', 'CVX', 'O',
                'V', 'MA', 'SPGI', 'MCD', 'BRK-B', 'CAT', 'JPM', 'XOM',
                'COST', 'T', 'PFE', 'ABT', 'LLY', 'WMT'
            ],
            'moderate': [
                'MSFT', 'GOOG', 'V', 'MA', 'ADP', 'AAPL', 'PG',
                'CAT', 'PGR', 'SPGI', 'AXP', 'ASML', 'AMAT', 'AMZN',
                'QCOM', 'WMT', 'JPM', 'BLK'
            ],
            'aggressive': [
                'NVDA', 'MSFT', 'GOOG', 'META', 'AMZN', 'ASML', 'CRM',
                'ORCL', 'AMAT', 'QCOM', 'ADP', 'AMD', 'MU', 'AVGO', 'HOOD'
            ]
        }[investing_style]

        # Combine all unique tickers we might use
        seen = set()
        all_tickers = [x for x in main_tickers + backup_tickers if not (x in seen or seen.add(x))]

        # Fetch data for all tickers at once
        df_meta = fetch_valid_tickers(all_tickers, premium=premium)

        # Define style-specific parameters (relaxed for conservative)
        style_filters = {
            'conservative': {
                'pe_max': 30,  # Increased from 30
                'debt_max': 100,  # Increased from 100
                'beta_max': 1.7,  # Increased from 1.7
                'div_min': 2.0,  # Reduced from 3
                'beta_min': 0.1  # Reduced from 0.2
            },
            'moderate': {
                'pe_max': 35,
                'debt_max': 150,
                'beta_max': 2.0,
                'div_min': 1.5,
                'beta_min': 0.4
            },
            'aggressive': {
                'pe_max': 50,
                'debt_max': 500,
                'beta_max': 3.5,
                'div_min': 0.0,
                'beta_min': 0.7
            }
        }
        limits = style_filters[investing_style]

        # Initial filtering
        filtered_df = df_meta[
            (df_meta['trailing_pe'].fillna(999) <= limits['pe_max']) &
            (df_meta['beta'].fillna(0).between(limits['beta_min'], limits['beta_max'], inclusive='both')) &
            (df_meta['dividend_yield'].fillna(-1) >= limits['div_min']) &
            (df_meta['debt_to_equity'].fillna(999) <= limits['debt_max'])
            ].copy()

        # Additional style-specific filtering
        if investing_style == 'aggressive':
            # Exclude conservative stocks from aggressive portfolios
            conservative_stocks = ['LLY', 'BRK-B', 'JNJ', 'PG', 'KO', 'PEP', 'MMM', 'SO', 'DUK', 'CVX', 'O', 'T', 'PFE', 'ABT', 'WMT', 'COST', 'MCD']
            filtered_df = filtered_df[~filtered_df['symbol'].isin(conservative_stocks)].copy()
            
            # Use categorization function for additional filtering
            filtered_df['risk_category'] = filtered_df.apply(
                lambda row: categorize_stock_risk(
                    row['symbol'], 
                    row['beta'], 
                    row['dividend_yield'], 
                    row['trailing_pe'], 
                    row['industry']
                ), axis=1
            )
            filtered_df = filtered_df[filtered_df['risk_category'].isin(['aggressive', 'moderate'])].copy()
            filtered_df = filtered_df.drop('risk_category', axis=1)  # Clean up
            
            # Prefer stocks with higher growth potential
            filtered_df = filtered_df[filtered_df['next_5y_eps_growth'].fillna(0) >= 0.05].copy()  # At least 5% growth
            
        elif investing_style == 'conservative':
            # Use categorization function for conservative filtering
            filtered_df['risk_category'] = filtered_df.apply(
                lambda row: categorize_stock_risk(
                    row['symbol'], 
                    row['beta'], 
                    row['dividend_yield'], 
                    row['trailing_pe'], 
                    row['industry']
                ), axis=1
            )
            filtered_df = filtered_df[filtered_df['risk_category'].isin(['conservative', 'moderate'])].copy()
            filtered_df = filtered_df.drop('risk_category', axis=1)  # Clean up
            
            # Prefer stable, established companies
            filtered_df = filtered_df[filtered_df['beta'].fillna(1.0) <= 1.2].copy()
            filtered_df = filtered_df[filtered_df['dividend_yield'].fillna(0) >= 1.5].copy()
            
        elif investing_style == 'moderate':
            # Balance between growth and stability
            filtered_df = filtered_df[filtered_df['beta'].fillna(1.0).between(0.6, 1.4)].copy()

        # Handle premium features
        if premium and sector_focus != 'all':
            sector_map = {
                'tech': ['Technology', 'Semiconductors', 'Software'],
                'healthcare': ['Healthcare', 'Biotechnology', 'Pharmaceuticals'],
                'finance': ['Financial Services', 'Banks', 'Insurance'],
                'consumer': ['Consumer Defensive', 'Consumer Cyclical'],
                'energy': ['Oil & Gas', 'Energy']
            }
            filtered_df = filtered_df[filtered_df['industry'].isin(sector_map[sector_focus])]

        # Build training set and get initial recommendations
        train_primary = build_training_set(filtered_df, time_horizon)
        recs = train_rank(
            train_primary,
            time_horizon,
            stocks_amount * 2,  # Get more candidates initially
            min_ann_return=6 if investing_style == 'conservative' else 10,  # Reduced for conservative
            max_pe=limits['pe_max'],
            max_ann_return=20 if investing_style == 'conservative' else 25,
            investing_style=investing_style
        )

        # Enhanced backup selection with style-appropriate filtering
        backup_df = pd.DataFrame()
        if len(recs) < stocks_amount:
            needed = max((stocks_amount - len(recs)) * 3, 0)
            existing_symbols = recs['symbol'].tolist() if not recs.empty else []

            # Get candidates from backup tickers only (not main tickers)
            fallback_tickers = [t for t in backup_tickers if t not in existing_symbols]
            to_add = fallback_tickers[:needed]

            if to_add:
                df_bu = fetch_valid_tickers(to_add, premium=premium)
                
                # Apply style-specific filtering to backup candidates
                if investing_style == 'aggressive':
                    # For aggressive, exclude conservative stocks from backup
                    conservative_backup = ['LLY', 'BRK-B', 'JNJ', 'PG', 'KO', 'PEP', 'MMM', 'SO', 'DUK', 'CVX', 'O', 'T', 'PFE', 'ABT', 'WMT']
                    df_bu = df_bu[~df_bu['symbol'].isin(conservative_backup)].copy()
                    
                    # Use categorization function
                    df_bu['risk_category'] = df_bu.apply(
                        lambda row: categorize_stock_risk(
                            row['symbol'], 
                            row['beta'], 
                            row['dividend_yield'], 
                            row['trailing_pe'], 
                            row['industry']
                        ), axis=1
                    )
                    df_bu = df_bu[df_bu['risk_category'].isin(['aggressive', 'moderate'])].copy()
                    df_bu = df_bu.drop('risk_category', axis=1)  # Clean up
                    
                    df_bu = df_bu[df_bu['beta'].fillna(0) >= 0.8].copy()
                    df_bu = df_bu[df_bu['dividend_yield'].fillna(0) <= 3.0].copy()
                    
                elif investing_style == 'conservative':
                    # Use categorization function for conservative filtering
                    df_bu['risk_category'] = df_bu.apply(
                        lambda row: categorize_stock_risk(
                            row['symbol'], 
                            row['beta'], 
                            row['dividend_yield'], 
                            row['trailing_pe'], 
                            row['industry']
                        ), axis=1
                    )
                    df_bu = df_bu[df_bu['risk_category'].isin(['conservative', 'moderate'])].copy()
                    df_bu = df_bu.drop('risk_category', axis=1)  # Clean up
                    
                    # For conservative, prefer stable stocks
                    df_bu = df_bu[df_bu['beta'].fillna(1.0) <= 1.3].copy()
                    df_bu = df_bu[df_bu['dividend_yield'].fillna(0) >= 1.5].copy()
                
                train_bu = build_training_set(df_bu, time_horizon)
                if not train_bu.empty:
                    backup_candidates = train_rank(
                        train_bu,
                        time_horizon,
                        len(to_add),
                        min_ann_return=0,  # No minimum for backups
                        max_pe=100,  # Higher P/E allowed
                        max_ann_return=None,
                        investing_style=investing_style
                    )
                    if not backup_candidates.empty:
                        if investing_style == 'conservative':
                            backup_candidates = backup_candidates.sort_values(
                                ['dividend_yield', 'beta', 'predicted_ann_return'],
                                ascending=[False, True, False]
                            )
                        elif investing_style == 'aggressive':
                            backup_candidates = backup_candidates.sort_values(
                                ['predicted_ann_return', 'beta', 'next_5y_eps_growth'],
                                ascending=[False, False, False]
                            )
                        backup_df = backup_candidates.head(stocks_amount - len(recs))

        # Combine and finalize recommendations
        final_recs = pd.concat([recs, backup_df], ignore_index=True).head(stocks_amount)

        # Remove top 3 for non-premium users
        if not premium:
            final_recs = final_recs.iloc[3:].head(original_stocks)

        # Sort by predicted_ann_return descending
        if not final_recs.empty and 'predicted_ann_return' in final_recs.columns:
            final_recs = final_recs.sort_values('predicted_ann_return', ascending=False).reset_index(drop=True)

        # Calculate suggested allocation
        if not final_recs.empty and final_recs['predicted_ann_return'].sum() > 0:
            # Avoid division by zero or missing beta
            adj_return = final_recs['predicted_ann_return'] / final_recs['beta'].replace(0, 1).fillna(1)
            final_recs['suggested_allocation'] = adj_return / adj_return.sum()
        else:
            final_recs['suggested_allocation'] = 1 / len(final_recs) if not final_recs.empty else 0

        def format_stocks(source_df):
            formatted = []
            if not source_df.empty:
                for _, row in source_df.iterrows():
                    formatted.append({
                        'symbol': row.get('symbol', 'N/A'),
                        'total_return': f"{row.get('predicted_total_return', 0):.0f}%",
                        'annual_return': f"{row.get('predicted_ann_return', 0):.2f}%",
                        'trailing_pe': f"{row.get('trailing_pe', 0):.2f}",
                        'forward_pe': f"{row.get('forward_pe', 0):.2f}",
                        'beta': f"{row.get('beta', 0):.2f}",
                        'dividend_yield': f"{row.get('dividend_yield', 0):.2f}%",
                        'debt_to_equity': f"{row.get('debt_to_equity', 0):.2f}%",
                        'ps_ratio': f"{row.get('ps_ratio', 0):.2f}",
                        'pb_ratio': f"{row.get('pb_ratio', 0):.2f}",
                        'roe': f"{row.get('roe', 0) * 100:.2f}%",
                        'next_5y_growth': (
                            f"{row.get('next_5y_eps_growth', 0) * 100:.1f}%"
                            if row.get('next_5y_eps_growth') not in [None, '', 0, np.nan] else 'N/A'
                        ),
                        'next_year_growth': (
                            f"{row.get('next_year_eps_growth', 0) * 100:.1f}%"
                            if row.get('next_year_eps_growth') not in [None, '', 0, np.nan] else 'N/A'
                        ),
                        'peg_ratio': f"{row.get('peg_ratio', 0):.2f}",
                        'suggested_allocation': f"{row.get('suggested_allocation', 0) * 100:.2f}%",
                        'industry': row.get('industry', 'N/A')
                    })
            return formatted

        recommendations = format_stocks(final_recs)

        # Calculate weighted averages
        if not final_recs.empty:
            avg_annual = (final_recs['predicted_ann_return'] * final_recs['suggested_allocation']).sum()
            avg_div = (final_recs['dividend_yield'] * final_recs['suggested_allocation']).sum()
            avg_total = (final_recs['predicted_total_return'] * final_recs['suggested_allocation']).sum()
        else:
            avg_annual = avg_div = avg_total = 0

        # Prepare session data
        csv_buffer = StringIO()
        final_recs.to_csv(csv_buffer, index=False)
        session['csv_data'] = csv_buffer.getvalue()
        session['premium'] = premium

        return {
            'recommendations': recommendations,
            'recs_count': len(final_recs),
            'averages': {
                'total': f"{avg_total:.0f}%",
                'annual': f"{avg_annual:.2f}%",
                'dividend': f"{avg_div:.2f}%"
            },
            'premium': premium,
            'sector_focus': sector_focus,
            'risk_tolerance': risk_tolerance,
            'dividend_preference': dividend_preference,
            'allocation': allocation,
            'investing_style': investing_style
        }

    except Exception as e:
        logger.error(f"Error in process_request: {str(e)}")
        return {
            'recommendations': [],
            'recs_count': 0,
            'averages': {
                'total': '0%',
                'annual': '0%',
                'dividend': '0%'
            },
            'premium': premium,
            'sector_focus': sector_focus,
            'risk_tolerance': risk_tolerance,
            'dividend_preference': dividend_preference,
            'allocation': allocation_recommendations.get(investing_style, {}),
            'investing_style': investing_style
        }


# ──── Email Helper Functions ─────────────────────────────────────────────
def send_password_reset_email(user, token):
    """Send password reset email to user"""
    try:
        reset_url = url_for('reset_password', token=token, _external=True)
        
        # Create email content
        subject = "Password Reset Request - Vallionis AI Finance"
        
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #2c3e50;">Password Reset Request</h2>
                <p>Hello,</p>
                <p>You have requested to reset your password for your Vallionis AI Finance account.</p>
                <p>Click the button below to reset your password:</p>
                <div style="text-align: center; margin: 30px 0;">
                    <a href="{reset_url}" 
                       style="background-color: #3498db; color: white; padding: 12px 30px; 
                              text-decoration: none; border-radius: 5px; display: inline-block;">
                        Reset Password
                    </a>
                </div>
                <p>Or copy and paste this link into your browser:</p>
                <p style="word-break: break-all; color: #3498db;">{reset_url}</p>
                <p><strong>This link will expire in 1 hour.</strong></p>
                <p>If you did not request this password reset, please ignore this email.</p>
                <hr style="border: none; border-top: 1px solid #eee; margin: 30px 0;">
                <p style="font-size: 12px; color: #666;">
                    This is an automated message from Vallionis AI Finance. Please do not reply to this email.
                </p>
            </div>
        </body>
        </html>
        """
        
        text_body = f"""
        Password Reset Request
        
        Hello,
        
        You have requested to reset your password for your Vallionis AI Finance account.
        
        Click this link to reset your password: {reset_url}
        
        This link will expire in 1 hour.
        
        If you did not request this password reset, please ignore this email.
        
        ---
        This is an automated message from Vallionis AI Finance.
        """
        
        # Try Flask-Mail first, fallback to SMTP
        try:
            msg = Message(
                subject=subject,
                recipients=[user.email],
                html=html_body,
                body=text_body
            )
            mail.send(msg)
            logger.info(f"Password reset email sent to {user.email} via Flask-Mail")
            return True
        except Exception as e:
            logger.warning(f"Flask-Mail failed for {user.email}: {e}")
        
        # Fallback to direct SMTP
        try:
            smtp_server = app.config.get('MAIL_SERVER', 'smtp.gmail.com')
            smtp_port = app.config.get('MAIL_PORT', 587)
            smtp_username = app.config.get('MAIL_USERNAME')
            smtp_password = app.config.get('MAIL_PASSWORD')
            
            if not smtp_username or not smtp_password:
                logger.error("Email credentials not configured")
                return False
            
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = smtp_username
            msg['To'] = user.email
            
            msg.attach(MIMEText(text_body, 'plain'))
            msg.attach(MIMEText(html_body, 'html'))
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.send_message(msg)
            
            logger.info(f"Password reset email sent to {user.email} via SMTP")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send password reset email to {user.email}: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Error in send_password_reset_email: {e}")
        return False


# ──── Routes ─────────────────────────────────────────────────────────────

@app.route('/connectivity-check')
def connectivity_check():
    # Check database connection
    try:
        db.engine.connect()
        db_status = 'ok'
    except Exception as e:
        db_status = f'failed: {e}'

    # Check DNS resolution for mail server
    try:
        addr = socket.gethostbyname('smtp.gmail.com')
        dns_status = f'ok - resolved to {addr}'
    except Exception as e:
        dns_status = f'failed: {e}'

    # Check SMTP port connectivity
    try:
        with socket.create_connection(('smtp.gmail.com', 587), timeout=5):
            smtp_status = 'ok'
    except Exception as e:
        smtp_status = f'failed: {e}'

    return jsonify({
        'database': db_status,
        'dns_resolution': dns_status,
        'smtp_connection': smtp_status
    })
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    # Check for maintenance mode
    if FEATURE_FLAGS['maintenance_mode']:
        return render_template('maintenance.html')
    
    form = RecommendationForm()

    # Set default values for premium fields
    if not form.premium.data:
        form.sector_focus.data = 'all'
        form.risk_tolerance.data = 'medium'
        form.dividend_preference.data = False

    if form.validate_on_submit():
        try:
            results = process_request(
                form.investing_style.data,
                form.time_horizon.data,
                form.stocks_amount.data,
                form.premium.data if current_user.get_subscription_status() else False,
                form.sector_focus.data if current_user.get_subscription_status() else 'all',
                form.risk_tolerance.data if current_user.get_subscription_status() else 'medium',
                form.dividend_preference.data if current_user.get_subscription_status() else False
            )

            return render_template('results.html', **results, current_year=datetime.now().year)

        except Exception as e:
            flash(f"Error generating recommendations: {str(e)}", "error")
            app.logger.error(f"Recommendation error: {str(e)}")
            return redirect(url_for('index'))

    # Use new design if feature flag is enabled
    template_name = 'index_new.html' if FEATURE_FLAGS['new_design'] else 'index.html'
    
    # Get currency configuration for the template
    currency_config = get_currency_config()
    
    return render_template(template_name,
                           form=form,
                           is_premium_user=current_user.get_subscription_status(),
                           feature_flags=FEATURE_FLAGS,
                           currency_config=currency_config)


@app.route('/download')
@login_required
def download():
    if not current_user.get_subscription_status():
        abort(403, description="Premium subscription required")
    if session.get('premium_type') == 'trial' and time.time() > session.get('trial_end', 0):
        session['premium'] = False
        abort(403, description="Your free trial has expired")

    if not session.get('premium', False):
        abort(403)
    csv_data = session.get('csv_data')
    if not csv_data:
        abort(404)
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=recommended_portfolio.csv"}
    )


def safe_stripe_call(func, *args, **kwargs):
    """Wrapper to handle Stripe ID inconsistencies"""
    try:
        return func(*args, **kwargs)
    except stripe.error.InvalidRequestError as e:
        if "No such customer" in str(e):
            if current_user.is_authenticated:
                app.logger.warning(f"Resetting invalid customer ID {current_user.stripe_customer_id}")
                current_user.stripe_customer_id = None
                current_user.subscription_status = None
                current_user.subscription_expires = None
                db.session.commit()
            return None
        raise
    except stripe.error.StripeError as e:
        app.logger.error(f"Stripe API error: {str(e)}")
        raise


@app.route('/subscribe', methods=['GET', 'POST'])
@login_required
def subscribe():
    if current_user.subscription_type == 'lifetime':
        flash('Cannot switch from lifetime to monthly subscription', 'warning')
        return redirect(url_for('subscription'))

    if request.method == 'POST':
        try:
            # Get user's currency preference (could be from form or session)
            currency = request.form.get('currency', 'GBP')
            currency_config = get_currency_config(currency)
            
            # Verify the price exists first
            try:
                price = stripe.Price.retrieve(currency_config['monthly_price_id'])
            except stripe.error.InvalidRequestError:
                flash('Subscription product not properly configured. Please contact support.', 'danger')
                return redirect(url_for('subscription'))

            checkout_session = stripe.checkout.Session.create(
                client_reference_id=current_user.id,
                customer_email=current_user.email,
                payment_method_types=['card'],
                line_items=[{
                    'price': currency_config['monthly_price_id'],
                    'quantity': 1,
                }],
                mode='subscription',
                subscription_data={
                    'trial_period_days': 7,
                    'trial_settings': {
                        'end_behavior': {
                            'missing_payment_method': 'cancel'
                        }
                    }
                },
                success_url=url_for('success', _external=True) + '?session_id={CHECKOUT_SESSION_ID}',
                cancel_url=url_for('subscription', _external=True),
            )
            return redirect(checkout_session.url, code=303)
        except Exception as e:
            logger.error(f"Subscription error: {str(e)}")
            flash(f'Error creating subscription: {str(e)}', 'danger')
            return redirect(url_for('subscription'))

    return redirect(url_for('subscription'))


@app.context_processor
def inject_env():
    env = Environment()
    env.filters['datetimeformat'] = lambda value: datetime.fromtimestamp(value).strftime('%Y-%m-%d %H:%M:%S')
    return {'env': env}


def handle_checkout_session(session):
    try:
        from datetime import datetime, timezone, timedelta

        user = db.session.get(User, session['client_reference_id'])
        if not user:
            logger.error(f"User not found for client_reference_id: {session['client_reference_id']}")
            return

        logger.info(f"Processing checkout for user {user.id}")

        if session['mode'] == 'subscription':
            subscription_id = session.get('subscription')
            if not subscription_id:
                logger.error("Subscription ID is missing in session data")
                return

            # Get subscription details directly from the expanded session object
            subscription = session.get('subscription')
            if not subscription:
                logger.error("No subscription data in session")
                return

            user.subscription_type = 'monthly'
            user.subscription_status = subscription['status']
            user.premium = subscription['status'] in ['active', 'trialing']

            if subscription['status'] == 'trialing':
                # Calculate trial end date (7 days from now)
                user.subscription_expires = datetime.now(timezone.utc) + timedelta(days=7)
            elif 'current_period_end' in subscription:
                # Use the actual subscription end date
                user.subscription_expires = datetime.fromtimestamp(
                    subscription['current_period_end'],
                    timezone.utc
                )

        elif session['mode'] == 'payment':  # Lifetime purchase
            user.subscription_type = 'lifetime'
            user.subscription_status = 'active'
            user.premium = True
            user.subscription_expires = datetime(2099, 1, 1, tzinfo=timezone.utc)

        if session.get('customer') and not user.stripe_customer_id:
            user.stripe_customer_id = session['customer']

        db.session.commit()
        logger.info(f"Successfully updated user {user.email} with premium access")

    except Exception as e:
        logger.error(f"Error in handle_checkout_session: {str(e)}", exc_info=True)
        db.session.rollback()
        raise  # Re-raise the exception for debugging


def update_subscription_status(subscription):
    """Handle subscription state changes"""
    user = User.query.filter_by(stripe_customer_id=subscription.customer).first()
    if not user:
        app.logger.error(f"No user found for customer {subscription.customer}")
        return

    user.subscription_status = subscription.status
    user.subscription_expires = datetime.utcfromtimestamp(subscription.current_period_end)

    if subscription.status == 'active':
        user.premium = True
    elif subscription.status in ['canceled', 'unpaid']:
        user.premium = False

    db.session.commit()


def handle_payment_intent(payment_intent):
    # Process the payment intent object
    print(f"Processing payment intent: {payment_intent['id']}")


@csrf.exempt
@app.route('/stripe-webhook', methods=['POST'])
def stripe_webhook():
    payload = request.get_data(as_text=True)
    sig_header = request.headers.get('Stripe-Signature')
    webhook_secret = STRIPE_WEBHOOK_SECRET

    logger.info("Webhook received")
    logger.debug(f"Payload: {payload[:200]}...")  # Log first 200 chars of payload
    logger.debug(f"Signature header: {sig_header}")

    try:
        # Verify and construct the event with Stripe
        event = stripe.Webhook.construct_event(
            payload,
            sig_header,
            webhook_secret,
            tolerance=300  # 5-minute clock skew tolerance
        )
    except ValueError as e:
        logger.error(f"Invalid payload: {str(e)}")
        return jsonify({"error": "Invalid payload"}), 400
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid signature: {str(e)}")
        return jsonify({"error": "Invalid signature"}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": str(e)}), 400

    logger.info(f"Processing event: {event['type']}")

    try:
        # Handle `checkout.session.completed` event
        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']
            logger.info(f"Checkout session completed: {session['id']}")

            if session['payment_status'] == 'paid':
                # Update user record
                user = db.session.get(User, session['client_reference_id'])
                if not user:
                    logger.error(f"User not found for ID: {session['client_reference_id']}")
                    return jsonify({"error": "User not found"}), 404

                if session['mode'] == 'payment':  # Lifetime purchase
                    logger.info("Processing lifetime purchase")
                    user.premium = True
                    user.subscription_type = 'lifetime'
                    user.subscription_status = 'active'
                    user.subscription_expires = datetime(2099, 1, 1)

                elif session['mode'] == 'subscription':  # Recurring subscription
                    logger.info("Processing subscription")
                    subscription_id = session.get('subscription')
                    if subscription_id:
                        subscription = stripe.Subscription.retrieve(subscription_id)
                    else:
                        logger.error("Subscription ID missing in checkout session")
                        subscription = None
                    user.subscription_type = 'monthly'
                    user.subscription_status = subscription['status']
                    user.premium = subscription['status'] in ['active', 'trialing']
                    if 'current_period_end' in subscription:
                        user.subscription_expires = datetime.fromtimestamp(subscription['current_period_end'])
                    else:
                        logger.error("current_period_end not found in subscription object")
                        user.subscription_expires = None

                # Set Stripe customer ID if not already set
                if session['customer'] and not user.stripe_customer_id:
                    user.stripe_customer_id = session['customer']

                db.session.commit()
                logger.info("User record updated successfully")

        # Handle `customer.subscription.updated` event
        elif event['type'] == 'customer.subscription.updated':
            subscription = event['data']['object']
            user = User.query.filter_by(stripe_customer_id=subscription['customer']).first()

            if user:
                logger.info(f"Updating subscription for user {user.email}")
                user.subscription_status = subscription['status']
                user.premium = subscription['status'] in ['active', 'trialing']

                # Set expiration to 7 days in the future for trialing subscriptions
                if subscription['status'] == 'trialing':
                    user.subscription_expires = datetime.utcnow() + timedelta(days=7)
                elif 'current_period_end' in subscription:
                    user.subscription_expires = datetime.fromtimestamp(subscription['current_period_end'])
                else:
                    user.subscription_expires = None

                db.session.commit()

        # Handle `customer.subscription.deleted` event
        elif event['type'] == 'customer.subscription.deleted':
            subscription = event['data']['object']
            user = User.query.filter_by(stripe_customer_id=subscription['customer']).first()

            if user:
                logger.info(f"Deleting subscription for user {user.email}")
                user.subscription_status = 'canceled'
                user.premium = False
                user.subscription_expires = None
                db.session.commit()

        # Handle `invoice.payment_failed` event
        elif event['type'] == 'invoice.payment_failed':
            invoice = event['data']['object']
            user = User.query.filter_by(stripe_customer_id=invoice['customer']).first()
            if user:
                logger.warning(f"Payment failed for user {user.email}")
                # Add notification logic here (e.g., email the user)

        # Handle `customer.updated` event
        elif event['type'] == 'customer.updated':
            customer = event['data']['object']
            user = User.query.filter_by(stripe_customer_id=customer['id']).first()
            if user:
                logger.info(f"Customer updated: {customer}")
                # Update user details if necessary (e.g., email or metadata)

        else:
            logger.warning(f"Unhandled event type: {event['type']}")
            logger.debug(f"Unhandled event payload: {event}")

        return jsonify({"status": "success"}), 200

    except Exception as e:
        logger.error(f"Error processing event {event['type']}: {str(e)}", exc_info=True)
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route('/success')
def success():
    session_id = request.args.get('session_id')
    if session_id:
        try:
            # Retrieve the checkout session with expanded subscription
            checkout_session = stripe.checkout.Session.retrieve(
                session_id,
                expand=['subscription']
            )
            handle_checkout_session(checkout_session)
        except Exception as e:
            logger.error(f"Error processing success: {str(e)}")
            flash('Error processing your subscription. Please contact support.', 'danger')

    return render_template('success.html')


@app.route('/purchase-lifetime', methods=['POST'])
@login_required
def purchase_lifetime():
    if current_user.subscription_type == 'lifetime' and current_user.get_subscription_status():
        flash('You already have lifetime access!', 'info')
        return redirect(url_for('subscription'))

    try:
        # Get user's currency preference
        currency = request.form.get('currency', 'GBP')
        currency_config = get_currency_config(currency)
        
        # Verify the price exists first
        try:
            price = stripe.Price.retrieve(currency_config['lifetime_price_id'])
            logger.info(f"Lifetime price verified: {price.id} - {price.unit_amount / 100} {price.currency}")
        except stripe.error.InvalidRequestError as e:
            logger.error(f"Invalid lifetime price: {str(e)}")
            flash('Lifetime product not properly configured. Please contact support.', 'danger')
            return redirect(url_for('subscription'))

        # Create checkout session
        checkout_session = stripe.checkout.Session.create(
            client_reference_id=current_user.id,
            customer_email=current_user.email,
            payment_method_types=['card'],
            line_items=[{
                'price': currency_config['lifetime_price_id'],
                'quantity': 1,
            }],
            mode='payment',
            success_url=url_for('success', _external=True) + '?session_id={CHECKOUT_SESSION_ID}',
            cancel_url=url_for('subscription', _external=True),
            metadata={
                'product_type': 'lifetime',
                'user_id': current_user.id,
                'currency': currency
            }
        )
        logger.info(f"Created checkout session: {checkout_session.id}")
        return redirect(checkout_session.url, code=303)

    except stripe.error.StripeError as e:
        logger.error(f"Stripe error in purchase_lifetime: {str(e)}")
        flash('Payment processing error. Please try again.', 'danger')
    except Exception as e:
        logger.error(f"Unexpected error in purchase_lifetime: {str(e)}")
        flash('An unexpected error occurred. Please contact support.', 'danger')

    return redirect(url_for('subscription'))


def check_expired_subscriptions():
    with app.app_context():
        now = datetime.utcnow()
        expired_users = User.query.filter(
            User.premium == True,
            User.subscription_expires != None,
            User.subscription_expires <= now
        ).all()

        for user in expired_users:
            logger.info(f"Removing premium for expired user: {user.email}")
            user.premium = False
            user.subscription_status = None
            user.subscription_expires = None
            db.session.commit()


@app.route('/subscription', methods=['GET'])
@login_required
def subscription():
    try:
        subscription_info = None
        if current_user.stripe_customer_id:
            # Verify customer exists through safe call
            cust = safe_stripe_call(stripe.Customer.retrieve, current_user.stripe_customer_id)

            if not cust:
                flash("Stripe customer record not found - reset local ID", "warning")
                return redirect(url_for('subscription'))

            # Get subscription data with safe call
            subscriptions = safe_stripe_call(
                stripe.Subscription.list,
                customer=current_user.stripe_customer_id,
                status='all',
                limit=1
            )

            if subscriptions and subscriptions.data:
                subscription_info = subscriptions.data[0]
                # Handle subscription status changes
                current_user.subscription_status = subscription_info.status
                if subscription_info.cancel_at_period_end:
                    current_user.subscription_expires = datetime.utcfromtimestamp(
                        subscription_info.current_period_end
                    )
                db.session.commit()

        # Get currency configuration for the template
        currency_config = get_currency_config()
        available_currencies = CURRENCY_CONFIG.keys()

        return render_template('subscription.html',
                           subscription_active=current_user.get_subscription_status(),
                           expires=current_user.subscription_expires,
                           subscription_info=subscription_info,
                           currency_config=currency_config,
                           available_currencies=available_currencies)
    except Exception as e:
        flash(f'Error loading subscription: {str(e)}', 'danger')
        app.logger.error(f"Subscription error: {str(e)}", exc_info=True)
        return redirect(url_for('index'))

scheduler = BackgroundScheduler()
scheduler.add_job(check_expired_subscriptions, 'interval', hours=1)
scheduler.start()


@app.route('/change-password', methods=['GET', 'POST'])
@login_required
def change_password():
    form = ChangePasswordForm()
    if form.validate_on_submit():
        if check_password_hash(current_user.password, form.old_password.data):
            current_user.password = generate_password_hash(form.new_password.data)
            db.session.commit()
            flash('Password updated successfully!', 'success')
            return redirect(url_for('account'))
        else:
            flash('Incorrect current password', 'danger')
    return render_template('change_password.html', form=form)


@app.route('/create-portal-session', methods=['POST'])
@login_required
def create_portal_session():
    if not current_user.stripe_customer_id:
        flash('No active subscription found', 'danger')
        return redirect(url_for('subscription'))

    try:
        portal_session = stripe.billing_portal.Session.create(
            customer=current_user.stripe_customer_id,
            return_url=url_for('subscription', _external=True)
        )
        return redirect(portal_session.url, code=303)
    except Exception as e:
        flash(f'Error accessing subscription portal: {str(e)}', 'danger')
        return redirect(url_for('subscription'))


@app.context_processor
def inject_user():
    return dict(
        is_premium=current_user.is_authenticated and current_user.get_subscription_status()
    )


@app.route('/account', methods=['GET', 'POST'])
@login_required
def account():
    form = ChangePasswordForm()
    if form.validate_on_submit():
        if check_password_hash(current_user.password, form.old_password.data):
            current_user.password = generate_password_hash(form.new_password.data)
            db.session.commit()
            flash('Password updated successfully!', 'success')
            return redirect(url_for('account'))
        else:
            flash('Incorrect current password', 'danger')
    return render_template('account.html', change_password_form=form)


@app.route('/ai-coach')
@login_required
def ai_coach():
    """Renders the AI Finance Coach page"""
    return render_template('ai_coach.html', title="AI Finance Coach", user=current_user)


@app.route('/api/ai/health', methods=['GET'])
@csrf.exempt
def ai_health_check():
    """Provides a health check for the AI service."""
    # In a real-world scenario, this would check the connection to the actual AI model.
    # For now, we'll assume it's always healthy to enable the UI.
    return jsonify({'status': 'healthy'})


@app.route('/api/ai/chat', methods=['POST'])
@login_required
@csrf.exempt
def ai_chat():
    """Handles chat messages from the AI Coach interface."""
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Invalid request. Message not provided.'}), 400

    user_message = data.get('message')

    # URL of your self-hosted AI service (running from ai_coach_api.py)
    AI_SERVICE_URL = os.environ.get('AI_SERVICE_URL', 'http://127.0.0.1:8000/chat')

    try:
        # Forward the message to the AI service
        response = requests.post(AI_SERVICE_URL, json={'message': user_message, 'user_id': current_user.id})
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        ai_data = response.json()
        ai_response = ai_data.get('response', 'Sorry, I could not process your request.')

    except requests.exceptions.RequestException as e:
        logger.error(f"Could not connect to AI service: {e}")
        ai_response = "Sorry, the AI Coach is currently unavailable. Please try again later."

    return jsonify({'response': ai_response})


@app.route('/fix-customer-id', methods=['POST'])
@login_required
def fix_customer_id():
    try:
        # First check local database consistency
        if current_user.subscription_status and not current_user.stripe_customer_id:
            current_user.subscription_status = None
            current_user.subscription_expires = None
            db.session.commit()
            flash("Reset invalid subscription status", "warning")

        # Search all possible customer associations
        customers = safe_stripe_call(
            stripe.Customer.search,
            query=f"email:'{current_user.email}'",
            limit=1
        )

        if customers and customers.data:
            valid_customer = None
            for cust in customers.data:
                # Verify customer has active subscriptions
                subs = safe_stripe_call(
                    stripe.Subscription.list,
                    customer=cust.id,
                    status='all',
                    limit=1
                )
                if subs and subs.data:
                    valid_customer = cust
                    break

            if valid_customer:
                current_user.stripe_customer_id = valid_customer.id
                db.session.commit()
                flash("Recovered customer ID from active subscription", "success")
                return redirect(url_for('subscription'))

        # Fallback to payment history
        charges = safe_stripe_call(
            stripe.Charge.list,
            customer=current_user.email,
            limit=1
        )
        if charges and charges.data:
            current_user.stripe_customer_id = charges.data[0].customer
            db.session.commit()
            flash("Recovered customer ID from payment history", "success")
            return redirect(url_for('subscription'))

        flash("No matching Stripe records found", "warning")

    except Exception as e:
        flash(f"Recovery failed: {str(e)}", "danger")
        app.logger.error(f"Customer ID recovery error: {str(e)}")

    return redirect(url_for('subscription'))


@app.route('/delete-account', methods=['POST'])
@login_required
def delete_account():
    try:
        # Get the user ID before deletion
        user_id = current_user.id

        # First delete the current_preferences record
        if current_user.current_preferences:
            db.session.delete(current_user.current_preferences)
            db.session.flush()  # Ensure deletion happens before user deletion

        # Then delete all preference history records
        UserPreferenceHistory.query.filter_by(user_id=user_id).delete()
        db.session.flush()

        # If the user has a Stripe subscription, cancel it first
        if current_user.stripe_customer_id:
            try:
                subscriptions = stripe.Subscription.list(customer=current_user.stripe_customer_id)
                for sub in subscriptions.data:
                    stripe.Subscription.delete(sub.id)
            except stripe.error.StripeError:
                pass  # Subscription might already be canceled

        # Now delete the user
        db.session.delete(current_user)
        db.session.commit()

        # Log the user out
        logout_user()

        flash('Your account has been permanently deleted.', 'success')
        return redirect(url_for('index'))

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting account: {str(e)}", exc_info=True)
        flash('Error deleting account. Please try again or contact support.', 'danger')
        return redirect(url_for('account'))


def check_customer_ids():
    with app.app_context():
        users = User.query.filter(
            (User.stripe_customer_id == None) |
            (User.stripe_customer_id == '')
        ).filter(
            User.subscription_status != None
        ).all()

        if users:
            app.logger.critical(f"""
                Found {len(users)} users with missing customer IDs:
                {[u.email for u in users]}
            """)


@app.route('/stripe-webhook', methods=['GET'])
def webhook_test():
    return "Webhook endpoint is working", 200


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()

        if user and check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Login unsuccessful. Please check email and password', 'danger')

    return render_template('login.html', title='Login', form=form)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    form = RegistrationForm()

    if form.validate_on_submit():
        # Check password length
        if len(form.password.data) < 8:
            flash('Password must be at least 8 characters long', 'danger')
            return redirect(url_for('register'))

        # Check if email exists
        existing_user = User.query.filter_by(email=form.email.data).first()
        if existing_user:
            flash('Email already registered. Please login instead.', 'danger')
            return redirect(url_for('login'))

        # Create new user with hashed password
        hashed_password = generate_password_hash(form.password.data)
        user = User(
            email=form.email.data,
            password=hashed_password,
            premium=False
        )

        try:
            db.session.add(user)
            db.session.commit()
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Registration error: {str(e)}")
            flash('Registration failed. Please try again.', 'danger')

    return render_template('register.html', form=form)


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    form = ForgotPasswordForm()
    
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            try:
                token = user.generate_reset_token()
                if send_password_reset_email(user, token):
                    flash('A password reset link has been sent to your email address.', 'info')
                else:
                    flash('Failed to send reset email. Please try again or contact support.', 'danger')
            except Exception as e:
                logger.error(f"Error generating reset token for {user.email}: {e}")
                flash('An error occurred. Please try again.', 'danger')
        else:
            # Still show success message for security (don't reveal if email exists)
            flash('If an account with that email exists, a password reset link has been sent.', 'info')
        
        return redirect(url_for('login'))
    
    return render_template('forgot_password.html', form=form)


@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    user = User.verify_reset_token(token)
    if not user:
        flash('Invalid or expired reset link.', 'danger')
        return redirect(url_for('forgot_password'))
    
    form = ResetPasswordForm()
    
    if form.validate_on_submit():
        try:
            hashed_password = generate_password_hash(form.password.data)
            user.password = hashed_password
            user.clear_reset_token()
            
            flash('Your password has been reset successfully. You can now log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            logger.error(f"Error resetting password for user {user.id}: {e}")
            flash('An error occurred while resetting your password. Please try again.', 'danger')
    
    return render_template('reset_password.html', form=form)


@app.route('/cancel-subscription', methods=['POST'])
@login_required
def cancel_subscription():
    try:
        # Check if user has a Stripe subscription
        if not current_user.stripe_customer_id:
            flash('No active subscription found', 'warning')
            return redirect(url_for('subscription'))

        # Get active subscriptions
        subscriptions = stripe.Subscription.list(
            customer=current_user.stripe_customer_id,
            status='active',
            limit=1
        )

        if not subscriptions.data:
            flash('No active subscription found', 'warning')
            return redirect(url_for('subscription'))

        subscription = subscriptions.data[0]

        # Cancel the subscription at period end
        updated_sub = stripe.Subscription.modify(
            subscription.id,
            cancel_at_period_end=True
        )

        # Update user status
        current_user.subscription_status = 'pending_cancel'
        current_user.subscription_expires = datetime.utcfromtimestamp(
            updated_sub.current_period_end
        )
        db.session.commit()

        flash(
            f'Subscription will remain active until {current_user.subscription_expires.strftime("%Y-%m-%d")}',
            'success'
        )

    except stripe.error.InvalidRequestError as e:
        if "No such customer" in str(e):
            flash("Payment system mismatch - contact support", "danger")
        elif "No such subscription" in str(e):
            current_user.subscription_status = None
            current_user.subscription_expires = None
            db.session.commit()
            flash("Subscription already canceled", "warning")
        else:
            flash("Payment system error - try again later", "danger")
    except Exception as e:
        db.session.rollback()
        flash(f'Error processing request: {str(e)}', 'danger')

    return redirect(url_for('subscription'))


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/privacy')
def privacy():
    return render_template('privacy.html')


@app.route('/terms')
def terms():
    return render_template('terms.html')


@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
    return response


@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html', now=datetime.now(timezone.utc)), 500


@app.errorhandler(403)
def forbidden_error(error):
    return render_template('403.html'), 403


@app.route('/favicon.ico')
def favicon():
    return '', 204


@app.route('/.well-known/appspecific/com.chrome.devtools.json')
def handle_chrome_devtools():
    return jsonify({}), 200


@app.route('/disclaimer')
def disclaimer():
    return render_template('disclaimer.html', current_year=datetime.now().year)


@app.route('/login/google')
def login_google():
    session.pop('oauth_state', None)
    session.pop('nonce', None)

    session['oauth_state'] = generate_token()
    session['nonce'] = generate_token()

    return google.authorize_redirect(
        redirect_uri=url_for('authorize_google', _external=True),
        nonce=session['nonce'],
        state=session['oauth_state'],
        access_type='offline',  # Request refresh token
        prompt='consent'  # Force consent screen to ensure refresh token
    )


@app.route('/authorize/google')
def authorize_google():
    try:
        # Verify state
        if request.args.get('state') != session.get('oauth_state'):
            flash('Invalid OAuth state', 'danger')
            return redirect(url_for('login'))

        # Get token - will use the special redirect URI for desktop apps
        token = google.authorize_access_token()
        if not token:
            flash('Failed to get access token', 'danger')
            return redirect(url_for('login'))

        # Parse and verify ID token
        user_info = google.parse_id_token(
            token,
            nonce=session.get('nonce'),
            claims_options={
                'iss': {
                    'values': ['https://accounts.google.com'],
                    'essential': True
                }
            }
        )

        # Find or create user
        user = User.query.filter_by(email=user_info['email']).first()
        if not user:
            user = User(
                email=user_info['email'],
                password=generate_password_hash(generate_token()),
                premium=False
            )
            db.session.add(user)
            db.session.commit()

        login_user(user)
        flash('Logged in successfully with Google', 'success')
        return redirect(url_for('index'))

    except Exception as e:
        logger.error(f"Google login error: {str(e)}")
        flash('Google login failed', 'danger')
        return redirect(url_for('login'))


@app.route('/preference-history')
@login_required
def preference_history():
    page = request.args.get('page', 1, type=int)
    per_page = 10

    history = current_user.preference_history.order_by(
        UserPreferenceHistory.timestamp.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)

    return render_template('preference_history.html', history=history)


@app.route('/api/preference-trends')
@login_required
def preference_trends():
    # Get all history records
    records = current_user.preference_history.order_by(
        UserPreferenceHistory.timestamp
    ).all()

    # Prepare data for charting
    data = {
        'dates': [r.timestamp.isoformat() for r in records],
        'time_horizon': [r.time_horizon for r in records],
        'investing_style': [r.investing_style for r in records],
        'risk_tolerance': [r.risk_tolerance or 'medium' for r in records]
    }

    return jsonify(data)


@app.route('/googlee527911ad856f67e.html')
def serve_verification():
    return send_from_directory('static', 'googlee527911ad856f67e.html')


@app.route('/verify')
def verify():
    return render_template('verify.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(check_customer_ids, 'interval', hours=1)
scheduler.start()

@app.route('/sitemap.xml')
def sitemap():
    """Generate sitemap for search engines"""
    from flask import make_response
    
    # Get the base URL
    base_url = request.url_root.rstrip('/')
    
    # Define your main pages with explicit URLs
    pages = [
        {'url': f"{base_url}/", 'priority': '1.0', 'changefreq': 'daily'},
        {'url': f"{base_url}/login", 'priority': '0.8', 'changefreq': 'monthly'},
        {'url': f"{base_url}/register", 'priority': '0.8', 'changefreq': 'monthly'},
        {'url': f"{base_url}/subscription", 'priority': '0.7', 'changefreq': 'weekly'},
        {'url': f"{base_url}/privacy", 'priority': '0.5', 'changefreq': 'monthly'},
        {'url': f"{base_url}/terms", 'priority': '0.5', 'changefreq': 'monthly'},
        {'url': f"{base_url}/disclaimer", 'priority': '0.5', 'changefreq': 'monthly'},
        {'url': f"{base_url}/contact", 'priority': '0.6', 'changefreq': 'monthly'},
    ]
    
    try:
        sitemap_xml = render_template('sitemap.xml', pages=pages, moment=datetime.now())
        response = make_response(sitemap_xml)
        response.headers["Content-Type"] = "application/xml"
        return response
    except Exception as e:
        logger.error(f"Error generating sitemap: {str(e)}")
        # Fallback to static sitemap
        return send_from_directory('static', 'sitemap.xml', mimetype='application/xml')

@app.route('/seo-status')
def seo_status():
    """SEO status monitoring page"""
    base_url = request.url_root.rstrip('/')
    
    # Define pages to check
    pages_to_check = [
        {'url': f"{base_url}/", 'name': 'Homepage'},
        {'url': f"{base_url}/login", 'name': 'Login'},
        {'url': f"{base_url}/register", 'name': 'Register'},
        {'url': f"{base_url}/subscription", 'name': 'Subscription'},
        {'url': f"{base_url}/privacy", 'name': 'Privacy'},
        {'url': f"{base_url}/terms", 'name': 'Terms'},
        {'url': f"{base_url}/disclaimer", 'name': 'Disclaimer'},
        {'url': f"{base_url}/contact", 'name': 'Contact'},
    ]
    
    return render_template('seo_status.html', 
                         pages=pages_to_check, 
                         base_url=base_url,
                         sitemap_url=f"{base_url}/sitemap.xml",
                         moment=datetime.now())

@app.route('/admin/feature-flags', methods=['GET', 'POST'])
@login_required
def admin_feature_flags():
    """Admin panel to control feature flags"""
    if not current_user.is_authenticated or not current_user.email.endswith('@admin.com'):
        abort(403)
    
    if request.method == 'POST':
        # Update feature flags via environment variables
        # Note: This would require server restart in production
        flash('Feature flags updated. Server restart required for changes to take effect.', 'info')
    
    return render_template('admin/feature_flags.html', feature_flags=FEATURE_FLAGS)

@app.route('/maintenance')
def maintenance():
    """Maintenance page"""
    return render_template('maintenance.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'feature_flags': FEATURE_FLAGS,
        'database': 'connected' if db.engine.pool.checkedin() > 0 else 'disconnected'
    })

# Currency configuration
CURRENCY_CONFIG = {
    'USD': {
        'symbol': '$',
        'monthly_price': 6.99,
        'lifetime_price': 129.99,
        'monthly_price_id': os.getenv('STRIPE_USD_MONTHLY_PRICE_ID'),
        'lifetime_price_id': os.getenv('STRIPE_USD_LIFETIME_PRICE_ID')
    },
    'EUR': {
        'symbol': '€',
        'monthly_price': 5.99,
        'lifetime_price': 109.99,
        'monthly_price_id': os.getenv('STRIPE_EUR_MONTHLY_PRICE_ID'),
        'lifetime_price_id': os.getenv('STRIPE_EUR_LIFETIME_PRICE_ID')
    },
    'GBP': {
        'symbol': '£',
        'monthly_price': 4.99,
        'lifetime_price': 99.99,
        'monthly_price_id': os.getenv('STRIPE_GBP_MONTHLY_PRICE_ID'),
        'lifetime_price_id': os.getenv('STRIPE_GBP_LIFETIME_PRICE_ID')
    }
}

# Default currency (fallback)
DEFAULT_CURRENCY = 'GBP'

def detect_user_currency():
    """Detect user's preferred currency based on IP or browser settings"""
    # Simple IP-based detection (you could use a service like ipapi.co)
    # For now, return default currency
    return DEFAULT_CURRENCY

def get_currency_config(currency_code=None):
    """Get currency configuration for the specified currency"""
    if not currency_code:
        currency_code = detect_user_currency()
    return CURRENCY_CONFIG.get(currency_code, CURRENCY_CONFIG[DEFAULT_CURRENCY])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
