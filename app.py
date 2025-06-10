from flask import Flask, render_template, request, session, Response, abort, redirect, url_for, jsonify, send_from_directory
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
from sqlalchemy.sql import text
import sklearn
from packaging import version
from sklearn.preprocessing import OneHotEncoder

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables
load_dotenv('.env')

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
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

# =============================================
# Configuration
# =============================================
app.secret_key = os.getenv('SECRET_KEY')
if not app.secret_key:
    raise ValueError("No SECRET_KEY set for Flask application")

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ['DATABASE_URL']

# Initialize database
db = SQLAlchemy(app)
migrate = Migrate(app, db)

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

# Database initialization function
def initialize_database():
    with app.app_context():
        try:
            # Test connection using text() wrapper
            db.session.execute(text('SELECT 1')).scalar()
            logger.info("✅ Database connection established")

            # Create tables if they don't exist
            db.create_all()
            logger.info("✅ Database initialized successfully")
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {str(e)}")
            # If using SQLite, ensure the directory exists
            if 'sqlite' in app.config['SQLALCHEMY_DATABASE_URI']:
                os.makedirs(os.path.dirname('instance/'), exist_ok=True)
                try:
                    db.create_all()
                except Exception as sqlite_error:
                    logger.error(f"❌ SQLite initialization failed: {str(sqlite_error)}")
                    raise
            else:
                raise

# Run database initialization at app startup
initialize_database()


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

# =============================================
# Database Models
# =============================================
class User(db.Model, UserMixin):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    premium = db.Column(db.Boolean, default=False)
    subscription_expires = db.Column(db.DateTime)
    stripe_customer_id = db.Column(db.String(50))
    subscription_type = db.Column(db.String(20), default=None)
    subscription_status = db.Column(db.String(20), default=None)

    def get_subscription_status(self):
        if self.premium:
            if self.subscription_type == 'lifetime':
                return True
            elif self.subscription_status in ['trialing', 'active']:
                if self.subscription_expires and self.subscription_expires > datetime.utcnow():
                    return True
        return False

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
        'Time Horizon (years, min: 5, max: 30)',
        validators=[DataRequired(), NumberRange(min=5, max=30)]
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
STRIPE_PRICE_ID = os.getenv('STRIPE_LIVE_PRICE_ID_MONTHLY')
STRIPE_LIFETIME_PRICE_ID = os.getenv('STRIPE_LIVE_PRICE_ID_LIFETIME')
STRIPE_WEBHOOK_SECRET = os.getenv('STRIPE_LIVE_WEBHOOK_SECRET')

if not all([stripe.api_key, STRIPE_PUBLISHABLE_KEY, STRIPE_PRICE_ID, STRIPE_LIFETIME_PRICE_ID, STRIPE_WEBHOOK_SECRET]):
    logger.error("Missing Stripe LIVE configuration in environment variables")
    raise ValueError("Stripe live credentials not configured")

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


# Database setup
with app.app_context():
    db.create_all()


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

# Login manager
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# Helper functions
def get_industry_pe_beta(symbol):
    try:
        profile = finnhub_client.company_profile2(symbol=symbol)
        industry = profile.get('finnhubIndustry', 'Unknown')
        metrics = finnhub_client.company_basic_financials(symbol=symbol, metric='all').get('metric', {})

        # Get data from both Finnhub and Yahoo Finance
        yf_data = yf.Ticker(symbol).info

        trailing_pe = metrics.get('peExclExtraTTM') or metrics.get('peInclExtraTTM') or metrics.get(
            'peBasicExclExtraTTM')
        forward_pe = metrics.get('forwardPE') or metrics.get('forwardPEInclExtraTTM') or yf_data.get('forwardPE')
        beta = metrics.get('beta') or yf_data.get('beta')
        dividend_yield = metrics.get('dividendYield') or yf_data.get('dividendYield')
        debt_to_equity = yf_data.get('debtToEquity')
        earnings_growth = yf_data.get('earningsQuarterlyGrowth')
        revenue_growth = yf_data.get('revenueQuarterlyGrowth')
        ps_ratio = yf_data.get('priceToSalesTrailing12Months')
        pb_ratio = yf_data.get('priceToBook')
        roe = yf_data.get('returnOnEquity')

        return {
            'symbol': symbol,
            'industry': industry,
            'trailing_pe': round(trailing_pe, 2) if trailing_pe else None,
            'forward_pe': round(forward_pe, 2) if forward_pe else None,
            'beta': round(beta, 2) if beta else None,
            'dividend_yield': round(dividend_yield, 4) if dividend_yield else None,
            'debt_to_equity': round(debt_to_equity, 2) if debt_to_equity else None,
            'earnings_growth': round(earnings_growth, 2) if earnings_growth else None,
            'ps_ratio': round(ps_ratio, 2) if ps_ratio else None,
            'pb_ratio': round(pb_ratio, 2) if pb_ratio else None,
            'roe': round(roe, 2) if roe else None
        }
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return {
            'symbol': symbol,
            'industry': 'Unknown'
        }


def fetch_valid_tickers(tickers, premium):
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
        'roe': None
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
            time.sleep(4)

    # Create DataFrame with guaranteed columns
    df = pd.DataFrame(rows, columns=base_template.keys())

    # Fill remaining missing values
    df['industry'] = df['industry'].fillna('Unknown')
    return df


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
    df, years, top_n, min_ann_return=10, max_pe=40, max_ann_return=25, investing_style=None, risk_free_rate=4
):
    # Early exit for invalid input
    if df.empty or 'annual_return' not in df.columns:
        print("Empty dataframe or missing annual_return column")
        return pd.DataFrame()

    required_columns = [
        'symbol', 'annual_return', 'trailing_pe', 'forward_pe', 'beta', 'dividend_yield', 'debt_to_equity',
        'earnings_growth', 'ps_ratio', 'pb_ratio', 'roe', 'industry'
    ]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return pd.DataFrame()

    try:
        # Prepare features
        numeric_features = [
            'trailing_pe', 'forward_pe', 'beta', 'dividend_yield', 'debt_to_equity', 'earnings_growth',
            'ps_ratio', 'pb_ratio', 'roe'
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

        # Apply style adjustments
        shrinkage_factors = {
            'conservative': 0.75,
            'moderate': 0.8,
            'aggressive': 0.9
        }
        if investing_style in shrinkage_factors:
            df['predicted_ann_return'] *= shrinkage_factors[investing_style]

        # Apply return caps and filters
        df['predicted_ann_return'] = np.where(
            df['predicted_ann_return'] > 25,
            df['predicted_ann_return'] * 0.87,
            df['predicted_ann_return']
        )
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

        df['predicted_ann_return'] = df['predicted_ann_return'] - 0.17 * np.maximum(df['forward_pe'] - 15, 0)
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
        # Force premium to False if user doesn't have premium status
        if premium and not current_user.get_subscription_status():
            premium = False
            flash('Premium features disabled - subscription required', 'warning')

        original_stocks = stocks_amount  # Store original requested amount
        if not premium:
            stocks_amount += 3

        # Define backup tickers for each investment style
        backup_tickers = {
            'conservative': ['JNJ', 'PG', 'KO', 'PEP', 'MMM', 'SO', 'DUK', 'CVX', 'LOW', 'O', 'V', 'MA', 'SPGI',
                             'MCD', 'BRK-B', 'CAT'],
            'moderate': ['MSFT', 'GOOG', 'V', 'MA', 'ADP', 'ORCL', 'CRM', 'AAPL', 'PG', 'CAT', 'PGR', 'SPGI', 'DELL',
                         'AXP', 'ASML', 'AMAT', 'AMZN', 'QCOM', 'WMT'],
            'aggressive': ['NVDA', 'MSFT', 'GOOG', 'AAPL', 'META', 'AMZN', 'ASML', 'CRM', 'ORCL', 'CAT',
                           'PGR', 'DELL', 'PFE', 'AXP', 'AMAT', 'MA', 'REGN', 'QCOM', 'ADP']
        }[investing_style]

        # Main ticker list with deduplication
        raw_tickers = [
            'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'BRK-B', 'AVGO', 'LLY',
            'WMT', 'JPM', 'V', 'MA', 'XOM', 'COST', 'PG', 'JNJ', 'ORCL',
            'KO', 'ABBV', 'TMUS', 'BAC', 'CVX', 'CRM', 'ABT', 'MCD',
            'ADP', 'WFC', 'PEP', 'AXP', 'MS', 'ISRG', 'NOW', 'BX', 'GS', 'PGR',
            'UBER', 'QCOM', 'BKNG', 'ADBE', 'TJX', 'BSX', 'AMD', 'CAT', 'NEE',
            'BLK', 'TXN', 'SYK', 'GILD', 'HON', 'BA', 'MMC', 'COP', 'PANW',
            'LMT', 'AMAT', 'AMT', 'SO', 'BMY', 'ELV', 'ABNB', 'ICE',
            'DELL', 'O', 'ASML', 'REGN', 'CDNS', 'HCA', 'FTNT',
            'SNPS', 'TSM', 'HOOD'
        ]

        seen = set()
        tickers = [sym for sym in raw_tickers if not (sym in seen or seen.add(sym))]
        seen_backup = set()
        backup_tickers = [sym for sym in backup_tickers if not (sym in seen_backup or seen_backup.add(sym))]

        # Fetch and filter data
        df_meta = fetch_valid_tickers(tickers, premium=premium)

        # Define style-specific parameters
        style_filters = {
            'conservative': {'pe_max': 30, 'debt_max': 100, 'beta_max': 1.7, 'div_min': 3, 'beta_min': 0.2},
            'moderate': {'pe_max': 35, 'debt_max': 150, 'beta_max': 2.0, 'div_min': 1.5, 'beta_min': 0.4},
            'aggressive': {'pe_max': 50, 'debt_max': 500, 'beta_max': 3.5, 'div_min': 0.0, 'beta_min': 0.7}
        }
        limits = style_filters[investing_style]

        style_params = {
            'conservative': {'min_ann_return': 8, 'max_pe': 30, 'max_ann_return': 20},
            'moderate': {'min_ann_return': 10, 'max_pe': 35, 'max_ann_return': 20},
            'aggressive': {'min_ann_return': 10, 'max_pe': 40, 'max_ann_return': 25}
        }
        params = style_params[investing_style]

        # Initial filtering
        filtered_df = df_meta[
            (df_meta['trailing_pe'].fillna(999) <= limits['pe_max']) &
            (df_meta['beta'].fillna(0).between(limits['beta_min'], limits['beta_max'], inclusive='both')) &
            (df_meta['dividend_yield'].fillna(-1) >= limits['div_min']) &
            (df_meta['debt_to_equity'].fillna(999) <= limits['debt_max'])
            ].copy()

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

        # Build training set and get recommendations
        train_primary = build_training_set(filtered_df, time_horizon)
        recs = train_rank(
            train_primary, time_horizon, stocks_amount,
            min_ann_return=params['min_ann_return'],
            max_pe=params['max_pe'],
            max_ann_return=params['max_ann_return'],
            investing_style=investing_style
        )

        # Apply premium adjustments
        if premium:
            if not recs.empty and 'predicted_ann_return' in recs.columns:
                recs['predicted_ann_return'] *= {
                    'low': 0.8,
                    'medium': 1.0,
                    'high': 1.2
                }[risk_tolerance]

            if dividend_preference and not recs.empty:
                recs = recs.sort_values(['dividend_yield', 'predicted_ann_return'], ascending=[False, False])

        # Non-premium adjustments
        elif investing_style == 'aggressive':
            filtered_df = filtered_df[
                ~filtered_df['industry'].isin(['Energy', 'Machinery', 'Industrial Machinery'])].copy()

        # Handle backup tickers if needed
        backup_df = pd.DataFrame()
        if len(recs) < stocks_amount:
            needed = max((stocks_amount - len(recs)) + 1, 0)
            existing_symbols = recs['symbol'].tolist() if not recs.empty else []
            to_add = [s for s in backup_tickers if s not in existing_symbols][:needed]

            if to_add:
                df_bu = fetch_valid_tickers(to_add, premium=premium)
                train_bu = build_training_set(df_bu, time_horizon)
                if not train_bu.empty:
                    backup_candidates = train_rank(
                        train_bu, time_horizon, len(to_add),
                        min_ann_return=0,
                        max_pe=100,
                        max_ann_return=None,
                        investing_style=investing_style
                    )
                    if not backup_candidates.empty:
                        backup_order = {sym: idx for idx, sym in enumerate(backup_tickers)}
                        backup_candidates['order'] = backup_candidates['symbol'].map(backup_order)
                        backup_df = backup_candidates.sort_values('order').dropna(subset=['order']).head(needed).drop(
                            columns=['order'])

        # Combine recommendations
        final_recs = pd.concat([recs, backup_df], ignore_index=True).head(stocks_amount)

        # Remove top 3 for non-premium users
        if not premium:
            final_recs = final_recs.iloc[3:].head(original_stocks)

        # --- SORT BY predicted_ann_return DESCENDING ---
        if not final_recs.empty and 'predicted_ann_return' in final_recs.columns:
            final_recs = final_recs.sort_values('predicted_ann_return', ascending=False).reset_index(drop=True)

        # --- CALCULATE SUGGESTED ALLOCATION (proportional to predicted_ann_return) ---
        if not final_recs.empty and final_recs['predicted_ann_return'].sum() > 0:
            final_recs['suggested_allocation'] = final_recs['predicted_ann_return'] / final_recs['predicted_ann_return'].sum()
        else:
            final_recs['suggested_allocation'] = 1 / len(final_recs) if not final_recs.empty else 0

        # Format recommendations
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
                        'suggested_allocation': f"{row.get('suggested_allocation', 0) * 100:.2f}%",
                        'industry': row.get('industry', 'N/A')
                    })
            return formatted

        recommendations = format_stocks(final_recs)

        # --- WEIGHTED AVERAGE STATS ---
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
            'dividend_preference': dividend_preference
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
            'dividend_preference': dividend_preference
        }


# ──── Routes ─────────────────────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
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

    return render_template('index.html',
                           form=form,
                           is_premium_user=current_user.get_subscription_status())


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
            # Verify the price exists first
            try:
                price = stripe.Price.retrieve(STRIPE_PRICE_ID)
            except stripe.error.InvalidRequestError:
                flash('Subscription product not properly configured. Please contact support.', 'danger')
                return redirect(url_for('subscription'))

            checkout_session = stripe.checkout.Session.create(
                client_reference_id=current_user.id,
                customer_email=current_user.email,
                payment_method_types=['card'],
                line_items=[{
                    'price': STRIPE_PRICE_ID,
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
        # Verify the price exists first
        try:
            price = stripe.Price.retrieve(STRIPE_LIFETIME_PRICE_ID)
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
                'price': STRIPE_LIFETIME_PRICE_ID,
                'quantity': 1,
            }],
            mode='payment',
            success_url=url_for('success', _external=True) + '?session_id={CHECKOUT_SESSION_ID}',
            cancel_url=url_for('subscription', _external=True),
            metadata={
                'product_type': 'lifetime',
                'user_id': current_user.id
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


@app.route('/subscription')
@login_required
def subscription():
    subscription_info = None
    try:
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

    except Exception as e:
        flash(f'Error loading subscription: {str(e)}', 'danger')
        app.logger.error(f"Subscription error: {str(e)}", exc_info=True)

    return render_template('subscription.html',
                           subscription_active=current_user.get_subscription_status(),
                           expires=current_user.subscription_expires,
                           subscription_info=subscription_info
                           )


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
        # If the user has a Stripe subscription, cancel it first
        if current_user.stripe_customer_id:
            try:
                subscriptions = stripe.Subscription.list(customer=current_user.stripe_customer_id)
                for sub in subscriptions.data:
                    stripe.Subscription.delete(sub.id)
            except stripe.error.StripeError:
                pass  # Subscription might already be canceled

        # Delete the user from the database
        db.session.delete(current_user)
        db.session.commit()

        # Log the user out
        logout_user()

        flash('Your account has been permanently deleted.', 'success')
        return redirect(url_for('index'))
    except Exception as e:
        db.session.rollback()
        flash('Error deleting account: ' + str(e), 'danger')
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


# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(check_customer_ids, 'interval', hours=1)
scheduler.start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
