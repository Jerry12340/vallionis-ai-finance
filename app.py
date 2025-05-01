from flask import Flask, render_template, request, session, Response, abort
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

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv('APIkey.env')
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key=os.getenv('FINNHUB_API_KEY'))


# ──── Core Functions from Original Script ────────────────────────────────

def get_industry_pe_beta(symbol):
    profile = finnhub_client.company_profile2(symbol=symbol)
    industry = profile.get('finnhubIndustry', 'Unknown')
    metrics = finnhub_client.company_basic_financials(symbol=symbol, metric='all').get('metric', {})

    # PE Ratios
    trailing = (metrics.get('peExclExtraTTM') or metrics.get('peInclExtraTTM') or metrics.get('peBasicExclExtraTTM'))
    forward = (metrics.get('forwardPE') or metrics.get('forwardPEInclExtraTTM') or metrics.get('forwardPEExclExtraTTM'))
    if not forward:
        try:
            forward = yf.Ticker(symbol).info.get('forwardPE')
        except:
            forward = None

    beta = metrics.get('beta')

    # Dividend Yield
    raw_dy = metrics.get('dividendYield')
    dividend_yield = round(raw_dy, 2) if raw_dy is not None else None
    if dividend_yield is None:
        try:
            yf_dy = yf.Ticker(symbol).info.get('dividendYield')
            if yf_dy is not None: dividend_yield = round(yf_dy, 2)
        except:
            pass

    # Debt to Equity
    try:
        de = yf.Ticker(symbol).info.get('debtToEquity')
        debt_to_equity = round(de, 2) if de is not None else None
    except:
        debt_to_equity = None

    # Earnings & Revenue Growth
    try:
        eg = yf.Ticker(symbol).info.get('earningsQuarterlyGrowth')
        earnings_growth = round(eg, 2) if eg is not None else None
    except:
        earnings_growth = None
    try:
        rg = yf.Ticker(symbol).info.get('revenueQuarterlyGrowth')
        revenue_growth = round(rg, 2) if rg is not None else None
    except:
        revenue_growth = None

    # P/S, P/B, ROE
    info = yf.Ticker(symbol).info
    ps_ratio = round(info.get('priceToSalesTrailing12Months', 0), 2) if info.get(
        'priceToSalesTrailing12Months') else None
    pb_ratio = round(info.get('priceToBook', 0), 2) if info.get('priceToBook') else None
    roe = round(info.get('returnOnEquity', 0), 2) if info.get('returnOnEquity') else None

    return {
        'symbol': symbol,
        'industry': industry,
        'trailing_pe': round(trailing, 2) if trailing else None,
        'forward_pe': round(forward, 2) if forward else None,
        'beta': round(beta, 2) if beta else None,
        'dividend_yield': dividend_yield,
        'debt_to_equity': debt_to_equity,
        'earnings_growth': earnings_growth,
        'ps_ratio': ps_ratio,
        'pb_ratio': pb_ratio,
        'roe': roe,
    }


def fetch_valid_tickers(tickers):
    rows = []
    skipped_tickers = []  # List to track skipped tickers
    for sym in tickers:
        try:
            # Fetch data for the ticker
            data = get_industry_pe_beta(sym)

            # Check if 'industry' is valid
            if data['industry'] not in ['Unknown', None]:
                rows.append(data)
            else:
                skipped_tickers.append(sym)  # Track skipped tickers
        except Exception as e:
            print(f"Error fetching data for {sym}: {e}")
            skipped_tickers.append(sym)  # In case of error, track the ticker

        time.sleep(1.2)

    # Create DataFrame from the collected rows
    df = pd.DataFrame(rows)

    # Fill missing 'industry' values with 'Unknown'
    df['industry'] = df['industry'].fillna('Unknown')

    # Ensure proper data types for columns
    numeric_cols = ['trailing_pe', 'forward_pe', 'beta', 'dividend_yield',
                    'debt_to_equity', 'earnings_growth', 'ps_ratio', 'pb_ratio', 'roe']
    df = df.infer_objects()

    return df


def build_training_set(df, years):
    records = []
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=years)
    for _, row in df.iterrows():
        sym = row['symbol']
        try:
            hist = yf.Ticker(sym).history(start=start, end=end)
            if hist.empty:
                continue
            ret = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]
            ann = ((1 + ret) ** (1 / years) - 1) * 100
            rec = row.to_dict()
            rec['annual_return'] = ann
            records.append(rec)
        except Exception:
            continue

    # Only drop rows that failed to compute a return, keep everything else
    return pd.DataFrame(records).dropna(subset=['annual_return']).copy()


def train_rank(df, years, top_n, min_ann_return=10):
    # Prepare features & target
    X = df.drop(columns=['symbol', 'annual_return'])
    y = df['annual_return']

    # Identify numeric and categorical features
    num_feats = ['trailing_pe', 'forward_pe', 'beta', 'dividend_yield',
                 'debt_to_equity', 'earnings_growth', 'ps_ratio', 'pb_ratio', 'roe']
    cat_feats = ['industry']

    # Build preprocessing pipeline with imputation
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_feats),
        ('cat', cat_pipeline, cat_feats)
    ])

    model = Pipeline([
        ('pre', preprocessor),
        ('rf', RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    # Handle cases with insufficient data
    if len(X) < 2:
        return pd.DataFrame()  # Return empty dataframe if not enough data

    # Fit model
    model.fit(X, y)

    # Generate predictions
    df['predicted_ann_return'] = model.predict(X) * 0.8  # Conservative shrinkage
    df['predicted_total_return'] = ((1 + df['predicted_ann_return'] / 100) ** years - 1) * 100
    if min_ann_return is not None:
        df = df[df['predicted_ann_return'] >= min_ann_return]

    return df.nlargest(top_n, 'predicted_ann_return')


# ──── Web Routes ─────────────────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        investing_style = request.form['investing_style']
        time_horizon = int(request.form['time_horizon'])
        stocks_amount = int(request.form['stocks_amount'])
        premium = 'premium' in request.form

        results = process_request(investing_style, time_horizon, stocks_amount, premium)

        return render_template(
            'results.html',
            investing_style=investing_style,
            time_horizon=time_horizon,
            **results
        )
    return render_template('index.html')


# ──── Processing Logic ───────────────────────────────────────────────────

def process_request(investing_style, time_horizon, stocks_amount, premium):
    # Ticker lists and deduplication
    backup_tickers = [
        'AAPL', 'MSFT', 'GOOG', 'AMZN', 'BRK-B', 'CAT', 'PGR', 'AMAT',
        'IBM', 'META', 'JPM', 'QCOM', 'JNJ', 'PG', 'XOM', 'V',
        'MA', 'PEP', 'CSCO', 'ADBE', 'CRM', 'UNH', 'ABT', 'WMT', 'BLK', 'ICE', 'MCD', 'CVX'
    ]

    raw_tickers = [
        'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'BRK-B', 'AVGO', 'LLY',
        'WMT', 'JPM', 'V', 'MA', 'XOM', 'COST', 'UNH', 'PG', 'JNJ', 'ORCL',
        'KO', 'ABBV', 'TMUS', 'BAC', 'PM', 'CVX', 'CRM', 'ABT', 'CSCO', 'IBM', 'MCD',
        'ADP', 'WFC', 'MRK', 'PEP', 'AXP', 'MS', 'ISRG', 'NOW', 'BX', 'GS', 'PGR',
        'UBER', 'QCOM', 'BKNG', 'ADBE', 'TJX', 'BSX', 'AMD', 'CAT', 'NEE',
        'BLK', 'TXN', 'SYK', 'GILD', 'HON', 'BA', 'MMC', 'COP', 'PANW',
        'LMT', 'AMAT', 'AMT', 'SO', 'BMY', 'ELV', 'ABNB', 'PYPL', 'ICE',
        'INTC', 'DASH', 'DELL', 'O', 'AVGO', 'ASML', 'REGN', 'HOOD', 'GIS', 'DUK',
        'CAT', 'PGR', 'BAC', 'PFE', 'KO', 'MRK', 'JPM'
    ]

    # Deduplicate tickers
    seen = set()
    tickers = [sym for sym in raw_tickers if not (sym in seen or seen.add(sym))]

    # Fetch and filter data
    df_meta = fetch_valid_tickers(tickers)
    style_filters = {
        'conservative': {'pe_max': 40, 'debt_max': 200, 'beta_max': 1.7, 'div_min': 2.0},
        'moderate': {'pe_max': 45, 'debt_max': 200, 'beta_max': 2.0, 'div_min': 1.0},
        'aggressive': {'pe_max': 50, 'debt_max': 200, 'beta_max': 3.5, 'div_min': 0.0}
    }
    limits = style_filters[investing_style]

    filtered_df = df_meta[
        (df_meta['trailing_pe'].fillna(0) <= limits['pe_max']) &
        (df_meta['beta'].fillna(0) <= limits['beta_max']) &
        (df_meta['dividend_yield'].fillna(0) >= limits['div_min'])
        ].copy()

    # Generate primary recommendations
    train_primary = build_training_set(filtered_df, time_horizon)
    recs = train_rank(train_primary, time_horizon, stocks_amount)
    recs['roe'] = recs['roe'] * 100
    recs = recs.fillna(0)

    # Handle backup tickers - MAIN CHANGE IS HERE
    backup_df = pd.DataFrame()
    if len(recs) < stocks_amount:
        needed = stocks_amount - len(recs)
        recommended_syms = set(recs['symbol'].tolist())

        # Maintain original backup_tickers order when selecting
        to_add = [s for s in backup_tickers if s not in recommended_syms][:needed]

        if to_add:
            df_bu = fetch_valid_tickers(to_add)
            train_bu = build_training_set(df_bu, time_horizon)

            # Get ALL available backup candidates first
            backup_candidates = train_rank(train_bu, time_horizon, len(to_add), min_ann_return=0)

            # Then order them according to our backup_tickers priority
            backup_order = {symbol: i for i, symbol in enumerate(backup_tickers)}
            backup_df = backup_candidates.copy()
            backup_df['order'] = backup_df['symbol'].map(backup_order)
            backup_df = backup_df.sort_values('order').head(needed).drop(columns=['order'])
            backup_df['roe'] = backup_df['roe'] * 100

    def format_stocks(source_df):
        if isinstance(source_df, pd.DataFrame) and not source_df.empty:
            formatted = []
            for _, row in source_df.iterrows():
                def fmt(key, fmt_spec):
                    try:
                        val = row.get(key)
                        if pd.isna(val):
                            return "N/A"
                        return format(float(val), fmt_spec)
                    except Exception:
                        return "N/A"

                item = {
                    'symbol': row.get('symbol', 'N/A'),
                    'total_return': fmt('predicted_total_return', '.0f') + '%',
                    'annual_return': fmt('predicted_ann_return', '.2f') + '%',
                    'trailing_pe': fmt('trailing_pe', '.2f'),
                    'forward_pe': fmt('forward_pe', '.2f'),
                    'beta': fmt('beta', '.2f'),
                    'dividend_yield': fmt('dividend_yield', '.2f') + '%',
                    'debt_to_equity': fmt('debt_to_equity', '.2f') + '%',
                    'ps_ratio': fmt('ps_ratio', '.2f'),
                    'pb_ratio': fmt('pb_ratio', '.2f'),
                    'roe': fmt('roe', '.2f') + '%'
                }
                formatted.append(item)
            return formatted
        return []

    # Combine and format recommendations
    final_recs = pd.concat([recs, backup_df], ignore_index=True).head(stocks_amount)

    # Sort by predicted annual return (descending) before formatting
    final_recs = final_recs.sort_values('predicted_ann_return', ascending=False)

    recommendations = format_stocks(final_recs)
    recs_count = len(recs)  # Store count of primary recommendations

    # Calculate averages from final selection
    avg_total = final_recs['predicted_total_return'].mean() if not final_recs.empty else 0
    avg_annual = final_recs['predicted_ann_return'].mean() if not final_recs.empty else 0
    avg_div = final_recs['dividend_yield'].mean() if not final_recs.empty else 0
    final_recs.to_csv('recommended_portfolio.csv', index=False)

    return {
        'recommendations': recommendations,
        'recs_count': recs_count,
        'averages': {
            'total': f"{avg_total:.0f}%",
            'annual': f"{avg_annual:.2f}%",
            'dividend': f"{avg_div:.2f}%"
        },
        'premium': premium
    }


@app.route('/download')
def download():
    # Check premium status from session
    if not session.get('premium', False):
        abort(403)  # Forbidden if not premium

    # Get CSV data from session
    csv_data = session.get('csv_data')
    if not csv_data:
        abort(404)  # Not found if no data

    # Create CSV response
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=recommended_portfolio.csv"}
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
