import finnhub
import time
import pandas as pd
import yfinance as yf
import numpy as np
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
from dotenv import load_dotenv
import os
from tabulate import tabulate

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv('.env')
api_key = os.getenv('FINNHUB_API_KEY')
finnhub_client = finnhub.Client(api_key)

# ——— Global user profile ———
investing_style = ''
time_horizon = 0
stocks_amount = 0
premium = True
df = None


# ——— User Questions ———
def questions():
    global investing_style, time_horizon, risk_tolerance, stocks_amount
    while True:
        investing_style = input('Style (conservative, moderate, aggressive)? ').lower()
        if investing_style in ['conservative', 'moderate', 'aggressive']:
            break
        print('Invalid.')
    while True:
        try:
            time_horizon = int(input('Time horizon (years, 30 max)? '))
            break
        except ValueError:
            print('Enter integer.')
        if 1 <= time_horizon <= 50:
            break
        print('Enter a valid amount between 1 and 30.')
    if premium:
        while True:
            try:
                stocks_amount = int(input('Amount of suggested stocks (max 30)? '))
            except ValueError:
                print('Enter integer.')
                continue
            if 1 <= stocks_amount <= 30:
                break
            print('Enter a valid amount between 1 and 30.')
    if not premium:
        while True:
            try:
                stocks_amount = int(input('Amount of suggested stocks (max 20)? '))
            except ValueError:
                print('Enter integer.')
                continue
            if 1 <= stocks_amount <= 20:
                break
            print('Enter a valid amount between 1 and 20.')


# ——— Finnhub Data Pull ———
def get_industry_pe_beta(symbol):
    try:
        profile = finnhub_client.company_profile2(symbol=symbol)
        industry = profile.get('finnhubIndustry', 'Unknown')
        metrics = finnhub_client.company_basic_financials(symbol=symbol, metric='all').get('metric', {})

        # Get data from both Finnhub and Yahoo Finance
        yf_ticker = yf.Ticker(symbol)
        yf_data = yf_ticker.info

        # Get fundamental metrics
        trailing_pe = metrics.get('peExclExtraTTM') or metrics.get('peInclExtraTTM') or metrics.get('peBasicExclExtraTTM')
        forward_pe = metrics.get('forwardPE') or metrics.get('forwardPEInclExtraTTM') or yf_data.get('forwardPE')
        beta = metrics.get('beta') or yf_data.get('beta')
        dividend_yield = metrics.get('dividendYield') or yf_data.get('dividendYield')
        debt_to_equity = yf_data.get('debtToEquity')
        earnings_growth = yf_data.get('earningsQuarterlyGrowth')
        revenue_growth = yf_data.get('revenueQuarterlyGrowth')
        ps_ratio = yf_data.get('priceToSalesTrailing12Months')
        pb_ratio = yf_data.get('priceToBook')
        roe = yf_data.get('returnOnEquity')

        # Get growth estimates
        next_5y_eps_growth = yf_data.get('next5YearsPerShareEarningsGrowthAnnual')
        next_year_eps_growth = yf_data.get('earningsGrowth')
        forward_eps = yf_data.get('forwardEps')
        peg_ratio = yf_data.get('pegRatio')

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
            'roe': round(roe, 2) if roe else None,
            'next_5y_eps_growth': round(next_5y_eps_growth, 4) if next_5y_eps_growth else None,
            'next_year_eps_growth': round(next_year_eps_growth, 4) if next_year_eps_growth else None,
            'forward_eps': round(forward_eps, 2) if forward_eps else None,
            'peg_ratio': round(peg_ratio, 2) if peg_ratio else None
        }
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return {
            'symbol': symbol,
            'industry': 'Unknown',
            'next_5y_eps_growth': None,
            'next_year_eps_growth': None,
            'forward_eps': None,
            'peg_ratio': None
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


# ——— Historical Returns & Annualization ———
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


# ——— Train, CV, Recommend ———
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

        # Apply growth-based adjustments
        growth_weight = {
            'conservative': 0.3,
            'moderate': 0.5,
            'aggressive': 0.7
        }.get(investing_style, 0.5)

        # Apply growth multiplier
        df['growth_multiplier'] = 1 + (df['next_5y_eps_growth'].fillna(0) * growth_weight
                                       df['predicted_ann_return'] = df['predicted_ann_return'] * df['growth_multiplier']

                                       # Add growth premium for reasonably priced growth stocks
                                       df['growth_premium'] = np.where(
                                       (df['peg_ratio'] < 2.0) & (df['next_5y_eps_growth'] > 0.1),
                                  df['next_5y_eps_growth'] * 0.25,  # 25% of growth rate as premium
        0
        )
        df['predicted_ann_return'] = df['predicted_ann_return'] + df['growth_premium']

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


def main():
    # 1. Gather user inputs
    questions()

    # 2. Your primary and backup lists
    backup_tickers = [
        'AAPL', 'MSFT', 'GOOG', 'AMZN', 'BRK-B', 'CAT', 'PGR', 'AMAT',
        'IBM', 'META', 'JPM', 'QCOM', 'JNJ', 'PG', 'XOM', 'V',
        'MA', 'PEP', 'CSCO', 'ADBE', 'CRM', 'ABT'
        , 'WMT', 'BLK', 'ICE', 'MCD', 'CVX', 'HD'
    ]
    raw_tickers = [
        'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'BRK-B', 'AVGO', 'LLY',
        'WMT', 'JPM', 'V', 'MA', 'XOM', 'COST', 'PG', 'JNJ', 'ORCL', 'HD',
        'KO', 'ABBV', 'TMUS', 'BAC', 'PM', 'CVX', 'CRM', 'ABT', 'CSCO', 'IBM', 'MCD',
        'ADP', 'WFC', 'MRK', 'PEP', 'AXP', 'MS', 'ISRG',
        'NOW', 'BX', 'GS', 'PGR', 'UBER', 'QCOM', 'BKNG', 'ADBE', 'AMGN',
        'TJX', 'BSX', 'AMD', 'CAT', 'NEE', 'BLK', 'TXN', 'SYK',
        'GILD', 'HON', 'VRTX', 'BA', 'MMC', 'COP',
        'PANW', 'LMT', 'AMAT', 'AMT', 'SO', 'BMY', 'ELV', 'ABNB', 'PYPL',
        'MNST', 'ICE', 'INTC', 'DASH', 'DELL', 'O', 'AVGO', 'ASML', 'REGN', 'HOOD',
        'GIS', 'DUK', 'CAT', 'PGR', 'BAC', 'PFE', 'KO', 'MRK', 'JPM'
    ]

    # 3. Deduplicate raw_tickers
    seen = set()
    tickers = []
    for sym in raw_tickers:
        if sym in seen:
            continue
        seen.add(sym)
        tickers.append(sym)

    # 4. Fetch metadata
    df_meta = fetch_valid_tickers(tickers)

    # 5. Apply style filters
    style_filters = {
        'conservative': {
            'pe_max': 40,  # Up from 25
            'debt_max': 200,  # Up from 50
            'beta_max': 1.7,  # Up from 1.2
            'div_min': 2.0,  # Down from 2.5
            'debt_to_equity_max': 200,  # Up from 50
            'ps_ratio_max': 20,  # Up from 5
            'pb_ratio_max': 14,  # Up from 5
            'roe_min': 0.08,  # Down from 0.10
            'div_max': 12  # Up from 10
        },
        'moderate': {
            'pe_max': 45,  # Up from 30
            'debt_max': 200,  # Up from 100
            'beta_max': 2.0,  # Up from 1.5
            'div_min': 1.0,  # Down from 1.5
            'debt_to_equity_max': 200,  # Up from 100
            'ps_ratio_max': 20,  # Up from 8
            'pb_ratio_max': 15,  # Up from 6
            'roe_min': 0.12,  # Down from 0.15
            'div_max': 9  # Up from 7
        },
        'aggressive': {
            'pe_max': 50,  # Up from 40
            'debt_max': 200,  # Up from 200
            'beta_max': 3.5,  # Up from 2.5
            'div_min': 0.0,  # Same
            'debt_to_equity_max': 200,  # Up from 200
            'ps_ratio_max': 20,  # Up from 10
            'pb_ratio_max': 25,  # Up from 15
            'roe_min': 0.25,  # Down from 0.25
            'div_max': 5  # Up from 3
        }
    }

    # Start with full meta set, then filter by style
    df = df_meta.copy()

    # Apply style filters
    limits = style_filters[investing_style]

    df = df[
        (df['trailing_pe'] <= limits['pe_max']) &
        (df['debt_to_equity'] <= limits['debt_to_equity_max']) &
        (df['beta'] <= limits['beta_max']) &
        (df['dividend_yield'] >= limits['div_min']) &
        (df['ps_ratio'] <= limits['ps_ratio_max']) &
        (df['pb_ratio'] <= limits['pb_ratio_max']) &
        (df['roe'] >= limits['roe_min']) &
        (df['dividend_yield'] <= limits['div_max'])
        ].copy()

    train_primary = build_training_set(df, time_horizon)

    back_up_needed = False
    if len(train_primary) < stocks_amount:
        back_up_needed = True
        needed = stocks_amount - len(train_primary)
        seen_backup = set()
        unique_backup = []
        for s in backup_tickers:
            if s not in seen_backup:
                seen_backup.add(s)
                unique_backup.append(s)
        to_add = [s for s in unique_backup if s not in df['symbol'].values][:needed]
        if to_add:
            df_bu = fetch_valid_tickers(to_add)

            train_bu = build_training_set(df_bu, time_horizon)
            train_bu = train_rank(train_bu, time_horizon, stocks_amount)
            train_bu_trailing_PE_avg = train_bu['trailing_pe'].mean()
            train_bu_forward_PE_avg = train_bu['forward_pe'].mean()
            train_bu_beta_avg = train_bu['beta'].mean()
            train_bu_de_avg = (train_bu['debt_to_equity'].mean())
            train_bu_ps_avg = train_bu['ps_ratio'].mean()
            train_bu_pb_avg = train_bu['pb_ratio'].mean()
            train_bu_roe_avg = train_bu['roe'].mean()
            train_bu['trailing_pe'].fillna(train_bu_trailing_PE_avg)
            train_bu['forward_pe'].fillna(train_bu_forward_PE_avg)
            train_bu['beta'].fillna(train_bu_beta_avg)
            train_bu['debt_to_equity'].fillna(train_bu_de_avg)
            train_bu['ps_ratio'].fillna(train_bu_ps_avg)
            train_bu['pb_ratio'].fillna(train_bu_pb_avg)
            train_bu['roe'].fillna(train_bu_roe_avg)
            train_bu['predicted_total_return'] = train_bu['predicted_total_return'].astype(int)
            train_bu['roe'] = train_bu['roe'] * 100
            train_bu['dividend_yield'] = train_bu['dividend_yield'].fillna(0)
            train_bu['predicted_total_return_fmt'] = train_bu['predicted_total_return'].map(lambda x: f"{x:,}%")
            train_bu['predicted_ann_return_fmt'] = train_bu['predicted_ann_return'].map(lambda x: f"{x:.2f}%")
            train_bu['dividend_yield_fmt'] = train_bu['dividend_yield'].map(lambda x: f"{x:.2f}%")
            train_bu['trailing_pe_fmt'] = train_bu['trailing_pe'].map(lambda x: f"{x:.2f}")
            train_bu['forward_pe_fmt'] = train_bu['forward_pe'].map(lambda x: f"{x:.2f}")
            train_bu['beta_fmt'] = train_bu['beta'].map(lambda x: f"{x:.2f}")
            train_bu['debt_to_equity_fmt'] = train_bu['debt_to_equity'].map(lambda x: f"{x:.2f}")
            train_bu['ps_ratio_fmt'] = train_bu['ps_ratio'].map(lambda x: f"{x:.2f}")
            train_bu['pb_ratio_fmt'] = train_bu['pb_ratio'].map(lambda x: f"{x:.2f}")
            train_bu['roe_fmt'] = train_bu['roe'].map(lambda x: f"{x:.2f}%")
            train_bu = train_bu.fillna(0)
            avg_total_return_train_bu = train_bu['predicted_total_return'].mean()
            avg_anu_return_train_bu = train_bu['predicted_ann_return'].mean()
            avg_div_return_train_bu = train_bu['dividend_yield'].mean()

    recs = train_rank(train_primary, time_horizon, stocks_amount)
    if investing_style == 'aggressive':
        recs = recs[recs['predicted_ann_return'] >= 10.0].copy()
    recs_trailing_PE_avg = recs['trailing_pe'].mean()
    recs_forward_PE_avg = recs['forward_pe'].mean()
    recs_beta_avg = recs['beta'].mean()
    recs_de_avg = recs['debt_to_equity'].mean()
    recs_ps_avg = recs['ps_ratio'].mean()
    recs_pb_avg = recs['pb_ratio'].mean()
    recs_roe_avg = recs['roe'].mean()
    recs['trailing_pe'].fillna(recs_trailing_PE_avg)
    recs['forward_pe'].fillna(recs_forward_PE_avg)
    recs['beta'].fillna(recs_beta_avg)
    recs['debt_to_equity'].fillna(recs_de_avg)
    recs['ps_ratio'].fillna(recs_ps_avg)
    recs['pb_ratio'].fillna(recs_pb_avg)
    recs['roe'].fillna(recs_roe_avg)
    recs['predicted_total_return'] = recs['predicted_total_return'].astype(int)
    avg_total_return = recs['predicted_total_return'].mean()
    avg_anu_return = recs['predicted_ann_return'].mean()
    avg_div_return = recs['dividend_yield'].mean()
    recs['roe'] = recs['roe'] * 100

    # ── FORMAT COLUMNS FOR DISPLAY ──
    recs['predicted_total_return_fmt'] = recs['predicted_total_return'].map(lambda x: f"{x:,}%")
    recs['predicted_ann_return_fmt'] = recs['predicted_ann_return'].map(lambda x: f"{x:.2f}%")
    recs['trailing_pe_fmt'] = recs['trailing_pe'].map(lambda x: f"{x:.2f}")
    recs['forward_pe_fmt'] = recs['forward_pe'].map(lambda x: f"{x:.2f}")
    recs['beta_fmt'] = recs['beta'].map(lambda x: f"{x:.2f}")
    recs['dividend_yield_fmt'] = recs['dividend_yield'].map(lambda x: f"{x:.2f}%")
    recs['debt_to_equity_fmt'] = recs['debt_to_equity'].map(lambda x: f"{x:.2f}")
    recs['ps_ratio_fmt'] = recs['ps_ratio'].map(lambda x: f"{x:.2f}")
    recs['pb_ratio_fmt'] = recs['pb_ratio'].map(lambda x: f"{x:.2f}")
    recs['roe_fmt'] = recs['roe'].map(lambda x: f"{x:.2f}%")

    if not premium:
        print(f"\nTop Recommendations ({time_horizon}y):")
        print(
            recs[['symbol', 'predicted_total_return_fmt', 'predicted_ann_return_fmt', 'dividend_yield_fmt']]
            .rename(columns={
                'predicted_total_return_fmt': ' Total Return ',
                'predicted_ann_return_fmt': ' Annual Return ',
                'dividend_yield_fmt': ' Dividends yield '
            })
            .to_string(index=False, col_space=10)
        )
        if back_up_needed:
            print('\nBackup Tickers:')
            print(
                train_bu[['symbol', 'predicted_total_return_fmt', 'predicted_ann_return_fmt', 'dividend_yield_fmt']]
                .rename(columns={
                    'predicted_total_return_fmt': ' Total Return ',
                    'predicted_ann_return_fmt': ' Annual Return ',
                    'dividend_yield_fmt': ' Dividends yield '
                })
                .to_string(header=False, index=False, col_space=11)
            )
    else:
        print(f"\nTop Recommendations (Premium) ({time_horizon}y):")
        cols = [
            'symbol',
            'predicted_total_return_fmt',
            'predicted_ann_return_fmt',
            'trailing_pe_fmt',
            'forward_pe_fmt',
            'beta_fmt',
            'dividend_yield_fmt',
            'debt_to_equity_fmt',
            'ps_ratio_fmt',
            'pb_ratio_fmt',
            'roe_fmt'
        ]

        display_names = {
            'symbol': ' Symbol ',
            'predicted_total_return_fmt': ' Total Return ',
            'predicted_ann_return_fmt': ' Annual Return ',
            'trailing_pe_fmt': ' Trailing P/E ',
            'forward_pe_fmt': ' Forward P/E ',
            'beta_fmt': ' Beta ',
            'dividend_yield_fmt': ' Dividend Yield ',
            'debt_to_equity_fmt': ' Debt/Equity ',
            'ps_ratio_fmt': ' P/S ',
            'pb_ratio_fmt': ' P/B ',
            'roe_fmt': ' ROE '
        }

        print(
            recs[cols]
            .rename(columns=display_names)
            .to_string(index=False, col_space=11)
        )
        train_bu = train_bu.fillna(0)
        if back_up_needed:
            print('\nBackup Tickers:')
            print(
                train_bu[cols]
                .rename(columns=display_names)
                .to_string(header=False, index=False, col_space=12)
            )

        avg_total_return = (avg_total_return_train_bu + avg_total_return) / 2 if back_up_needed else avg_total_return
        avg_anu_return = (avg_anu_return_train_bu + avg_anu_return) / 2 if back_up_needed else avg_anu_return
        avg_div_return = (avg_div_return_train_bu + avg_div_return) / 2 if back_up_needed else avg_div_return

        print(f'\nAverage total return: {avg_total_return:.0f}%')
        print(f'Average annual return: {avg_anu_return:.2f}%')
        print(f'Average dividends: {avg_div_return:.2f}%')


if __name__ == "__main__":
    main()
