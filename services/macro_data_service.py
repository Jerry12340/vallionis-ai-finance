import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import json
from flask import jsonify, render_template_string, make_response
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io
import csv
import time
import hashlib
import redis
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(self, redis_url=None):
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.redis_client = None
        self.connect()

    def connect(self):
        """Establish connection to Redis"""
        try:
            self.redis_client = redis.Redis.from_url(
                self.redis_url,
                decode_responses=False,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            self.redis_client = None

    def get(self, key):
        """Get value from Redis cache"""
        if not self.redis_client:
            return None

        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                return pickle.loads(cached_data)
        except Exception as e:
            logger.error(f"Error getting from Redis cache: {str(e)}")
        return None

    def set(self, key, value, timeout=None):
        """Set value in Redis cache with optional timeout"""
        if not self.redis_client:
            return False

        try:
            serialized = pickle.dumps(value)
            if timeout:
                self.redis_client.setex(key, timeout, serialized)
            else:
                self.redis_client.set(key, serialized)
            return True
        except Exception as e:
            logger.error(f"Error setting Redis cache: {str(e)}")
            return False

    def delete(self, key):
        """Delete a key from Redis cache"""
        if not self.redis_client:
            return False

        try:
            return self.redis_client.delete(key) > 0
        except Exception as e:
            logger.error(f"Error deleting from Redis cache: {str(e)}")
            return False

    def clear(self):
        """Clear all cached data (use with caution!)"""
        if not self.redis_client:
            return False

        try:
            self.redis_client.flushdb()
            return True
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {str(e)}")
            return False

    def get_stats(self):
        """Get Redis cache statistics"""
        if not self.redis_client:
            return {"connected": False}

        try:
            info = self.redis_client.info()
            return {
                "connected": True,
                "keys": info.get('db0', {}).get('keys', 0),
                "memory_used": info.get('used_memory', 0),
                "hits": info.get('keyspace_hits', 0),
                "misses": info.get('keyspace_misses', 0),
                "uptime": info.get('uptime_in_seconds', 0)
            }
        except Exception as e:
            logger.error(f"Error getting Redis stats: {str(e)}")
            return {"connected": False}


class MacroDataService:
    def __init__(self):
        self.fred_api_key = os.getenv('FRED_API_KEY') or 'demo'
        if self.fred_api_key == 'demo':
            logger.warning("FRED_API_KEY not found; using demo data fallback.")
        else:
            logger.info("FRED_API_KEY detected; live FRED API will be used.")
        self.using_demo_data = (self.fred_api_key == 'demo')
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # FRED Series IDs for common economic indicators
        self.series_ids = {
            'inflation': 'CPIAUCSL',
            'gdp': 'GDPC1',
            'nominal_gdp': 'GDPA',
            'unemployment': 'UNRATE',
            'fed_funds': 'FEDFUNDS',
            'treasury_10y': 'DGS10',
            'treasury_2y': 'DGS2',
            'jolts': 'JTSJOL',
            'jobless_claims': 'ICSA',
        }

        # Redis cache configuration
        self.cache_enabled = True
        self.cache_timeout = int(os.getenv('CACHE_TIMEOUT_SECONDS', '86400'))  # 1 day default
        self.redis_cache = RedisCache()

        # In-memory fallback cache if Redis is unavailable
        self._fallback_cache = {}
        self._fallback_timestamps = {}

        # Cache statistics
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'expired': 0,
            'total_requests': 0,
            'redis_connected': self.redis_cache.redis_client is not None
        }

        logger.info(
            f"Cache initialized: enabled={self.cache_enabled}, timeout={self.cache_timeout}s, Redis connected={self.cache_stats['redis_connected']}")

    def _get_cache_key(self, func_name, *args, **kwargs):
        """Generate a unique cache key for function call with arguments"""
        # Convert args and kwargs to a consistent string representation
        args_str = str(args)
        kwargs_str = str(sorted(kwargs.items()))
        key_str = f"{func_name}_{args_str}_{kwargs_str}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _check_cache(self, key):
        """Check if cache entry exists and is still valid"""
        if not self.cache_enabled:
            return None

        self.cache_stats['total_requests'] += 1

        # Try Redis first
        if self.redis_cache.redis_client:
            cached_data = self.redis_cache.get(key)
            if cached_data is not None:
                self.cache_stats['hits'] += 1
                logger.info(f"Redis cache HIT for key: {key[:20]}...")
                return cached_data

        # Fallback to in-memory cache if Redis is unavailable
        if key in self._fallback_cache and key in self._fallback_timestamps:
            if time.time() - self._fallback_timestamps[key] < self.cache_timeout:
                self.cache_stats['hits'] += 1
                logger.info(f"Fallback cache HIT for key: {key[:20]}...")
                return self._fallback_cache[key]
            else:
                # Cache expired, remove entry
                self.cache_stats['expired'] += 1
                logger.info(f"Fallback cache EXPIRED for key: {key[:20]}...")
                del self._fallback_cache[key]
                del self._fallback_timestamps[key]

        self.cache_stats['misses'] += 1
        logger.info(f"Cache MISS for key: {key[:20]}...")
        return None

    def _set_cache(self, key, value, timeout=None):
        """Store value in cache with timestamp"""
        if not self.cache_enabled:
            return

        # Use specific timeout or default
        cache_timeout = timeout if timeout is not None else self.cache_timeout

        # Try Redis first
        if self.redis_cache.redis_client:
            if self.redis_cache.set(key, value, cache_timeout):
                logger.info(f"Redis cache SET for key: {key[:20]}...")
                return

        # Fallback to in-memory cache if Redis is unavailable
        self._fallback_cache[key] = value
        self._fallback_timestamps[key] = time.time()
        logger.info(f"Fallback cache SET for key: {key[:20]}...")

    def _delete_from_cache(self, key):
        """Delete a key from cache"""
        # Try Redis first
        if self.redis_cache.redis_client:
            self.redis_cache.delete(key)

        # Fallback to in-memory cache
        if key in self._fallback_cache:
            del self._fallback_cache[key]
        if key in self._fallback_timestamps:
            del self._fallback_timestamps[key]

    def clear_cache(self):
        """Clear all cached data"""
        # Clear Redis cache
        if self.redis_cache.redis_client:
            self.redis_cache.clear()

        # Clear fallback cache
        self._fallback_cache = {}
        self._fallback_timestamps = {}

        logger.info("Cache cleared")

    def clear_indicator_cache(self, indicator_key):
        """Clear cache for a specific indicator"""
        # Clear series data
        series_key = self._get_cache_key("get_indicator_series", indicator_key, 3650)
        self._delete_from_cache(series_key)

        # Clear chart data
        chart_key = self._get_cache_key("get_indicator_chart", indicator_key, 3650)
        self._delete_from_cache(chart_key)

        # Clear latest value
        latest_key = self._get_cache_key("get_latest_indicator_value", indicator_key, 3650)
        self._delete_from_cache(latest_key)

        # Clear FRED data if it's a base indicator
        base_key = indicator_key
        if indicator_key in ['inflation_yoy', 'inflation_mom']:
            base_key = 'inflation'

        if base_key in self.series_ids:
            fred_key = self._get_cache_key("get_fred_data", self.series_ids[base_key], 3650)
            self._delete_from_cache(fred_key)

        logger.info(f"Cleared cache for indicator: {indicator_key}")

    def get_cache_stats(self):
        """Return cache statistics"""
        total_requests = self.cache_stats['total_requests']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0

        # Get Redis stats if available
        redis_stats = self.redis_cache.get_stats() if self.redis_cache.redis_client else {}

        return {
            **self.cache_stats,
            'hit_rate': round(hit_rate, 2),
            'current_size': len(self._fallback_cache),
            'redis_stats': redis_stats,
            'enabled': self.cache_enabled,
            'timeout': self.cache_timeout
        }

    def inspect_cache(self):
        """Return information about current cache contents"""
        cache_info = []
        now = time.time()

        # Get Redis cache info if available
        if self.redis_cache.redis_client:
            try:
                # Use SCAN instead of KEYS for production safety
                cursor = '0'
                keys = []
                while cursor != 0:
                    cursor, partial_keys = self.redis_cache.redis_client.scan(cursor=cursor, count=100)
                    keys.extend(partial_keys)

                for key in keys[:100]:  # Limit to first 100 keys
                    ttl = self.redis_cache.redis_client.ttl(key)
                    cache_info.append({
                        'key': key.decode()[:30] + '...' if len(key) > 30 else key.decode(),
                        'age_seconds': 'N/A',
                        'expires_in_seconds': ttl,
                        'data_type': 'Redis',
                        'data_length': 'N/A'
                    })
            except Exception as e:
                logger.error(f"Error inspecting Redis cache: {str(e)}")

        # Add fallback cache info
        for key, value in self._fallback_cache.items():
            timestamp = self._fallback_timestamps.get(key, 0)
            age = now - timestamp
            expires_in = self.cache_timeout - age
            cache_info.append({
                'key': key[:30] + '...' if len(key) > 30 else key,
                'age_seconds': round(age, 2),
                'expires_in_seconds': round(expires_in, 2),
                'data_type': type(value).__name__,
                'data_length': len(value) if hasattr(value, '__len__') else 1
            })

        return sorted(cache_info, key=lambda x: x.get('age_seconds', 0), reverse=True)

    def set_cache_timeout(self, seconds):
        """Dynamically change the cache timeout"""
        if seconds < 0:
            logger.warning(f"Invalid cache timeout {seconds}, must be positive")
            return False

        old_timeout = self.cache_timeout
        self.cache_timeout = seconds
        logger.info(f"Cache timeout changed from {old_timeout} to {seconds} seconds")

        # Clear cache when timeout changes significantly
        if abs(old_timeout - seconds) > 300:
            self.clear_cache()

        return True

    def get_fred_data(self, series_id, days=365 * 5):
        """Fetch economic data from FRED API with improved error handling and caching"""
        # Check cache first
        cache_key = self._get_cache_key("get_fred_data", series_id, days)
        cached_data = self._check_cache(cache_key)
        if cached_data is not None:
            logger.info(f"Cache hit for FRED data: {series_id}")
            return cached_data

        logger.info(f"Cache miss for FRED data: {series_id}, fetching from API")

        # If using demo key, return demo data immediately
        if self.fred_api_key == 'demo':
            demo_data = self._get_demo_data(series_id)
            self._set_cache(cache_key, demo_data)
            return demo_data

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            params = {
                'series_id': series_id,
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'observation_start': start_date.strftime('%Y-%m-%d'),
                'observation_end': end_date.strftime('%Y-%m-%d'),
                'sort_order': 'desc',
                'limit': 10000
            }

            # Add retry mechanism for 500 errors
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(
                        'https://api.stlouisfed.org/fred/series/observations',
                        params=params,
                        timeout=15,
                        headers=self.headers
                    )

                    # Check for rate limiting or server errors
                    if response.status_code == 429:
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limited. Waiting {wait_time} seconds before retry {attempt + 1}")
                        time.sleep(wait_time)
                        continue
                    elif response.status_code >= 500:
                        logger.warning(f"Server error {response.status_code}. Retrying in {attempt + 1} seconds")
                        time.sleep(attempt + 1)
                        continue

                    response.raise_for_status()

                    data = response.json()

                    if 'observations' in data and data['observations']:
                        df = pd.DataFrame(data['observations'])
                        df['date'] = pd.to_datetime(df['date'])
                        df['value'] = pd.to_numeric(df['value'], errors='coerce')
                        df = df.dropna(subset=['value'])

                        result = [
                            {'date': row['date'].strftime('%Y-%m-%d'),
                             'value': float(row['value'])}
                            for _, row in df.iterrows()
                        ]

                        result.sort(key=lambda x: x['date'])
                        # Cache the result
                        self._set_cache(cache_key, result)

                        return result

                    # Cache empty result too
                    self._set_cache(cache_key, [])
                    return []

                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        raise e
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(1)

            # Cache empty result
            self._set_cache(cache_key, [])
            return []

        except Exception as e:
            logger.error(f"Error fetching FRED data for {series_id}: {str(e)}")
            demo_data = self._get_demo_data(series_id)
            # Cache demo data as fallback
            self._set_cache(cache_key, demo_data)
            return demo_data

    def get_indicator_series(self, indicator_key, days=365 * 10):
        """Return transformed time series for an indicator with smart caching"""
        # Define cache timeouts based on data frequency
        cache_timeouts = {
            'gdp': 86400 * 90,  # 90 days (quarterly data)
            'nominal_gdp': 86400 * 90,  # 90 days
            'inflation_yoy': 86400 * 30,  # 30 days (monthly)
            'inflation_mom': 86400 * 30,  # 30 days
            'unemployment': 86400 * 7,  # 7 days (weekly)
            'fed_funds': 86400 * 1,  # 1 day (can change frequently)
            'treasury_10y': 86400 * 1,  # 1 day
            'treasury_2y': 86400 * 1,  # 1 day
            'yield_curve_2_10': 86400 * 1,  # 1 day
            'jolts': 86400 * 30,  # 30 days (monthly)
            'jobless_claims': 86400 * 7,  # 7 days (weekly)
        }

        # Use specific timeout or default
        timeout = cache_timeouts.get(indicator_key, self.cache_timeout)

        # Check cache first
        cache_key = self._get_cache_key("get_indicator_series", indicator_key, days)
        cached_data = self._check_cache(cache_key)
        if cached_data is not None:
            logger.info(f"Cache hit for indicator series: {indicator_key}")
            return cached_data

        logger.info(f"Cache miss for indicator series: {indicator_key}, calculating")

        inflation_variant = None
        base_key = indicator_key
        if indicator_key in ['inflation_yoy', 'inflation_mom']:
            base_key = 'inflation'
            inflation_variant = indicator_key.split('_')[1]
        elif indicator_key == 'yield_curve_2_10':
            result = self._get_yield_curve_2_10(days)
            self._set_cache(cache_key, result, timeout)
            return result

        series_id = self.series_ids.get(base_key)
        if not series_id:
            self._set_cache(cache_key, [], timeout)
            return []

        data = self.get_fred_data(series_id, days)
        if not data:
            self._set_cache(cache_key, [], timeout)
            return []

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        if base_key == 'inflation':
            if inflation_variant == 'mom':
                df['value'] = df['value'].pct_change(1) * 100.0
            else:
                df['value'] = df['value'].pct_change(12) * 100.0
            df = df.dropna(subset=['value'])

        result = [
            {'date': row['date'].strftime('%Y-%m-%d'), 'value': float(row['value'])}
            for _, row in df.iterrows()
        ]

        # Cache the result with appropriate timeout
        self._set_cache(cache_key, result, timeout)
        return result

    def _get_yield_curve_2_10(self, days=365 * 10):
        """Calculate 2/10 yield curve (10Y - 2Y spread) with caching"""
        # Check cache first
        cache_key = self._get_cache_key("_get_yield_curve_2_10", days)
        cached_data = self._check_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            data_2y = self.get_fred_data('DGS2', days)
            data_10y = self.get_fred_data('DGS10', days)

            if not data_2y or not data_10y:
                self._set_cache(cache_key, [])
                return []

            df_2y = pd.DataFrame(data_2y)
            df_10y = pd.DataFrame(data_10y)
            df_2y['date'] = pd.to_datetime(df_2y['date'])
            df_10y['date'] = pd.to_datetime(df_10y['date'])

            merged = pd.merge(df_2y, df_10y, on='date', suffixes=('_2y', '_10y'))
            merged = merged.sort_values('date')

            merged['value'] = merged['value_10y'] - merged['value_2y']
            merged = merged.dropna(subset=['value'])

            result = [
                {'date': row['date'].strftime('%Y-%m-%d'), 'value': float(row['value'])}
                for _, row in merged.iterrows()
            ]

            # Cache the result with 1-day timeout (frequently changing data)
            self._set_cache(cache_key, result, 86400)
            return result
        except Exception as e:
            logger.error(f"Error calculating 2/10 yield curve: {str(e)}")
            self._set_cache(cache_key, [])
            return []

    def get_indicator_chart(self, indicator_key, days=365 * 10):
        """Generate an interactive chart for a specific indicator with caching"""
        # Check cache first
        cache_key = self._get_cache_key("get_indicator_chart", indicator_key, days)
        cached_data = self._check_cache(cache_key)
        if cached_data is not None:
            logger.info(f"Cache hit for indicator chart: {indicator_key}")
            return cached_data

        logger.info(f"Cache miss for indicator chart: {indicator_key}, generating")

        try:
            series_id = self.series_ids.get(
                'inflation' if indicator_key in ['inflation_yoy', 'inflation_mom'] else indicator_key)
            if indicator_key == 'yield_curve_2_10':
                series_id = 'DGS2'
            if not series_id:
                error_msg = "<p>Invalid indicator specified.</p>"
                self._set_cache(cache_key, error_msg)
                return error_msg

            data = self.get_indicator_series(indicator_key, days)
            if not data:
                error_msg = f"<p>No data available for {indicator_key}.</p>"
                self._set_cache(cache_key, error_msg)
                return error_msg

            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])

            fig = go.Figure()

            chart_config = self._get_chart_config(indicator_key, df)

            if chart_config['type'] == 'line':
                fig.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=df['value'],
                        name=chart_config['name'],
                        line=dict(color=chart_config['color']),
                        hovertemplate='%{x|%Y-%m-%d}<br>%{y:,.2f}<extra></extra>'
                    )
                )
            elif chart_config['type'] == 'bar':
                fig.add_trace(
                    go.Bar(
                        x=df['date'],
                        y=df['value'],
                        name=chart_config['name'],
                        marker_color=chart_config['color'],
                        opacity=0.7,
                        hovertemplate='%{x|%Y-%m-%d}<br>%{y:,.2f}<extra></extra>'
                    )
                )

            fig.update_layout(
                title=dict(
                    text=chart_config['title'],
                    font=dict(size=18)
                ),
                xaxis=dict(
                    title="Date",
                    rangeslider=dict(visible=True),
                    type="date",
                    gridcolor='rgba(200, 200, 200, 0.3)'
                ),
                yaxis=dict(
                    title=chart_config['yaxis_title'],
                    title_font=dict(color=chart_config['color']),
                    tickfont=dict(color=chart_config['color']),
                    gridcolor='rgba(200, 200, 200, 0.3)',
                    zeroline=True,
                    zerolinecolor='#666',
                    zerolinewidth=1
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=500,
                margin=dict(l=80, r=80, t=80, b=80),
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial"
                )
            )

            fig.update_xaxes(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(count=5, label="5y", step="year", stepmode="backward"),
                        dict(count=10, label="10y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )

            chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

            # Cache the result
            self._set_cache(cache_key, chart_html)
            return chart_html

        except Exception as e:
            logger.error(f"Error generating {indicator_key} chart: {str(e)}")
            error_msg = f"<p>Error generating {indicator_key} chart. Please try again later.</p>"
            self._set_cache(cache_key, error_msg)
            return error_msg

    def _get_chart_config(self, indicator_key, df):
        """Get configuration for different chart types"""
        configs = {
            'gdp': {
                'type': 'line',
                'name': 'Real GDP',
                'title': 'Real Gross Domestic Product',
                'yaxis_title': 'Billions of 2012 $',
                'color': '#1f77b4'
            },
            'nominal_gdp': {
                'type': 'line',
                'name': 'Nominal GDP',
                'title': 'Nominal Gross Domestic Product',
                'yaxis_title': 'Billions of Current $',
                'color': '#ff7f0e'
            },
            'inflation': {
                'type': 'line',
                'name': 'Inflation (YoY %)',
                'title': 'Inflation Rate (YoY % from CPI)',
                'yaxis_title': 'Percent',
                'color': '#2ca02c'
            },
            'inflation_yoy': {
                'type': 'line',
                'name': 'Inflation (YoY %)',
                'title': 'Inflation Rate (YoY % from CPI)',
                'yaxis_title': 'Percent',
                'color': '#2ca02c'
            },
            'inflation_mom': {
                'type': 'line',
                'name': 'Inflation (MoM %)',
                'title': 'Inflation Rate (MoM % from CPI)',
                'yaxis_title': 'Percent',
                'color': '#17becf'
            },
            'unemployment': {
                'type': 'line',
                'name': 'Unemployment Rate',
                'title': 'Unemployment Rate',
                'yaxis_title': 'Percent',
                'color': '#d62728'
            },
            'fed_funds': {
                'type': 'line',
                'name': 'Federal Funds Rate',
                'title': 'Federal Funds Rate',
                'yaxis_title': 'Percent',
                'color': '#9467bd'
            },
            'treasury_10y': {
                'type': 'line',
                'name': '10-Year Treasury Yield',
                'title': '10-Year Treasury Constant Maturity Rate',
                'yaxis_title': 'Percent',
                'color': '#8c564b'
            },
            'treasury_2y': {
                'type': 'line',
                'name': '2-Year Treasury Yield',
                'title': '2-Year Treasury Constant Maturity Rate',
                'yaxis_title': 'Percent',
                'color': '#bcbd22'
            },
            'yield_curve_2_10': {
                'type': 'line',
                'name': '2/10 Yield Curve',
                'title': '2/10 Year Treasury Yield Curve (10Y - 2Y)',
                'yaxis_title': 'Basis Points',
                'color': '#e377c2'
            },
            'jolts': {
                'type': 'line',
                'name': 'Job Openings',
                'title': 'Job Openings (JOLTS)',
                'yaxis_title': 'Thousands',
                'color': '#17becf'
            },
            'jobless_claims': {
                'type': 'line',
                'name': 'Jobless Claims',
                'title': 'Initial Jobless Claims',
                'yaxis_title': 'Thousands',
                'color': '#ff7f0e'
            }
        }

        return configs.get(indicator_key, {
            'type': 'line',
            'name': indicator_key.replace('_', ' ').title(),
            'title': indicator_key.replace('_', ' ').title(),
            'yaxis_title': 'Value',
            'color': '#7f7f7f'
        })

    def export_to_csv(self, indicator_key, days=365 * 10):
        """Export indicator data as CSV"""
        try:
            series_id = self.series_ids.get(indicator_key)
            if not series_id:
                return None

            data = self.get_indicator_series(indicator_key, days)
            if not data:
                return None

            df = pd.DataFrame(data)

            si = io.StringIO()
            cw = csv.writer(si)

            cw.writerow(['Date', 'Value'])

            for row in data:
                cw.writerow([row['date'], row['value']])

            output = make_response(si.getvalue())
            output.headers["Content-Disposition"] = f"attachment; filename={indicator_key}_data.csv"
            output.headers["Content-type"] = "text/csv"

            return output

        except Exception as e:
            logger.error(f"Error exporting {indicator_key} to CSV: {str(e)}")
            return None

    def export_all_to_csv(self, days=365 * 10):
        """Export all supported indicator series to one CSV file."""
        try:
            keys = [
                'nominal_gdp', 'gdp', 'inflation_yoy', 'inflation_mom',
                'unemployment', 'fed_funds', 'treasury_10y', 'treasury_2y',
                'yield_curve_2_10', 'jolts', 'jobless_claims'
            ]
            frames = []
            for key in keys:
                series = self.get_indicator_series(key, days)
                if series:
                    df = pd.DataFrame(series)
                    df = df.rename(columns={'value': key})
                    frames.append(df)
            if not frames:
                return None
            from functools import reduce
            merged = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), frames)
            merged = merged.sort_values('date')

            si = io.StringIO()
            merged.to_csv(si, index=False)
            output = make_response(si.getvalue())
            output.headers["Content-Disposition"] = "attachment; filename=all_indicators.csv"
            output.headers["Content-type"] = "text/csv"
            return output
        except Exception as e:
            logger.error(f"Error exporting all indicators to CSV: {str(e)}")
            return None

    def _get_demo_data(self, series_id):
        """Generate realistic sample data for demo purposes"""
        end_date = datetime.now()

        base_values = {
            'CPIAUCSL': 314.0,
            'GDPC1': 22000.0,
            'GDPA': 28500.0,
            'UNRATE': 4.2,
            'FEDFUNDS': 5.25,
            'DGS10': 4.0,
            'DGS2': 4.2,
            'JTSJOL': 8500.0,
            'ICSA': 220.0
        }

        base_value = base_values.get(series_id, 100.0)

        import random
        data = []
        months = 120
        for i in range(months):
            months_back = (months - 1 - i)
            date = (end_date - timedelta(days=months_back * 30)).strftime('%Y-%m-%d')
            if series_id == 'CPIAUCSL':
                value = base_value * (1 + (i * 0.0020))
                current_dt = end_date - timedelta(days=months_back * 30)
                if current_dt.year == 2021:
                    value *= 1.004 + random.uniform(-0.001, 0.001)
                value += random.uniform(-0.8, 0.8)
            elif series_id in ['GDPC1', 'GDPA']:
                quarterly_factor = (1 + ((i % 3) == 0) * 0.002)
                value = base_value * (1 + (i * 0.001)) * quarterly_factor + random.uniform(-80, 120)
            elif series_id == 'UNRATE':
                current_dt = end_date - timedelta(days=months_back * 30)
                value = base_value + 0.4 * np.sin(i / 6.0) + random.uniform(-0.2, 0.2)
                if current_dt.year == 2020 and 3 <= current_dt.month <= 6:
                    spike = 10 + 5 * np.exp(-abs(current_dt.month - 4.5))
                    value = max(value, spike)
            elif series_id in ['FEDFUNDS', 'DGS10', 'DGS2']:
                value = base_value + 0.6 * np.sin(i / 8.0) + random.uniform(-0.25, 0.25)
                value = max(0.0, value)
            elif series_id == 'JTSJOL':
                value = base_value + 500 * np.sin(i / 12.0) + random.uniform(-200, 300)
                value = max(5000, value)
            elif series_id == 'ICSA':
                value = base_value - 50 * np.sin(i / 12.0) + random.uniform(-30, 50)
                value = max(150, value)
            else:
                value = base_value + i + random.uniform(-1, 1)
            data.append({
                'date': date,
                'value': round(float(value), 2)
            })

        return data

    def get_latest_indicator_value(self, indicator_key, days=365 * 10):
        """Get latest transformed value for an indicator with caching"""
        # Check cache first
        cache_key = self._get_cache_key("get_latest_indicator_value", indicator_key, days)
        cached_data = self._check_cache(cache_key)
        if cached_data is not None:
            return cached_data

        series = self.get_indicator_series(indicator_key, days)
        if not series:
            self._set_cache(cache_key, None)
            return None
        latest = series[-1]
        result = {'latest_value': latest['value'], 'latest_date': latest['date']}

        # Cache the result
        self._set_cache(cache_key, result)
        return result

    def get_inflation_metrics(self, days=365 * 15):
        """Compute inflation YoY% and MoM% from CPI level series, plus latest CPI level with caching"""
        # Check cache first
        cache_key = self._get_cache_key("get_inflation_metrics", days)
        cached_data = self._check_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            series_id = self.series_ids.get('inflation')
            if not series_id:
                self._set_cache(cache_key, None)
                return None
            data = self.get_fred_data(series_id, days)
            if not data or len(data) < 14:
                self._set_cache(cache_key, None)
                return None
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])
            if len(df) < 14:
                self._set_cache(cache_key, None)
                return None
            last = df.iloc[-1]
            prev = df.iloc[-2]
            last_12 = df.iloc[-13]
            yoy = ((last['value'] / last_12['value']) - 1.0) * 100.0
            mom = ((last['value'] / prev['value']) - 1.0) * 100.0
            result = {
                'latest_date': last['date'].strftime('%Y-%m-%d'),
                'cpi_level': float(last['value']),
                'yoy_percent': float(round(yoy, 2)),
                'mom_percent': float(round(mom, 2))
            }

            # Cache the result
            self._set_cache(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Error computing inflation metrics: {str(e)}")
            self._set_cache(cache_key, None)
            return None

    def get_macro_data(self):
        """Fetch all macro indicators"""
        macro_data = {}

        for indicator, series_id in self.series_ids.items():
            data = self.get_fred_data(series_id)
            if data:
                macro_data[indicator] = {
                    'data': data,
                    'name': self._get_indicator_name(indicator),
                    'unit': self._get_indicator_unit(indicator)
                }

        return macro_data

    def _get_indicator_name(self, indicator):
        """Get display name for indicator"""
        names = {
            'inflation': 'Inflation (CPI)',
            'gdp': 'Real GDP',
            'unemployment': 'Unemployment Rate',
            'fed_funds': 'Federal Funds Rate',
            'treasury_10y': '10-Year Treasury Yield'
        }
        return names.get(indicator, indicator)

    def _get_indicator_unit(self, indicator):
        """Get unit for indicator"""
        units = {
            'inflation': 'Index',
            'gdp': 'Billions of Chained 2012 USD',
            'unemployment': 'Percent',
            'fed_funds': 'Percent',
            'treasury_10y': 'Percent'
        }
        return units.get(indicator, '')

    def analyze_macro_environment(self, macro_data):
        """Generate AI analysis of the macro environment"""
        try:
            analysis_input = {}
            for indicator, data in macro_data.items():
                if data and 'data' in data and len(data['data']) > 0:
                    latest = data['data'][-1]
                    analysis_input[indicator] = {
                        'value': latest['value'],
                        'unit': data.get('unit', ''),
                        'name': data.get('name', indicator)
                    }

            analysis = {
                'overview': self._generate_overview(analysis_input),
                'risks': self._identify_risks(analysis_input),
                'opportunities': self._identify_opportunities(analysis_input)
            }

            return analysis

        except Exception as e:
            logger.error(f"Error in macro analysis: {str(e)}")
            return None

    def _generate_overview(self, data):
        """Generate overview of current macro environment"""
        if not data:
            return "Unable to generate overview due to missing data."

        gdp = data.get('gdp', {})
        inflation = data.get('inflation', {})
        unemployment = data.get('unemployment', {})

        overview = "The current macroeconomic environment shows "

        if gdp:
            overview += f"GDP at {gdp['value']} {gdp.get('unit', '')}, "

        if inflation:
            inflation_val = inflation.get('value', 0)
            if inflation_val > 3.5:
                inflation_status = "elevated"
            elif inflation_val < 2:
                inflation_status = "low"
            else:
                inflation_status = "moderate"
            overview += f"with {inflation_status} inflation at {inflation_val}%, "

        if unemployment:
            unemployment_val = unemployment.get('value', 0)
            if unemployment_val < 4:
                employment_status = "tight"
            elif unemployment_val > 6:
                employment_status = "loose"
            else:
                employment_status = "balanced"
            overview += f"and a {employment_status} labor market with unemployment at {unemployment_val}%."

        return overview

    def _identify_risks(self, data):
        """Identify potential risks in the macro environment"""
        risks = []

        if 'inflation' in data:
            inflation = data['inflation']['value']
            if inflation > 4:
                risks.append(f"High inflation ({inflation}%) may lead to aggressive monetary tightening")
            elif inflation < 1.5:
                risks.append("Very low inflation may indicate weak demand")

        if 'unemployment' in data:
            unemployment = data['unemployment']['value']
            if unemployment > 6:
                risks.append(f"Elevated unemployment rate ({unemployment}%) may signal economic weakness")

        if 'fed_funds' in data and 'treasury_10y' in data:
            fed_rate = data['fed_funds']['value']
            treasury_10y = data['treasury_10y']['value']
            if treasury_10y < fed_rate:
                risks.append("Inverted yield curve may signal potential economic slowdown")

        return risks if risks else ["No significant immediate risks identified"]

    def _identify_opportunities(self, data):
        """Identify potential opportunities in the macro environment"""
        opportunities = []

        if 'inflation' in data and 'fed_funds' in data:
            inflation = data['inflation']['value']
            fed_rate = data['fed_funds']['value']

            if inflation < 2 and fed_rate > 2:
                opportunities.append("Low inflation and high rates may present buying opportunities in bonds")
            elif inflation > 3 and fed_rate < 1:
                opportunities.append("High inflation and low rates may favor inflation-protected assets")

        if 'gdp' in data and len(data['gdp'].get('data', [])) > 1:
            gdp_growth = data['gdp']['data'][-1]['value'] - data['gdp']['data'][-2]['value']
            if gdp_growth > 0.5:
                opportunities.append(f"Strong GDP growth of {gdp_growth:.2f}% may support equity markets")

        return opportunities if opportunities else ["Consider a diversified portfolio approach"]

# Create a singleton instance
macro_data_service = MacroDataService()
