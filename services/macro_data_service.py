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

logger = logging.getLogger(__name__)

class MacroDataService:
    def __init__(self):
        self.fred_api_key = os.getenv('FRED_API_KEY', 'demo')  # Using demo key as fallback
        self.yahoo_finance_base = 'https://query1.finance.yahoo.com/v8/finance/chart/'
        self.fred_base_url = 'https://api.stlouisfed.org/fred/series/observations'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # FRED Series IDs for common economic indicators
        self.series_ids = {
            'inflation': 'CPIAUCSL',  # CPI for All Urban Consumers
            'gdp': 'GDPC1',           # Real GDP (2012 chained dollars)
            'nominal_gdp': 'GDPA',    # Nominal GDP (current dollars)
            'unemployment': 'UNRATE',  # Unemployment Rate
            'fed_funds': 'FEDFUNDS',   # Federal Funds Rate
            'treasury_10y': 'DGS10',   # 10-Year Treasury Yield
        }

    def get_fred_data(self, series_id, days=365*5):
        """Fetch economic data from FRED API"""
        try:
            end_date = datetime.now()
            # For demo key, get more data to ensure we have recent points
            if self.fred_api_key == 'demo':
                days = max(days, 365*10)  # Get at least 10 years of data for demo
            
            start_date = end_date - timedelta(days=days)
            
            params = {
                'series_id': series_id,
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'observation_start': start_date.strftime('%Y-%m-%d'),
                'observation_end': end_date.strftime('%Y-%m-%d'),
                'sort_order': 'desc',  # Get most recent data first
                'limit': 2000  # Increase limit for demo key to get more data points
            }
            
            response = requests.get('https://api.stlouisfed.org/fred/series/observations', 
                                 params=params, 
                                 timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Process data into a clean format
            if 'observations' in data and data['observations']:
                df = pd.DataFrame(data['observations'])
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                
                # Filter out any rows with missing values
                df = df.dropna(subset=['value'])
                
                # Convert to list of dicts with proper date formatting
                result = [
                    {'date': row['date'].strftime('%Y-%m-%d'), 
                     'value': float(row['value'])}
                    for _, row in df.iterrows()
                ]
                
                # Sort by date to ensure chronological order
                result.sort(key=lambda x: x['date'])
                
                return result
                        
            return []
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400 and 'observation_start' in str(e):
                # For demo key, try to return the most recent data point we can get
                if self.fred_api_key == 'demo':
                    demo_data = self._get_demo_data(series_id)
                    if demo_data:
                        # Sort demo data by date to ensure we have the latest
                        demo_data_sorted = sorted(demo_data, key=lambda x: x['date'], reverse=True)
                        return demo_data_sorted
            logger.error(f"HTTP Error fetching FRED data for {series_id}: {str(e)}")
            return self._get_demo_data(series_id) if self.fred_api_key == 'demo' else []
            
        except Exception as e:
            logger.error(f"Error fetching FRED data for {series_id}: {str(e)}")
            return self._get_demo_data(series_id) if self.fred_api_key == 'demo' else []
            
    def get_indicator_series(self, indicator_key, days=365*10):
        """Return transformed time series for an indicator (e.g., CPI -> YoY inflation%). Supports inflation_yoy and inflation_mom."""
        # Support synthetic keys for inflation transformations
        inflation_variant = None
        base_key = indicator_key
        if indicator_key in ['inflation_yoy', 'inflation_mom']:
            base_key = 'inflation'
            inflation_variant = indicator_key.split('_')[1]

        series_id = self.series_ids.get(base_key)
        if not series_id:
            return []
        data = self.get_fred_data(series_id, days)
        if not data:
            return []
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        # Transformations per indicator
        if base_key == 'inflation':
            df = df.sort_values('date')
            if inflation_variant == 'mom':
                # Month-over-month % change
                df['value'] = df['value'].pct_change(1) * 100.0
                df = df.dropna(subset=['value'])
            else:
                # Default to YoY % change
                df['value'] = df['value'].pct_change(12) * 100.0
                df = df.dropna(subset=['value'])
        # Ensure ascending order and standard format
        df = df.sort_values('date')
        return [
            {'date': row['date'].strftime('%Y-%m-%d'), 'value': float(row['value'])}
            for _, row in df.iterrows()
        ]

    def get_indicator_chart(self, indicator_key, days=365*10):
        """Generate an interactive chart for a specific indicator"""
        try:
            series_id = self.series_ids.get('inflation' if indicator_key in ['inflation_yoy','inflation_mom'] else indicator_key)
            if not series_id:
                return "<p>Invalid indicator specified.</p>"
                
            # Get indicator data (transformed if needed)
            data = self.get_indicator_series(indicator_key, days)
            if not data:
                return f"<p>No data available for {indicator_key}.</p>"
                
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            
            # Create figure
            fig = go.Figure()
            
            # Define chart settings based on indicator type
            chart_config = self._get_chart_config(indicator_key, df)
            
            # Add trace based on chart type
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
            
            # Update layout
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
            
            # Add range selector buttons
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
            
            # Render chart only; CSV download is handled in the card header
            chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
            return chart_html
            
        except Exception as e:
            logger.error(f"Error generating {indicator_key} chart: {str(e)}")
            return f"<p>Error generating {indicator_key} chart. Please try again later.</p>"
    
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
            }
        }
        
        return configs.get(indicator_key, {
            'type': 'line',
            'name': indicator_key.replace('_', ' ').title(),
            'title': indicator_key.replace('_', ' ').title(),
            'yaxis_title': 'Value',
            'color': '#7f7f7f'
        })
    
    def export_to_csv(self, indicator_key, days=365*10):
        """Export indicator data as CSV"""
        try:
            series_id = self.series_ids.get(indicator_key)
            if not series_id:
                return None
                
            data = self.get_indicator_series(indicator_key, days)
            if not data:
                return None
                
            df = pd.DataFrame(data)
            
            # Create CSV in memory
            si = io.StringIO()
            cw = csv.writer(si)
            
            # Write header
            cw.writerow(['Date', 'Value'])
            
            # Write data
            for row in data:
                cw.writerow([row['date'], row['value']])
            
            # Create response
            output = make_response(si.getvalue())
            output.headers["Content-Disposition"] = f"attachment; filename={indicator_key}_data.csv"
            output.headers["Content-type"] = "text/csv"
            
            return output
            
        except Exception as e:
            logger.error(f"Error exporting {indicator_key} to CSV: {str(e)}")
            return None
    
    def _get_demo_data(self, series_id):
        """Generate realistic sample data for demo purposes when API calls fail"""
        end_date = datetime.now()
        
        # Base values for different indicators (as of November 2024)
        base_values = {
            'CPIAUCSL': 314.0,    # CPI index (1982-84=100)
            'GDPC1': 22000.0,     # Real GDP (billions of chained 2012 $)
            'GDPA': 28500.0,      # Nominal GDP (billions current $)
            'UNRATE': 4.2,        # Unemployment Rate (%)
            'FEDFUNDS': 5.25,     # Federal Funds Rate (%)
            'DGS10': 4.0          # 10-Year Treasury Yield (%)
        }
        
        # Get the base value for this series, default to 100 if not found
        base_value = base_values.get(series_id, 100.0)
        
        # Generate realistic looking data with some random variation
        import random
        data = []
        # Generate ~10 years monthly (120 points)
        months = 120
        for i in range(months):
            months_back = (months - 1 - i)
            date = (end_date - timedelta(days=months_back * 30)).strftime('%Y-%m-%d')
            # Add some realistic variation based on indicator type
            if series_id == 'CPIAUCSL':  # CPI: slow upward trend with small fluctuations
                value = base_value * (1 + (i * 0.0025)) + random.uniform(-0.8, 0.8)
            elif series_id in ['GDPC1', 'GDPA']:  # GDP: slow growth with quarterly variation
                quarterly_factor = (1 + ((i % 3) == 0) * 0.002)
                value = base_value * (1 + (i * 0.001)) * quarterly_factor + random.uniform(-80, 120)
            elif series_id == 'UNRATE':  # Unemployment: small fluctuations around base
                value = base_value + 0.4 * np.sin(i / 6.0) + random.uniform(-0.2, 0.2)
                value = max(2.5, min(9.0, value))
            elif series_id in ['FEDFUNDS', 'DGS10']:  # Interest rates: more volatile
                value = base_value + 0.6 * np.sin(i / 8.0) + random.uniform(-0.25, 0.25)
                value = max(0.0, value)
            else:  # Default case
                value = base_value + i + random.uniform(-1, 1)
            data.append({
                'date': date,
                'value': round(float(value), 2)
            })
            
        return data

    def get_latest_indicator_value(self, indicator_key, days=365*10):
        """Get latest transformed value for an indicator."""
        series = self.get_indicator_series(indicator_key, days)
        if not series:
            return None
        latest = series[-1]
        return {'latest_value': latest['value'], 'latest_date': latest['date']}

    def get_inflation_metrics(self, days=365*15):
        """Compute inflation YoY% and MoM% from CPI level series, plus latest CPI level."""
        try:
            series_id = self.series_ids.get('inflation')
            if not series_id:
                return None
            data = self.get_fred_data(series_id, days)
            if not data or len(data) < 14:
                return None
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])
            if len(df) < 14:
                return None
            last = df.iloc[-1]
            prev = df.iloc[-2]
            last_12 = df.iloc[-13]
            yoy = ((last['value'] / last_12['value']) - 1.0) * 100.0
            mom = ((last['value'] / prev['value']) - 1.0) * 100.0
            return {
                'latest_date': last['date'].strftime('%Y-%m-%d'),
                'cpi_level': float(last['value']),
                'yoy_percent': float(round(yoy, 2)),
                'mom_percent': float(round(mom, 2))
            }
        except Exception as e:
            logger.error(f"Error computing inflation metrics: {str(e)}")
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
            # Convert data to a more AI-friendly format
            analysis_input = {}
            for indicator, data in macro_data.items():
                if data and 'data' in data and len(data['data']) > 0:
                    latest = data['data'][-1]
                    analysis_input[indicator] = {
                        'value': latest['value'],
                        'unit': data.get('unit', ''),
                        'name': data.get('name', indicator)
                    }
            
            # Simple rule-based analysis (can be enhanced with ML model)
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
        
        # GDP analysis
        if gdp:
            overview += f"GDP at {gdp['value']} {gdp.get('unit', '')}, "
        
        # Inflation analysis
        if inflation:
            inflation_val = inflation.get('value', 0)
            if inflation_val > 3.5:
                inflation_status = "elevated"
            elif inflation_val < 2:
                inflation_status = "low"
            else:
                inflation_status = "moderate"
            overview += f"with {inflation_status} inflation at {inflation_val}%, "
        
        # Unemployment analysis
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
