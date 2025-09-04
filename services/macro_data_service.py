import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import json
from flask import jsonify
import logging

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
            'gdp': 'GDPC1',           # Real GDP
            'unemployment': 'UNRATE',  # Unemployment Rate
            'fed_funds': 'FEDFUNDS',   # Federal Funds Rate
            'treasury_10y': 'DGS10',   # 10-Year Treasury Yield
        }

    def get_fred_data(self, series_id, days=365*5):
        """Fetch economic data from FRED API"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            params = {
                'series_id': series_id,
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'observation_start': start_date.strftime('%Y-%m-%d'),
                'observation_end': end_date.strftime('%Y-%m-%d')
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
                # Try with a different date range for demo key
                if self.fred_api_key == 'demo':
                    return self._get_demo_data(series_id)
            logger.error(f"HTTP Error fetching FRED data for {series_id}: {str(e)}")
            return self._get_demo_data(series_id) if self.fred_api_key == 'demo' else []
            
        except Exception as e:
            logger.error(f"Error fetching FRED data for {series_id}: {str(e)}")
            return self._get_demo_data(series_id) if self.fred_api_key == 'demo' else []
    
    def _get_demo_data(self, series_id):
        """Generate realistic sample data for demo purposes when API calls fail"""
        end_date = datetime.now()
        
        # Base values for different indicators (as of 2023)
        base_values = {
            'CPIAUCSL': 300.0,  # CPI index (2023 level)
            'GDPC1': 20000.0,   # Real GDP in billions (2023 level)
            'UNRATE': 3.5,      # Unemployment rate in %
            'FEDFUNDS': 5.25,   # Federal funds rate in %
            'DGS10': 4.0        # 10-year treasury yield in %
        }
        
        # Get the base value for this series, default to 100 if not found
        base_value = base_values.get(series_id, 100.0)
        
        # Generate realistic looking data with some random variation
        import random
        data = []
        for i in range(12):  # Last 12 months of data
            date = (end_date - timedelta(days=(11-i)*30)).strftime('%Y-%m-%d')
            
            # Add some realistic variation based on indicator type
            if series_id == 'CPIAUCSL':  # CPI: slow upward trend with small fluctuations
                value = base_value * (1 + (i * 0.002)) + random.uniform(-0.5, 0.5)
            elif series_id == 'GDPC1':  # GDP: slow growth with some quarterly variation
                value = base_value * (1 + (i * 0.001)) + random.uniform(-50, 100)
            elif series_id == 'UNRATE':  # Unemployment: small fluctuations around base
                value = base_value + random.uniform(-0.3, 0.3)
            elif series_id in ['FEDFUNDS', 'DGS10']:  # Interest rates: more volatile
                value = base_value + (i-6)*0.1 + random.uniform(-0.2, 0.2)
            else:  # Default case
                value = base_value + i + random.uniform(-1, 1)
                
            data.append({
                'date': date,
                'value': round(value, 2)  # Round to 2 decimal places
            })
            
        return data

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
