"""
Wave Detection System - Professional Stock Analysis Platform
Author: AI Assistant
Version: 1.1.0
Last Updated: December 2024

Features:
- Wave Detection (Active & Forming)
- Volume Acceleration Analysis (NEW!)
- Category & Sector Filtering
- Multi-Sheet Excel Export
- Professional Analytics & Reports
- Real-time Google Sheets Integration

Volume Acceleration Strategy:
- Calculates: (30d/90d volume) - (30d/180d volume)
- Positive values indicate institutional accumulation
- Values > 0.2 suggest imminent breakout
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import logging
from io import BytesIO
import xlsxwriter
from typing import Dict, List, Tuple, Optional
import requests

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Wave Detection System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .wave-active {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
    .wave-forming {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1e3c72;
        color: white;
    }
    .diagnostic-info {
        font-size: 0.8rem;
        color: #6c757d;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA LOADING AND CLEANING FUNCTIONS
# ============================================

def safe_float(value, default=0.0):
    """Safely convert value to float"""
    try:
        if pd.isna(value):
            return default
        if isinstance(value, str):
            # Remove any remaining formatting
            value = value.replace('₹', '').replace(',', '').replace('%', '').strip()
            if value == '' or value == '-':
                return default
        return float(value)
    except:
        return default

def safe_compare(value1, value2, operation='gt'):
    """Safely compare two values"""
    v1 = safe_float(value1)
    v2 = safe_float(value2)
    
    if operation == 'gt':
        return v1 > v2
    elif operation == 'lt':
        return v1 < v2
    elif operation == 'gte':
        return v1 >= v2
    elif operation == 'lte':
        return v1 <= v2
    else:
        return v1 == v2

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_google_sheets_data(sheet_url: str, gid: str) -> pd.DataFrame:
    """Load data from Google Sheets with error handling"""
    try:
        # Convert to CSV export URL
        csv_url = f"{sheet_url.split('/edit')[0]}/export?format=csv&gid={gid}"
        
        # Load data
        df = pd.read_csv(csv_url)
        
        # Log success
        logger.info(f"Successfully loaded {len(df)} rows from Google Sheets")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error(f"Failed to load data: {str(e)}")
        return pd.DataFrame()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare the data with comprehensive error handling"""
    try:
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Define column types
        currency_cols = ['price', 'low_52w', 'high_52w', 'sma_20d', 'sma_50d', 'sma_200d', 'prev_close']
        percentage_cols = ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
                          'from_low_pct', 'from_high_pct', 'eps_change_pct',
                          'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
                          'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 'vol_ratio_90d_180d']
        volume_cols = ['volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d']
        numeric_cols = ['rvol', 'pe', 'eps_current', 'eps_last_qtr', 'year']
        
        # Clean all numeric columns
        all_numeric_cols = currency_cols + percentage_cols + volume_cols + numeric_cols
        
        for col in all_numeric_cols:
            if col in df.columns:
                # First, handle string values
                if df[col].dtype == 'object':
                    # Remove currency symbols, percentage signs, and commas
                    df[col] = df[col].astype(str).str.replace('₹', '', regex=False)
                    df[col] = df[col].str.replace('%', '', regex=False)
                    df[col] = df[col].str.replace(',', '', regex=False)
                    df[col] = df[col].str.strip()
                    
                    # Replace empty strings and dashes with NaN
                    df[col] = df[col].replace(['', '-', 'N/A', 'n/a', '#N/A'], np.nan)
                
                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean market cap separately (it has 'Cr' suffix)
        if 'market_cap' in df.columns:
            df['market_cap_clean'] = df['market_cap'].astype(str).str.replace('₹', '', regex=False)
            df['market_cap_clean'] = df['market_cap_clean'].str.replace(' Cr', '', regex=False)
            df['market_cap_clean'] = df['market_cap_clean'].str.replace(',', '', regex=False)
            df['market_cap_clean'] = pd.to_numeric(df['market_cap_clean'], errors='coerce')
        
        # Ensure categorical columns are strings
        categorical_cols = ['ticker', 'company_name', 'category', 'sector']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                # Replace 'nan' string with empty string
                df[col] = df[col].replace('nan', '')
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['ticker'], keep='first')
        
        # Remove extreme outliers and data errors
        if 'rvol' in df.columns:
            df = df[df['rvol'] < 100]  # Remove data errors
        
        # Fill NaN values with appropriate defaults
        # For percentage columns, NaN means 0
        for col in percentage_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # For volume columns, NaN means 0
        for col in volume_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # For rvol, NaN means 0
        if 'rvol' in df.columns:
            df['rvol'] = df['rvol'].fillna(0)
        
        # Remove rows with critical missing data
        critical_cols = ['ticker', 'price']
        for col in critical_cols:
            if col in df.columns:
                if col == 'price':
                    df = df[df[col].notna() & (df[col] > 0)]
                else:
                    df = df[df[col].notna() & (df[col] != '')]
        
        logger.info(f"Data cleaned successfully. Final shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        st.error(f"Data cleaning failed: {str(e)}")
        return df

# ============================================
# WAVE DETECTION ALGORITHMS
# ============================================

def safe_float(value, default=0.0):
    """Safely convert value to float"""
    try:
        if pd.isna(value):
            return default
        if isinstance(value, str):
            # Remove any remaining formatting
            value = value.replace('₹', '').replace(',', '').replace('%', '').strip()
            if value == '' or value == '-':
                return default
        return float(value)
    except:
        return default

def safe_compare(value1, value2, operation='gt'):
    """Safely compare two values"""
    v1 = safe_float(value1)
    v2 = safe_float(value2)
    
    if operation == 'gt':
        return v1 > v2
    elif operation == 'lt':
        return v1 < v2
    elif operation == 'gte':
        return v1 >= v2
    elif operation == 'lte':
        return v1 <= v2
    else:
        return v1 == v2

def calculate_volume_acceleration(row: pd.Series) -> float:
    """
    Calculate Volume Acceleration - the rate of change in volume
    Positive = Accelerating (bullish)
    Negative = Decelerating (bearish)
    """
    try:
        vol_30d = safe_float(row.get('volume_30d', 0))
        vol_90d = safe_float(row.get('volume_90d', 0))
        vol_180d = safe_float(row.get('volume_180d', 0))
        
        if vol_90d > 0 and vol_180d > 0:
            ratio_30_90 = vol_30d / vol_90d
            ratio_30_180 = vol_30d / vol_180d
            acceleration = ratio_30_90 - ratio_30_180
            return acceleration
        return 0
    except:
        return 0

def calculate_wave_score(row: pd.Series) -> float:
    """Calculate comprehensive wave score with robust error handling"""
    try:
        score = 0
        
        # Volume Power (20% - reduced from 25%)
        rvol = safe_float(row.get('rvol', 0))
        if rvol > 0:
            vol_ratio_1d_90d = safe_float(row.get('vol_ratio_1d_90d', 0))
            vol_ratio_7d_90d = safe_float(row.get('vol_ratio_7d_90d', 0))
            vol_ratio_30d_90d = safe_float(row.get('vol_ratio_30d_90d', 0))
            vol_ratio_90d_180d = safe_float(row.get('vol_ratio_90d_180d', 0))
            
            vol_score = (
                min(rvol, 5) / 5 * 0.2 +
                (vol_ratio_1d_90d / 100 * 0.15 if vol_ratio_1d_90d > 0 else 0) +
                (vol_ratio_7d_90d / 100 * 0.15 if vol_ratio_7d_90d > 0 else 0) +
                (vol_ratio_30d_90d / 100 * 0.15 if vol_ratio_30d_90d > 0 else 0) +
                (vol_ratio_90d_180d / 100 * 0.35 if vol_ratio_90d_180d > 0 else 0)
            )
            score += vol_score * 20
        
        # NEW: Volume Acceleration (10% weight) - KEY INDICATOR
        vol_acceleration = calculate_volume_acceleration(row)
        if vol_acceleration > 0:
            # Positive acceleration is bullish
            accel_score = min(vol_acceleration * 100, 10)  # Cap at 10 points
            score += accel_score
        
        # Position Opportunity (25%)
        from_low = safe_float(row.get('from_low_pct', 0))
        from_high = safe_float(row.get('from_high_pct', 0))
        
        if from_low >= 0:  # Valid from_low_pct
            pos_score = (
                max(0, (100 - from_low) / 100) * 0.6 +
                max(0, (100 + from_high) / 100) * 0.4
            )
            score += pos_score * 25
        
        # Momentum Cascade (20% - reduced from 25%)
        mom_score = 0
        ret_1d = safe_float(row.get('ret_1d', 0))
        ret_7d = safe_float(row.get('ret_7d', 0))
        ret_30d = safe_float(row.get('ret_30d', 0))
        ret_3m = safe_float(row.get('ret_3m', 0))
        
        if ret_1d > 0: 
            mom_score += 5
        if ret_7d > 0 and ret_30d != 0 and ret_7d > ret_30d / 4: 
            mom_score += 10
        if ret_30d > 0 and ret_3m != 0 and ret_30d > ret_3m / 3: 
            mom_score += 5
        score += mom_score
        
        # Technical Alignment (20% - reduced from 25%)
        tech_score = 0
        price = safe_float(row.get('price', 0))
        sma_20d = safe_float(row.get('sma_20d', 0))
        sma_50d = safe_float(row.get('sma_50d', 0))
        sma_200d = safe_float(row.get('sma_200d', 0))
        
        if price > 0 and sma_20d > 0:
            if price > sma_20d: 
                tech_score += 7
            if sma_50d > 0 and sma_20d > sma_50d: 
                tech_score += 7
            if sma_200d > 0 and sma_50d > sma_200d: 
                tech_score += 6
        score += tech_score
        
        # Bonus factors
        eps_change = safe_float(row.get('eps_change_pct', 0))
        pe = safe_float(row.get('pe', 0))
        
        if eps_change > 20: 
            score += 5
        if 0 < pe < 25: 
            score += 5
        
        return min(score, 100)  # Cap at 100
        
    except Exception as e:
        logger.error(f"Error calculating wave score for {row.get('ticker', 'Unknown')}: {str(e)}")
        return 0

def detect_wave_stage(row: pd.Series) -> Tuple[str, str, float]:
    """Detect current wave stage with robust error handling"""
    try:
        # Safely get all values
        rvol = safe_float(row.get('rvol', 0))
        ret_7d = safe_float(row.get('ret_7d', 0))
        from_low_pct = safe_float(row.get('from_low_pct', 0))
        vol_ratio_90d_180d = safe_float(row.get('vol_ratio_90d_180d', 0))
        from_high_pct = safe_float(row.get('from_high_pct', 0))
        ret_3d = safe_float(row.get('ret_3d', 0))
        
        # NEW: Calculate volume acceleration
        vol_acceleration = calculate_volume_acceleration(row)
        
        # Check for ACCELERATING VOLUME (New Early Detection!)
        if (vol_acceleration > 0.2 and  # Strong acceleration
            from_low_pct < 40 and
            vol_ratio_90d_180d > 0.95):
            return "🚨 Volume Accelerating", "Forming", 85
        
        # Check for active waves
        elif (rvol > 2 and ret_7d > 3 and from_low_pct < 50):
            momentum_strength = ret_7d / 7 if ret_7d > 0 else 0
            if momentum_strength > 1:
                return "🚀 Explosive Wave", "Active", 90
            else:
                return "🏄 Riding Wave", "Active", 75
        
        # Check for forming waves (with acceleration check)
        elif (vol_ratio_90d_180d > 0.9 and 
              from_low_pct < 30 and
              abs(ret_7d) < 2):
            
            if vol_acceleration > 0.1:  # Mild acceleration
                return "⚡ Accelerating Formation", "Forming", 80
            else:
                pressure = vol_ratio_90d_180d * 100
                if pressure > 110:
                    return "🌊 High Pressure", "Forming", 70
                else:
                    return "💫 Building Wave", "Forming", 60
        
        # Check for exhausted waves
        elif (from_high_pct > -10 and rvol < 0.5):
            return "⚠️ Exhausted", "Danger", 20
        
        # Check for deceleration
        elif vol_acceleration < -0.1:
            return "📉 Decelerating", "Danger", 25
        
        # Default
        else:
            return "😴 Dormant", "Inactive", 10
            
    except Exception as e:
        logger.error(f"Error detecting wave stage for {row.get('ticker', 'Unknown')}: {str(e)}")
        return "❓ Unknown", "Error", 0

# ============================================
# FILTERING AND ANALYSIS FUNCTIONS
# ============================================

def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Apply user-selected filters with safe comparisons"""
    filtered_df = df.copy()
    
    # Category filter
    if filters['categories'] and 'All' not in filters['categories']:
        filtered_df = filtered_df[filtered_df['category'].isin(filters['categories'])]
    
    # Sector filter
    if filters['sectors'] and 'All' not in filters['sectors']:
        filtered_df = filtered_df[filtered_df['sector'].isin(filters['sectors'])]
    
    # Volume filter (safe comparison)
    if filters['min_volume'] > 0:
        filtered_df['volume_30d_safe'] = filtered_df['volume_30d'].apply(safe_float)
        filtered_df = filtered_df[filtered_df['volume_30d_safe'] >= filters['min_volume']]
        filtered_df = filtered_df.drop('volume_30d_safe', axis=1)
    
    # Price filter (safe comparison)
    if filters['min_price'] > 0:
        filtered_df['price_safe'] = filtered_df['price'].apply(safe_float)
        filtered_df = filtered_df[filtered_df['price_safe'] >= filters['min_price']]
        filtered_df = filtered_df.drop('price_safe', axis=1)
    
    # Wave stage filter
    if filters['wave_stages'] and 'All' not in filters['wave_stages']:
        filtered_df['wave_type'] = filtered_df.apply(lambda x: detect_wave_stage(x)[1], axis=1)
        filtered_df = filtered_df[filtered_df['wave_type'].isin(filters['wave_stages'])]
    
    return filtered_df

def get_top_opportunities(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Get top opportunities based on wave score"""
    # Calculate scores and stages
    df['wave_score'] = df.apply(calculate_wave_score, axis=1)
    df[['wave_stage', 'wave_type', 'confidence']] = df.apply(
        lambda x: pd.Series(detect_wave_stage(x)), axis=1
    )
    
    # Sort by score and get top n
    top_df = df.nlargest(n, 'wave_score')
    
    return top_df

# ============================================
# REPORT GENERATION FUNCTIONS
# ============================================

def generate_excel_report(df: pd.DataFrame, analysis_df: pd.DataFrame) -> BytesIO:
    """Generate comprehensive Excel report"""
    output = BytesIO()
    
    # Calculate volume acceleration if not already present
    if 'vol_acceleration' not in analysis_df.columns:
        analysis_df['vol_acceleration'] = analysis_df.apply(calculate_volume_acceleration, axis=1)
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#1e3c72',
            'font_color': 'white',
            'border': 1
        })
        
        number_format = workbook.add_format({'num_format': '#,##0.00'})
        percent_format = workbook.add_format({'num_format': '0.00%'})
        currency_format = workbook.add_format({'num_format': '₹#,##0.00'})
        
        # Sheet 1: Executive Summary
        summary_data = {
            'Metric': ['Total Stocks Analyzed', 'Active Waves', 'Forming Waves', 
                      'Volume Accelerating', 'Average Wave Score', 'Top Opportunity Score'],
            'Value': [
                len(df),
                len(analysis_df[analysis_df['wave_type'] == 'Active']),
                len(analysis_df[analysis_df['wave_type'] == 'Forming']),
                len(analysis_df[analysis_df['vol_acceleration'] > 0.1]),
                f"{analysis_df['wave_score'].mean():.2f}",
                f"{analysis_df['wave_score'].max():.2f}"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
        
        # Sheet 2: Top Opportunities
        analysis_df.to_excel(writer, sheet_name='Top Opportunities', index=False)
        
        # Sheet 3: Active Waves
        active_waves = analysis_df[analysis_df['wave_type'] == 'Active']
        active_waves.to_excel(writer, sheet_name='Active Waves', index=False)
        
        # Sheet 4: Forming Waves
        forming_waves = analysis_df[analysis_df['wave_type'] == 'Forming']
        forming_waves.to_excel(writer, sheet_name='Forming Waves', index=False)
        
        # Sheet 5: Volume Acceleration
        accel_stocks = analysis_df[analysis_df['vol_acceleration'] > 0.05].sort_values('vol_acceleration', ascending=False)
        accel_stocks.to_excel(writer, sheet_name='Volume Acceleration', index=False)
        
        # Sheet 6: Full Data
        df.to_excel(writer, sheet_name='Full Data', index=False)
        
        # Format sheets
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            worksheet.set_zoom(90)
            
            # Auto-fit columns
            for i, col in enumerate(df.columns):
                column_width = max(df[col].astype(str).str.len().max(), len(col)) + 2
                worksheet.set_column(i, i, min(column_width, 50))
    
    output.seek(0)
    return output

def generate_diagnostic_report(df: pd.DataFrame) -> Dict:
    """Generate system diagnostic information"""
    # Calculate volume acceleration for diagnostics
    df['vol_acceleration_temp'] = df.apply(calculate_volume_acceleration, axis=1)
    
    diagnostics = {
        'data_quality': {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_tickers': df['ticker'].duplicated().sum(),
            'data_freshness': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'coverage': {
            'categories': df['category'].value_counts().to_dict(),
            'sectors': df['sector'].nunique(),
            'price_range': f"₹{df['price'].min():.2f} - ₹{df['price'].max():.2f}",
            'market_cap_range': f"₹{df['market_cap_clean'].min():.0f} Cr - ₹{df['market_cap_clean'].max():.0f} Cr"
        },
        'wave_analysis': {
            'high_rvol_stocks': len(df[df['rvol'] > 2]),
            'accumulation_candidates': len(df[(df['vol_ratio_90d_180d'] > 0.9) & (df['from_low_pct'] < 30)]),
            'momentum_stocks': len(df[(df['ret_7d'] > 5) & (df['ret_30d'] > 10)]),
            'oversold_stocks': len(df[df['from_low_pct'] < 20]),
            'volume_accelerating': len(df[df['vol_acceleration_temp'] > 0.1]),
            'strong_acceleration': len(df[df['vol_acceleration_temp'] > 0.2])
        }
    }
    
    # Clean up temporary column
    df.drop('vol_acceleration_temp', axis=1, inplace=True, errors='ignore')
    
    return diagnostics

# ============================================
# VISUALIZATION FUNCTIONS
# ============================================

def create_wave_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create wave stage distribution chart"""
    if 'wave_type' not in df.columns:
        df['wave_type'] = df.apply(lambda x: detect_wave_stage(x)[1], axis=1)
    
    wave_counts = df['wave_type'].value_counts()
    
    # Define colors for each wave type
    color_map = {
        'Active': '#28a745',
        'Forming': '#ffc107', 
        'Danger': '#dc3545',
        'Inactive': '#6c757d',
        'Error': '#000000'
    }
    
    colors = [color_map.get(x, '#6c757d') for x in wave_counts.index]
    
    fig = go.Figure(data=[
        go.Bar(
            x=wave_counts.index,
            y=wave_counts.values,
            marker_color=colors
        )
    ])
    
    fig.update_layout(
        title="Wave Stage Distribution",
        xaxis_title="Wave Stage",
        yaxis_title="Number of Stocks",
        showlegend=False,
        height=400
    )
    
    return fig

def create_sector_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create sector performance heatmap"""
    # Calculate sector performance safely
    sector_groups = df.groupby('sector')['ret_30d'].agg(['mean', 'count'])
    # Filter sectors with at least 3 stocks
    sector_groups = sector_groups[sector_groups['count'] >= 3]
    sector_perf = sector_groups['mean'].sort_values(ascending=False).head(15)
    
    # Create color scale
    max_val = max(abs(sector_perf.max()), abs(sector_perf.min()))
    colors = []
    for val in sector_perf.values:
        if val > 0:
            colors.append(f'rgba(40, 167, 69, {min(val/max_val, 1)})')
        else:
            colors.append(f'rgba(220, 53, 69, {min(abs(val)/max_val, 1)})')
    
    fig = go.Figure(data=[
        go.Bar(
            x=sector_perf.values,
            y=sector_perf.index,
            orientation='h',
            marker_color=colors,
            text=[f"{x:.1f}%" for x in sector_perf.values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Top 15 Sectors by 30-Day Return",
        xaxis_title="Average 30-Day Return (%)",
        yaxis_title="Sector",
        height=600,
        margin=dict(l=200)
    )
    
    return fig

def create_wave_score_scatter(df: pd.DataFrame) -> go.Figure:
    """Create wave score vs return scatter plot"""
    # Ensure we have wave scores
    if 'wave_score' not in df.columns:
        df['wave_score'] = df.apply(calculate_wave_score, axis=1)
    if 'confidence' not in df.columns:
        df['confidence'] = df.apply(lambda x: detect_wave_stage(x)[2], axis=1)
    
    # Filter out invalid data
    plot_df = df[(df['wave_score'] > 0) & (df['ret_30d'].notna())].copy()
    
    # Calculate marker sizes safely
    plot_df['marker_size'] = plot_df['rvol'].apply(lambda x: min(safe_float(x, 1) * 5, 50))
    
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=plot_df['wave_score'],
        y=plot_df['ret_30d'],
        mode='markers',
        marker=dict(
            size=plot_df['marker_size'],
            color=plot_df['confidence'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Confidence"),
            line=dict(width=1, color='white')
        ),
        text=plot_df['ticker'] + '<br>' + plot_df['company_name'].str[:30] + '...',
        hovertemplate='%{text}<br>Wave Score: %{x:.1f}<br>30d Return: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Wave Score vs 30-Day Return",
        xaxis_title="Wave Score",
        yaxis_title="30-Day Return (%)",
        height=500,
        hovermode='closest'
    )
    
    return fig

# ============================================
# MAIN STREAMLIT APPLICATION
# ============================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🌊 Wave Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6c757d;">Advanced Stock Analysis Platform</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Data Source")
        sheet_url = st.text_input(
            "Google Sheets URL",
            value="https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk/edit?usp=sharing",
            help="Enter your Google Sheets URL"
        )
        gid = st.text_input("Sheet GID", value="2026492216", help="Sheet identifier")
        
        if st.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Load data
        df = load_google_sheets_data(sheet_url, gid)
        
        if not df.empty:
            df = clean_data(df)
            
            # Filters
            st.markdown("## 🔍 Filters")
            
            # Category filter
            categories = ['All'] + sorted(df['category'].dropna().unique().tolist())
            selected_categories = st.multiselect("Category", categories, default=['All'])
            
            # Sector filter
            sectors = ['All'] + sorted(df['sector'].dropna().unique().tolist())
            selected_sectors = st.multiselect("Sector", sectors, default=['All'])
            
            # Volume filter
            min_volume = st.number_input("Min 30-day Volume", min_value=0, value=50000, step=10000)
            
            # Price filter
            min_price = st.number_input("Min Price (₹)", min_value=0.0, value=0.0, step=10.0)
            
            # Wave stage filter
            wave_stages = ['All', 'Active', 'Forming', 'Danger', 'Inactive']
            selected_wave_stages = st.multiselect("Wave Stages", wave_stages, default=['Active', 'Forming'])
            
            # Analysis depth
            st.markdown("---")
            st.markdown("## ⚙️ Settings")
            top_n = st.slider("Top N Opportunities", min_value=5, max_value=50, value=20, step=5)
            
            # Quick insights
            st.markdown("---")
            st.markdown("## 💡 Quick Insights")
            if not df.empty:
                # Calculate quick metrics
                temp_df = df.copy()
                temp_df['vol_accel'] = temp_df.apply(calculate_volume_acceleration, axis=1)
                
                accel_count = len(temp_df[temp_df['vol_accel'] > 0.1])
                strong_accel = len(temp_df[temp_df['vol_accel'] > 0.2])
                
                if strong_accel > 0:
                    st.success(f"🚨 {strong_accel} stocks showing STRONG volume acceleration!")
                elif accel_count > 0:
                    st.warning(f"📈 {accel_count} stocks showing volume acceleration")
                else:
                    st.info("No significant volume acceleration detected")
                
                temp_df.drop('vol_accel', axis=1, inplace=True)
    
    # Main content
    if not df.empty:
        # Apply filters
        filters = {
            'categories': selected_categories,
            'sectors': selected_sectors,
            'min_volume': min_volume,
            'min_price': min_price,
            'wave_stages': selected_wave_stages
        }
        
        filtered_df = apply_filters(df, filters)
        
        # Get top opportunities
        analysis_df = get_top_opportunities(filtered_df, top_n)
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🏄 Active Waves", "🌊 Forming Waves", "🚨 Volume Acceleration", 
            "📊 Analytics", "📈 Reports", "🔧 Diagnostics"
        ])
        
        with tab1:
            st.markdown("## 🏄 Active Waves - Ride These Now!")
            
            active_waves = analysis_df[analysis_df['wave_type'] == 'Active'].sort_values('wave_score', ascending=False)
            
            if not active_waves.empty:
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Active Waves", len(active_waves))
                with col2:
                    st.metric("Avg Score", f"{active_waves['wave_score'].mean():.1f}")
                with col3:
                    st.metric("Best Score", f"{active_waves['wave_score'].max():.1f}")
                with col4:
                    st.metric("Avg 7d Return", f"{active_waves['ret_7d'].mean():.1f}%")
                
                st.markdown("---")
                
                # Display active waves
                for _, row in active_waves.iterrows():
                    with st.container():
                        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{row['ticker']}** - {row['company_name'][:40]}...")
                            st.markdown(f"<small>{row['sector']}</small>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"**Score: {row['wave_score']:.1f}**")
                            st.markdown(f"{row['wave_stage']}")
                        
                        with col3:
                            price = safe_float(row['price'], 0)
                            rvol = safe_float(row['rvol'], 0)
                            st.markdown(f"**₹{price:.2f}**")
                            st.markdown(f"Vol: {rvol:.1f}x")
                        
                        with col4:
                            ret_7d = safe_float(row['ret_7d'], 0)
                            ret_30d = safe_float(row['ret_30d'], 0)
                            color = "green" if ret_7d > 0 else "red"
                            st.markdown(f"<span style='color:{color}'>7d: {ret_7d:.1f}%</span>", 
                                      unsafe_allow_html=True)
                            st.markdown(f"30d: {ret_30d:.1f}%")
                        
                        with col5:
                            from_low = safe_float(row['from_low_pct'], 0)
                            confidence = safe_float(row['confidence'], 0)
                            target = max(40 - from_low, 10)
                            st.markdown(f"**Target: +{target:.0f}%**")
                            st.markdown(f"Confidence: {confidence:.0f}%")
                        
                        st.markdown("---")
            else:
                st.info("No active waves found with current filters")
        
        with tab2:
            st.markdown("## 🌊 Forming Waves - Prepare for Entry")
            
            forming_waves = analysis_df[analysis_df['wave_type'] == 'Forming'].sort_values('wave_score', ascending=False)
            
            if not forming_waves.empty:
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Forming Waves", len(forming_waves))
                with col2:
                    st.metric("Avg Score", f"{forming_waves['wave_score'].mean():.1f}")
                with col3:
                    st.metric("Best Setup", f"{forming_waves['wave_score'].max():.1f}")
                with col4:
                    st.metric("Avg Position", f"{forming_waves['from_low_pct'].mean():.0f}%")
                
                st.markdown("---")
                
                # Display forming waves
                for _, row in forming_waves.iterrows():
                    with st.container():
                        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{row['ticker']}** - {row['company_name'][:40]}...")
                            st.markdown(f"<small>{row['sector']}</small>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"**Score: {row['wave_score']:.1f}**")
                            st.markdown(f"{row['wave_stage']}")
                        
                        with col3:
                            price = safe_float(row['price'], 0)
                            vol_ratio = safe_float(row['vol_ratio_90d_180d'], 0)
                            st.markdown(f"**₹{price:.2f}**")
                            st.markdown(f"90/180: {vol_ratio:.2f}")
                        
                        with col4:
                            from_low = safe_float(row['from_low_pct'], 0)
                            st.markdown(f"From Low: {from_low:.0f}%")
                            st.markdown(f"Room: {100 - from_low:.0f}%")
                        
                        with col5:
                            vol_ratio = safe_float(row['vol_ratio_90d_180d'], 0)
                            pressure = vol_ratio * 100
                            st.markdown(f"**Pressure: {pressure:.0f}%**")
                            eta = "1-3 days" if pressure > 100 else "3-7 days"
                            st.markdown(f"ETA: {eta}")
                        
                        st.markdown("---")
            else:
                st.info("No forming waves found with current filters")
        
        with tab3:
            st.markdown("## 🚨 Volume Acceleration - Early Wave Detection")
            st.markdown("*Stocks showing accelerating volume patterns - institutional accumulation signals*")
            
            # Calculate volume acceleration for all stocks
            analysis_df['vol_acceleration'] = analysis_df.apply(calculate_volume_acceleration, axis=1)
            
            # Filter for positive acceleration
            accel_stocks = analysis_df[analysis_df['vol_acceleration'] > 0.05].sort_values('vol_acceleration', ascending=False)
            
            if not accel_stocks.empty:
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accelerating Stocks", len(accel_stocks))
                with col2:
                    st.metric("Max Acceleration", f"{accel_stocks['vol_acceleration'].max():.3f}")
                with col3:
                    st.metric("Avg Acceleration", f"{accel_stocks['vol_acceleration'].mean():.3f}")
                with col4:
                    strong_accel = len(accel_stocks[accel_stocks['vol_acceleration'] > 0.2])
                    st.metric("Strong Acceleration (>0.2)", strong_accel)
                
                st.markdown("---")
                
                # Explanation
                st.info("**Volume Acceleration** = (30d/90d ratio) - (30d/180d ratio). "
                       "Positive values indicate institutions are accelerating their buying.")
                
                # Display accelerating stocks
                for _, row in accel_stocks.head(20).iterrows():
                    with st.container():
                        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{row['ticker']}** - {row['company_name'][:40]}...")
                            st.markdown(f"<small>{row['sector']}</small>", unsafe_allow_html=True)
                        
                        with col2:
                            accel = row['vol_acceleration']
                            color = "green" if accel > 0.2 else "orange"
                            st.markdown(f"<span style='color:{color}'>**Accel: {accel:.3f}**</span>", 
                                      unsafe_allow_html=True)
                            st.markdown(f"Score: {row['wave_score']:.1f}")
                        
                        with col3:
                            vol_30d = safe_float(row['volume_30d'], 0)
                            vol_90d = safe_float(row['volume_90d'], 0)
                            ratio_30_90 = vol_30d / vol_90d if vol_90d > 0 else 0
                            st.markdown(f"30d/90d: {ratio_30_90:.2f}")
                            st.markdown(f"From Low: {safe_float(row['from_low_pct'], 0):.0f}%")
                        
                        with col4:
                            price = safe_float(row['price'], 0)
                            ret_7d = safe_float(row['ret_7d'], 0)
                            st.markdown(f"**₹{price:.2f}**")
                            color = "green" if ret_7d > 0 else "red"
                            st.markdown(f"<span style='color:{color}'>7d: {ret_7d:.1f}%</span>", 
                                      unsafe_allow_html=True)
                        
                        with col5:
                            if accel > 0.3:
                                st.markdown("**🔥 URGENT**")
                                st.markdown("Act Now")
                            elif accel > 0.2:
                                st.markdown("**⚡ HIGH**")
                                st.markdown("1-2 days")
                            elif accel > 0.1:
                                st.markdown("**📈 MEDIUM**")
                                st.markdown("3-5 days")
                            else:
                                st.markdown("**👀 WATCH**")
                                st.markdown("5-7 days")
                        
                        st.markdown("---")
                
                # Volume Acceleration Chart
                st.markdown("### 📊 Volume Acceleration Distribution")
                
                fig_accel = go.Figure()
                fig_accel.add_trace(go.Histogram(
                    x=accel_stocks['vol_acceleration'],
                    nbinsx=30,
                    marker_color='rgba(55, 128, 191, 0.7)',
                    marker_line_color='rgb(8,48,107)',
                    marker_line_width=1.5
                ))
                
                fig_accel.update_layout(
                    title="Distribution of Volume Acceleration Values",
                    xaxis_title="Volume Acceleration",
                    yaxis_title="Number of Stocks",
                    height=400,
                    showlegend=False
                )
                
                # Add vertical lines for thresholds
                fig_accel.add_vline(x=0.1, line_dash="dash", line_color="orange", 
                                   annotation_text="Medium", annotation_position="top")
                fig_accel.add_vline(x=0.2, line_dash="dash", line_color="red", 
                                   annotation_text="Strong", annotation_position="top")
                
                st.plotly_chart(fig_accel, use_container_width=True)
                
            else:
                st.info("No stocks showing significant volume acceleration at this time.")
        
        with tab4:
            st.markdown("## 📊 Market Analytics")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_wave_distribution_chart(analysis_df), use_container_width=True)
            
            with col2:
                st.plotly_chart(create_sector_heatmap(filtered_df), use_container_width=True)
            
            # Wave Score Analysis
            st.plotly_chart(create_wave_score_scatter(analysis_df), use_container_width=True)
            
            # Market Statistics
            st.markdown("### 📈 Market Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Safe comparison for SMA
                above_sma20 = 0
                for _, row in filtered_df.iterrows():
                    if safe_compare(row['price'], row['sma_20d'], 'gt'):
                        above_sma20 += 1
                
                total_stocks = len(filtered_df)
                pct_above = (above_sma20 / total_stocks * 100) if total_stocks > 0 else 0
                st.metric("Stocks Above SMA20", 
                         f"{above_sma20} ({pct_above:.1f}%)")
            
            with col2:
                # Safe comparison for rvol
                high_vol = 0
                for _, row in filtered_df.iterrows():
                    if safe_compare(row['rvol'], 2, 'gt'):
                        high_vol += 1
                
                st.metric("High Volume Stocks (rvol > 2)", f"{high_vol}")
            
            with col3:
                # Safe comparison for from_low_pct
                near_low = 0
                for _, row in filtered_df.iterrows():
                    if safe_compare(row['from_low_pct'], 20, 'lt'):
                        near_low += 1
                
                st.metric("Near 52w Low (<20%)", f"{near_low}")
            
            with col4:
                # Safe comparison for from_high_pct
                near_high = 0
                for _, row in filtered_df.iterrows():
                    if safe_compare(row['from_high_pct'], -10, 'gt'):
                        near_high += 1
                
                st.metric("Near 52w High (>-10%)", f"{near_high}")
        
        with tab5:
            st.markdown("## 📈 Download Reports")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Generate Excel Report", type="primary"):
                    with st.spinner("Generating report..."):
                        excel_file = generate_excel_report(filtered_df, analysis_df)
                        
                        st.download_button(
                            label="📥 Download Excel Report",
                            data=excel_file,
                            file_name=f"wave_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            
            with col2:
                if st.button("📄 Generate PDF Summary", type="secondary"):
                    st.info("PDF generation coming soon!")
            
            # Quick summary
            st.markdown("### 📋 Quick Summary")
            
            # Calculate volume acceleration if needed
            if 'vol_acceleration' not in analysis_df.columns:
                analysis_df['vol_acceleration'] = analysis_df.apply(calculate_volume_acceleration, axis=1)
            
            accel_stocks = analysis_df[analysis_df['vol_acceleration'] > 0.1]
            
            summary_text = f"""
            **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
            
            **Stocks Analyzed:** {len(filtered_df)}
            **Volume Accelerating:** {len(accel_stocks)}
            
            **Top 5 Opportunities:**
            """
            
            for i, (_, row) in enumerate(analysis_df.head(5).iterrows(), 1):
                vol_accel = row.get('vol_acceleration', 0)
                accel_text = f" | Accel: {vol_accel:.3f}" if vol_accel > 0.05 else ""
                summary_text += f"\n{i}. **{row['ticker']}** (Score: {row['wave_score']:.1f}) - {row['wave_stage']}{accel_text}"
            
            st.text_area("Summary", summary_text, height=300)
        
        with tab6:
            st.markdown("## 🔧 System Diagnostics")
            
            diagnostics = generate_diagnostic_report(df)
            
            # Data Quality
            st.markdown("### 📊 Data Quality")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Rows", f"{diagnostics['data_quality']['total_rows']:,}")
            with col2:
                st.metric("Total Columns", diagnostics['data_quality']['total_columns'])
            with col3:
                st.metric("Missing Values", f"{diagnostics['data_quality']['missing_values']:,}")
            with col4:
                st.metric("Duplicate Tickers", diagnostics['data_quality']['duplicate_tickers'])
            
            # Coverage
            st.markdown("### 📈 Market Coverage")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Unique Sectors", diagnostics['coverage']['sectors'])
                st.metric("Price Range", diagnostics['coverage']['price_range'])
            
            with col2:
                categories_text = "\n".join([f"{k}: {v}" for k, v in diagnostics['coverage']['categories'].items()])
                st.text_area("Category Distribution", categories_text, height=150)
            
            # Wave Analysis Stats
            st.markdown("### 🌊 Wave Analysis Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("High Volume Stocks", diagnostics['wave_analysis']['high_rvol_stocks'])
                st.metric("Volume Accelerating", diagnostics['wave_analysis']['volume_accelerating'])
            
            with col2:
                st.metric("Accumulation Candidates", diagnostics['wave_analysis']['accumulation_candidates'])
                st.metric("Strong Acceleration", diagnostics['wave_analysis']['strong_acceleration'])
            
            with col3:
                st.metric("Momentum Stocks", diagnostics['wave_analysis']['momentum_stocks'])
                st.metric("Oversold Stocks", diagnostics['wave_analysis']['oversold_stocks'])
            
            # System Info
            st.markdown("### 💻 System Information")
            st.markdown(f"<p class='diagnostic-info'>Last Update: {diagnostics['data_quality']['data_freshness']}</p>", 
                       unsafe_allow_html=True)
            st.markdown(f"<p class='diagnostic-info'>Data Source: Google Sheets (GID: {gid})</p>", 
                       unsafe_allow_html=True)
            st.markdown(f"<p class='diagnostic-info'>Version: 1.0.0</p>", 
                       unsafe_allow_html=True)
    
    else:
        st.error("No data loaded. Please check your Google Sheets URL and GID.")

if __name__ == "__main__":
    main()
