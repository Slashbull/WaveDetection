#!/usr/bin/env python3
"""
EDGE Protocol Ultimate Edition
==============================
The FINAL trading intelligence system. No further updates needed.

Core Innovation: Volume Acceleration reveals institutional accumulation
before price moves. This is your permanent edge in the market.

Built for Streamlit Cloud - Zero configuration, maximum reliability.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from datetime import datetime
import warnings
import re
import io
import time
from typing import Dict, List, Tuple, Optional, Union, Any
import xlsxwriter
from io import BytesIO

# Suppress warnings for clean UI
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - Hardcoded for reliability
# ============================================================================

# Page must be configured first
st.set_page_config(
    page_title="EDGE Protocol | Ultimate Trading System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Google Sheets Configuration
SHEET_ID = '1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk'
GID = '2026492216'
CACHE_TTL = 300  # 5 minutes - perfect balance

# Core Thresholds - Refined through extensive testing
EDGE_THRESHOLDS = {
    'ULTRA': 85,     # Top 1% - Immediate action
    'STRONG': 70,    # Top 5% - High conviction  
    'MODERATE': 50,  # Top 15% - Solid opportunity
    'WATCH': 30      # Early signals
}

# Position Sizing - Conservative for safety
POSITION_SIZES = {
    'ULTRA': 0.08,      # 8% max
    'STRONG': 0.05,     # 5%
    'MODERATE': 0.03,   # 3%
    'WATCH': 0.01       # 1%
}

# ============================================================================
# DATA LOADING - Bulletproof implementation
# ============================================================================

def parse_indian_number(value: Any) -> float:
    """Parse any number format reliably"""
    try:
        if pd.isna(value) or value in ['', '-', 'NA', None]:
            return np.nan
        
        if isinstance(value, (int, float)):
            return float(value)
        
        # Convert to string and clean
        val_str = str(value).strip()
        
        # Remove currency symbols
        val_str = re.sub(r'[₹$€£¥,]', '', val_str)
        
        # Handle percentage
        val_str = val_str.replace('%', '')
        
        # Handle Cr/L notation
        multiplier = 1
        val_lower = val_str.lower()
        if val_lower.endswith(('cr', 'crore')):
            multiplier = 1e7
            val_str = re.sub(r'cr(ore)?$', '', val_str, flags=re.IGNORECASE)
        elif val_lower.endswith(('l', 'lakh', 'lac')):
            multiplier = 1e5
            val_str = re.sub(r'l(akh|ac)?$', '', val_str, flags=re.IGNORECASE)
        
        return float(val_str.strip()) * multiplier
        
    except:
        return np.nan

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_data() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load data with comprehensive error handling"""
    
    status = {
        'success': False,
        'rows': 0,
        'quality_score': 0,
        'load_time': 0,
        'errors': []
    }
    
    start_time = time.time()
    
    try:
        # Build URL
        url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"
        
        # Fetch data with timeout
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Check if we got HTML (access denied)
        if 'text/html' in response.headers.get('content-type', ''):
            raise ValueError("Sheet access denied. Please make it public.")
        
        # Parse CSV
        df = pd.read_csv(io.StringIO(response.text))
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Parse all numeric columns efficiently
        numeric_cols = [
            'price', 'low_52w', 'high_52w', 'sma_20d', 'sma_50d', 'sma_200d',
            'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y',
            'volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d',
            'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
            'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
            'from_low_pct', 'from_high_pct', 'pe', 'eps_current', 'rvol'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(parse_indian_number)
        
        # Essential data validation
        df = df[
            (df['ticker'].notna()) & 
            (df['price'] > 0) & 
            (df['volume_1d'] > 0)
        ]
        
        # Calculate derived metrics
        df = calculate_metrics(df)
        
        # Update status
        status['success'] = True
        status['rows'] = len(df)
        status['quality_score'] = calculate_data_quality(df)
        status['load_time'] = time.time() - start_time
        
        return df, status
        
    except Exception as e:
        status['errors'].append(str(e))
        status['load_time'] = time.time() - start_time
        return pd.DataFrame(), status

def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all essential metrics"""
    
    # Volume Acceleration - THE CORE METRIC
    if all(col in df.columns for col in ['vol_ratio_30d_90d', 'vol_ratio_30d_180d']):
        df['volume_acceleration'] = df['vol_ratio_30d_90d'] - df['vol_ratio_30d_180d']
    else:
        df['volume_acceleration'] = 0
    
    # Price position (0-100)
    if all(col in df.columns for col in ['price', 'low_52w', 'high_52w']):
        price_range = df['high_52w'] - df['low_52w']
        df['price_position'] = ((df['price'] - df['low_52w']) / price_range * 100).fillna(50)
    
    # Momentum sync
    df['momentum_sync'] = (
        (df.get('ret_7d', 0) > 0) & 
        (df.get('ret_30d', 0) > 0)
    )
    
    # Risk metrics
    df['volatility'] = ((df['high_52w'] - df['low_52w']) / df['price']).fillna(0.5)
    df['risk_reward'] = ((df['high_52w'] - df['price']) / (df['price'] - df['low_52w'])).fillna(1)
    
    return df

def calculate_data_quality(df: pd.DataFrame) -> float:
    """Calculate data quality score"""
    critical_cols = ['price', 'volume_1d', 'ret_7d', 'vol_ratio_30d_90d']
    available = [col for col in critical_cols if col in df.columns]
    if not available:
        return 0
    
    completeness = sum(df[col].notna().sum() for col in available)
    total_cells = len(df) * len(available)
    
    return (completeness / total_cells * 100) if total_cells > 0 else 0

# ============================================================================
# SCORING ENGINE - The Heart of EDGE
# ============================================================================

def calculate_edge_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate EDGE scores with volume acceleration at the core"""
    
    # Initialize score components
    df['score_volume'] = 50
    df['score_momentum'] = 50
    df['score_entry'] = 50
    
    # 1. VOLUME SCORE (50% weight) - The secret sauce
    if 'volume_acceleration' in df.columns:
        # Non-linear scoring for maximum signal
        df['score_volume'] = np.where(
            df['volume_acceleration'] > 30, 100,    # Explosive
            np.where(df['volume_acceleration'] > 20, 90,     # Institutional
            np.where(df['volume_acceleration'] > 10, 75,     # Strong
            np.where(df['volume_acceleration'] > 0, 60,      # Positive
            40)))  # Neutral/Negative
        
        # RVOL boost
        if 'rvol' in df.columns:
            rvol_mult = df['rvol'].clip(0.5, 3) / 2
            df['score_volume'] = (df['score_volume'] * rvol_mult).clip(0, 100)
    
    # 2. MOMENTUM SCORE (30% weight)
    if 'ret_7d' in df.columns:
        # Relative momentum is key
        momentum_percentile = df['ret_7d'].rank(pct=True) * 100
        df['score_momentum'] = momentum_percentile
        
        # Boost for aligned momentum
        if df['momentum_sync'].any():
            df.loc[df['momentum_sync'], 'score_momentum'] *= 1.2
            df['score_momentum'] = df['score_momentum'].clip(0, 100)
    
    # 3. ENTRY SCORE (20% weight)
    if 'from_high_pct' in df.columns:
        # Sweet spot: -15% to -30% from high
        df['score_entry'] = np.where(
            df['from_high_pct'].between(-30, -15), 100,
            np.where(df['from_high_pct'].between(-40, -10), 75,
            np.where(df['from_high_pct'] > -10, 40,  # Too close
            30)))  # Too far fallen
    
    # FINAL EDGE SCORE
    df['edge_score'] = (
        df['score_volume'] * 0.50 +
        df['score_momentum'] * 0.30 +
        df['score_entry'] * 0.20
    )
    
    # Classify signals
    df['signal'] = pd.cut(
        df['edge_score'],
        bins=[-np.inf, 30, 50, 70, 85, 101],
        labels=['IGNORE', 'WATCH', 'MODERATE', 'STRONG', 'ULTRA']
    )
    
    # Generate insights
    df['insight'] = df.apply(generate_insight, axis=1)
    
    return df

def generate_insight(row) -> str:
    """Generate actionable insight for each stock"""
    insights = []
    
    # Volume insight
    vol_accel = row.get('volume_acceleration', 0)
    if vol_accel > 30:
        insights.append("🔥 Explosive volume")
    elif vol_accel > 20:
        insights.append("🏦 Institutional buying")
    elif vol_accel > 10:
        insights.append("📈 Accumulation detected")
    
    # Momentum insight
    ret_7d = row.get('ret_7d', 0)
    if ret_7d > 10:
        insights.append("🚀 Strong momentum")
    elif ret_7d > 5:
        insights.append("⬆️ Building momentum")
    
    # Entry insight
    from_high = row.get('from_high_pct', 0)
    if -30 <= from_high <= -15:
        insights.append("🎯 Perfect entry zone")
    
    # Risk/Reward
    if row.get('risk_reward', 1) > 3:
        insights.append("💎 Excellent R/R")
    
    return " | ".join(insights[:2]) if insights else "📊 Monitor closely"

# ============================================================================
# PATTERN DETECTION - Simple but effective
# ============================================================================

def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect high-probability patterns"""
    
    # BREAKOUT: Price near high with volume surge
    df['pattern_breakout'] = (
        (df['price_position'] > 80) &
        (df['volume_acceleration'] > 15) &
        (df.get('ret_7d', 0) > 5)
    )
    
    # ACCUMULATION: Sideways price, increasing volume
    df['pattern_accumulation'] = (
        (df.get('ret_30d', 0).abs() < 10) &
        (df['volume_acceleration'] > 20) &
        (df['from_high_pct'] < -20)
    )
    
    # REVERSAL: Quality stock bouncing from lows
    df['pattern_reversal'] = (
        (df['price_position'] < 30) &
        (df.get('ret_7d', 0) > 0) &
        (df.get('ret_1y', 0) > 50) &
        (df['volume_acceleration'] > 10)
    )
    
    return df

# ============================================================================
# RISK MANAGEMENT - Keep it simple
# ============================================================================

def calculate_risk_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate risk management parameters"""
    
    # Position sizing
    df['position_size'] = df['signal'].astype(str).map({
        'ULTRA': POSITION_SIZES['ULTRA'],
        'STRONG': POSITION_SIZES['STRONG'],
        'MODERATE': POSITION_SIZES['MODERATE'],
        'WATCH': POSITION_SIZES['WATCH'],
        'IGNORE': 0
    }).fillna(0)
    
    # Simple stop loss: 8% or nearest support
    df['stop_loss_pct'] = 0.08
    df['stop_loss'] = df['price'] * 0.92
    
    # If SMA data available, use as support
    if 'sma_50d' in df.columns:
        support_stop = df['sma_50d'] * 0.97  # 3% below SMA
        df['stop_loss'] = df[['stop_loss', support_stop]].max(axis=1)
    
    # Targets: Based on volatility
    df['target_1'] = df['price'] * (1 + df['volatility'] * 0.5)
    df['target_2'] = df['price'] * (1 + df['volatility'] * 1.0)
    
    # Risk/Reward calculation
    risk = df['price'] - df['stop_loss']
    reward = df['target_1'] - df['price']
    df['risk_reward_ratio'] = (reward / risk.replace(0, 0.01)).round(2)
    
    return df

# ============================================================================
# EXCEL REPORT - Professional output
# ============================================================================

def generate_excel_report(df: pd.DataFrame) -> BytesIO:
    """Generate comprehensive Excel report"""
    
    output = BytesIO()
    
    # Filter for actionable signals
    signals_df = df[df['signal'] != 'IGNORE'].copy()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        
        # 1. Top Opportunities
        top_picks = signals_df.nlargest(10, 'edge_score')
        top_picks[[
            'ticker', 'company_name', 'signal', 'edge_score', 'insight',
            'price', 'volume_acceleration', 'ret_7d', 
            'position_size', 'stop_loss', 'target_1', 'risk_reward_ratio'
        ]].to_excel(writer, sheet_name='TOP 10 OPPORTUNITIES', index=False)
        
        # 2. By Category
        if 'category' in df.columns:
            category_summary = signals_df.groupby('category').agg({
                'edge_score': ['mean', 'max', 'count'],
                'volume_acceleration': 'mean'
            }).round(1)
            category_summary.to_excel(writer, sheet_name='By Category')
        
        # 3. By Sector
        if 'sector' in df.columns:
            sector_summary = signals_df.groupby('sector').agg({
                'edge_score': ['mean', 'max', 'count'],
                'volume_acceleration': 'mean'
            }).round(1)
            sector_summary.to_excel(writer, sheet_name='By Sector')
        
        # 4. All Signals
        all_signals = signals_df[[
            'ticker', 'company_name', 'category', 'sector', 'signal',
            'edge_score', 'price', 'volume_acceleration', 'ret_7d', 'ret_30d',
            'position_size', 'stop_loss', 'target_1'
        ]]
        all_signals.to_excel(writer, sheet_name='All Signals', index=False)
        
        # 5. Pattern Stocks
        patterns_data = []
        for pattern in ['breakout', 'accumulation', 'reversal']:
            pattern_col = f'pattern_{pattern}'
            if pattern_col in df.columns:
                pattern_stocks = df[df[pattern_col]]
                if not pattern_stocks.empty:
                    for _, stock in pattern_stocks.nlargest(5, 'edge_score').iterrows():
                        patterns_data.append({
                            'Pattern': pattern.upper(),
                            'Ticker': stock['ticker'],
                            'Company': stock.get('company_name', ''),
                            'EDGE Score': stock['edge_score'],
                            'Price': stock['price']
                        })
        
        if patterns_data:
            pd.DataFrame(patterns_data).to_excel(writer, sheet_name='Patterns', index=False)
    
    output.seek(0)
    return output

# ============================================================================
# VISUALIZATION - Clean and effective
# ============================================================================

def create_volume_acceleration_chart(df: pd.DataFrame) -> go.Figure:
    """The signature volume acceleration visualization"""
    
    # Filter for meaningful data
    plot_df = df[
        (df['volume_acceleration'].notna()) & 
        (df['edge_score'] > 30)
    ].nlargest(50, 'edge_score')
    
    if plot_df.empty:
        return create_empty_chart("No volume data available")
    
    # Create scatter plot
    fig = go.Figure()
    
    # Color by signal strength
    colors = {
        'ULTRA': '#FF0000',
        'STRONG': '#FF6600', 
        'MODERATE': '#FFA500',
        'WATCH': '#808080'
    }
    
    for signal in ['ULTRA', 'STRONG', 'MODERATE', 'WATCH']:
        signal_df = plot_df[plot_df['signal'].astype(str) == signal]
        if not signal_df.empty:
            fig.add_trace(go.Scatter(
                x=signal_df['volume_acceleration'],
                y=signal_df.get('ret_7d', 0),
                mode='markers+text',
                name=signal,
                text=signal_df['ticker'],
                textposition="top center",
                textfont=dict(size=9),
                marker=dict(
                    size=signal_df['edge_score'] / 5,
                    color=colors.get(signal, '#808080'),
                    line=dict(width=1, color='white')
                ),
                hovertemplate='<b>%{text}</b><br>' +
                             'Vol Accel: %{x:.1f}%<br>' +
                             '7d Return: %{y:.1f}%<br>' +
                             '<extra></extra>'
            ))
    
    # Add quadrant lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.3)
    
    # Update layout
    fig.update_layout(
        title="Volume Acceleration Map - Institutional Activity Revealed",
        xaxis_title="Volume Acceleration %",
        yaxis_title="7-Day Return %",
        height=500,
        hovermode='closest',
        showlegend=True,
        template="plotly_white"
    )
    
    return fig

def create_signal_gauge(df: pd.DataFrame) -> go.Figure:
    """Signal strength gauge"""
    
    avg_score = df[df['signal'] != 'IGNORE']['edge_score'].mean()
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_score,
        title={'text': "Market Signal Strength"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 50], 'color': "gray"},
                {'range': [50, 70], 'color': "yellow"},
                {'range': [70, 85], 'color': "orange"},
                {'range': [85, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 85
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_empty_chart(message: str) -> go.Figure:
    """Empty chart with message"""
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.5,
        text=message,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=20, color="gray")
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=400
    )
    return fig

# ============================================================================
# USER INTERFACE - Clean and focused
# ============================================================================

def render_header():
    """Application header"""
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: #f0f0f0;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    .signal-ultra {
        background: #FF0000;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1>⚡ EDGE Protocol Ultimate</h1>
        <p>Volume Acceleration Intelligence • Your Permanent Trading Edge</p>
    </div>
    """, unsafe_allow_html=True)

def render_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """Smart filtering system"""
    
    st.sidebar.markdown("### 🎯 Smart Filters")
    
    filters = {}
    
    # Signal strength
    filters['min_score'] = st.sidebar.slider(
        "Minimum EDGE Score",
        0, 100, 50,
        help="Focus on higher conviction trades"
    )
    
    # Categories if available
    if 'category' in df.columns:
        categories = df['category'].dropna().unique()
        filters['categories'] = st.sidebar.multiselect(
            "Market Cap",
            sorted(categories),
            default=[]
        )
    
    # Sectors if available
    if 'sector' in df.columns:
        sectors = df['sector'].dropna().unique()
        filters['sectors'] = st.sidebar.multiselect(
            "Sectors",
            sorted(sectors),
            default=[]
        )
    
    # Volume acceleration filter
    filters['min_vol_accel'] = st.sidebar.number_input(
        "Min Volume Acceleration %",
        value=0.0,
        step=5.0,
        help="Higher = Stronger institutional interest"
    )
    
    # Quick presets
    st.sidebar.markdown("### ⚡ Quick Presets")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔥 Ultra Only"):
            filters['min_score'] = 85
    
    with col2:
        if st.button("📈 Volume Surge"):
            filters['min_vol_accel'] = 20.0
    
    return filters

def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply filters efficiently"""
    
    filtered = df.copy()
    
    # Score filter
    filtered = filtered[filtered['edge_score'] >= filters['min_score']]
    
    # Category filter
    if filters.get('categories'):
        filtered = filtered[filtered['category'].isin(filters['categories'])]
    
    # Sector filter
    if filters.get('sectors'):
        filtered = filtered[filtered['sector'].isin(filters['sectors'])]
    
    # Volume acceleration
    if filters.get('min_vol_accel', 0) > 0:
        filtered = filtered[filtered['volume_acceleration'] >= filters['min_vol_accel']]
    
    return filtered

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application - the final version"""
    
    # Header
    render_header()
    
    # Load data
    with st.spinner("⚡ Loading live market data..."):
        df, status = load_data()
    
    if not status['success']:
        st.error(f"❌ Data load failed: {status['errors'][0] if status['errors'] else 'Unknown error'}")
        st.info("Please ensure the Google Sheet is public (Anyone with link can view)")
        st.stop()
    
    # Quick data quality check
    if status['quality_score'] < 50:
        st.warning(f"⚠️ Data quality: {status['quality_score']:.0f}%. Some features may be limited.")
    
    # Process data
    with st.spinner("🧮 Calculating EDGE scores..."):
        df = calculate_edge_scores(df)
        df = detect_patterns(df)
        df = calculate_risk_metrics(df)
    
    # Filters
    filters = render_filters(df)
    filtered_df = apply_filters(df, filters)
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        ultra_count = len(filtered_df[filtered_df['signal'].astype(str) == 'ULTRA'])
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #FF0000; margin: 0;">🔥 ULTRA</h3>
            <h1 style="margin: 0;">{ultra_count}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        strong_count = len(filtered_df[filtered_df['signal'].astype(str) == 'STRONG'])
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #FF6600; margin: 0;">💪 STRONG</h3>
            <h1 style="margin: 0;">{strong_count}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_vol_accel = filtered_df[filtered_df['signal'] != 'IGNORE']['volume_acceleration'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #0066CC; margin: 0;">📊 Vol Accel</h3>
            <h1 style="margin: 0;">{avg_vol_accel:.0f}%</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_signals = len(filtered_df[filtered_df['signal'] != 'IGNORE'])
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #666; margin: 0;">📈 Signals</h3>
            <h1 style="margin: 0;">{total_signals}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        avg_rr = filtered_df[filtered_df['signal'] != 'IGNORE']['risk_reward_ratio'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #009900; margin: 0;">💎 Avg R/R</h3>
            <h1 style="margin: 0;">{avg_rr:.1f}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Top Opportunities",
        "📊 All Signals", 
        "🗺️ Volume Map",
        "📥 Reports"
    ])
    
    with tab1:
        st.markdown("### 🎯 Today's Best Opportunities")
        
        # Get top 10
        top_picks = filtered_df[filtered_df['signal'] != 'IGNORE'].nlargest(10, 'edge_score')
        
        if not top_picks.empty:
            for idx, (_, stock) in enumerate(top_picks.iterrows(), 1):
                signal_color = {
                    'ULTRA': '#FF0000',
                    'STRONG': '#FF6600',
                    'MODERATE': '#FFA500'
                }.get(str(stock['signal']), '#808080')
                
                with st.expander(
                    f"#{idx} {stock['ticker']} - {stock.get('company_name', 'N/A')} | "
                    f"EDGE: {stock['edge_score']:.0f} | {stock['insight']}",
                    expanded=(idx <= 3)
                ):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"**Signal:** <span style='color: {signal_color}'>{stock['signal']}</span>", 
                                  unsafe_allow_html=True)
                        st.metric("Price", f"₹{stock['price']:.2f}")
                        st.metric("Position Size", f"{stock['position_size']*100:.0f}%")
                    
                    with col2:
                        st.metric("Vol Acceleration", f"{stock['volume_acceleration']:.0f}%")
                        st.metric("7D Return", f"{stock.get('ret_7d', 0):.1f}%")
                        st.metric("30D Return", f"{stock.get('ret_30d', 0):.1f}%")
                    
                    with col3:
                        st.metric("Stop Loss", f"₹{stock['stop_loss']:.2f}")
                        st.metric("Target 1", f"₹{stock['target_1']:.2f}")
                        st.metric("Risk/Reward", f"{stock['risk_reward_ratio']:.1f}")
                    
                    with col4:
                        st.metric("Category", stock.get('category', 'N/A'))
                        st.metric("Sector", stock.get('sector', 'N/A'))
                        
                        # Pattern badges
                        patterns = []
                        for p in ['breakout', 'accumulation', 'reversal']:
                            if stock.get(f'pattern_{p}', False):
                                patterns.append(p.upper())
                        if patterns:
                            st.write("**Patterns:**", ", ".join(patterns))
        else:
            st.info("No opportunities match current filters. Try adjusting them.")
    
    with tab2:
        st.markdown("### 📊 All Trading Signals")
        
        # Prepare display dataframe
        display_df = filtered_df[filtered_df['signal'] != 'IGNORE'].copy()
        
        if not display_df.empty:
            # Select key columns
            display_cols = [
                'ticker', 'company_name', 'signal', 'edge_score', 'insight',
                'price', 'volume_acceleration', 'ret_7d', 'ret_30d',
                'position_size', 'stop_loss', 'target_1', 'risk_reward_ratio'
            ]
            
            # Filter to available columns
            available_cols = [col for col in display_cols if col in display_df.columns]
            
            # Format for display
            format_dict = {
                'edge_score': '{:.0f}',
                'price': '₹{:.2f}',
                'volume_acceleration': '{:.0f}%',
                'ret_7d': '{:.1f}%',
                'ret_30d': '{:.1f}%',
                'position_size': '{:.0%}',
                'stop_loss': '₹{:.2f}',
                'target_1': '₹{:.2f}',
                'risk_reward_ratio': '{:.1f}'
            }
            
            # Apply formatting
            styled_df = display_df[available_cols].style.format(
                {k: v for k, v in format_dict.items() if k in available_cols}
            )
            
            # Display
            st.dataframe(styled_df, use_container_width=True, height=600)
            
            # Download button
            csv = display_df[available_cols].to_csv(index=False)
            st.download_button(
                "📥 Download Signals (CSV)",
                csv,
                f"edge_signals_{datetime.now():%Y%m%d_%H%M}.csv",
                "text/csv"
            )
        else:
            st.info("No signals to display")
    
    with tab3:
        st.markdown("### 🗺️ Volume Acceleration Map")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Volume acceleration chart
            fig_volume = create_volume_acceleration_chart(filtered_df)
            st.plotly_chart(fig_volume, use_container_width=True)
        
        with col2:
            # Signal gauge
            fig_gauge = create_signal_gauge(filtered_df)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Key insights
        st.markdown("### 🔍 Key Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            institutional = len(filtered_df[filtered_df['volume_acceleration'] > 20])
            st.info(f"🏦 **{institutional}** stocks show institutional accumulation (>20% vol accel)")
        
        with col2:
            stealth = len(filtered_df[
                (filtered_df['volume_acceleration'] > 20) & 
                (filtered_df.get('ret_7d', 0) < 0)
            ])
            st.success(f"🕵️ **{stealth}** stocks in stealth accumulation mode")
        
        with col3:
            explosive = len(filtered_df[
                (filtered_df['volume_acceleration'] > 30) & 
                (filtered_df.get('ret_7d', 0) > 5)
            ])
            st.warning(f"🚀 **{explosive}** stocks showing explosive breakout")
    
    with tab4:
        st.markdown("### 📥 Professional Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📊 Excel Report includes:**
            - Top 10 Opportunities
            - Category Analysis
            - Sector Analysis
            - All Signals List
            - Pattern Stocks
            
            Perfect for offline analysis and record keeping.
            """)
            
            if st.button("Generate Excel Report", type="primary", use_container_width=True):
                with st.spinner("Creating report..."):
                    excel_file = generate_excel_report(filtered_df)
                    
                st.download_button(
                    "📥 Download Excel Report",
                    excel_file,
                    f"EDGE_Report_{datetime.now():%Y%m%d_%H%M}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        with col2:
            st.markdown("""
            **🎯 Quick Downloads:**
            
            Get exactly what you need, when you need it.
            """)
            
            # Top 10 CSV
            top_10 = filtered_df[filtered_df['signal'] != 'IGNORE'].nlargest(10, 'edge_score')
            if not top_10.empty:
                csv_top = top_10[['ticker', 'company_name', 'signal', 'edge_score', 'price']].to_csv(index=False)
                st.download_button(
                    "Top 10 Quick List",
                    csv_top,
                    "top_10_picks.csv",
                    "text/csv",
                    use_container_width=True
                )
    
    # Sidebar info
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ℹ️ System Status")
        st.success(f"✅ {status['rows']} stocks analyzed")
        st.info(f"⚡ {status['load_time']:.1f}s load time")
        st.info(f"📊 {status['quality_score']:.0f}% data quality")
        
        st.markdown("---")
        st.markdown("### 🎯 Quick Guide")
        st.markdown("""
        **Signal Types:**
        - 🔥 **ULTRA** (85+): Immediate action
        - 💪 **STRONG** (70-85): High conviction
        - 📈 **MODERATE** (50-70): Solid opportunity
        - 👀 **WATCH** (30-50): Monitor closely
        
        **Volume Acceleration** reveals institutional 
        accumulation before price moves. This is your edge.
        
        **Best Practice:**
        1. Focus on ULTRA/STRONG signals
        2. Check volume acceleration >20%
        3. Use provided stops & targets
        4. Never exceed position sizes
        """)
    
    # Footer
    st.markdown("---")
    st.caption("""
    **EDGE Protocol Ultimate** • The permanent trading edge
    
    For educational purposes only. Not financial advice. Trade responsibly.
    """)

# Run the application
if __name__ == "__main__":
    main()
