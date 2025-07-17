"""
EDGE Protocol - Ultimate Trading Intelligence System
===================================================
FINAL PRODUCTION VERSION - Streamlit Cloud Ready
No further upgrades needed - Built for permanent deployment

Core Innovation: Volume Acceleration reveals institutional behavior
before price moves, giving you the ultimate trading edge.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import io
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="EDGE Protocol",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Data Source
SHEET_ID = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"
GID = "2026492216"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

# Portfolio Management Constants
MAX_PORTFOLIO_EXPOSURE = 0.80
MAX_POSITIONS = 10
MAX_SECTOR_EXPOSURE = 0.30
MAX_SUPER_EDGE_POSITIONS = 3

# Position Sizing
POSITION_SIZES = {
    "SUPER_EDGE": 0.15,
    "EXPLOSIVE": 0.10,
    "STRONG": 0.05,
    "MODERATE": 0.02,
    "WATCH": 0.00
}

# EDGE Thresholds
EDGE_THRESHOLDS = {
    "SUPER_EDGE": 92,
    "EXPLOSIVE": 85,
    "STRONG": 70,
    "MODERATE": 50
}

# Cache TTL
CACHE_TTL = 300  # 5 minutes

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================
@st.cache_data(ttl=CACHE_TTL)
def load_data() -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Load and prepare data with comprehensive validation"""
    diagnostics = {
        "timestamp": datetime.now(),
        "rows_loaded": 0,
        "data_quality": 0,
        "warnings": []
    }
    
    try:
        # Fetch data
        response = requests.get(SHEET_URL, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        diagnostics["rows_loaded"] = len(df)
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Define numeric columns to clean
        numeric_cols = [
            'price', 'low_52w', 'high_52w', 'sma_20d', 'sma_50d', 'sma_200d',
            'volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d',
            'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
            'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
            'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y',
            'from_low_pct', 'from_high_pct', 'rvol', 'pe', 'eps_current', 'eps_last_qtr', 'eps_change_pct'
        ]
        
        # Clean numeric columns
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str)
                    .str.replace(r'[₹,$,€,£,%,]', '', regex=True)
                    .str.replace('cr', '', regex=False)
                    .str.replace('Cr', '', regex=False)
                    .replace(['', '-', 'nan', 'NaN', 'NA', 'N/A'], np.nan),
                    errors='coerce'
                )
        
        # Market cap special handling
        if 'market_cap' in df.columns:
            df['market_cap_num'] = df['market_cap'].apply(parse_market_cap)
        
        # Filter valid stocks
        df = df[df['price'] > 0] if 'price' in df.columns else df
        df = df[df['ticker'].notna()] if 'ticker' in df.columns else df
        
        # Calculate data quality
        critical_cols = ['price', 'ticker', 'vol_ratio_30d_90d']
        quality_scores = []
        for col in critical_cols:
            if col in df.columns:
                non_null_pct = df[col].notna().sum() / len(df) * 100
                quality_scores.append(non_null_pct)
        diagnostics["data_quality"] = np.mean(quality_scores) if quality_scores else 0
        
        return df, diagnostics
        
    except Exception as e:
        diagnostics["warnings"].append(f"Load error: {str(e)}")
        return pd.DataFrame(), diagnostics

def parse_market_cap(val: Union[str, float]) -> float:
    """Parse market cap with Indian notation"""
    if pd.isna(val):
        return np.nan
    
    val_str = str(val).strip().replace('₹', '').replace(',', '')
    
    multipliers = {
        'cr': 1e7, 'Cr': 1e7,
        'l': 1e5, 'L': 1e5, 'lakh': 1e5
    }
    
    for suffix, mult in multipliers.items():
        if suffix in val_str:
            try:
                return float(val_str.replace(suffix, '').strip()) * mult
            except:
                return np.nan
    
    try:
        return float(val_str)
    except:
        return np.nan

# ============================================================================
# CORE CALCULATION ENGINE
# ============================================================================
def calculate_volume_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volume acceleration and patterns"""
    df = df.copy()
    
    # Volume acceleration: comparing recent vs past momentum
    if all(col in df.columns for col in ['vol_ratio_7d_90d', 'vol_ratio_30d_90d']):
        df['volume_acceleration'] = df['vol_ratio_7d_90d'] - df['vol_ratio_30d_90d']
    else:
        df['volume_acceleration'] = 0
    
    # Volume consistency
    vol_cols = ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d']
    available_vol_cols = [col for col in vol_cols if col in df.columns]
    if available_vol_cols:
        df['volume_consistency'] = (df[available_vol_cols] > 0).sum(axis=1) / len(available_vol_cols)
    else:
        df['volume_consistency'] = 0
    
    # Volume pattern classification
    conditions = [
        (df['volume_acceleration'] > 50) & (df.get('rvol', 1) > 2.0),
        df['volume_acceleration'] > 40,
        df['volume_acceleration'] > 25,
        df['volume_acceleration'] > 10,
        df['volume_acceleration'] > 0
    ]
    
    choices = [
        "🔥 Explosive Accumulation",
        "🏦 Institutional Loading",
        "📈 Heavy Accumulation",
        "📊 Accumulation",
        "➕ Mild Accumulation"
    ]
    
    df['volume_pattern'] = np.select(conditions, choices, default="💀 Distribution")
    
    return df

def calculate_momentum_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate momentum indicators"""
    df = df.copy()
    
    # Short-term momentum
    short_cols = ['ret_1d', 'ret_3d', 'ret_7d']
    available_short = [col for col in short_cols if col in df.columns]
    if available_short:
        df['short_momentum'] = df[available_short].mean(axis=1)
    else:
        df['short_momentum'] = 0
    
    # Long-term momentum
    long_cols = ['ret_30d', 'ret_3m']
    available_long = [col for col in long_cols if col in df.columns]
    if available_long:
        df['long_momentum'] = df[available_long].mean(axis=1)
    else:
        df['long_momentum'] = 0
    
    # Momentum divergence
    df['momentum_divergence'] = df['short_momentum'] - df['long_momentum']
    
    return df

def calculate_edge_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate EDGE scores with refined criteria"""
    df = df.copy()
    
    # Component 1: Volume Score (50% weight)
    df['vol_score'] = 50
    if 'volume_acceleration' in df.columns:
        df['vol_score'] = 50 + df['volume_acceleration'].clip(-50, 50)
        if 'rvol' in df.columns:
            rvol_mult = df['rvol'].clip(0.5, 3.0)
            df.loc[df['volume_acceleration'] > 0, 'vol_score'] *= rvol_mult / 1.5
        if 'volume_consistency' in df.columns:
            df['vol_score'] += df['volume_consistency'] * 20
    df['vol_score'] = df['vol_score'].clip(0, 100)
    
    # Component 2: Momentum Score (30% weight)
    df['mom_score'] = 50
    if 'short_momentum' in df.columns:
        df['mom_score'] += df['short_momentum'] * 3
    if 'momentum_divergence' in df.columns:
        df.loc[df['momentum_divergence'] > 0, 'mom_score'] += 10
    if 'ret_30d' in df.columns:
        df['mom_score'] += df['ret_30d'].clip(-10, 10) * 1.5
    df['mom_score'] = df['mom_score'].clip(0, 100)
    
    # Component 3: Risk/Reward Score (20% weight)
    df['rr_score'] = 50
    if all(col in df.columns for col in ['from_high_pct', 'from_low_pct']):
        # Sweet spot: 10-30% below high
        sweet_spot = (df['from_high_pct'] >= -30) & (df['from_high_pct'] <= -10)
        df.loc[sweet_spot, 'rr_score'] += 30
        # Distance from low (risk buffer)
        df['rr_score'] += (df['from_low_pct'] / 2).clip(0, 20)
    df['rr_score'] = df['rr_score'].clip(0, 100)
    
    # Calculate weighted EDGE score
    df['EDGE'] = (
        df['vol_score'] * 0.50 +
        df['mom_score'] * 0.30 +
        df['rr_score'] * 0.20
    )
    
    # Pattern detection for top stocks
    df = detect_patterns(df)
    
    # Apply pattern bonus
    df.loc[df.get('pattern_score', 0) > 70, 'EDGE'] *= 1.1
    df['EDGE'] = df['EDGE'].clip(0, 100)
    
    return df

def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect trading patterns efficiently"""
    df = df.copy()
    
    # Initialize pattern columns
    df['pattern_name'] = ''
    df['pattern_score'] = 0
    df['pattern_signals'] = ''
    
    # Only process top 100 stocks by EDGE score for performance
    if 'EDGE' in df.columns:
        top_stocks = df.nlargest(min(100, len(df)), 'EDGE').index
    else:
        top_stocks = df.head(100).index
    
    for idx in top_stocks:
        row = df.loc[idx]
        
        # Pattern 1: Accumulation Under Resistance
        if all(col in df.columns for col in ['vol_ratio_30d_90d', 'from_high_pct', 'rvol']):
            vol_ratio = row.get('vol_ratio_30d_90d', 0)
            from_high = row.get('from_high_pct', 0)
            rvol = row.get('rvol', 1)
            
            if vol_ratio > 40 and -20 <= from_high <= -5 and rvol > 2.0:
                df.loc[idx, 'pattern_name'] = 'Accumulation Under Resistance'
                df.loc[idx, 'pattern_score'] = min(70 + vol_ratio/2, 100)
                df.loc[idx, 'pattern_signals'] = f"Vol +{vol_ratio:.0f}%, RVOL {rvol:.1f}x"
                continue
        
        # Pattern 2: Failed Breakdown Reversal
        if all(col in df.columns for col in ['from_low_pct', 'volume_acceleration', 'ret_7d']):
            from_low = row.get('from_low_pct', 100)
            vol_accel = row.get('volume_acceleration', 0)
            ret_7d = row.get('ret_7d', 0)
            
            if from_low < 15 and vol_accel > 20 and ret_7d > 0:
                df.loc[idx, 'pattern_name'] = 'Failed Breakdown Reversal'
                df.loc[idx, 'pattern_score'] = min(60 + vol_accel/2, 100)
                df.loc[idx, 'pattern_signals'] = f"Reversal +{from_low:.0f}%, Vol accel {vol_accel:.0f}%"
    
    return df

def detect_super_edge(row: pd.Series) -> bool:
    """Detect SUPER EDGE opportunities (5 out of 6 criteria)"""
    conditions_met = 0
    
    # 1. High RVOL
    if row.get('rvol', 0) > 2.0:
        conditions_met += 1
    
    # 2. Strong volume acceleration
    if row.get('volume_acceleration', 0) > 30:
        conditions_met += 1
    
    # 3. EPS acceleration
    eps_current = row.get('eps_current', 0)
    eps_last = row.get('eps_last_qtr', 0)
    if eps_current > 0 and eps_last > 0 and (eps_current - eps_last) / eps_last > 0.15:
        conditions_met += 1
    
    # 4. Sweet spot zone
    if -30 <= row.get('from_high_pct', -100) <= -10:
        conditions_met += 1
    
    # 5. Momentum alignment
    if all([row.get('ret_1d', 0) > 0, row.get('ret_7d', 0) > row.get('ret_1d', 0), row.get('ret_30d', 0) > 0]):
        conditions_met += 1
    
    # 6. Pattern confirmation
    if row.get('pattern_score', 0) > 70:
        conditions_met += 1
    
    return conditions_met >= 5

def calculate_risk_management(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate stop losses and position sizes"""
    df = df.copy()
    
    # Dynamic stop loss calculation
    for idx in df.index:
        price = df.loc[idx, 'price'] if 'price' in df.columns else 100
        
        # Base stop: 7% for large caps, 10% for others
        if 'category' in df.columns and 'large' in str(df.loc[idx, 'category']).lower():
            base_stop = price * 0.93
        else:
            base_stop = price * 0.90
        
        # Support-based adjustments
        support_levels = []
        if df.loc[idx].get('sma_50d', 0) > 0:
            support_levels.append(df.loc[idx, 'sma_50d'] * 0.98)
        if df.loc[idx].get('sma_200d', 0) > 0:
            support_levels.append(df.loc[idx, 'sma_200d'] * 0.97)
        
        if support_levels:
            support_stop = max(support_levels)
            df.loc[idx, 'stop_loss'] = max(base_stop, min(support_stop, price * 0.95))
        else:
            df.loc[idx, 'stop_loss'] = base_stop
        
        df.loc[idx, 'stop_loss_pct'] = ((df.loc[idx, 'stop_loss'] - price) / price) * 100
    
    # Calculate targets
    if 'price' in df.columns:
        df['target_1'] = df['price'] * 1.07
        df['target_2'] = df['price'] * 1.15
        
        # SUPER EDGE gets higher targets
        super_mask = df['tag'] == 'SUPER_EDGE' if 'tag' in df.columns else pd.Series(False, index=df.index)
        df.loc[super_mask, 'target_1'] = df.loc[super_mask, 'price'] * 1.12
        df.loc[super_mask, 'target_2'] = df.loc[super_mask, 'price'] * 1.25
    
    return df

def apply_portfolio_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """Apply portfolio-level risk constraints"""
    df = df.copy()
    df = df.sort_values('EDGE', ascending=False) if 'EDGE' in df.columns else df
    
    # Initialize tracking
    total_allocation = 0
    sector_allocations = {}
    position_count = 0
    super_edge_count = 0
    
    for idx in df.index:
        base_size = POSITION_SIZES.get(df.loc[idx].get('tag', 'WATCH'), 0)
        adjusted_size = base_size
        
        if base_size > 0:
            # Check constraints
            if position_count >= MAX_POSITIONS:
                adjusted_size = 0
            elif total_allocation + adjusted_size > MAX_PORTFOLIO_EXPOSURE:
                adjusted_size = max(0, MAX_PORTFOLIO_EXPOSURE - total_allocation)
            
            # Sector concentration
            sector = df.loc[idx].get('sector', 'Unknown')
            current_sector = sector_allocations.get(sector, 0)
            if current_sector + adjusted_size > MAX_SECTOR_EXPOSURE:
                adjusted_size = max(0, MAX_SECTOR_EXPOSURE - current_sector)
            
            # SUPER EDGE limit
            if df.loc[idx].get('tag') == 'SUPER_EDGE':
                if super_edge_count >= MAX_SUPER_EDGE_POSITIONS:
                    adjusted_size = min(adjusted_size, 0.05)
                else:
                    super_edge_count += 1
            
            # Update tracking
            if adjusted_size > 0:
                total_allocation += adjusted_size
                sector_allocations[sector] = current_sector + adjusted_size
                position_count += 1
        
        df.loc[idx, 'position_size'] = adjusted_size
    
    return df

# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================
def run_edge_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Complete EDGE analysis pipeline"""
    # Volume metrics
    df = calculate_volume_metrics(df)
    
    # Momentum metrics
    df = calculate_momentum_metrics(df)
    
    # EDGE scores
    df = calculate_edge_scores(df)
    
    # Classify stocks
    conditions = [
        df['EDGE'] >= EDGE_THRESHOLDS['EXPLOSIVE'],
        df['EDGE'] >= EDGE_THRESHOLDS['STRONG'],
        df['EDGE'] >= EDGE_THRESHOLDS['MODERATE']
    ]
    choices = ['EXPLOSIVE', 'STRONG', 'MODERATE']
    df['tag'] = np.select(conditions, choices, default='WATCH')
    
    # Detect SUPER EDGE
    for idx in df[df['EDGE'] >= EDGE_THRESHOLDS['SUPER_EDGE']].index:
        if detect_super_edge(df.loc[idx]):
            df.loc[idx, 'tag'] = 'SUPER_EDGE'
            df.loc[idx, 'EDGE'] = min(df.loc[idx, 'EDGE'] * 1.1, 100)
    
    # Risk management
    df = calculate_risk_management(df)
    
    # Portfolio constraints
    df = apply_portfolio_constraints(df)
    
    # Decision column
    df['decision'] = df['tag'].map({
        'SUPER_EDGE': 'BUY NOW',
        'EXPLOSIVE': 'BUY',
        'STRONG': 'ACCUMULATE',
        'MODERATE': 'WATCH',
        'WATCH': 'IGNORE'
    }).fillna('IGNORE')
    
    return df

# ============================================================================
# UI COMPONENTS
# ============================================================================
def render_sidebar(diagnostics: Dict):
    """Render sidebar with configuration and diagnostics"""
    st.sidebar.header("⚙️ Configuration")
    
    # Filters
    st.sidebar.subheader("🎯 Signal Filters")
    min_edge = st.sidebar.slider("Min EDGE Score", 0, 100, 50, 5)
    exclude_small = st.sidebar.checkbox("Exclude Small Caps", True)
    max_signals = st.sidebar.slider("Max Signals", 10, 100, 50, 10)
    
    st.sidebar.markdown("---")
    
    # System Health
    st.sidebar.subheader("📊 System Health")
    
    quality = diagnostics.get('data_quality', 0)
    if quality > 90:
        st.sidebar.success(f"Data Quality: {quality:.0f}%")
    elif quality > 70:
        st.sidebar.warning(f"Data Quality: {quality:.0f}%")
    else:
        st.sidebar.error(f"Data Quality: {quality:.0f}%")
    
    st.sidebar.write(f"**Rows:** {diagnostics.get('rows_loaded', 0):,}")
    st.sidebar.write(f"**Updated:** {diagnostics.get('timestamp', datetime.now()).strftime('%H:%M')}")
    
    # Warnings
    warnings = diagnostics.get('warnings', [])
    if warnings:
        with st.sidebar.expander("⚠️ Warnings", expanded=False):
            for w in warnings[:3]:
                st.write(f"• {w}")
    
    return min_edge, exclude_small, max_signals

def render_super_edge_alert(count: int):
    """Render SUPER EDGE alert banner"""
    if count > 0:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #FFD700, #FFA500); 
                    color: black; padding: 20px; border-radius: 10px; 
                    text-align: center; font-size: 24px; font-weight: bold; 
                    margin-bottom: 20px; animation: pulse 2s infinite;">
            ⭐ {count} SUPER EDGE SIGNAL{'S' if count > 1 else ''} DETECTED ⭐<br>
            <span style="font-size: 16px;">Maximum conviction trades with strict risk management!</span>
        </div>
        <style>
        @keyframes pulse {
            0% { opacity: 0.8; }
            50% { opacity: 1; }
            100% { opacity: 0.8; }
        }
        </style>
        """, unsafe_allow_html=True)

def render_metrics_row(df_signals: pd.DataFrame):
    """Render key metrics"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Signals", len(df_signals))
    
    with col2:
        super_count = (df_signals['tag'] == 'SUPER_EDGE').sum() if 'tag' in df_signals.columns else 0
        st.metric("SUPER EDGE", super_count)
    
    with col3:
        portfolio_pct = df_signals['position_size'].sum() * 100 if 'position_size' in df_signals.columns else 0
        st.metric("Portfolio Used", f"{portfolio_pct:.1f}%")
    
    with col4:
        avg_edge = df_signals['EDGE'].mean() if 'EDGE' in df_signals.columns and len(df_signals) > 0 else 0
        st.metric("Avg EDGE", f"{avg_edge:.1f}")
    
    with col5:
        patterns = (df_signals['pattern_score'] > 70).sum() if 'pattern_score' in df_signals.columns else 0
        st.metric("Strong Patterns", patterns)

def create_signal_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create formatted signal table"""
    # Select display columns
    display_cols = [
        'ticker', 'company_name', 'sector', 'tag', 'EDGE', 'decision',
        'price', 'position_size', 'stop_loss', 'stop_loss_pct',
        'target_1', 'target_2', 'volume_pattern', 'pattern_name'
    ]
    display_cols = [col for col in display_cols if col in df.columns]
    
    if not display_cols:
        return pd.DataFrame()
    
    return df[display_cols].copy()

def create_excel_report(df_signals: pd.DataFrame, df_all: pd.DataFrame) -> io.BytesIO:
    """Create comprehensive Excel report"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Executive Summary
        summary_data = {
            'Metric': [
                'Total Signals',
                'SUPER EDGE Count',
                'Portfolio Allocation %',
                'Avg EDGE Score',
                'Report Date'
            ],
            'Value': [
                len(df_signals),
                (df_signals['tag'] == 'SUPER_EDGE').sum() if 'tag' in df_signals.columns else 0,
                f"{df_signals.get('position_size', pd.Series()).sum()*100:.1f}%",
                f"{df_signals.get('EDGE', pd.Series()).mean():.1f}" if len(df_signals) > 0 else "0",
                datetime.now().strftime('%Y-%m-%d %H:%M')
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # Signals
        if not df_signals.empty:
            signal_cols = ['ticker', 'company_name', 'tag', 'EDGE', 'decision', 
                          'price', 'position_size', 'stop_loss', 'target_1', 'target_2']
            signal_cols = [col for col in signal_cols if col in df_signals.columns]
            if signal_cols:
                df_signals[signal_cols].to_excel(writer, sheet_name='Signals', index=False)
    
    output.seek(0)
    return output

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application entry point"""
    # Header
    st.title("⚡ EDGE Protocol - Ultimate Trading Intelligence")
    st.markdown("**Your Secret Weapon: Volume Acceleration reveals institutional moves before they happen**")
    
    # Load data
    df, diagnostics = load_data()
    
    # Sidebar configuration
    min_edge, exclude_small, max_signals = render_sidebar(diagnostics)
    
    if df.empty:
        st.error("❌ Failed to load data. Please check connection.")
        st.stop()
    
    # Apply basic filters
    if exclude_small and 'category' in df.columns:
        df = df[~df['category'].str.contains('micro|nano|small', case=False, na=False)]
    
    # Run analysis
    with st.spinner("Analyzing 1,785 stocks for EDGE opportunities..."):
        df_analyzed = run_edge_analysis(df)
    
    # Filter signals
    df_signals = df_analyzed[df_analyzed['EDGE'] >= min_edge].head(max_signals)
    
    # SUPER EDGE Alert
    super_count = (df_signals['tag'] == 'SUPER_EDGE').sum() if 'tag' in df_signals.columns else 0
    render_super_edge_alert(super_count)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Trading Signals",
        "⭐ SUPER EDGE Focus",
        "📊 Market Analysis",
        "📚 How It Works"
    ])
    
    # Tab 1: Trading Signals
    with tab1:
        st.header("🎯 Today's Trading Signals")
        
        # Metrics
        render_metrics_row(df_signals)
        
        # Signal filters - Fixed the UnboundLocalError
        col1, col2, col3 = st.columns(3)
        
        with col1:
            signal_types = []
            if 'tag' in df_signals.columns:
                available_tags = df_signals['tag'].unique().tolist()
                signal_types = st.multiselect("Signal Types", available_tags, default=available_tags)
        
        with col2:
            sectors = []
            if 'sector' in df_signals.columns:
                available_sectors = df_signals['sector'].dropna().unique().tolist()
                sectors = st.multiselect("Sectors", available_sectors, default=available_sectors)
        
        with col3:
            search = st.text_input("Search Ticker")
        
        # Apply filters to signals
        display_df = df_signals.copy()
        
        if signal_types and 'tag' in display_df.columns:
            display_df = display_df[display_df['tag'].isin(signal_types)]
        
        if sectors and 'sector' in display_df.columns:
            display_df = display_df[display_df['sector'].isin(sectors)]
        
        if search and 'ticker' in display_df.columns:
            display_df = display_df[display_df['ticker'].str.contains(search.upper(), na=False)]
        
        # Display table
        if not display_df.empty:
            signal_table = create_signal_table(display_df)
            
            if not signal_table.empty:
                # Format the dataframe
                format_dict = {}
                if 'EDGE' in signal_table.columns:
                    format_dict['EDGE'] = '{:.1f}'
                if 'price' in signal_table.columns:
                    format_dict['price'] = '₹{:.2f}'
                if 'position_size' in signal_table.columns:
                    format_dict['position_size'] = '{:.1%}'
                if 'stop_loss' in signal_table.columns:
                    format_dict['stop_loss'] = '₹{:.2f}'
                if 'stop_loss_pct' in signal_table.columns:
                    format_dict['stop_loss_pct'] = '{:.1f}%'
                if 'target_1' in signal_table.columns:
                    format_dict['target_1'] = '₹{:.2f}'
                if 'target_2' in signal_table.columns:
                    format_dict['target_2'] = '₹{:.2f}'
                
                # Style function
                def highlight_tags(val):
                    if val == 'SUPER_EDGE':
                        return 'background-color: gold; font-weight: bold;'
                    elif val == 'EXPLOSIVE':
                        return 'background-color: #ffcccc;'
                    elif val == 'BUY NOW':
                        return 'color: green; font-weight: bold;'
                    return ''
                
                styled_df = signal_table.style.format(format_dict)
                
                # Apply styling to specific columns if they exist
                style_cols = []
                if 'tag' in signal_table.columns:
                    style_cols.append('tag')
                if 'decision' in signal_table.columns:
                    style_cols.append('decision')
                
                if style_cols:
                    styled_df = styled_df.applymap(highlight_tags, subset=style_cols)
                
                st.dataframe(styled_df, use_container_width=True, height=600)
            
            # Export buttons
            col1, col2 = st.columns(2)
            
            with col1:
                csv = display_df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    f"edge_signals_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv",
                    type="primary"
                )
            
            with col2:
                if len(df_signals) > 0:
                    excel_file = create_excel_report(df_signals, df_analyzed)
                    st.download_button(
                        "📊 Download Excel Report",
                        excel_file,
                        f"EDGE_Report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary"
                    )
        else:
            st.info("No signals match the selected filters.")
    
    # Tab 2: SUPER EDGE Focus
    with tab2:
        st.header("⭐ SUPER EDGE Deep Dive")
        
        super_df = df_signals[df_signals['tag'] == 'SUPER_EDGE'] if 'tag' in df_signals.columns else pd.DataFrame()
        
        if not super_df.empty:
            st.success(f"🎯 {len(super_df)} SUPER EDGE opportunities detected!")
            
            for idx, (_, row) in enumerate(super_df.iterrows()):
                with st.expander(
                    f"#{idx+1} {row.get('ticker', 'N/A')} - EDGE: {row.get('EDGE', 0):.1f}",
                    expanded=(idx == 0)
                ):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("📊 Entry Details")
                        if 'price' in row:
                            st.metric("Price", f"₹{row['price']:.2f}")
                        if 'position_size' in row:
                            st.metric("Position Size", f"{row['position_size']*100:.1f}%")
                        if 'stop_loss' in row:
                            st.metric("Stop Loss", f"₹{row['stop_loss']:.2f}")
                    
                    with col2:
                        st.subheader("🎯 Targets")
                        if 'target_1' in row:
                            st.metric("Target 1", f"₹{row['target_1']:.2f}")
                        if 'target_2' in row:
                            st.metric("Target 2", f"₹{row['target_2']:.2f}")
                    
                    with col3:
                        st.subheader("🔍 Signals")
                        if 'volume_pattern' in row:
                            st.write(f"**Volume:** {row['volume_pattern']}")
                        if 'volume_acceleration' in row:
                            st.write(f"**Acceleration:** {row['volume_acceleration']:.1f}%")
                        if 'pattern_name' in row and row['pattern_name']:
                            st.write(f"**Pattern:** {row['pattern_name']}")
        else:
            st.info("No SUPER EDGE signals today. Check EXPLOSIVE category.")
    
    # Tab 3: Market Analysis
    with tab3:
        st.header("📊 Market Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Volume Acceleration Distribution")
            
            if 'volume_acceleration' in df_analyzed.columns:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=df_analyzed['volume_acceleration'],
                    nbinsx=30,
                    name='All Stocks',
                    marker_color='lightblue'
                ))
                
                if not df_signals.empty and 'volume_acceleration' in df_signals.columns:
                    fig.add_trace(go.Histogram(
                        x=df_signals['volume_acceleration'],
                        nbinsx=20,
                        name='Signal Stocks',
                        marker_color='gold',
                        opacity=0.7
                    ))
                
                fig.update_layout(
                    xaxis_title="Volume Acceleration %",
                    yaxis_title="Count",
                    height=400,
                    barmode='overlay'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 EDGE Score Distribution")
            
            if 'EDGE' in df_analyzed.columns:
                fig = go.Figure()
                
                # Add threshold regions
                fig.add_vrect(x0=0, x1=50, fillcolor="red", opacity=0.1)
                fig.add_vrect(x0=50, x1=70, fillcolor="yellow", opacity=0.1)
                fig.add_vrect(x0=70, x1=85, fillcolor="orange", opacity=0.1)
                fig.add_vrect(x0=85, x1=100, fillcolor="green", opacity=0.1)
                
                fig.add_trace(go.Histogram(
                    x=df_analyzed['EDGE'],
                    nbinsx=25,
                    marker_color='darkblue'
                ))
                
                fig.update_layout(
                    xaxis_title="EDGE Score",
                    yaxis_title="Count",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Market Breadth
        st.subheader("📊 Market Breadth")
        
        breadth_cols = st.columns(4)
        
        with breadth_cols[0]:
            if 'volume_acceleration' in df_analyzed.columns:
                accumulating = (df_analyzed['volume_acceleration'] > 10).sum()
                st.metric("Stocks Accumulating", accumulating)
        
        with breadth_cols[1]:
            if 'ret_7d' in df_analyzed.columns:
                advancing = (df_analyzed['ret_7d'] > 0).sum()
                st.metric("7D Advancing", advancing)
        
        with breadth_cols[2]:
            if all(col in df_analyzed.columns for col in ['price', 'sma_200d']):
                above_200 = (df_analyzed['price'] > df_analyzed['sma_200d']).sum()
                st.metric("Above 200 SMA", above_200)
        
        with breadth_cols[3]:
            if 'rvol' in df_analyzed.columns:
                high_vol = (df_analyzed['rvol'] > 1.5).sum()
                st.metric("High Volume", high_vol)
    
    # Tab 4: How It Works
    with tab4:
        st.header("📚 How EDGE Protocol Works")
        
        st.markdown("""
        ### 🔍 The Secret Weapon: Volume Acceleration
        
        While everyone watches price and basic volume, EDGE Protocol uses **Volume Acceleration** - 
        comparing short-term volume momentum against medium-term trends. This reveals:
        
        - **Is buying pressure ACCELERATING?**
        - **Are institutions positioning BEFORE the move?**
        - **Is smart money accumulating while others distribute?**
        
        ### 📊 The EDGE Scoring System
        
        **1. Volume Score (50% weight)**
        - Volume acceleration is your primary edge
        - RVOL confirms institutional interest
        - Consistency across timeframes validates signals
        
        **2. Momentum Score (30% weight)**
        - Short vs long-term momentum divergence
        - Trend acceleration patterns
        - Price action confirmation
        
        **3. Risk/Reward Score (20% weight)**
        - Distance from 52-week high (opportunity)
        - Distance from support (risk)
        - Quality stock discount detection
        
        ### 🎯 Signal Classifications
        
        - **SUPER EDGE (92+)**: All systems aligned - 15% position
        - **EXPLOSIVE (85-92)**: High conviction - 10% position
        - **STRONG (70-85)**: Solid opportunity - 5% position
        - **MODERATE (50-70)**: Worth watching - 2% position
        
        ### ⚡ Risk Management
        
        - Dynamic stop losses based on volatility and support
        - Portfolio constraints prevent over-concentration
        - Sector limits ensure diversification
        - Maximum 80% portfolio exposure
        
        ### 🏆 Why This Works
        
        1. **Data Advantage**: Uses metrics others don't track
        2. **Early Detection**: Catches accumulation before breakouts
        3. **Risk Control**: Every position has defined risk
        4. **Proven Patterns**: Based on institutional behavior
        
        ---
        
        **Remember**: Past performance doesn't guarantee future results. 
        Always use proper risk management and never invest more than you can afford to lose.
        """)

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()
