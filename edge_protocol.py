"""
EDGE Protocol - Ultimate Trading Intelligence System
===================================================
Your unfair advantage: Volume acceleration reveals institutional behavior
before price moves. This is the signal others can't see.

Version: 1.0 FINAL
Author: EDGE Protocol Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import io
from datetime import datetime
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Centralized configuration for EDGE Protocol"""
    
    # Data source
    SHEET_ID = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"
    GID = "2026492216"
    SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"
    
    # Trading parameters
    SWEET_SPOT_HIGH = -15  # % from 52w high
    SWEET_SPOT_LOW = -30   # % from 52w high
    
    # EDGE thresholds
    EDGE_THRESHOLDS = {
        'SUPER_EDGE': 85,
        'EXPLOSIVE': 70,
        'STRONG': 50,
        'MODERATE': 30
    }
    
    # Position sizing (% of capital)
    POSITION_SIZES = {
        'SUPER_EDGE': 0.15,  # 15%
        'EXPLOSIVE': 0.10,   # 10%
        'STRONG': 0.05,      # 5%
        'MODERATE': 0.02,    # 2%
        'WATCH': 0.00        # 0%
    }
    
    # Scoring weights - Simplified as recommended
    WEIGHTS = {
        'volume': 0.50,      # Your secret weapon
        'risk_reward': 0.30, # Entry quality
        'momentum': 0.20     # Timing confirmation
    }
    
    # Pattern thresholds
    PATTERN_STRONG = 70
    PATTERN_CONFLUENCE_SUPER = 85
    
    # Display settings
    CACHE_TTL = 300  # 5 minutes
    MAX_DISPLAY_ROWS = 50

# ============================================================================
# DATA LOADING & CLEANING
# ============================================================================
@st.cache_data(ttl=Config.CACHE_TTL)
def load_data() -> pd.DataFrame:
    """Load and clean data from Google Sheets with robust error handling"""
    try:
        # Fetch data
        response = requests.get(Config.SHEET_URL, timeout=30)
        response.raise_for_status()
        
        # Parse CSV
        df = pd.read_csv(io.StringIO(response.text))
        
        # Clean column names - standardize to lowercase
        df.columns = df.columns.str.strip().str.lower()
        
        # Define numeric columns based on provided headers
        numeric_cols = [
            'price', 'ret_1d', 'low_52w', 'high_52w', 'from_low_pct', 'from_high_pct',
            'sma_20d', 'sma_50d', 'sma_200d', 'ret_3d', 'ret_7d', 'ret_30d', 
            'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
            'volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d',
            'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
            'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 
            'vol_ratio_90d_180d', 'rvol', 'prev_close', 'pe', 
            'eps_current', 'eps_last_qtr', 'eps_change_pct'
        ]
        
        # Clean and convert numeric columns
        for col in numeric_cols:
            if col in df.columns:
                # Remove currency symbols and convert
                df[col] = pd.to_numeric(
                    df[col].astype(str)
                    .str.replace('₹', '', regex=False)
                    .str.replace(',', '', regex=False)
                    .str.replace('%', '', regex=False)
                    .str.strip(),
                    errors='coerce'
                )
        
        # Handle market cap separately (has 'Cr' suffix)
        if 'market_cap' in df.columns:
            df['market_cap_value'] = pd.to_numeric(
                df['market_cap'].astype(str)
                .str.replace('₹', '', regex=False)
                .str.replace(',', '', regex=False)
                .str.replace('Cr', '', regex=False)
                .str.strip(),
                errors='coerce'
            ) * 10000000  # Convert Cr to actual value
        
        # Ensure critical columns exist and have valid data
        df = df[df['ticker'].notna()]  # Remove rows without ticker
        df = df[df['price'] > 0]        # Remove invalid prices
        
        # Fill missing values intelligently
        df['rvol'] = df['rvol'].fillna(1.0)
        df['volume_1d'] = df['volume_1d'].fillna(0)
        
        return df
        
    except requests.RequestException as e:
        st.error(f"Failed to fetch data: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return pd.DataFrame()

# ============================================================================
# CORE CALCULATIONS
# ============================================================================
def calculate_volume_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volume acceleration - THE SECRET WEAPON"""
    df = df.copy()
    
    # Calculate acceleration (30d/90d vs 30d/180d comparison)
    df['volume_acceleration'] = df['vol_ratio_30d_90d'] - df['vol_ratio_30d_180d']
    
    # Classify volume patterns
    conditions = [
        df['volume_acceleration'] > 30,
        df['volume_acceleration'] > 20,
        df['volume_acceleration'] > 10,
        df['volume_acceleration'] > 0,
        df['volume_acceleration'] > -10,
        df['volume_acceleration'] <= -10
    ]
    
    choices = [
        'Institutional Loading',
        'Heavy Accumulation',
        'Accumulation',
        'Mild Accumulation',
        'Distribution',
        'Exodus'
    ]
    
    df['volume_pattern'] = np.select(conditions, choices, default='Neutral')
    
    # Sweet spot detection
    df['is_sweet_spot'] = (
        (df['from_high_pct'] >= Config.SWEET_SPOT_LOW) & 
        (df['from_high_pct'] <= Config.SWEET_SPOT_HIGH)
    )
    
    return df

# ============================================================================
# PATTERN DETECTION - ONLY TOP 3 PATTERNS
# ============================================================================
def detect_accumulation_under_resistance(row: pd.Series) -> Dict:
    """Pattern 1: Volume explodes but price stays flat near resistance"""
    score = 0
    signals = []
    
    # Volume explosion check
    vol_ratio = row.get('vol_ratio_30d_90d', 0)
    if vol_ratio > 50:
        score += 40
        signals.append(f"Vol +{vol_ratio:.0f}%")
    elif vol_ratio > 30:
        score += 25
        signals.append(f"Vol +{vol_ratio:.0f}%")
    
    # Price flat check
    ret_30d = row.get('ret_30d', 0)
    if abs(ret_30d) < 5:
        score += 30
        signals.append("Flat price")
    
    # Near resistance check
    from_high = row.get('from_high_pct', -100)
    if -10 <= from_high <= 0:
        score += 30
        signals.append("At resistance")
    
    return {
        'name': 'Accumulation Under Resistance',
        'score': min(score, 100),
        'signals': ', '.join(signals) if signals else 'None'
    }

def detect_failed_breakdown_reversal(row: pd.Series) -> Dict:
    """Pattern 2: Near 52w low but volume accelerating"""
    score = 0
    signals = []
    
    # Near 52w low check
    from_low = row.get('from_low_pct', 100)
    if from_low < 10:
        score += 40
        signals.append(f"Near low +{from_low:.0f}%")
    elif from_low < 20:
        score += 25
        signals.append(f"Recent low +{from_low:.0f}%")
    
    # Volume acceleration check
    vol_accel = row.get('volume_acceleration', 0)
    if vol_accel > 20:
        score += 40
        signals.append(f"Vol accel {vol_accel:.0f}%")
    elif vol_accel > 10:
        score += 25
        signals.append(f"Vol building")
    
    # Momentum reversal
    if row.get('ret_7d', 0) > 0:
        score += 20
        signals.append("Momentum +ve")
    
    return {
        'name': 'Failed Breakdown Reversal',
        'score': min(score, 100),
        'signals': ', '.join(signals) if signals else 'None'
    }

def detect_coiled_spring(row: pd.Series) -> Dict:
    """Pattern 3: Volume up but price in tight range"""
    score = 0
    signals = []
    
    # Volume increase check
    vol_ratio = row.get('vol_ratio_30d_90d', 0)
    if vol_ratio > 30:
        score += 35
        signals.append(f"Vol +{vol_ratio:.0f}%")
    elif vol_ratio > 15:
        score += 20
        signals.append("Vol building")
    
    # Tight range check
    ret_7d = row.get('ret_7d', 0)
    ret_30d = row.get('ret_30d', 0)
    if abs(ret_7d) < 5 and abs(ret_30d) < 10:
        score += 35
        signals.append("Tight range")
    
    # Above key SMAs
    price = row.get('price', 1)
    sma_50d = row.get('sma_50d', price)
    sma_200d = row.get('sma_200d', price)
    
    if price > sma_50d and price > sma_200d:
        score += 30
        signals.append("Above SMAs")
    
    return {
        'name': 'Coiled Spring',
        'score': min(score, 100),
        'signals': ', '.join(signals) if signals else 'None'
    }

def detect_best_pattern(row: pd.Series) -> Tuple[str, float, str]:
    """Run all patterns and return the strongest one"""
    patterns = [
        detect_accumulation_under_resistance(row),
        detect_failed_breakdown_reversal(row),
        detect_coiled_spring(row)
    ]
    
    # Sort by score
    best_pattern = max(patterns, key=lambda x: x['score'])
    
    # Check for confluence
    strong_patterns = [p for p in patterns if p['score'] >= Config.PATTERN_STRONG]
    confluence_bonus = len(strong_patterns) * 10
    
    final_score = min(best_pattern['score'] + confluence_bonus, 100)
    
    # Add confluence info to signals
    if len(strong_patterns) > 1:
        best_pattern['signals'] += f" | {len(strong_patterns)} patterns!"
    
    return best_pattern['name'], final_score, best_pattern['signals']

# ============================================================================
# SCORING ENGINE
# ============================================================================
def calculate_component_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate individual component scores"""
    df = df.copy()
    
    # 1. Volume Score (50% weight) - THE SECRET WEAPON
    df['score_volume'] = 50  # Base score
    
    # Volume acceleration bonus
    vol_accel = df['volume_acceleration']
    df.loc[vol_accel > 30, 'score_volume'] = 100
    df.loc[(vol_accel > 20) & (vol_accel <= 30), 'score_volume'] = 85
    df.loc[(vol_accel > 10) & (vol_accel <= 20), 'score_volume'] = 70
    df.loc[(vol_accel > 0) & (vol_accel <= 10), 'score_volume'] = 60
    
    # RVOL multiplier
    high_rvol = df['rvol'] > 2.0
    df.loc[high_rvol, 'score_volume'] = df.loc[high_rvol, 'score_volume'] * 1.2
    df['score_volume'] = df['score_volume'].clip(0, 100)
    
    # 2. Risk/Reward Score (30% weight)
    # Calculate upside to 52w high
    df['upside_potential'] = ((df['high_52w'] - df['price']) / df['price'] * 100).clip(0, 100)
    
    # Calculate downside to 52w low  
    df['downside_risk'] = ((df['price'] - df['low_52w']) / df['price'] * 100).clip(0, 100)
    
    # Risk/Reward ratio
    df['rr_ratio'] = (df['upside_potential'] / (df['downside_risk'] + 1)).clip(0, 5)
    df['score_risk_reward'] = (df['rr_ratio'] * 20).clip(0, 100)
    
    # Sweet spot bonus
    df.loc[df['is_sweet_spot'], 'score_risk_reward'] = df.loc[df['is_sweet_spot'], 'score_risk_reward'] * 1.3
    df['score_risk_reward'] = df['score_risk_reward'].clip(0, 100)
    
    # 3. Momentum Score (20% weight)
    # Simple and effective momentum calculation
    df['momentum_short'] = (df['ret_1d'] + df['ret_3d'] + df['ret_7d']) / 3
    df['momentum_mid'] = df['ret_30d']
    
    # Combined momentum
    df['momentum_combined'] = df['momentum_short'] * 0.6 + df['momentum_mid'] * 0.4
    df['score_momentum'] = (50 + df['momentum_combined'] * 3).clip(0, 100)
    
    # Momentum alignment bonus
    aligned = (
        (df['ret_1d'] > 0) & 
        (df['ret_3d'] > df['ret_1d']) & 
        (df['ret_7d'] > df['ret_3d']) & 
        (df['ret_30d'] > 0)
    )
    df.loc[aligned, 'score_momentum'] = df.loc[aligned, 'score_momentum'] * 1.2
    df['score_momentum'] = df['score_momentum'].clip(0, 100)
    
    return df

def calculate_edge_score(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate final EDGE score with pattern detection"""
    df = df.copy()
    
    # Calculate component scores
    df = calculate_component_scores(df)
    
    # Calculate weighted EDGE score
    df['edge_score'] = (
        df['score_volume'] * Config.WEIGHTS['volume'] +
        df['score_risk_reward'] * Config.WEIGHTS['risk_reward'] +
        df['score_momentum'] * Config.WEIGHTS['momentum']
    )
    
    # Detect patterns for high-potential stocks only (optimization)
    df['pattern_detected'] = ''
    df['pattern_strength'] = 0
    df['pattern_signals'] = ''
    
    # Only run pattern detection on stocks with EDGE > 30
    high_potential = df[df['edge_score'] > 30].index
    
    for idx in high_potential:
        row = df.loc[idx]
        pattern_name, pattern_score, pattern_signals = detect_best_pattern(row)
        
        df.loc[idx, 'pattern_detected'] = pattern_name
        df.loc[idx, 'pattern_strength'] = pattern_score
        df.loc[idx, 'pattern_signals'] = pattern_signals
    
    # Apply pattern bonus to EDGE score
    pattern_bonus = df['pattern_strength'] * 0.1  # Up to 10 point bonus
    df['edge_score'] = (df['edge_score'] + pattern_bonus).clip(0, 100)
    
    # SUPER EDGE detection - Multiple strong signals aligned
    super_edge_conditions = (
        (df['edge_score'] >= Config.EDGE_THRESHOLDS['SUPER_EDGE']) &
        (df['volume_acceleration'] > 30) &
        (df['rvol'] > 2.0) &
        (df['is_sweet_spot'] == True) &
        (df['pattern_strength'] >= Config.PATTERN_STRONG)
    )
    
    # Classify signals
    df['signal_type'] = 'WATCH'
    df.loc[df['edge_score'] >= Config.EDGE_THRESHOLDS['MODERATE'], 'signal_type'] = 'MODERATE'
    df.loc[df['edge_score'] >= Config.EDGE_THRESHOLDS['STRONG'], 'signal_type'] = 'STRONG'
    df.loc[df['edge_score'] >= Config.EDGE_THRESHOLDS['EXPLOSIVE'], 'signal_type'] = 'EXPLOSIVE'
    df.loc[super_edge_conditions, 'signal_type'] = 'SUPER_EDGE'
    
    # Position sizing
    df['position_size'] = df['signal_type'].map(Config.POSITION_SIZES)
    
    # Risk management levels
    df['stop_loss'] = df['price'] * 0.93  # 7% stop
    df['target_1'] = df['price'] * 1.08   # 8% target
    df['target_2'] = df['price'] * 1.15   # 15% target
    
    # Adjust for SUPER EDGE
    super_mask = df['signal_type'] == 'SUPER_EDGE'
    df.loc[super_mask, 'target_1'] = df.loc[super_mask, 'price'] * 1.12
    df.loc[super_mask, 'target_2'] = df.loc[super_mask, 'price'] * 1.20
    
    return df

# ============================================================================
# VISUALIZATION
# ============================================================================
def create_edge_radar(row: pd.Series) -> go.Figure:
    """Create radar chart for EDGE components"""
    categories = ['Volume<br>Power', 'Risk/<br>Reward', 'Momentum']
    values = [
        row.get('score_volume', 0),
        row.get('score_risk_reward', 0),
        row.get('score_momentum', 0)
    ]
    
    # Different colors for different signal types
    color_map = {
        'SUPER_EDGE': 'gold',
        'EXPLOSIVE': 'red',
        'STRONG': 'orange',
        'MODERATE': 'green',
        'WATCH': 'lightblue'
    }
    
    line_color = color_map.get(row.get('signal_type', 'WATCH'), 'lightblue')
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        line_color=line_color,
        line_width=3
    ))
    
    title = f"{row.get('ticker', 'Stock')} - EDGE Components"
    if row.get('signal_type') == 'SUPER_EDGE':
        title = "⭐ " + title + " ⭐"
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title=title,
        height=400
    )
    
    return fig

# ============================================================================
# MAIN UI
# ============================================================================
def main():
    st.set_page_config(
        page_title="EDGE Protocol",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for visual enhancements
    st.markdown("""
    <style>
    .super-edge-alert {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: black;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .stMetric {
        background-color: rgba(28, 131, 225, 0.1);
        padding: 10px;
        border-radius: 5px;
    }
    div[data-testid="stDataFrame"] {
        height: 600px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("⚡ EDGE Protocol - Ultimate Trading Intelligence")
    st.markdown("**Volume Acceleration + Sweet Spot + Patterns = Your EDGE**")
    
    # Load and process data
    with st.spinner("Calculating EDGE across the market..."):
        df = load_data()
        
        if df.empty:
            st.error("❌ Failed to load data. Please check your connection and try again.")
            return
        
        # Calculate all metrics
        df = calculate_volume_acceleration(df)
        df = calculate_edge_score(df)
    
    # Sidebar filters
    with st.sidebar:
        st.header("🎯 Filters")
        
        # Signal type filter
        signal_types = ['SUPER_EDGE', 'EXPLOSIVE', 'STRONG', 'MODERATE']
        selected_signals = st.multiselect(
            "Signal Types",
            signal_types,
            default=['SUPER_EDGE', 'EXPLOSIVE', 'STRONG']
        )
        
        # Minimum EDGE score
        min_edge = st.slider("Minimum EDGE Score", 0, 100, 50, 5)
        
        # Volume pattern filter
        vol_patterns = df['volume_pattern'].unique()
        selected_vol_patterns = st.multiselect(
            "Volume Patterns",
            vol_patterns,
            default=['Institutional Loading', 'Heavy Accumulation', 'Accumulation']
        )
        
        # Sector filter
        sectors = sorted(df['sector'].dropna().unique())
        selected_sectors = st.multiselect("Sectors", sectors, default=sectors)
    
    # Apply filters
    filtered_df = df[
        (df['signal_type'].isin(selected_signals)) &
        (df['edge_score'] >= min_edge) &
        (df['volume_pattern'].isin(selected_vol_patterns)) &
        (df['sector'].isin(selected_sectors))
    ].sort_values('edge_score', ascending=False)
    
    # SUPER EDGE Alert
    super_edge_count = (filtered_df['signal_type'] == 'SUPER_EDGE').sum()
    if super_edge_count > 0:
        st.markdown(f"""
        <div class="super-edge-alert">
            ⭐ {super_edge_count} SUPER EDGE SIGNAL{'S' if super_edge_count > 1 else ''} DETECTED! ⭐
        </div>
        """, unsafe_allow_html=True)
    
    # Main content - 3 tabs only as recommended
    tab1, tab2, tab3 = st.tabs([
        "📊 Daily Signals",
        "⭐ SUPER EDGE Analysis", 
        "🔍 Deep Dive"
    ])
    
    # Tab 1: Daily Signals
    with tab1:
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Signals", len(filtered_df))
        with col2:
            st.metric("Avg EDGE Score", f"{filtered_df['edge_score'].mean():.1f}" if len(filtered_df) > 0 else "0")
        with col3:
            explosive_count = (filtered_df['signal_type'].isin(['SUPER_EDGE', 'EXPLOSIVE'])).sum()
            st.metric("Explosive Signals", explosive_count)
        with col4:
            if len(filtered_df) > 0:
                total_allocation = filtered_df.head(10)['position_size'].sum()
                st.metric("Top 10 Allocation", f"{total_allocation*100:.0f}%")
            else:
                st.metric("Top 10 Allocation", "0%")
        
        # Main signals table
        st.subheader("🎯 Today's EDGE Signals")
        
        if not filtered_df.empty:
            # Select display columns
            display_cols = [
                'ticker', 'company_name', 'signal_type', 'edge_score',
                'pattern_detected', 'pattern_strength', 'pattern_signals',
                'price', 'volume_acceleration', 'volume_pattern', 'rvol',
                'position_size', 'stop_loss', 'target_1', 'target_2'
            ]
            
            # Ensure columns exist
            display_cols = [col for col in display_cols if col in filtered_df.columns]
            
            # Style function for highlighting
            def highlight_signals(row):
                if row['signal_type'] == 'SUPER_EDGE':
                    return ['background-color: gold'] * len(row)
                elif row['signal_type'] == 'EXPLOSIVE':
                    return ['background-color: #ffcccb'] * len(row)
                elif row['signal_type'] == 'STRONG':
                    return ['background-color: #ffd8b1'] * len(row)
                return [''] * len(row)
            
            # Format the dataframe
            styled_df = filtered_df[display_cols].style.apply(highlight_signals, axis=1).format({
                'edge_score': '{:.1f}',
                'pattern_strength': '{:.0f}',
                'price': '₹{:.2f}',
                'volume_acceleration': '{:.1f}%',
                'rvol': '{:.1f}x',
                'position_size': '{:.1%}',
                'stop_loss': '₹{:.2f}',
                'target_1': '₹{:.2f}',
                'target_2': '₹{:.2f}'
            })
            
            # Add pattern icons
            st.dataframe(styled_df, use_container_width=True, height=600)
            
            # Pattern legend
            st.markdown("""
            **Pattern Icons:** 🎯 = Pattern Detected | 🔥 = Strong Pattern (>70) | ⚡ = Multiple Patterns
            """)
            
            # Export button
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export Signals",
                csv,
                f"edge_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv",
                type="primary"
            )
        else:
            st.info("No signals match your filters. Try adjusting the criteria.")
    
    # Tab 2: SUPER EDGE Analysis
    with tab2:
        super_edge_df = filtered_df[filtered_df['signal_type'] == 'SUPER_EDGE']
        
        if not super_edge_df.empty:
            st.success(f"🎯 {len(super_edge_df)} SUPER EDGE opportunities ready for maximum conviction!")
            
            # Display each SUPER EDGE signal
            for idx, (_, row) in enumerate(super_edge_df.iterrows()):
                with st.expander(
                    f"#{idx+1} {row['ticker']} - {row['company_name']} | EDGE: {row['edge_score']:.1f}",
                    expanded=(idx == 0)
                ):
                    # Metrics columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Price", f"₹{row['price']:.2f}")
                        st.metric("Position Size", f"{row['position_size']*100:.0f}%")
                        st.metric("Volume Acceleration", f"{row['volume_acceleration']:.1f}%")
                    
                    with col2:
                        st.metric("RVOL", f"{row['rvol']:.1f}x")
                        st.metric("From 52w High", f"{row['from_high_pct']:.1f}%")
                        st.metric("Pattern", row['pattern_detected'])
                    
                    with col3:
                        st.metric("Stop Loss", f"₹{row['stop_loss']:.2f}")
                        st.metric("Target 1", f"₹{row['target_1']:.2f}")
                        st.metric("Target 2", f"₹{row['target_2']:.2f}")
                    
                    # Radar chart
                    fig = create_edge_radar(row)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Pattern signals
                    if row['pattern_signals']:
                        st.info(f"**Pattern Signals:** {row['pattern_signals']}")
                    
                    # Why SUPER EDGE?
                    st.markdown("""
                    **Why this is SUPER EDGE:**
                    - ✅ Volume acceleration > 30% (Institutional loading)
                    - ✅ RVOL > 2.0x (Unusual activity)
                    - ✅ In sweet spot zone (-15% to -30% from high)
                    - ✅ Strong pattern detected
                    - ✅ All components aligned
                    """)
        else:
            st.info("No SUPER EDGE signals today. Check EXPLOSIVE category for high-conviction trades.")
    
    # Tab 3: Deep Dive
    with tab3:
        st.subheader("🔍 Stock Deep Dive Analysis")
        
        # Stock selector
        all_stocks = df[df['edge_score'] > 0].sort_values('edge_score', ascending=False)
        
        if not all_stocks.empty:
            ticker_list = all_stocks['ticker'].tolist()
            
            # Mark SUPER EDGE stocks
            ticker_display = []
            for t in ticker_list:
                signal = all_stocks[all_stocks['ticker'] == t]['signal_type'].iloc[0]
                if signal == 'SUPER_EDGE':
                    ticker_display.append(f"⭐ {t}")
                elif signal == 'EXPLOSIVE':
                    ticker_display.append(f"🔥 {t}")
                else:
                    ticker_display.append(t)
            
            selected_idx = st.selectbox(
                "Select Stock for Analysis",
                range(len(ticker_list)),
                format_func=lambda x: ticker_display[x]
            )
            
            selected_ticker = ticker_list[selected_idx]
            stock_data = all_stocks[all_stocks['ticker'] == selected_ticker].iloc[0]
            
            # Header
            if stock_data['signal_type'] == 'SUPER_EDGE':
                st.markdown("""
                <div style="background: gold; padding: 15px; border-radius: 10px; text-align: center;">
                    <h2 style="margin: 0;">⭐ SUPER EDGE SIGNAL ⭐</h2>
                </div>
                """, unsafe_allow_html=True)
            
            st.subheader(f"{stock_data['company_name']} ({stock_data['ticker']})")
            
            # Comprehensive metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("**Price & Action**")
                st.metric("Price", f"₹{stock_data['price']:.2f}")
                st.metric("Signal Type", stock_data['signal_type'])
                st.metric("EDGE Score", f"{stock_data['edge_score']:.1f}")
            
            with col2:
                st.markdown("**Volume Intelligence**")
                st.metric("Volume Accel", f"{stock_data['volume_acceleration']:.1f}%")
                st.metric("RVOL", f"{stock_data['rvol']:.1f}x")
                st.metric("Volume Pattern", stock_data['volume_pattern'])
            
            with col3:
                st.markdown("**Risk/Reward**")
                st.metric("Stop Loss", f"₹{stock_data['stop_loss']:.2f}")
                st.metric("Target 1", f"₹{stock_data['target_1']:.2f}")
                st.metric("RR Ratio", f"{stock_data.get('rr_ratio', 0):.1f}")
            
            with col4:
                st.markdown("**Position**")
                st.metric("Position Size", f"{stock_data['position_size']*100:.0f}%")
                risk_amt = stock_data['position_size'] * 7  # 7% stop
                st.metric("Risk Amount", f"{risk_amt:.1f}%")
                st.metric("From High", f"{stock_data['from_high_pct']:.1f}%")
            
            # Radar visualization
            fig = create_edge_radar(stock_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Pattern analysis
            if stock_data['pattern_detected']:
                st.subheader("🎯 Pattern Analysis")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric("Pattern", stock_data['pattern_detected'])
                    st.metric("Strength", f"{stock_data['pattern_strength']:.0f}")
                
                with col2:
                    st.info(f"**Signals:** {stock_data['pattern_signals']}")
            
            # Trading checklist
            st.subheader("✅ Pre-Trade Checklist")
            
            checklist = {
                "Volume Acceleration > 20%": stock_data['volume_acceleration'] > 20,
                "RVOL > 1.5x": stock_data['rvol'] > 1.5,
                "In Sweet Spot (-15% to -30%)": stock_data['is_sweet_spot'],
                "Pattern Detected": stock_data['pattern_strength'] > 0,
                "Above 50 SMA": stock_data['price'] > stock_data.get('sma_50d', 0),
                "Momentum Positive": stock_data['ret_7d'] > 0
            }
            
            col1, col2 = st.columns(2)
            items = list(checklist.items())
            
            for i, (check, passed) in enumerate(items):
                with col1 if i < 3 else col2:
                    if passed:
                        st.success(f"✅ {check}")
                    else:
                        st.error(f"❌ {check}")
            
            # Entry instructions
            if stock_data['signal_type'] in ['SUPER_EDGE', 'EXPLOSIVE']:
                st.markdown("""
                ### 📌 Entry Instructions:
                1. **Entry**: Current price or better
                2. **Position**: {} of capital
                3. **Stop Loss**: ₹{:.2f} (strict)
                4. **Target 1**: ₹{:.2f} (book 50%)
                5. **Target 2**: ₹{:.2f} (trail rest)
                """.format(
                    f"{stock_data['position_size']*100:.0f}%",
                    stock_data['stop_loss'],
                    stock_data['target_1'],
                    stock_data['target_2']
                ))
        else:
            st.info("No stocks available for analysis.")
    
    # Footer
    st.markdown("---")
    st.caption("""
    **EDGE Protocol v1.0** | Volume Acceleration reveals what others can't see
    
    Position sizes are maximum suggestions. Never risk more than you can afford to lose.
    Always use stop losses. Past performance doesn't guarantee future results.
    """)

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
