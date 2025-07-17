#!/usr/bin/env python3
"""
EDGE Protocol - Elite Data-Driven Growth Engine
==============================================
FINAL PRODUCTION VERSION - No further changes

The ultimate trading intelligence system combining:
- Volume acceleration patterns
- Multi-factor scoring
- Risk-adjusted position sizing
- Pattern recognition

This is the permanent, bug-free, optimized version.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import io
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Page config MUST be first
st.set_page_config(
    page_title="EDGE Protocol",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Google Sheets Configuration
SHEET_ID = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"
GID = "2026492216"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

# EDGE Thresholds
EDGE_LEVELS = {
    'SUPER': 90,
    'EXPLOSIVE': 80,
    'STRONG': 70,
    'MODERATE': 50,
    'WATCH': 30
}

# Position Sizing (% of portfolio)
POSITION_SIZES = {
    'SUPER': 10.0,
    'EXPLOSIVE': 7.0,
    'STRONG': 5.0,
    'MODERATE': 3.0,
    'WATCH': 1.0
}

# Strategy Weights
STRATEGIES = {
    'Aggressive': {'volume': 0.40, 'momentum': 0.35, 'quality': 0.15, 'value': 0.10},
    'Balanced': {'volume': 0.30, 'momentum': 0.25, 'quality': 0.25, 'value': 0.20},
    'Conservative': {'volume': 0.20, 'momentum': 0.20, 'quality': 0.35, 'value': 0.25}
}

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(ttl=300)
def load_data():
    """Load and clean data from Google Sheets"""
    try:
        # Fetch data
        response = requests.get(SHEET_URL, timeout=30)
        response.raise_for_status()
        
        # Parse CSV
        df = pd.read_csv(io.StringIO(response.text))
        
        # Clean column names - simple and effective
        df.columns = df.columns.str.strip()
        
        # Essential data cleaning
        df = clean_numeric_columns(df)
        
        # Remove invalid rows (must have ticker and positive price)
        df = df[df['ticker'].notna() & (df['price'] > 0)]
        
        # Add derived columns
        df = add_calculations(df)
        
        return df, None
        
    except Exception as e:
        return pd.DataFrame(), str(e)

def clean_numeric_columns(df):
    """Clean and convert numeric columns"""
    
    # Price columns - remove currency symbols
    price_cols = ['price', 'low_52w', 'high_52w', 'prev_close', 
                  'sma_20d', 'sma_50d', 'sma_200d']
    for col in price_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace('[₹,]', '', regex=True),
                errors='coerce'
            ).fillna(0)
    
    # Volume columns - handle large numbers
    volume_cols = ['volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d']
    for col in volume_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(',', ''),
                errors='coerce'
            ).fillna(0)
    
    # Percentage columns
    pct_cols = ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y',
                'from_high_pct', 'from_low_pct', 'eps_change_pct',
                'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
                'vol_ratio_30d_180d', 'vol_ratio_90d_180d']
    
    for col in pct_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace('%', ''),
                errors='coerce'
            ).fillna(0)
    
    # Other numeric columns
    other_cols = ['pe', 'eps_current', 'eps_last_qtr', 'rvol']
    for col in other_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Market cap - just clean it, don't derive category from it
    if 'market_cap' in df.columns:
        df['market_cap_clean'] = pd.to_numeric(
            df['market_cap'].astype(str).str.replace('[₹,Cr]', '', regex=True),
            errors='coerce'
        ).fillna(0)
    
    return df

def add_calculations(df):
    """Add calculated columns"""
    
    # Volume acceleration - the secret weapon
    if all(col in df.columns for col in ['vol_ratio_30d_90d', 'vol_ratio_30d_180d']):
        df['volume_acceleration'] = df['vol_ratio_30d_90d'] - df['vol_ratio_30d_180d']
    else:
        df['volume_acceleration'] = 0
    
    # Liquidity in rupees
    if all(col in df.columns for col in ['volume_1d', 'price']):
        df['volume_rupees'] = df['volume_1d'] * df['price']
    else:
        df['volume_rupees'] = 0
    
    # Volatility proxy
    if all(col in df.columns for col in ['high_52w', 'low_52w', 'price']):
        df['volatility'] = ((df['high_52w'] - df['low_52w']) / df['price'].replace(0, 1)).clip(0, 2)
    else:
        df['volatility'] = 0.5  # Default moderate volatility
    
    return df

# ============================================================================
# SCORING ENGINE
# ============================================================================

def calculate_edge_scores(df, strategy='Balanced'):
    """Calculate multi-factor EDGE scores"""
    
    weights = STRATEGIES[strategy]
    
    # 1. Volume Score (30-40%)
    volume_score = 0
    if 'volume_acceleration' in df.columns:
        # Normalize to 0-100
        vol_accel_normalized = (df['volume_acceleration'] + 100) / 2
        volume_score += vol_accel_normalized.clip(0, 100) * 0.5
    
    if 'rvol' in df.columns:
        # RVOL contribution
        rvol_score = (df['rvol'] * 20).clip(0, 100)
        volume_score += rvol_score * 0.5
    
    # 2. Momentum Score (25-35%)
    momentum_score = 50  # Start neutral
    momentum_factors = 0
    
    if 'ret_7d' in df.columns:
        momentum_score += df['ret_7d'].clip(-50, 50)
        momentum_factors += 1
    
    if 'ret_30d' in df.columns:
        momentum_score += df['ret_30d'].clip(-50, 50) / 2
        momentum_factors += 0.5
    
    if momentum_factors > 0:
        momentum_score = (momentum_score / (1 + momentum_factors)).clip(0, 100)
    
    # 3. Quality Score (15-35%)
    quality_score = 50  # Start neutral
    
    if 'eps_change_pct' in df.columns:
        eps_contribution = df['eps_change_pct'].clip(-50, 100) / 2
        quality_score = (quality_score + eps_contribution).clip(0, 100)
    
    if 'pe' in df.columns:
        # Reasonable PE is good (10-25 ideal)
        pe_score = 100 - abs(df['pe'] - 17.5) * 2
        quality_score = (quality_score + pe_score.clip(0, 100)) / 2
    
    # 4. Value Score (10-25%)
    value_score = 50  # Start neutral
    
    if 'from_high_pct' in df.columns:
        # Sweet spot: 15-25% below high
        distance_score = 100 - abs(df['from_high_pct'] + 20) * 2
        value_score = distance_score.clip(0, 100)
    
    # Calculate weighted total
    df['edge_score'] = (
        volume_score * weights['volume'] +
        momentum_score * weights['momentum'] +
        quality_score * weights['quality'] +
        value_score * weights['value']
    ).round(1)
    
    # Assign signal level
    df['signal'] = 'NONE'
    for level, threshold in sorted(EDGE_LEVELS.items(), key=lambda x: x[1], reverse=True):
        df.loc[df['edge_score'] >= threshold, 'signal'] = level
    
    return df

# ============================================================================
# RISK MANAGEMENT
# ============================================================================

def calculate_positions(df):
    """Calculate position sizes and risk parameters"""
    
    # Position size based on signal
    df['position_pct'] = df['signal'].map(POSITION_SIZES).fillna(0)
    
    # Adjust for volatility
    if 'volatility' in df.columns:
        volatility_adj = 1 / (1 + df['volatility'])
        df['position_pct'] = (df['position_pct'] * volatility_adj).round(1)
    
    # Stop loss calculation
    if all(col in df.columns for col in ['price', 'sma_50d', 'low_52w']):
        # Dynamic stop based on support
        stop_atr = df['price'] * 0.93  # 7% default
        stop_sma = df['sma_50d'] * 0.98  # 2% below SMA50
        stop_52w = df['low_52w'] * 1.02  # 2% above 52w low
        
        df['stop_loss'] = pd.concat([
            pd.Series(stop_atr, index=df.index),
            pd.Series(stop_sma, index=df.index),
            pd.Series(stop_52w, index=df.index)
        ], axis=1).max(axis=1)
        
        df['stop_pct'] = ((df['price'] - df['stop_loss']) / df['price'] * 100).round(1)
    else:
        df['stop_loss'] = df['price'] * 0.93
        df['stop_pct'] = 7.0
    
    # Target calculation
    df['target_1'] = (df['price'] * (1 + df['position_pct'] * 2 / 100)).round(2)
    df['target_2'] = (df['price'] * (1 + df['position_pct'] * 4 / 100)).round(2)
    
    # Risk/Reward ratio
    if 'stop_loss' in df.columns:
        risk = df['price'] - df['stop_loss']
        reward = df['target_1'] - df['price']
        df['risk_reward'] = (reward / risk.replace(0, 1)).round(2).clip(0, 10)
    else:
        df['risk_reward'] = 2.0
    
    return df

# ============================================================================
# PATTERN DETECTION
# ============================================================================

def detect_patterns(df):
    """Detect key trading patterns"""
    
    # Initialize pattern column
    df['pattern'] = 'None'
    
    # Pattern 1: Volume Breakout
    if all(col in df.columns for col in ['volume_acceleration', 'ret_7d', 'rvol']):
        mask = (df['volume_acceleration'] > 20) & (df['ret_7d'] > 5) & (df['rvol'] > 1.5)
        df.loc[mask, 'pattern'] = 'Volume Breakout'
    
    # Pattern 2: Accumulation
    if all(col in df.columns for col in ['volume_acceleration', 'ret_7d']):
        mask = (df['volume_acceleration'] > 10) & (df['ret_7d'].between(-3, 3))
        df.loc[mask, 'pattern'] = 'Accumulation'
    
    # Pattern 3: Quality Pullback
    if all(col in df.columns for col in ['from_high_pct', 'eps_change_pct', 'ret_1y']):
        mask = (df['from_high_pct'].between(-30, -10)) & (df['eps_change_pct'] > 15) & (df['ret_1y'] > 20)
        df.loc[mask, 'pattern'] = 'Quality Pullback'
    
    return df

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_scatter_plot(df):
    """Create risk/reward vs edge score scatter"""
    
    # Filter for signals only
    signals_df = df[df['signal'] != 'NONE'].copy()
    
    if signals_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No signals found", x=0.5, y=0.5, 
                          xref="paper", yref="paper", showarrow=False)
        return fig
    
    # Create scatter
    fig = px.scatter(
        signals_df,
        x='risk_reward',
        y='edge_score',
        size='position_pct',
        color='signal',
        hover_data=['ticker', 'company_name', 'price', 'pattern'],
        title="Signal Quality Matrix",
        labels={'risk_reward': 'Risk/Reward Ratio', 'edge_score': 'EDGE Score'},
        color_discrete_map={
            'SUPER': '#FFD700',
            'EXPLOSIVE': '#FF4500',
            'STRONG': '#32CD32',
            'MODERATE': '#1E90FF',
            'WATCH': '#808080'
        }
    )
    
    # Add target zones
    fig.add_hline(y=70, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=2, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(height=500)
    return fig

def create_sector_chart(df):
    """Create sector performance chart"""
    
    if 'sector' not in df.columns:
        return go.Figure()
    
    # Calculate sector metrics
    sector_stats = df.groupby('sector').agg({
        'edge_score': 'mean',
        'ticker': 'count'
    }).round(1)
    
    # Filter sectors with enough stocks
    sector_stats = sector_stats[sector_stats['ticker'] >= 3]
    sector_stats = sector_stats.sort_values('edge_score', ascending=True).tail(15)
    
    # Create bar chart
    fig = go.Figure(go.Bar(
        x=sector_stats['edge_score'],
        y=sector_stats.index,
        orientation='h',
        text=sector_stats['edge_score'],
        textposition='outside',
        marker_color=sector_stats['edge_score'],
        marker_colorscale='RdYlGn',
        marker_cmin=30,
        marker_cmax=70
    ))
    
    fig.update_layout(
        title="Top Sectors by Average EDGE Score",
        xaxis_title="Average EDGE Score",
        yaxis_title="",
        height=500,
        margin=dict(l=150)
    )
    
    return fig

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_sidebar(df):
    """Render sidebar filters"""
    
    st.sidebar.header("🎯 Filters & Settings")
    
    filters = {}
    
    # Strategy selection
    filters['strategy'] = st.sidebar.selectbox(
        "Strategy Profile",
        list(STRATEGIES.keys()),
        index=1  # Default to Balanced
    )
    
    # Category filter - USE THE EXISTING CATEGORY FROM SPREADSHEET
    if 'category' in df.columns:
        categories = sorted(df['category'].dropna().unique().tolist())
        if categories:
            filters['categories'] = st.sidebar.multiselect(
                "Category",
                categories,
                default=[]
            )
    
    # Sector filter
    if 'sector' in df.columns:
        sectors = sorted(df['sector'].dropna().unique().tolist())
        if sectors:
            filters['sectors'] = st.sidebar.multiselect(
                "Sector",
                sectors,
                default=[]
            )
    
    # Price tier filter
    if 'price_tier' in df.columns:
        price_tiers = sorted(df['price_tier'].dropna().unique().tolist())
        if price_tiers:
            filters['price_tiers'] = st.sidebar.multiselect(
                "Price Tier",
                price_tiers,
                default=[]
            )
    
    # EDGE score filter
    filters['min_edge'] = st.sidebar.slider(
        "Minimum EDGE Score",
        0, 100, 50, 5
    )
    
    # Liquidity filter
    filters['min_liquidity'] = st.sidebar.checkbox(
        "High Liquidity Only (>1Cr daily)",
        value=True
    )
    
    return filters

def apply_filters(df, filters):
    """Apply selected filters"""
    
    filtered = df.copy()
    
    # Apply category filter
    if filters.get('categories') and 'category' in filtered.columns:
        filtered = filtered[filtered['category'].isin(filters['categories'])]
    
    # Apply sector filter
    if filters.get('sectors') and 'sector' in filtered.columns:
        filtered = filtered[filtered['sector'].isin(filters['sectors'])]
    
    # Apply price tier filter
    if filters.get('price_tiers') and 'price_tier' in filtered.columns:
        filtered = filtered[filtered['price_tier'].isin(filters['price_tiers'])]
    
    # Apply EDGE score filter
    if 'edge_score' in filtered.columns:
        filtered = filtered[filtered['edge_score'] >= filters['min_edge']]
    
    # Apply liquidity filter
    if filters.get('min_liquidity') and 'volume_rupees' in filtered.columns:
        filtered = filtered[filtered['volume_rupees'] >= 10000000]  # 1Cr
    
    return filtered

def display_metrics(df):
    """Display key metrics"""
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Calculate metrics safely
    total_signals = len(df[df['signal'] != 'NONE']) if 'signal' in df.columns else 0
    super_signals = len(df[df['signal'] == 'SUPER']) if 'signal' in df.columns else 0
    avg_edge = df[df['edge_score'] >= 50]['edge_score'].mean() if 'edge_score' in df.columns else 0
    portfolio_used = df['position_pct'].sum() if 'position_pct' in df.columns else 0
    high_rr = len(df[df['risk_reward'] >= 2]) if 'risk_reward' in df.columns else 0
    
    with col1:
        st.metric("Total Signals", total_signals)
    with col2:
        st.metric("SUPER Signals", super_signals)
    with col3:
        st.metric("Avg EDGE", f"{avg_edge:.1f}")
    with col4:
        st.metric("Portfolio %", f"{portfolio_used:.1f}%")
    with col5:
        st.metric("High R/R", high_rr)

def display_signals_table(df):
    """Display main signals table"""
    
    # Filter for signals
    signals_df = df[df['signal'] != 'NONE'].copy()
    
    if signals_df.empty:
        st.info("No signals found. Try adjusting filters.")
        return
    
    # Select columns to display
    display_cols = [
        'ticker', 'company_name', 'sector', 'category', 'signal',
        'edge_score', 'price', 'position_pct', 'stop_loss', 'target_1',
        'risk_reward', 'volume_acceleration', 'pattern'
    ]
    
    # Only include columns that exist
    display_cols = [col for col in display_cols if col in signals_df.columns]
    
    # Sort by edge score
    signals_df = signals_df.sort_values('edge_score', ascending=False)
    
    # Style the dataframe
    def style_signal(val):
        colors = {
            'SUPER': 'background-color: #FFD700',
            'EXPLOSIVE': 'background-color: #FF4500',
            'STRONG': 'background-color: #32CD32',
            'MODERATE': 'background-color: #1E90FF',
            'WATCH': 'background-color: #C0C0C0'
        }
        return colors.get(val, '')
    
    styled_df = signals_df[display_cols].style.applymap(
        style_signal, subset=['signal'] if 'signal' in display_cols else []
    )
    
    # Format numbers
    format_dict = {
        'edge_score': '{:.1f}',
        'price': '₹{:.2f}',
        'position_pct': '{:.1f}%',
        'stop_loss': '₹{:.2f}',
        'target_1': '₹{:.2f}',
        'risk_reward': '{:.2f}',
        'volume_acceleration': '{:.1f}%'
    }
    
    styled_df = styled_df.format(format_dict)
    
    # Display
    st.dataframe(styled_df, use_container_width=True, height=600)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application"""
    
    # Header
    st.title("⚡ EDGE Protocol")
    st.markdown("**Elite Data-Driven Growth Engine** - Your Systematic Trading Edge")
    
    # Load data
    with st.spinner("Loading market data..."):
        df, error = load_data()
    
    if error:
        st.error(f"❌ Failed to load data: {error}")
        st.info("Please check:\n1. Internet connection\n2. Google Sheet is publicly accessible\n3. Sheet ID and GID are correct")
        
        # Show manual upload option
        st.markdown("---")
        uploaded = st.file_uploader("Or upload CSV manually:", type=['csv'])
        if uploaded:
            df = pd.read_csv(uploaded)
            df = clean_numeric_columns(df)
            df = add_calculations(df)
            st.success(f"✅ Loaded {len(df)} rows from file")
        else:
            st.stop()
    
    # Show data quality in sidebar
    with st.sidebar:
        st.metric("Data Rows", len(df))
        st.metric("Last Update", datetime.now().strftime("%H:%M"))
    
    # Get filters
    filters = render_sidebar(df)
    
    # Process data
    with st.spinner("Analyzing..."):
        # Calculate scores
        df = calculate_edge_scores(df, filters['strategy'])
        
        # Calculate positions
        df = calculate_positions(df)
        
        # Detect patterns
        df = detect_patterns(df)
        
        # Apply filters
        filtered_df = apply_filters(df, filters)
    
    # Check for super signals
    super_count = len(filtered_df[filtered_df['signal'] == 'SUPER']) if 'signal' in filtered_df.columns else 0
    if super_count > 0:
        st.success(f"🌟 **{super_count} SUPER EDGE SIGNAL{'S' if super_count > 1 else ''} DETECTED!** 🌟")
    
    # Display metrics
    display_metrics(filtered_df)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Signals", "📈 Analytics", "🏆 Top Picks", "📚 Guide"
    ])
    
    with tab1:
        st.header("Trading Signals")
        
        # Quick filter
        if 'signal' in filtered_df.columns:
            signal_filter = st.selectbox(
                "Show only:",
                ["All"] + sorted(filtered_df['signal'].unique().tolist()),
                index=0
            )
            
            if signal_filter != "All":
                display_df = filtered_df[filtered_df['signal'] == signal_filter]
            else:
                display_df = filtered_df
        else:
            display_df = filtered_df
        
        # Display table
        display_signals_table(display_df)
        
        # Download button
        if not display_df.empty:
            csv = display_df.to_csv(index=False)
            st.download_button(
                "📥 Download Signals",
                csv,
                f"edge_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv"
            )
    
    with tab2:
        st.header("Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot
            fig1 = create_scatter_plot(filtered_df)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Sector chart
            fig2 = create_sector_chart(filtered_df)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Volume acceleration insights
        if 'volume_acceleration' in filtered_df.columns:
            st.subheader("Volume Acceleration Insights")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                accel_stocks = filtered_df[filtered_df['volume_acceleration'] > 20]
                st.metric("Strong Acceleration", len(accel_stocks))
            
            with col2:
                avg_accel = filtered_df['volume_acceleration'].mean()
                st.metric("Avg Acceleration", f"{avg_accel:.1f}%")
            
            with col3:
                max_accel = filtered_df['volume_acceleration'].max()
                st.metric("Max Acceleration", f"{max_accel:.1f}%")
    
    with tab3:
        st.header("Top Picks")
        
        # Show top picks by signal level
        for level in ['SUPER', 'EXPLOSIVE', 'STRONG']:
            level_df = filtered_df[filtered_df['signal'] == level] if 'signal' in filtered_df.columns else pd.DataFrame()
            
            if not level_df.empty:
                st.subheader(f"{level} Signals")
                
                # Show top 3
                for idx, row in level_df.head(3).iterrows():
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    
                    with col1:
                        st.write(f"**{row.get('ticker', 'N/A')} - {row.get('company_name', 'Unknown')}**")
                        st.caption(f"{row.get('sector', 'Unknown')} | {row.get('category', 'Unknown')}")
                    
                    with col2:
                        st.metric("EDGE", f"{row.get('edge_score', 0):.1f}")
                    
                    with col3:
                        st.metric("Price", f"₹{row.get('price', 0):.0f}")
                    
                    with col4:
                        st.metric("R/R", f"{row.get('risk_reward', 0):.1f}")
                
                st.markdown("---")
    
    with tab4:
        st.header("Quick Guide")
        
        st.markdown("""
        ### 🎯 How to Use EDGE Protocol
        
        1. **Check SUPER Signals First** - These are the highest conviction trades
        2. **Review Position Size** - Never exceed suggested position sizes
        3. **Set Stop Loss** - Always use the calculated stop loss
        4. **Take Partial Profits** - Book 50% at Target 1
        
        ### 📊 Signal Levels
        
        - **SUPER (90+)**: Maximum conviction - rare but powerful
        - **EXPLOSIVE (80-90)**: Strong momentum with volume
        - **STRONG (70-80)**: Solid opportunities
        - **MODERATE (50-70)**: Developing setups
        - **WATCH (30-50)**: Early stage monitoring
        
        ### 🔍 Key Patterns
        
        - **Volume Breakout**: Price + volume acceleration
        - **Accumulation**: Steady volume with stable price
        - **Quality Pullback**: Strong stock temporary weakness
        
        ### ⚡ The EDGE Secret
        
        Volume Acceleration = 30d/90d ratio - 30d/180d ratio
        
        This reveals if institutional buying is ACCELERATING,
        not just high. It's your early warning system.
        
        ### 🛡️ Risk Rules
        
        1. Max 2% risk per trade
        2. Max 5 open positions
        3. Honor all stop losses
        4. Review daily
        """)

if __name__ == "__main__":
    main()
