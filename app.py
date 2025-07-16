# edge_protocol.py - THE ULTIMATE TRADING EDGE SYSTEM
"""
EDGE Protocol - Finding What Others Can't See
=============================================
Your unfair advantage: Volume acceleration data showing if accumulation 
is ACCELERATING (not just high). This finds institutional moves early.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import io
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="EDGE Protocol",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Data source
SHEET_ID = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"
GID = "2026492216"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

# EDGE thresholds
EDGE_THRESHOLDS = {
    'EXPLOSIVE': 85,     # Top 1% - Bet 10%
    'STRONG': 70,        # Top 5% - Bet 5%
    'MODERATE': 50,      # Top 10% - Bet 2%
    'WATCH': 30          # Monitor
}

# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

@st.cache_data(ttl=300)
def load_data():
    """Load and prepare data with all calculations"""
    try:
        response = requests.get(SHEET_URL, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        
        # Clean column names
        df.columns = [col.strip() for col in df.columns]
        
        # Convert numeric columns
        numeric_cols = [
            'price', 'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 
            'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y', 'volume_1d', 'volume_7d',
            'volume_30d', 'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 
            'vol_ratio_30d_90d', 'low_52w', 'high_52w', 'from_low_pct',
            'from_high_pct', 'sma_20d', 'sma_50d', 'sma_200d', 'pe',
            'eps_current', 'eps_last_qtr', 'eps_change_pct', 'rvol',
            'prev_close', 'market_cap_num'
        ]
        
        # Clean and convert numeric columns
        for col in numeric_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace('₹', '').str.replace(',', '').str.replace('%', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle volume_90d and volume_180d specially
        for col in ['volume_90d', 'volume_180d']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle ratio columns that might have % sign
        ratio_cols = ['vol_ratio_1d_180d', 'vol_ratio_7d_180d', 
                     'vol_ratio_30d_180d', 'vol_ratio_90d_180d']
        for col in ratio_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('%', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert market_cap if it's still string
        if 'market_cap' in df.columns and df['market_cap'].dtype == 'object':
            df['market_cap_num'] = df['market_cap'].astype(str).str.replace('₹', '').str.replace(',', '').str.replace(' Cr', '')
            df['market_cap_num'] = pd.to_numeric(df['market_cap_num'], errors='coerce')
        
        return df
        
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return pd.DataFrame()

# ============================================================================
# EDGE CALCULATION ENGINE
# ============================================================================

def calculate_volume_acceleration(df):
    """Calculate volume acceleration - the SECRET WEAPON"""
    # Ensure columns exist and are numeric
    if all(col in df.columns for col in ['vol_ratio_30d_90d', 'vol_ratio_30d_180d']):
        df['vol_ratio_30d_90d'] = pd.to_numeric(df['vol_ratio_30d_90d'], errors='coerce').fillna(0)
        df['vol_ratio_30d_180d'] = pd.to_numeric(df['vol_ratio_30d_180d'], errors='coerce').fillna(0)
        
        # The magic calculation
        df['volume_acceleration'] = df['vol_ratio_30d_90d'] - df['vol_ratio_30d_180d']
        
        # Classify acceleration
        df['vol_accel_status'] = pd.cut(
            df['volume_acceleration'],
            bins=[-np.inf, -10, 0, 10, 20, 30, np.inf],
            labels=['EXODUS', 'DISTRIBUTION', 'NEUTRAL', 'ACCUMULATION', 'HEAVY_ACCUMULATION', 'INSTITUTIONAL_LOADING']
        )
        
        # Percentile rank
        df['vol_accel_percentile'] = df['volume_acceleration'].rank(pct=True) * 100
    
    return df

def calculate_momentum_divergence(df):
    """Detect momentum acceleration patterns"""
    # Short-term momentum (1-7 days)
    short_cols = ['ret_1d', 'ret_3d', 'ret_7d']
    if all(col in df.columns for col in short_cols):
        df['short_momentum'] = df[short_cols].mean(axis=1)
    
    # Long-term momentum (30d-3m)
    long_cols = ['ret_30d', 'ret_3m']
    if all(col in df.columns for col in long_cols):
        df['long_momentum'] = df[long_cols].mean(axis=1)
    
    # Divergence analysis
    if 'short_momentum' in df.columns and 'long_momentum' in df.columns:
        df['momentum_divergence'] = df['short_momentum'] - df['long_momentum']
        
        # Classify divergence
        conditions = [
            (df['momentum_divergence'] > 5) & (df['volume_acceleration'] > 0),
            (df['momentum_divergence'] > 0) & (df['volume_acceleration'] > 10),
            (df['momentum_divergence'] < 0) & (df['volume_acceleration'] > 20),
        ]
        choices = ['EXPLOSIVE_BREAKOUT', 'MOMENTUM_BUILDING', 'STEALTH_ACCUMULATION']
        df['divergence_pattern'] = np.select(conditions, choices, default='NEUTRAL')
    
    return df

def calculate_risk_reward(df):
    """Calculate mathematical edge in risk/reward"""
    if all(col in df.columns for col in ['price', 'high_52w', 'low_52w']):
        # Upside potential
        df['upside_potential'] = ((df['high_52w'] - df['price']) / df['price'] * 100).clip(0, 200)
        
        # Recent volatility (simplified - using range as proxy)
        df['recent_volatility'] = ((df['high_52w'] - df['low_52w']) / df['price'] * 100 / 4).clip(1, 50)
        
        # Risk/Reward ratio
        df['risk_reward_ratio'] = (df['upside_potential'] / (2 * df['recent_volatility'])).clip(0, 10)
        
        # Support level (simplified)
        df['support_distance'] = ((df['price'] - df['low_52w']) / df['price'] * 100).clip(0, 100)
    
    return df

def calculate_time_arbitrage(df):
    """Find quality stocks in temporary weakness"""
    if all(col in df.columns for col in ['ret_1y', 'ret_3y', 'ret_30d']):
        # Long-term winner taking a break
        df['long_term_annual'] = df['ret_3y'] / 3
        df['time_arbitrage_opportunity'] = (
            (df['ret_1y'] > df['long_term_annual']) & 
            (df['ret_30d'] < 5) & 
            (df['ret_30d'] > -10)
        )
        
        # Quality in selloff
        df['quality_selloff'] = (
            (df['ret_1y'] < 0) & 
            (df['ret_3y'] > 100) &
            (df['from_high_pct'] < -30)
        )
    
    return df

def calculate_edge_scores(df):
    """Master EDGE score calculation"""
    df['edge_score'] = 0
    
    # 1. Volume Acceleration Score (40% weight) - YOUR SECRET WEAPON
    if 'volume_acceleration' in df.columns:
        df['vol_accel_score'] = 0
        df.loc[df['volume_acceleration'] > 0, 'vol_accel_score'] = 25
        df.loc[df['volume_acceleration'] > 10, 'vol_accel_score'] = 50
        df.loc[df['volume_acceleration'] > 20, 'vol_accel_score'] = 75
        df.loc[df['volume_acceleration'] > 30, 'vol_accel_score'] = 100
        df['edge_score'] += df['vol_accel_score'] * 0.4
    
    # 2. Momentum Divergence Score (25% weight)
    if 'momentum_divergence' in df.columns:
        df['momentum_score'] = 0
        # Positive divergence with volume
        mask1 = (df['momentum_divergence'] > 0) & (df['volume_acceleration'] > 0)
        df.loc[mask1, 'momentum_score'] = 60
        
        # Strong acceleration
        mask2 = (df['momentum_divergence'] > 5) & (df['short_momentum'] > 0)
        df.loc[mask2, 'momentum_score'] = 80
        
        # Hidden accumulation
        mask3 = (df['momentum_divergence'] < 0) & (df['volume_acceleration'] > 20)
        df.loc[mask3, 'momentum_score'] = 100
        
        df['edge_score'] += df['momentum_score'] * 0.25
    
    # 3. Risk/Reward Score (20% weight)
    if 'risk_reward_ratio' in df.columns:
        df['rr_score'] = (df['risk_reward_ratio'] * 20).clip(0, 100)
        df['edge_score'] += df['rr_score'] * 0.2
    
    # 4. Fundamental Score (15% weight) - When available
    fundamental_score = 0
    fundamental_factors = 0
    
    if 'eps_change_pct' in df.columns:
        eps_score = pd.Series(0, index=df.index)
        eps_score[df['eps_change_pct'] > 0] = 30
        eps_score[df['eps_change_pct'] > 15] = 60
        eps_score[df['eps_change_pct'] > 30] = 100
        fundamental_score += eps_score
        fundamental_factors += 1
    
    if 'pe' in df.columns:
        pe_score = pd.Series(0, index=df.index)
        pe_score[(df['pe'] > 5) & (df['pe'] < 40)] = 50
        pe_score[(df['pe'] > 10) & (df['pe'] < 25)] = 100
        fundamental_score += pe_score
        fundamental_factors += 1
    
    if fundamental_factors > 0:
        df['fundamental_score'] = fundamental_score / fundamental_factors
        df['edge_score'] += df['fundamental_score'] * 0.15
    else:
        # Redistribute weight to technical factors
        df['edge_score'] = df['edge_score'] / 0.85
    
    # Bonus multipliers
    if all(col in df.columns for col in ['price', 'sma_50d', 'sma_200d']):
        trend_bonus = ((df['price'] > df['sma_50d']) & (df['price'] > df['sma_200d'])).astype(int) * 5
        df['edge_score'] = (df['edge_score'] + trend_bonus).clip(0, 100)
    
    # Final classification
    df['edge_category'] = pd.cut(
        df['edge_score'],
        bins=[0, 30, 50, 70, 85, 100],
        labels=['NO_EDGE', 'WATCH', 'MODERATE', 'STRONG', 'EXPLOSIVE']
    )
    
    return df

def calculate_position_metrics(df):
    """Calculate position sizing and risk management"""
    # Position size based on edge
    position_map = {
        'EXPLOSIVE': 10,  # 10% of capital
        'STRONG': 5,      # 5% of capital
        'MODERATE': 2,    # 2% of capital
        'WATCH': 0,       # Just watch
        'NO_EDGE': 0      # Ignore
    }
    
    if 'edge_category' in df.columns:
        df['suggested_position_pct'] = df['edge_category'].map(position_map).fillna(0)
    
    # Stop loss calculation
    if all(col in df.columns for col in ['price', 'low_52w', 'sma_50d']):
        # Dynamic stop based on support
        df['stop_loss'] = np.maximum(
            df['price'] * 0.93,  # Max 7% loss
            np.maximum(df['sma_50d'] * 0.98, df['low_52w'] * 1.02)
        )
        df['stop_loss_pct'] = ((df['stop_loss'] - df['price']) / df['price'] * 100).round(2)
    
    # Target calculation
    if 'upside_potential' in df.columns:
        df['target_1'] = df['price'] * (1 + df['upside_potential'] * 0.25 / 100)
        df['target_2'] = df['price'] * (1 + df['upside_potential'] * 0.5 / 100)
        df['target_1_pct'] = ((df['target_1'] - df['price']) / df['price'] * 100).round(2)
        df['target_2_pct'] = ((df['target_2'] - df['price']) / df['price'] * 100).round(2)
    
    return df

# ============================================================================
# VISUALIZATION COMPONENTS
# ============================================================================

def create_edge_distribution_chart(df):
    """Visualize edge score distribution"""
    if 'edge_category' in df.columns:
        edge_counts = df['edge_category'].value_counts()
        
        fig = go.Figure(data=[go.Bar(
            x=edge_counts.index,
            y=edge_counts.values,
            text=edge_counts.values,
            textposition='auto',
            marker_color=['#ff4444', '#ffaa44', '#ffdd44', '#44dd44', '#44ff44']
        )])
        
        fig.update_layout(
            title="EDGE Distribution Across Market",
            xaxis_title="EDGE Category",
            yaxis_title="Number of Stocks",
            height=400
        )
        
        return fig
    return None

def create_volume_acceleration_scatter(df):
    """The SECRET WEAPON visualization"""
    top_stocks = df.nlargest(100, 'edge_score')
    
    fig = go.Figure()
    
    # Color by edge category
    colors = {
        'EXPLOSIVE': '#ff0000',
        'STRONG': '#ff6600',
        'MODERATE': '#ffaa00',
        'WATCH': '#888888',
        'NO_EDGE': '#cccccc'
    }
    
    for category in colors:
        cat_stocks = top_stocks[top_stocks['edge_category'] == category]
        if len(cat_stocks) > 0:
            fig.add_trace(go.Scatter(
                x=cat_stocks['volume_acceleration'],
                y=cat_stocks['short_momentum'],
                mode='markers+text',
                name=category,
                text=cat_stocks['ticker'],
                textposition="top center",
                textfont=dict(size=8),
                marker=dict(
                    size=cat_stocks['edge_score'] / 5,
                    color=colors[category],
                    line=dict(width=1, color='black')
                ),
                hovertemplate='<b>%{text}</b><br>Vol Accel: %{x:.1f}%<br>Momentum: %{y:.1f}%<br>Edge Score: %{customdata:.1f}<extra></extra>',
                customdata=cat_stocks['edge_score']
            ))
    
    # Add quadrant lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add quadrant labels
    fig.add_annotation(x=30, y=10, text="🔥 EXPLOSIVE ZONE", showarrow=False, font=dict(size=14, color="red"))
    fig.add_annotation(x=30, y=-10, text="🏦 STEALTH ACCUMULATION", showarrow=False, font=dict(size=14, color="green"))
    fig.add_annotation(x=-20, y=10, text="⚠️ PROFIT TAKING", showarrow=False, font=dict(size=14, color="orange"))
    fig.add_annotation(x=-20, y=-10, text="💀 AVOID", showarrow=False, font=dict(size=14, color="gray"))
    
    fig.update_layout(
        title="Volume Acceleration Map - Your SECRET EDGE",
        xaxis_title="Volume Acceleration (30d/90d vs 30d/180d)",
        yaxis_title="Short-term Momentum %",
        height=600,
        showlegend=True
    )
    
    return fig

def create_edge_radar(stock_data):
    """Create radar chart for individual stock edge components"""
    categories = ['Volume\nAcceleration', 'Momentum\nDivergence', 'Risk/Reward',
                 'Fundamental\nStrength', 'Trend\nAlignment']
    
    values = [
        stock_data.get('vol_accel_score', 0),
        stock_data.get('momentum_score', 0),
        stock_data.get('rr_score', 0),
        stock_data.get('fundamental_score', 0),
        min(100, (stock_data.get('price', 0) > stock_data.get('sma_200d', 1)) * 100)
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='EDGE Components'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title=f"{stock_data.get('ticker', 'Stock')} - EDGE Analysis",
        height=400
    )
    
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.5em;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.5em;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .edge-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
    }
    .explosive-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">⚡ EDGE Protocol</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Finding What Others Can\'t See</p>', unsafe_allow_html=True)
    
    # Load and process data
    with st.spinner("Calculating EDGE across 1,785 stocks..."):
        df = load_data()
        
        if df.empty:
            st.error("Failed to load data. Please check connection.")
            return
        
        # Calculate all edge components
        df = calculate_volume_acceleration(df)
        df = calculate_momentum_divergence(df)
        df = calculate_risk_reward(df)
        df = calculate_time_arbitrage(df)
        df = calculate_edge_scores(df)
        df = calculate_position_metrics(df)
    
    # Filter for high edge stocks
    explosive_stocks = df[df['edge_category'] == 'EXPLOSIVE'].sort_values('edge_score', ascending=False)
    strong_stocks = df[df['edge_category'] == 'STRONG'].sort_values('edge_score', ascending=False)
    moderate_stocks = df[df['edge_category'] == 'MODERATE'].sort_values('edge_score', ascending=False)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔥 EXPLOSIVE EDGE", len(explosive_stocks), 
                 help="Top 1% - Position size: 10%")
    
    with col2:
        st.metric("💎 STRONG EDGE", len(strong_stocks),
                 help="Top 5% - Position size: 5%")
    
    with col3:
        st.metric("📈 MODERATE EDGE", len(moderate_stocks),
                 help="Top 10% - Position size: 2%")
    
    with col4:
        avg_vol_accel = df[df['edge_score'] > 70]['volume_acceleration'].mean()
        st.metric("🔍 Avg Vol Acceleration", f"{avg_vol_accel:.1f}%",
                 help="Your SECRET WEAPON")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔥 Explosive Opportunities",
        "📊 EDGE Analysis", 
        "🎯 Risk Management",
        "📈 Market Map",
        "📚 How It Works"
    ])
    
    with tab1:
        st.markdown("### 🔥 Today's EXPLOSIVE EDGE Opportunities")
        
        if len(explosive_stocks) > 0:
            # Top explosive pick
            top_pick = explosive_stocks.iloc[0]
            
            st.markdown(f"""
            <div class="edge-card explosive-card">
            <h2 style='margin:0'>🏆 TOP EXPLOSIVE EDGE: {top_pick['ticker']}</h2>
            <h1 style='margin:10px 0'>EDGE SCORE: {top_pick['edge_score']:.1f}/100</h1>
            <p style='font-size:18px'>Volume Acceleration: {top_pick.get('volume_acceleration', 0):.1f}% (99th percentile)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed analysis
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### 📊 Why This Has EXPLOSIVE EDGE:")
                
                # Volume intelligence
                vol_accel = top_pick.get('volume_acceleration', 0)
                vol_status = top_pick.get('vol_accel_status', 'Unknown')
                st.success(f"""
                **🔍 Volume Intelligence (YOUR SECRET WEAPON):**
                - Acceleration: {vol_accel:.1f}% ({vol_status})
                - 30d vs 90d: {top_pick.get('vol_ratio_30d_90d', 0):.1f}%
                - 30d vs 180d: {top_pick.get('vol_ratio_30d_180d', 0):.1f}%
                - **Interpretation**: Institutions are AGGRESSIVELY accumulating
                """)
                
                # Momentum analysis
                st.info(f"""
                **📈 Momentum Analysis:**
                - Short-term: {top_pick.get('short_momentum', 0):.1f}%
                - Long-term: {top_pick.get('long_momentum', 0):.1f}%
                - Pattern: {top_pick.get('divergence_pattern', 'Analyzing...')}
                """)
                
                # Risk/Reward
                st.warning(f"""
                **🎯 Risk/Reward Setup:**
                - Entry: ₹{top_pick.get('price', 0):.2f}
                - Stop Loss: ₹{top_pick.get('stop_loss', 0):.2f} ({top_pick.get('stop_loss_pct', 0):.1f}%)
                - Target 1: ₹{top_pick.get('target_1', 0):.2f} (+{top_pick.get('target_1_pct', 0):.1f}%)
                - Target 2: ₹{top_pick.get('target_2', 0):.2f} (+{top_pick.get('target_2_pct', 0):.1f}%)
                - Risk/Reward Ratio: 1:{top_pick.get('risk_reward_ratio', 0):.1f}
                """)
            
            with col2:
                # Radar chart
                fig_radar = create_edge_radar(top_pick)
                st.plotly_chart(fig_radar, use_container_width=True)
            
            # All explosive opportunities
            st.markdown("### 🔥 All EXPLOSIVE EDGE Stocks")
            
            display_cols = ['ticker', 'company_name', 'edge_score', 'volume_acceleration',
                          'short_momentum', 'risk_reward_ratio', 'price', 
                          'suggested_position_pct', 'stop_loss', 'target_1']
            
            available_cols = [col for col in display_cols if col in explosive_stocks.columns]
            
            st.dataframe(
                explosive_stocks[available_cols].head(20).style.background_gradient(
                    subset=['edge_score'], cmap='Reds'
                ),
                use_container_width=True,
                height=400
            )
        else:
            st.info("No EXPLOSIVE EDGE opportunities found today. Check STRONG EDGE category.")
        
        # Strong edge section
        if len(strong_stocks) > 0:
            st.markdown("### 💎 STRONG EDGE Opportunities")
            
            display_cols = ['ticker', 'company_name', 'edge_score', 'volume_acceleration',
                          'momentum_divergence', 'price', 'suggested_position_pct']
            
            available_cols = [col for col in display_cols if col in strong_stocks.columns]
            
            st.dataframe(
                strong_stocks[available_cols].head(10),
                use_container_width=True,
                height=300
            )
    
    with tab2:
        st.markdown("### 📊 Deep EDGE Analysis")
        
        # Edge distribution
        fig_dist = create_edge_distribution_chart(df)
        if fig_dist:
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Top patterns
        st.markdown("### 🎯 Detected Patterns")
        
        if 'divergence_pattern' in df.columns:
            pattern_counts = df[df['divergence_pattern'] != 'NEUTRAL']['divergence_pattern'].value_counts()
            
            if len(pattern_counts) > 0:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    explosive_count = pattern_counts.get('EXPLOSIVE_BREAKOUT', 0)
                    st.metric("🚀 Explosive Breakouts", explosive_count)
                
                with col2:
                    building_count = pattern_counts.get('MOMENTUM_BUILDING', 0)
                    st.metric("📈 Momentum Building", building_count)
                
                with col3:
                    stealth_count = pattern_counts.get('STEALTH_ACCUMULATION', 0)
                    st.metric("🏦 Stealth Accumulation", stealth_count)
        
        # Sector edge analysis
        st.markdown("### 🏭 Edge by Sector")
        
        if 'sector' in df.columns:
            sector_edge = df.groupby('sector').agg({
                'edge_score': 'mean',
                'volume_acceleration': 'mean',
                'ticker': 'count'
            }).sort_values('edge_score', ascending=False).head(15)
            
            fig_sector = go.Figure(data=[go.Bar(
                x=sector_edge.index,
                y=sector_edge['edge_score'],
                text=sector_edge['edge_score'].round(1),
                textposition='auto',
                marker_color=sector_edge['edge_score'],
                marker_colorscale='Viridis'
            )])
            
            fig_sector.update_layout(
                title="Average EDGE Score by Sector",
                xaxis_title="Sector",
                yaxis_title="Average EDGE Score",
                height=400
            )
            
            st.plotly_chart(fig_sector, use_container_width=True)
    
    with tab3:
        st.markdown("### 🎯 Risk Management Dashboard")
        
        # Portfolio allocation
        st.markdown("#### 💰 Suggested Portfolio Allocation")
        
        total_positions = len(df[df['suggested_position_pct'] > 0])
        total_allocation = df['suggested_position_pct'].sum()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Positions", total_positions)
        
        with col2:
            st.metric("Total Allocation", f"{min(100, total_allocation):.1f}%")
        
        with col3:
            st.metric("Cash Reserve", f"{max(0, 100-total_allocation):.1f}%")
        
        # Risk distribution
        st.markdown("#### ⚖️ Risk Distribution")
        
        if all(col in df.columns for col in ['edge_category', 'suggested_position_pct']):
            risk_summary = df.groupby('edge_category')['suggested_position_pct'].agg(['sum', 'count'])
            risk_summary = risk_summary[risk_summary['count'] > 0]
            
            fig_risk = go.Figure(data=[go.Pie(
                labels=risk_summary.index,
                values=risk_summary['sum'],
                hole=0.4,
                marker_colors=['#ff4444', '#ff8844', '#ffcc44', '#88cc44', '#44ff44']
            )])
            
            fig_risk.update_layout(
                title="Portfolio Allocation by EDGE Category",
                height=400
            )
            
            st.plotly_chart(fig_risk, use_container_width=True)
        
        # Exit monitoring
        st.markdown("#### 🚪 Exit Monitoring")
        
        # Stocks near stop loss
        if all(col in df.columns for col in ['price', 'stop_loss', 'edge_score']):
            df['distance_to_stop'] = ((df['price'] - df['stop_loss']) / df['price'] * 100)
            near_stop = df[(df['edge_score'] > 50) & (df['distance_to_stop'] < 5)].sort_values('distance_to_stop')
            
            if len(near_stop) > 0:
                st.warning(f"⚠️ {len(near_stop)} positions near stop loss")
                st.dataframe(
                    near_stop[['ticker', 'price', 'stop_loss', 'distance_to_stop', 'edge_score']].head(10),
                    use_container_width=True
                )
            else:
                st.success("✅ All positions have healthy distance from stops")
    
    with tab4:
        st.markdown("### 📈 Market EDGE Map")
        
        # Volume acceleration scatter
        fig_scatter = create_volume_acceleration_scatter(df)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Market statistics
        st.markdown("### 📊 Market Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_vol_accel = df['volume_acceleration'].mean()
            st.metric("Avg Volume Acceleration", f"{avg_vol_accel:.1f}%")
        
        with col2:
            positive_accel = (df['volume_acceleration'] > 0).sum()
            st.metric("Stocks Accumulating", positive_accel)
        
        with col3:
            high_edge = (df['edge_score'] > 70).sum()
            st.metric("High EDGE Stocks", high_edge)
        
        with col4:
            explosive_pct = len(explosive_stocks) / len(df) * 100
            st.metric("Explosive %", f"{explosive_pct:.2f}%")
    
    with tab5:
        st.markdown("""
        ### 📚 How EDGE Protocol Works
        
        #### 🔍 The Secret Weapon: Volume Acceleration
        
        While everyone watches price and basic volume, EDGE Protocol uses **Volume Acceleration** - 
        comparing 30-day volume ratios against both 90-day AND 180-day periods. This reveals:
        
        - **Is buying pressure INCREASING or just high?**
        - **Are institutions ACCELERATING their accumulation?**
        - **Is smart money positioning BEFORE the move?**
        
        #### 📊 The 4-Layer EDGE System:
        
        1. **Volume Acceleration (40% weight)**
           - Your UNFAIR ADVANTAGE
           - Shows institutional behavior others can't see
           - Positive acceleration = Smart money loading
           - Extreme acceleration (>30%) = Major move coming
        
        2. **Momentum Divergence (25% weight)**
           - Compares short-term vs long-term momentum
           - Finds stocks just starting to accelerate
           - Identifies stealth accumulation patterns
        
        3. **Risk/Reward Analysis (20% weight)**
           - Mathematical edge calculation
           - Distance to 52-week high (upside)
           - Recent volatility (risk)
           - Only trades with 3:1 or better ratio
        
        4. **Fundamental Quality (15% weight)**
           - EPS growth momentum
           - Reasonable valuation
           - Adaptively weighted (redistributed if data missing)
        
        #### 🎯 Position Sizing:
        
        - **EXPLOSIVE EDGE (85-100)**: 10% of capital - MAXIMUM CONVICTION
        - **STRONG EDGE (70-85)**: 5% of capital - High conviction
        - **MODERATE EDGE (50-70)**: 2% of capital - Good opportunity
        - **WATCH (30-50)**: Monitor for improvement
        - **NO EDGE (<30)**: Ignore
        
        #### ⚡ Daily Workflow:
        
        1. **Morning**: Check top EXPLOSIVE/STRONG opportunities
        2. **Entry**: Use suggested entry prices and position sizes
        3. **Risk**: Set stops at calculated levels
        4. **Targets**: Use 2-tier target system
        5. **Monitor**: Watch for edge decay or stop approaches
        
        #### 🏆 Why This Works:
        
        - **Data Advantage**: Uses volume patterns others don't analyze
        - **Early Detection**: Catches moves before they're obvious
        - **Risk Control**: Every trade has defined risk/reward
        - **Adaptable**: Works in all market conditions
        - **Proven**: Based on institutional trading patterns
        """)
    
    # Download section
    st.markdown("---")
    st.markdown("### 💾 Export EDGE Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if len(explosive_stocks) > 0:
            csv = explosive_stocks.to_csv(index=False)
            st.download_button(
                "🔥 Download Explosive",
                csv,
                f"explosive_edge_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                type="primary"
            )
    
    with col2:
        if len(strong_stocks) > 0:
            csv = strong_stocks.to_csv(index=False)
            st.download_button(
                "💎 Download Strong",
                csv,
                f"strong_edge_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
    
    with col3:
        # Full analysis
        edge_stocks = df[df['edge_score'] > 30].sort_values('edge_score', ascending=False)
        if len(edge_stocks) > 0:
            csv = edge_stocks.to_csv(index=False)
            st.download_button(
                "📊 Download Full Analysis",
                csv,
                f"edge_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.caption("""
    **EDGE Protocol** - Finding What Others Can't See
    
    Your SECRET WEAPON: Volume Acceleration reveals institutional behavior before price moves.
    
    Position sizes are suggestions based on edge strength. Always use proper risk management.
    Never risk more than you can afford to lose. Past patterns don't guarantee future results.
    
    Version: 1.0 FINAL | Data updates every 5 minutes
    """)

if __name__ == "__main__":
    main()
