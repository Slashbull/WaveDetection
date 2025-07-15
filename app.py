# mantra_ultimate_fixed.py - PRODUCTION READY WITH DATA FORMAT FIXES
"""
M.A.N.T.R.A. Ultimate - Fixed for Your Data Format
=================================================
Handles Unicode issues, percentage formats, and all data quirks
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import re
import requests
from datetime import datetime
import warnings
from scipy import stats
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
@st.cache_resource
def get_config():
    return {
        'SHEET_URL': "https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk/export?format=csv&gid=2026492216",
        'CACHE_TTL': 300,
        'REQUEST_TIMEOUT': 30,
        
        # Ultra-conservative thresholds
        'SIGNAL_THRESHOLDS': {
            "ALPHA_EXTREME": 0.90,   # Top 10%
            "ALPHA_HIGH": 0.80,      # Top 20%
            "ALPHA_MODERATE": 0.65,  # Top 35%
            "MONITOR": 0.50,         # Monitor zone
            "NEUTRAL": 0.35,         # No action
            "CAUTION": 0.20,         # Risk present
            "AVOID": 0.0            # High risk
        },
        
        # Signal colors
        'SIGNAL_COLORS': {
            "ALPHA_EXTREME": "#00ff41",
            "ALPHA_HIGH": "#28a745",
            "ALPHA_MODERATE": "#40c057",
            "MONITOR": "#ffd43b",
            "NEUTRAL": "#868e96",
            "CAUTION": "#ff6b6b",
            "AVOID": "#c92a2a"
        }
    }

CONFIG = get_config()

# ============================================================================
# PAGE SETUP
# ============================================================================
st.set_page_config(
    page_title="M.A.N.T.R.A. Ultimate",
    page_icon="🔱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FIXED DATA LOADING WITH PROPER ENCODING
# ============================================================================
@st.cache_data(ttl=CONFIG['CACHE_TTL'], show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load data with proper handling of Unicode and percentage formats"""
    try:
        # Download with proper encoding
        response = requests.get(CONFIG['SHEET_URL'], timeout=CONFIG['REQUEST_TIMEOUT'])
        response.encoding = 'utf-8'  # Force UTF-8 encoding
        
        # Read CSV
        df = pd.read_csv(io.StringIO(response.text))
        
        # Remove unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Clean column names (already clean in your data)
        df.columns = [col.strip() for col in df.columns]
        
        # Define column patterns for conversion
        # Price columns (with rupee symbol issues)
        price_cols = ['price', 'prev_close', 'low_52w', 'high_52w', 
                      'sma_20d', 'sma_50d', 'sma_200d']
        
        # Percentage columns  
        pct_cols = ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 
                    'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
                    'from_low_pct', 'from_high_pct', 'eps_change_pct',
                    'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d']
        
        # Volume columns
        vol_cols = ['volume_1d', 'volume_7d', 'volume_30d', 'volume_3m']
        
        # Other numeric columns
        other_numeric = ['pe', 'eps_current', 'eps_last_qtr', 'eps_duplicate', 'rvol', 'year']
        
        # Convert price columns (handle rupee symbol)
        for col in price_cols:
            if col in df.columns:
                # Remove any currency symbols and commas
                df[col] = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert percentage columns
        for col in pct_cols:
            if col in df.columns:
                # Remove % sign and convert
                df[col] = df[col].astype(str).str.replace('%', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert volume columns
        for col in vol_cols:
            if col in df.columns:
                # Remove commas
                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert other numeric columns
        for col in other_numeric:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle market cap specially (with Cr notation)
        if 'market_cap' in df.columns:
            # Extract number and multiplier
            df['market_cap_num'] = df['market_cap'].astype(str).str.extract(r'([\d,]+\.?\d*)')
            df['market_cap_num'] = df['market_cap_num'].str.replace(',', '').astype(float)
            
            # Apply multiplier
            df['market_cap'] = df.apply(
                lambda x: x['market_cap_num'] * 1 if 'Cr' in str(x['market_cap']) else x['market_cap_num'],
                axis=1
            )
        
        # Ensure ticker is clean
        if 'ticker' in df.columns:
            df['ticker'] = df['ticker'].astype(str).str.strip().str.upper()
            df = df[df['ticker'].notna() & (df['ticker'] != 'NAN') & (df['ticker'] != '')]
        
        # Remove rows where price is null
        if 'price' in df.columns:
            df = df[df['price'].notna() & (df['price'] > 0)]
        
        # Add timestamp
        df['data_timestamp'] = datetime.now()
        
        return df
        
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return pd.DataFrame()

# ============================================================================
# SIMPLIFIED BUT POWERFUL ANALYSIS ENGINE
# ============================================================================
class SimpleAlphaEngine:
    """Simplified but robust alpha scoring"""
    
    @staticmethod
    def calculate_alpha_score(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate alpha score with available data"""
        
        # Initialize scores
        df['momentum_score'] = 0.5
        df['volume_score'] = 0.5
        df['technical_score'] = 0.5
        df['fundamental_score'] = 0.5
        df['alpha_score'] = 0.5
        df['signal'] = 'NEUTRAL'
        
        # 1. MOMENTUM SCORE - Multi-timeframe analysis
        momentum_components = []
        
        # Short-term momentum
        if all(col in df.columns for col in ['ret_1d', 'ret_7d']):
            short_momentum = (df['ret_1d'] + df['ret_7d']) / 2
            momentum_components.append(short_momentum)
        
        # Medium-term momentum  
        if all(col in df.columns for col in ['ret_30d', 'ret_3m']):
            medium_momentum = (df['ret_30d'] * 0.6 + df['ret_3m'] * 0.4)
            momentum_components.append(medium_momentum)
        
        # Long-term momentum
        if 'ret_1y' in df.columns:
            momentum_components.append(df['ret_1y'])
        
        if momentum_components:
            # Combine all momentum signals
            combined_momentum = sum(momentum_components) / len(momentum_components)
            # Normalize to 0-1 using percentile rank
            df['momentum_score'] = combined_momentum.rank(pct=True)
        
        # 2. VOLUME SCORE - Smart money detection
        volume_components = []
        
        # Volume surge detection
        if all(col in df.columns for col in ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d']):
            # Recent volume surge
            surge_score = (
                (df['vol_ratio_1d_90d'] > 50).astype(float) * 0.3 +
                (df['vol_ratio_7d_90d'] > 30).astype(float) * 0.3 +
                (df['vol_ratio_30d_90d'] > 20).astype(float) * 0.4
            )
            volume_components.append(surge_score)
            
            # Smart money accumulation pattern
            accumulation = (
                (df['vol_ratio_30d_90d'] > 20) & 
                (df['vol_ratio_1d_90d'] < 0)
            ).astype(float)
            volume_components.append(accumulation)
        
        if 'rvol' in df.columns:
            rvol_score = (df['rvol'] > 1.5).astype(float)
            volume_components.append(rvol_score)
        
        if volume_components:
            df['volume_score'] = sum(volume_components) / len(volume_components)
        
        # 3. TECHNICAL SCORE - Price action and MAs
        technical_components = []
        
        # MA alignment
        if all(col in df.columns for col in ['price', 'sma_20d', 'sma_50d', 'sma_200d']):
            ma_score = (
                (df['price'] > df['sma_20d']).astype(float) * 0.4 +
                (df['price'] > df['sma_50d']).astype(float) * 0.3 +
                (df['price'] > df['sma_200d']).astype(float) * 0.3
            )
            technical_components.append(ma_score)
        
        # 52-week range position
        if all(col in df.columns for col in ['from_low_pct', 'from_high_pct']):
            # Normalize position in range
            range_position = df['from_low_pct'] / (df['from_low_pct'] - df['from_high_pct'] + 0.01)
            range_position = range_position.clip(0, 1)
            technical_components.append(range_position)
        
        if technical_components:
            df['technical_score'] = sum(technical_components) / len(technical_components)
        
        # 4. FUNDAMENTAL SCORE - Value and growth
        fundamental_components = []
        
        # PE valuation
        if 'pe' in df.columns:
            # Good PE between 10-30, normalize inversely
            pe_score = np.where(
                (df['pe'] > 0) & (df['pe'] < 50),
                1 - (df['pe'] / 50),
                0
            )
            fundamental_components.append(pe_score)
        
        # EPS growth
        if 'eps_change_pct' in df.columns:
            # Normalize EPS change
            eps_score = df['eps_change_pct'].clip(-50, 100)
            eps_score = (eps_score + 50) / 150
            fundamental_components.append(eps_score)
        
        # EPS tier scoring
        if 'eps_tier' in df.columns:
            tier_map = {
                '5↓': 0, '5↑': 0.3, '15↑': 0.5, '35↑': 0.7,
                '55↑': 0.85, '75↑': 0.95, '95↑': 1.0
            }
            eps_tier_score = df['eps_tier'].map(tier_map).fillna(0.3)
            fundamental_components.append(eps_tier_score)
        
        if fundamental_components:
            df['fundamental_score'] = sum(fundamental_components) / len(fundamental_components)
        
        # 5. CALCULATE FINAL ALPHA SCORE
        # Weighted combination of all factors
        weights = {
            'momentum_score': 0.35,
            'volume_score': 0.25,
            'technical_score': 0.20,
            'fundamental_score': 0.20
        }
        
        df['alpha_score'] = sum(
            df[factor] * weight 
            for factor, weight in weights.items() 
            if factor in df.columns
        )
        
        # Normalize alpha score
        df['alpha_score'] = df['alpha_score'].rank(pct=True)
        
        # Generate signals based on percentile
        conditions = [
            (df['alpha_score'] >= CONFIG['SIGNAL_THRESHOLDS']['ALPHA_EXTREME']),
            (df['alpha_score'] >= CONFIG['SIGNAL_THRESHOLDS']['ALPHA_HIGH']),
            (df['alpha_score'] >= CONFIG['SIGNAL_THRESHOLDS']['ALPHA_MODERATE']),
            (df['alpha_score'] >= CONFIG['SIGNAL_THRESHOLDS']['MONITOR']),
            (df['alpha_score'] >= CONFIG['SIGNAL_THRESHOLDS']['NEUTRAL']),
            (df['alpha_score'] >= CONFIG['SIGNAL_THRESHOLDS']['CAUTION'])
        ]
        
        choices = ['ALPHA_EXTREME', 'ALPHA_HIGH', 'ALPHA_MODERATE', 
                  'MONITOR', 'NEUTRAL', 'CAUTION']
        
        df['signal'] = np.select(conditions, choices, default='AVOID')
        
        # Add confidence level
        df['confidence'] = pd.cut(
            df['alpha_score'],
            bins=[0, 0.3, 0.5, 0.7, 0.85, 1.0],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Extreme']
        )
        
        # Special situations
        df['special_setup'] = 'None'
        
        # Volume explosion
        if all(col in df.columns for col in ['vol_ratio_1d_90d', 'momentum_score']):
            mask = (df['vol_ratio_1d_90d'] > 100) & (df['momentum_score'] > 0.7)
            df.loc[mask, 'special_setup'] = 'Volume Explosion'
        
        # Momentum surge
        if all(col in df.columns for col in ['ret_30d', 'ret_3m']):
            mask = (df['ret_30d'] > 20) & (df['ret_3m'] > 30) & (df['special_setup'] == 'None')
            df.loc[mask, 'special_setup'] = 'Momentum Surge'
        
        # Value emergence
        if all(col in df.columns for col in ['pe', 'eps_change_pct']):
            mask = (df['pe'] > 0) & (df['pe'] < 20) & (df['eps_change_pct'] > 25) & (df['special_setup'] == 'None')
            df.loc[mask, 'special_setup'] = 'Value Opportunity'
        
        # Sort by alpha score
        df = df.sort_values('alpha_score', ascending=False)
        
        return df

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def create_alpha_landscape(df: pd.DataFrame, limit: int = 50) -> go.Figure:
    """Create 3D visualization of alpha landscape"""
    
    plot_df = df.head(limit)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=plot_df['momentum_score'],
        y=plot_df['volume_score'],
        z=plot_df['alpha_score'],
        mode='markers+text',
        marker=dict(
            size=10,
            color=plot_df['alpha_score'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Alpha Score"),
            line=dict(width=1, color='DarkSlateGray')
        ),
        text=plot_df['ticker'],
        textposition="top center",
        hovertemplate='<b>%{text}</b><br>' +
                      'Momentum: %{x:.2f}<br>' +
                      'Volume: %{y:.2f}<br>' +
                      'Alpha: %{z:.3f}<br>' +
                      '<extra></extra>'
    )])
    
    fig.update_layout(
        title=f"Alpha Landscape (Top {limit})",
        scene=dict(
            xaxis_title="Momentum Score",
            yaxis_title="Volume Score",
            zaxis_title="Alpha Score",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=600,
        margin=dict(r=20, b=10, l=10, t=40)
    )
    
    return fig

def create_signal_distribution(df: pd.DataFrame) -> go.Figure:
    """Create signal distribution chart"""
    
    signal_counts = df['signal'].value_counts()
    
    fig = go.Figure(data=[go.Bar(
        x=signal_counts.index,
        y=signal_counts.values,
        marker_color=[CONFIG['SIGNAL_COLORS'].get(s, '#gray') for s in signal_counts.index],
        text=signal_counts.values,
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Signal Distribution",
        xaxis_title="Signal",
        yaxis_title="Count",
        height=400,
        showlegend=False
    )
    
    return fig

def create_sector_performance(df: pd.DataFrame) -> go.Figure:
    """Create sector performance chart"""
    
    if 'sector' not in df.columns:
        return go.Figure()
    
    # Top sectors by alpha score
    sector_stats = df.groupby('sector').agg({
        'alpha_score': ['mean', 'count'],
        'ret_30d': 'mean' if 'ret_30d' in df.columns else lambda x: 0
    })
    
    sector_stats.columns = ['avg_alpha', 'count', 'avg_return']
    sector_stats = sector_stats.sort_values('avg_alpha', ascending=False).head(20)
    
    fig = go.Figure(data=[go.Bar(
        y=sector_stats.index,
        x=sector_stats['avg_alpha'],
        orientation='h',
        marker_color=sector_stats['avg_alpha'],
        marker_colorscale='Viridis',
        text=sector_stats['avg_alpha'].round(3),
        textposition='outside'
    )])
    
    fig.update_layout(
        title="Top Sectors by Average Alpha Score",
        xaxis_title="Average Alpha Score",
        yaxis_title="Sector",
        height=600,
        margin=dict(l=200)
    )
    
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Header
    st.markdown("""
    <h1 style='text-align: center; color: #1f77b4;'>
    🔱 M.A.N.T.R.A. Ultimate
    </h1>
    <p style='text-align: center; font-size: 18px;'>
    Market Analysis Neural Trading Research Assistant<br>
    <i>Data-Driven Alpha Discovery</i>
    </p>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading market data..."):
        df = load_data()
    
    if df.empty:
        st.error("❌ Failed to load data. Please check your connection.")
        st.stop()
    
    # Run analysis
    with st.spinner("Running alpha analysis..."):
        analyzed_df = SimpleAlphaEngine.calculate_alpha_score(df)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Data info
        st.success(f"""
        📊 **Data Loaded**
        - Stocks: {len(analyzed_df):,}
        - Columns: {len(analyzed_df.columns)}
        - Updated: {analyzed_df['data_timestamp'].iloc[0].strftime('%H:%M')}
        """)
        
        # Filters
        st.header("🔍 Filters")
        
        # Signal filter
        signal_options = list(CONFIG['SIGNAL_THRESHOLDS'].keys())
        selected_signals = st.multiselect(
            "Signal Types",
            signal_options,
            default=['ALPHA_EXTREME', 'ALPHA_HIGH', 'ALPHA_MODERATE']
        )
        
        # Sector filter
        if 'sector' in analyzed_df.columns:
            sectors = sorted(analyzed_df['sector'].dropna().unique())
            selected_sectors = st.multiselect("Sectors", sectors, default=sectors[:10])
        else:
            selected_sectors = []
        
        # Special setups
        if 'special_setup' in analyzed_df.columns:
            special_setups = analyzed_df['special_setup'].unique()
            show_special_only = st.checkbox("Show Special Setups Only", False)
        else:
            show_special_only = False
        
        # Advanced filters
        with st.expander("Advanced Filters"):
            min_alpha = st.slider("Min Alpha Score", 0.0, 1.0, 0.0, 0.05)
            
            if 'pe' in analyzed_df.columns:
                max_pe = st.number_input("Max P/E Ratio", value=100.0, min_value=0.0)
            else:
                max_pe = 9999
        
        # Refresh button
        if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Apply filters
    filtered_df = analyzed_df.copy()
    
    if selected_signals:
        filtered_df = filtered_df[filtered_df['signal'].isin(selected_signals)]
    
    if selected_sectors and 'sector' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['sector'].isin(selected_sectors)]
    
    if show_special_only:
        filtered_df = filtered_df[filtered_df['special_setup'] != 'None']
    
    filtered_df = filtered_df[filtered_df['alpha_score'] >= min_alpha]
    
    if 'pe' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['pe'] > 0) & (filtered_df['pe'] <= max_pe)]
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        extreme_alpha = len(filtered_df[filtered_df['signal'] == 'ALPHA_EXTREME'])
        st.metric("🎯 Extreme Alpha", extreme_alpha)
    
    with col2:
        high_alpha = len(filtered_df[filtered_df['signal'] == 'ALPHA_HIGH'])
        st.metric("💎 High Alpha", high_alpha)
    
    with col3:
        special_count = len(filtered_df[filtered_df['special_setup'] != 'None'])
        st.metric("🌟 Special Setups", special_count)
    
    with col4:
        avg_alpha = filtered_df['alpha_score'].mean() if len(filtered_df) > 0 else 0
        st.metric("⚡ Avg Alpha", f"{avg_alpha:.3f}")
    
    with col5:
        if 'ret_30d' in filtered_df.columns and len(filtered_df) > 0:
            avg_return = filtered_df['ret_30d'].mean()
            st.metric("📈 Avg 30D Return", f"{avg_return:.1f}%")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Signals", "📊 Analytics", "🔬 Analysis", "💾 Export"])
    
    with tab1:
        # Search
        search_term = st.text_input("🔍 Search stocks", placeholder="Enter ticker...")
        
        display_df = filtered_df.copy()
        if search_term:
            display_df = display_df[
                display_df['ticker'].str.contains(search_term.upper(), na=False) |
                display_df.get('company_name', '').str.contains(search_term, case=False, na=False)
            ]
        
        # Display columns
        display_cols = ['ticker', 'company_name', 'signal', 'alpha_score', 'confidence',
                       'price', 'ret_30d', 'pe', 'eps_change_pct', 'special_setup']
        display_cols = [col for col in display_cols if col in display_df.columns]
        
        if len(display_df) > 0:
            st.dataframe(
                display_df[display_cols].head(100),
                use_container_width=True,
                height=600,
                column_config={
                    'alpha_score': st.column_config.ProgressColumn(
                        'Alpha Score',
                        format='%.3f',
                        min_value=0,
                        max_value=1
                    ),
                    'price': st.column_config.NumberColumn('Price', format='₹%.2f'),
                    'ret_30d': st.column_config.NumberColumn('30D %', format='%.1f%%'),
                    'pe': st.column_config.NumberColumn('P/E', format='%.1f'),
                    'eps_change_pct': st.column_config.NumberColumn('EPS Chg%', format='%.1f%%')
                }
            )
            
            # Top pick
            top_pick = display_df.iloc[0]
            st.success(f"""
            **🏆 Top Pick: {top_pick['ticker']}**  
            Signal: {top_pick['signal']} | Alpha: {top_pick['alpha_score']:.3f}
            """)
        else:
            st.info("No stocks match the current filters")
    
    with tab2:
        # Analytics
        col1, col2 = st.columns(2)
        
        with col1:
            if len(filtered_df) > 0:
                fig_3d = create_alpha_landscape(filtered_df, 50)
                st.plotly_chart(fig_3d, use_container_width=True)
        
        with col2:
            if len(analyzed_df) > 0:
                fig_signals = create_signal_distribution(analyzed_df)
                st.plotly_chart(fig_signals, use_container_width=True)
        
        # Sector performance
        if 'sector' in filtered_df.columns and len(filtered_df) > 0:
            fig_sector = create_sector_performance(filtered_df)
            st.plotly_chart(fig_sector, use_container_width=True)
    
    with tab3:
        # Deep Analysis
        st.header("🔬 Stock Analysis")
        
        if len(filtered_df) > 0:
            # Select stock
            stock_ticker = st.selectbox(
                "Select stock for analysis",
                filtered_df['ticker'].head(50).tolist()
            )
            
            if stock_ticker:
                stock_data = filtered_df[filtered_df['ticker'] == stock_ticker].iloc[0]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 📊 Scores")
                    st.metric("Alpha Score", f"{stock_data['alpha_score']:.3f}")
                    st.metric("Momentum", f"{stock_data['momentum_score']:.3f}")
                    st.metric("Volume", f"{stock_data['volume_score']:.3f}")
                    st.metric("Technical", f"{stock_data['technical_score']:.3f}")
                
                with col2:
                    st.markdown("### 💰 Fundamentals")
                    if 'pe' in stock_data and pd.notna(stock_data['pe']):
                        st.metric("P/E Ratio", f"{stock_data['pe']:.1f}")
                    if 'eps_change_pct' in stock_data and pd.notna(stock_data['eps_change_pct']):
                        st.metric("EPS Change", f"{stock_data['eps_change_pct']:.1f}%")
                    if 'eps_tier' in stock_data:
                        st.metric("EPS Tier", stock_data['eps_tier'])
                
                with col3:
                    st.markdown("### 📈 Returns")
                    returns = ['ret_30d', 'ret_3m', 'ret_1y']
                    for ret in returns:
                        if ret in stock_data and pd.notna(stock_data[ret]):
                            period = ret.replace('ret_', '').upper()
                            st.metric(f"{period} Return", f"{stock_data[ret]:.1f}%")
                
                # Special setup
                if stock_data.get('special_setup', 'None') != 'None':
                    st.info(f"🌟 **Special Setup:** {stock_data['special_setup']}")
    
    with tab4:
        # Export
        st.header("💾 Export Data")
        
        if len(filtered_df) > 0:
            # Prepare export
            export_df = filtered_df.copy()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # CSV downloads
            col1, col2 = st.columns(2)
            
            with col1:
                csv = export_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Full Results",
                    csv,
                    f"mantra_alpha_{timestamp}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                top20_csv = export_df.head(20).to_csv(index=False)
                st.download_button(
                    "🏆 Download Top 20",
                    top20_csv,
                    f"mantra_top20_{timestamp}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            # Summary stats
            st.subheader("📊 Summary")
            
            summary = {
                "Timestamp": timestamp,
                "Total Analyzed": len(analyzed_df),
                "Filtered": len(filtered_df),
                "Extreme Alpha": extreme_alpha,
                "High Alpha": high_alpha,
                "Special Setups": special_count,
                "Avg Alpha Score": float(avg_alpha) if avg_alpha > 0 else 0
            }
            
            st.json(summary)

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
