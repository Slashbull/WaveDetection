# quantum_mantra_ultimate.py - Single Sheet Edition
"""
Quantum M.A.N.T.R.A. Ultimate Trading Intelligence System
========================================================
Simplified to use ONLY the watchlist sheet
All analysis from one data source - clean and fast!
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
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Simplified configuration - single sheet only"""
    
    # Single Data Source
    SHEET_ID = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"
    WATCHLIST_GID = "2026492216"
    SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={WATCHLIST_GID}"
    
    # Quantum Signal Thresholds (Ultra-conservative)
    SIGNAL_THRESHOLDS = {
        "QUANTUM_BUY": 0.85,      # Top 2-3% only
        "STRONG_BUY": 0.75,       # Top 8-10%
        "BUY": 0.65,              # Top 20%
        "WATCH": 0.50,            # Monitor
        "NEUTRAL": 0.35,          # No signal
        "AVOID": 0.20,            # Risk present
        "STRONG_AVOID": 0.0       # High risk
    }
    
    # Colors
    SIGNAL_COLORS = {
        "QUANTUM_BUY": "#00ff00",   # Bright green
        "STRONG_BUY": "#28a745",    # Dark green
        "BUY": "#40c057",           # Green
        "WATCH": "#ffd43b",         # Yellow
        "NEUTRAL": "#868e96",       # Gray
        "AVOID": "#fa5252",         # Light red
        "STRONG_AVOID": "#e03131"   # Dark red
    }
    
    # Performance
    CACHE_TTL = 300  # 5 minutes
    REQUEST_TIMEOUT = 30
    
    # App Info
    APP_NAME = "Quantum M.A.N.T.R.A."
    APP_VERSION = "Ultimate 2.0"
    APP_ICON = "🌌"

# Global config
CONFIG = Config()

# ============================================================================
# PAGE SETUP
# ============================================================================

st.set_page_config(
    page_title=CONFIG.APP_NAME,
    page_icon=CONFIG.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .quantum-signal {
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.8; }
        100% { opacity: 1; }
    }
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(ttl=CONFIG.CACHE_TTL)
def load_watchlist_data():
    """Load and clean watchlist data from Google Sheets"""
    try:
        # Download CSV
        response = requests.get(CONFIG.SHEET_URL, timeout=CONFIG.REQUEST_TIMEOUT)
        response.raise_for_status()
        
        # Read into dataframe
        df = pd.read_csv(io.StringIO(response.text))
        
        # Remove empty columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed|^_|^$', regex=True)]
        
        # Clean column names
        df.columns = [re.sub(r"\s+", "_", re.sub(r"[^\w\s]", "", col.strip().lower())) 
                     for col in df.columns]
        
        # Clean numeric columns
        numeric_cols = ['price', 'prev_close', 'pe', 'eps_current', 'eps_last_qtr', 
                       'eps_change_pct', 'low_52w', 'high_52w', 'from_low_pct', 
                       'from_high_pct', 'sma_20d', 'sma_50d', 'sma_200d',
                       'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 
                       'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
                       'volume_1d', 'volume_7d', 'volume_30d', 'volume_3m',
                       'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d', 'rvol']
        
        for col in numeric_cols:
            if col in df.columns:
                # Convert to string and clean
                s = df[col].astype(str)
                # Remove symbols
                for symbol in ['₹', '$', '€', '£', 'Cr', 'L', 'K', 'M', 'B', '%', ',', '↑', '↓']:
                    s = s.str.replace(symbol, '', regex=False)
                s = s.str.strip().replace('', 'NaN')
                df[col] = pd.to_numeric(s, errors='coerce')
        
        # Special handling for market_cap
        if 'market_cap' in df.columns:
            s = df['market_cap'].astype(str)
            # Extract numeric value
            s = s.str.replace('₹', '').str.replace(',', '').str.strip()
            # Handle Cr suffix (multiply by 1)
            s = s.str.replace('Cr', '').str.strip()
            df['market_cap'] = pd.to_numeric(s, errors='coerce')
        
        # Clean text columns
        text_cols = ['ticker', 'exchange', 'company_name', 'sector', 'category', 
                     'eps_tier', 'price_tier', 'trading_under']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Normalize ticker
        if 'ticker' in df.columns:
            df['ticker'] = df['ticker'].str.upper()
        
        # Drop rows with no ticker
        if 'ticker' in df.columns:
            df = df[df['ticker'].notna() & (df['ticker'] != 'nan')]
        
        return df
        
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return pd.DataFrame()

# ============================================================================
# QUANTUM ANALYSIS ENGINE (All 6 Layers)
# ============================================================================

def quantum_state_analyzer(df: pd.DataFrame) -> pd.DataFrame:
    """Layer 1: Quantum state detection"""
    
    # Momentum quantum state - using acceleration of acceleration (jerk)
    if all(col in df.columns for col in ['ret_1d', 'ret_3d', 'ret_7d']):
        df['momentum_jerk'] = (
            df['ret_1d'] - df['ret_3d']/3
        ) - (
            df['ret_3d']/3 - df['ret_7d']/7
        )
        
        conditions = [
            (df['momentum_jerk'] > 0) & (df['ret_1d'] > 0) & (df['ret_3d'] > df['ret_7d']),
            (df['momentum_jerk'] < 0) & (df['ret_1d'] > 0),
            (df['momentum_jerk'] > 0) & (df['ret_1d'] < 0)
        ]
        choices = ['ACCELERATING', 'DECELERATING', 'REVERSING']
        df['momentum_quantum'] = np.select(conditions, choices, default='CONSOLIDATING')
    
    # Volume intelligence - reveals WHO is trading
    vol_cols = ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d']
    if all(col in df.columns for col in vol_cols):
        df['volume_intelligence'] = (
            (df['vol_ratio_30d_90d'] - df['vol_ratio_1d_90d']) * 0.4 +  # Stealth accumulation
            (df[vol_cols].mean(axis=1)) * 0.3 +  # Overall activity
            (df['vol_ratio_7d_90d'] - df['vol_ratio_30d_90d']) * 0.3  # Recent surge
        )
    
    # Price memory - how price behaves near historical levels
    if all(col in df.columns for col in ['from_high_pct', 'from_low_pct']):
        df['price_memory'] = (
            np.exp(-df['from_high_pct']/100) * 0.5 +  # Resistance memory
            (1 - np.exp(df['from_low_pct']/100)) * 0.5  # Support memory
        )
    
    return df

def temporal_harmonics(df: pd.DataFrame) -> pd.DataFrame:
    """Layer 2: Temporal harmonic analysis - Fibonacci patterns across time"""
    
    # Fibonacci weights for different timeframes
    fib_weights = [1, 1, 2, 3, 5, 8, 13, 21, 34]
    returns = ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 
               'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y']
    
    available_returns = [r for r in returns if r in df.columns]
    
    if len(available_returns) > 0:
        df['harmonic_momentum'] = 0
        total_weight = 0
        
        for i, ret in enumerate(available_returns):
            weight = fib_weights[i] if i < len(fib_weights) else fib_weights[-1]
            normalized = (df[ret] + 100) / 200  # Normalize to [0,1]
            df['harmonic_momentum'] += weight * normalized
            total_weight += weight
        
        df['harmonic_momentum'] = df['harmonic_momentum'] / total_weight
        
        # Phase alignment - are all timeframes moving together?
        df['phase_alignment'] = 0
        for i in range(len(available_returns)-1):
            df['phase_alignment'] += (df[available_returns[i]] > df[available_returns[i+1]]).astype(int)
        
        df['phase_alignment'] = df['phase_alignment'] / max(1, len(available_returns)-1)
        
        # Temporal energy - stored momentum potential
        if 'volume_intelligence' in df.columns:
            df['temporal_energy'] = (
                df['harmonic_momentum'] * df['phase_alignment'] * 
                np.sqrt(df['volume_intelligence'].clip(0).fillna(0) + 1)
            )
        else:
            df['temporal_energy'] = df['harmonic_momentum'] * df['phase_alignment']
    
    return df

def wave_function_analyzer(df: pd.DataFrame) -> pd.DataFrame:
    """Layer 3: Wave function collapse analysis - when probabilities become certainties"""
    
    # Technical wave - MA alignment probability
    ma_cols = ['sma_20d', 'sma_50d', 'sma_200d']
    if 'price' in df.columns and all(col in df.columns for col in ma_cols):
        df['technical_wave'] = 0
        for ma in ma_cols:
            df['technical_wave'] += (df['price'] > df[ma]).astype(float) / len(ma_cols)
        
        # MA alignment bonus
        df['ma_aligned'] = (
            (df['price'] > df['sma_20d']) & 
            (df['sma_20d'] > df['sma_50d']) & 
            (df['sma_50d'] > df['sma_200d'])
        ).astype(float)
        
        df['technical_wave'] = df['technical_wave'] * (0.7 + 0.3 * df['ma_aligned'])
    
    # Fundamental wave - quality and growth
    if 'eps_tier' in df.columns:
        eps_tier_scores = {
            '5↓': -1, '5↑': 0, '15↑': 0.2, '35↑': 0.4, 
            '55↑': 0.6, '75↑': 0.8, '95↑': 1.0,
            '<5': -0.5, '5-15': 0.1, '15-35': 0.3, '35-55': 0.5,
            '55-75': 0.7, '75-95': 0.85, '>95': 0.95
        }
        df['eps_score'] = df['eps_tier'].map(eps_tier_scores).fillna(0)
    else:
        df['eps_score'] = 0.5
    
    if all(col in df.columns for col in ['eps_score', 'eps_change_pct', 'pe']):
        df['fundamental_wave'] = (
            df['eps_score'] * 0.4 +
            (df['eps_change_pct'].clip(-50, 100) + 50) / 150 * 0.3 +
            (1 - df['pe'].clip(0, 50) / 50) * 0.3
        )
    
    # Market structure wave - sector and category relative performance
    if 'sector' in df.columns and 'ret_30d' in df.columns:
        df['sector_relative'] = df.groupby('sector')['ret_30d'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-6)
        ).fillna(0)
        
        df['structure_wave'] = df['sector_relative'] * 0.5
        
        if 'category' in df.columns and 'ret_3m' in df.columns:
            df['category_relative'] = df.groupby('category')['ret_3m'].transform(
                lambda x: x.rank(pct=True)
            ).fillna(0.5)
            df['structure_wave'] = df['structure_wave'] + df['category_relative'] * 0.5
    else:
        df['structure_wave'] = 0.5
    
    # Wave alignment and collapse probability
    wave_cols = ['technical_wave', 'fundamental_wave', 'structure_wave']
    available_waves = [w for w in wave_cols if w in df.columns]
    
    if len(available_waves) > 0:
        df['wave_alignment'] = df[available_waves].mean(axis=1)
        # Sigmoid function for probability collapse
        df['collapse_probability'] = 1 / (1 + np.exp(-10 * (df['wave_alignment'] - 0.5)))
    
    return df

def energy_field_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """Layer 4: Energy field analysis - capital flow and attraction"""
    
    # Field strength - how much attention/capital a stock attracts
    field_components = []
    
    if 'harmonic_momentum' in df.columns:
        field_components.append(df['harmonic_momentum'] ** 2 * 0.3)
    
    if 'rvol' in df.columns:
        field_components.append((df['rvol'] / (df['rvol'].max() + 1e-6)) * 0.3)
    
    if 'fundamental_wave' in df.columns:
        field_components.append(df['fundamental_wave'] * 0.4)
    
    if field_components:
        df['field_strength'] = sum(field_components) / len(field_components)
    else:
        df['field_strength'] = 0.5
    
    # Field coherence - how stable/consistent is the trend
    returns_cols = [col for col in df.columns if col.startswith('ret_') and col in df.columns]
    if len(returns_cols) >= 3:
        # Calculate consistency across timeframes
        returns_subset = df[returns_cols[:5]]  # Use up to 5 timeframes
        df['field_coherence'] = returns_subset.apply(
            lambda x: 1 - (x.std() / (x.abs().mean() + 1e-6)), axis=1
        ).clip(0, 1)
    else:
        df['field_coherence'] = 0.5
    
    # Energy flow direction - is capital flowing in or out?
    if 'field_strength' in df.columns and 'volume_intelligence' in df.columns:
        conditions = [
            (df['field_strength'] > df['field_strength'].quantile(0.7)) & 
            (df['volume_intelligence'] > 0),
            (df['field_strength'] < df['field_strength'].quantile(0.3)) &
            (df['volume_intelligence'] < 0)
        ]
        choices = ['ATTRACTING', 'REPELLING']
        df['energy_flow'] = np.select(conditions, choices, default='NEUTRAL')
    else:
        df['energy_flow'] = 'NEUTRAL'
    
    # Sector interference - how does the stock relate to its sector
    if 'sector' in df.columns and 'field_strength' in df.columns:
        df['sector_interference'] = df.groupby('sector')['field_strength'].transform(
            lambda x: x / (x.sum() + 1e-6)
        )
    
    return df

def critical_point_detector(df: pd.DataFrame) -> pd.DataFrame:
    """Layer 5: Critical point detection - explosive setups"""
    
    # Price compression - coiled spring effect
    if all(col in df.columns for col in ['high_52w', 'low_52w', 'price']):
        df['price_compression'] = 1 - (
            (df['high_52w'] - df['low_52w']) / (df['price'] + 1e-6)
        ).clip(0, 2) / 2
    
    # MA compression - converging averages signal big move coming
    if all(col in df.columns for col in ['sma_20d', 'sma_200d', 'price']):
        df['ma_compression'] = 1 - (
            ((df['sma_20d'] - df['sma_200d']).abs() / (df['price'] + 1e-6)).clip(0, 0.2) / 0.2
        )
    
    # Volatility compression
    short_term_returns = ['ret_1d', 'ret_3d', 'ret_7d']
    long_term_returns = ['ret_30d', 'ret_3m', 'ret_6m']
    
    short_available = [r for r in short_term_returns if r in df.columns]
    long_available = [r for r in long_term_returns if r in df.columns]
    
    if len(short_available) >= 2 and len(long_available) >= 2:
        short_vol = df[short_available].std(axis=1)
        long_vol = df[long_available].std(axis=1)
        df['volatility_compression'] = 1 - (short_vol / (long_vol + 1e-6)).clip(0, 1)
    
    # Critical mass calculation
    critical_components = []
    for comp in ['price_compression', 'ma_compression', 'volatility_compression', 'wave_alignment']:
        if comp in df.columns:
            critical_components.append(df[comp])
    
    if critical_components:
        df['critical_mass'] = sum(critical_components) / len(critical_components)
    else:
        df['critical_mass'] = 0.5
    
    # Trigger conditions - what ignites the explosion
    df['trigger_conditions'] = False
    
    if 'vol_ratio_1d_90d' in df.columns and 'critical_mass' in df.columns:
        df['trigger_conditions'] |= (df['vol_ratio_1d_90d'] > 50) & (df['critical_mass'] > 0.7)
    
    if 'eps_change_pct' in df.columns and 'technical_wave' in df.columns:
        df['trigger_conditions'] |= (df['eps_change_pct'] > 25) & (df['technical_wave'] > 0.7)
    
    if all(col in df.columns for col in ['energy_flow', 'momentum_quantum', 'from_low_pct']):
        df['trigger_conditions'] |= (
            (df['energy_flow'] == 'ATTRACTING') & 
            (df['momentum_quantum'] == 'REVERSING') &
            (df['from_low_pct'] < 30)
        )
    
    # Critical score with amplification
    if 'critical_mass' in df.columns:
        df['critical_score'] = np.where(
            df['trigger_conditions'],
            df['critical_mass'] * 1.5,  # Amplify when triggered
            df['critical_mass']
        ).clip(0, 1)
    
    return df

def quantum_signal_synthesis(df: pd.DataFrame) -> pd.DataFrame:
    """Layer 6: Final signal synthesis - bringing it all together"""
    
    # Build quantum score from available components
    score_components = []
    weights = []
    
    component_map = {
        'temporal_energy': 0.25,
        'collapse_probability': 0.25,
        'field_strength': 0.20,
        'field_coherence': 0.05,
        'critical_score': 0.25
    }
    
    for comp, weight in component_map.items():
        if comp in df.columns:
            score_components.append(df[comp] * weight)
            weights.append(weight)
    
    if score_components:
        df['quantum_score'] = sum(score_components) / sum(weights)
    else:
        # Fallback scoring if advanced components missing
        basic_score = 0.5
        if 'ret_30d' in df.columns:
            basic_score += (df['ret_30d'] > 0) * 0.2
        if 'volume_intelligence' in df.columns:
            basic_score += (df['volume_intelligence'] > 0) * 0.2
        if 'eps_change_pct' in df.columns:
            basic_score += (df['eps_change_pct'] > 10) * 0.1
        df['quantum_score'] = basic_score
    
    # Risk adjustment
    if all(col in df.columns for col in ['ret_1d', 'ret_3d', 'ret_7d']):
        volatility = df[['ret_1d', 'ret_3d', 'ret_7d']].std(axis=1)
        df['risk_factor'] = (1 - volatility / 10).clip(0, 1) * 0.5
    else:
        df['risk_factor'] = 0.7
    
    if 'rvol' in df.columns:
        liquidity_factor = (df['rvol'].clip(0, 2) / 2)
        df['risk_factor'] = (df['risk_factor'] + liquidity_factor * 0.5)
    
    # Risk-adjusted quantum score
    df['risk_adjusted_quantum_score'] = df['quantum_score'] * (0.5 + 0.5 * df['risk_factor'])
    
    # Generate signals based on thresholds
    df['signal'] = 'NEUTRAL'
    
    for signal, threshold in sorted(CONFIG.SIGNAL_THRESHOLDS.items(), key=lambda x: x[1], reverse=True):
        df.loc[df['risk_adjusted_quantum_score'] >= threshold, 'signal'] = signal
    
    # Special setups detection
    df['special_setup'] = 'NONE'
    
    # Quantum convergence - all stars align
    if all(col in df.columns for col in ['quantum_score', 'critical_score', 'wave_alignment']):
        mask = (df['quantum_score'] > 0.9) & (df['critical_score'] > 0.8) & (df['wave_alignment'] > 0.8)
        df.loc[mask, 'special_setup'] = 'QUANTUM_CONVERGENCE'
    
    # Institutional accumulation pattern
    if all(col in df.columns for col in ['energy_flow', 'vol_ratio_7d_90d', 'eps_tier']):
        mask = ((df['energy_flow'] == 'ATTRACTING') & 
                (df['vol_ratio_7d_90d'] > 100) & 
                (df['eps_tier'].isin(['55↑', '75↑', '95↑', '55-75', '75-95', '>95'])))
        df.loc[mask & (df['special_setup'] == 'NONE'), 'special_setup'] = 'INSTITUTIONAL_ACCUMULATION'
    
    # Value reversal setup
    if all(col in df.columns for col in ['momentum_quantum', 'from_low_pct', 'fundamental_wave']):
        mask = ((df['momentum_quantum'] == 'REVERSING') & 
                (df['from_low_pct'] < 20) & 
                (df['fundamental_wave'] > 0.7))
        df.loc[mask & (df['special_setup'] == 'NONE'), 'special_setup'] = 'VALUE_REVERSAL'
    
    # Momentum explosion setup
    if all(col in df.columns for col in ['momentum_quantum', 'phase_alignment', 'vol_ratio_1d_90d']):
        mask = ((df['momentum_quantum'] == 'ACCELERATING') & 
                (df['phase_alignment'] > 0.8) & 
                (df['vol_ratio_1d_90d'] > 100))
        df.loc[mask & (df['special_setup'] == 'NONE'), 'special_setup'] = 'MOMENTUM_EXPLOSION'
    
    return df

@st.cache_data(ttl=60)
def run_quantum_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Run complete quantum analysis pipeline"""
    
    if df.empty:
        return df
    
    # Make a copy to avoid modifying cached data
    df = df.copy()
    
    # Run through all 6 layers
    with st.spinner("Layer 1: Quantum State Analysis..."):
        df = quantum_state_analyzer(df)
    
    with st.spinner("Layer 2: Temporal Harmonics..."):
        df = temporal_harmonics(df)
    
    with st.spinner("Layer 3: Wave Function Analysis..."):
        df = wave_function_analyzer(df)
    
    with st.spinner("Layer 4: Energy Field Dynamics..."):
        df = energy_field_dynamics(df)
    
    with st.spinner("Layer 5: Critical Point Detection..."):
        df = critical_point_detector(df)
    
    with st.spinner("Layer 6: Signal Synthesis..."):
        df = quantum_signal_synthesis(df)
    
    # Final ranking within signal groups
    df['final_rank'] = df.groupby('signal')['risk_adjusted_quantum_score'].rank(
        ascending=False, method='dense'
    )
    
    # Sort by quantum score
    df = df.sort_values('risk_adjusted_quantum_score', ascending=False)
    
    return df

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_quantum_landscape(df: pd.DataFrame) -> go.Figure:
    """Create 3D quantum landscape visualization"""
    
    # Select top stocks for clarity
    plot_df = df.head(100)
    
    # Determine axes based on available columns
    x_col = 'temporal_energy' if 'temporal_energy' in df.columns else 'ret_30d'
    y_col = 'collapse_probability' if 'collapse_probability' in df.columns else 'volume_intelligence'
    z_col = 'field_strength' if 'field_strength' in df.columns else 'quantum_score'
    
    # Create 3D scatter
    fig = go.Figure(data=[go.Scatter3d(
        x=plot_df[x_col] if x_col in plot_df.columns else plot_df.index,
        y=plot_df[y_col] if y_col in plot_df.columns else plot_df.index,
        z=plot_df[z_col] if z_col in plot_df.columns else plot_df['quantum_score'],
        mode='markers+text',
        marker=dict(
            size=8 + (np.log1p(plot_df['market_cap']) if 'market_cap' in plot_df.columns else 5),
            color=plot_df['quantum_score'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Quantum<br>Score", x=1.02),
            line=dict(width=1, color='white')
        ),
        text=plot_df['ticker'],
        textposition="top center",
        textfont=dict(size=9, color='white'),
        hovertemplate=(
            '<b>%{text}</b><br>' +
            f'{x_col}: %{{x:.3f}}<br>' +
            f'{y_col}: %{{y:.3f}}<br>' +
            f'{z_col}: %{{z:.3f}}<br>' +
            'Quantum Score: %{marker.color:.3f}<br>' +
            '<extra></extra>'
        )
    )])
    
    # Update layout
    fig.update_layout(
        title={
            'text': "Quantum Trading Landscape (Top 100 Stocks)",
            'x': 0.5,
            'xanchor': 'center'
        },
        scene=dict(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            zaxis_title=z_col.replace('_', ' ').title(),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                up=dict(x=0, y=0, z=1)
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig

def create_sector_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create sector performance heatmap"""
    
    if 'sector' not in df.columns:
        return go.Figure().add_annotation(text="Sector data not available", showarrow=False)
    
    # Calculate sector metrics
    metrics = ['quantum_score', 'temporal_energy', 'field_strength', 'critical_score']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        available_metrics = ['quantum_score']
    
    # Aggregate by sector
    sector_stats = df.groupby('sector')[available_metrics].agg(['mean', 'count'])
    
    # Prepare heatmap data
    heatmap_data = []
    metric_labels = []
    
    for metric in available_metrics:
        if (metric, 'mean') in sector_stats.columns:
            heatmap_data.append(sector_stats[(metric, 'mean')].values)
            metric_labels.append(metric.replace('_', ' ').title())
    
    if not heatmap_data:
        return go.Figure().add_annotation(text="Insufficient data for heatmap", showarrow=False)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=sector_stats.index,
        y=metric_labels,
        colorscale='RdYlGn',
        text=np.round(heatmap_data, 3),
        texttemplate='%{text:.3f}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Sector Quantum Metrics Heatmap",
        xaxis_title="Sector",
        yaxis_title="Metric",
        height=400,
        xaxis={'tickangle': -45}
    )
    
    return fig

def create_signal_gauge(df: pd.DataFrame) -> go.Figure:
    """Create a gauge showing overall market quantum state"""
    
    avg_quantum = df['quantum_score'].mean() * 100
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg_quantum,
        title={'text': "Market Quantum State"},
        delta={'reference': 50, 'relative': True},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 20], 'color': "#e03131"},
                {'range': [20, 35], 'color': "#fa5252"},
                {'range': [35, 50], 'color': "#868e96"},
                {'range': [50, 65], 'color': "#ffd43b"},
                {'range': [65, 75], 'color': "#40c057"},
                {'range': [75, 85], 'color': "#28a745"},
                {'range': [85, 100], 'color': "#00ff00"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': avg_quantum
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header with custom styling
    st.markdown(f"""
    <h1 style='text-align: center; color: #1f77b4; font-size: 3em; margin-bottom: 0;'>
    {CONFIG.APP_ICON} {CONFIG.APP_NAME}
    </h1>
    <p style='text-align: center; color: #666; font-size: 1.2em; margin-top: 0;'>
    Single-Sheet Quantum Trading Intelligence v{CONFIG.APP_VERSION}
    </p>
    <hr style='margin: 20px 0; border: none; border-top: 2px solid #1f77b4;'>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Quantum Control Panel")
        
        # Data refresh button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", type="primary", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        with col2:
            if st.button("ℹ️ Help", use_container_width=True):
                st.session_state.show_help = not st.session_state.get('show_help', False)
        
        # Load data
        with st.spinner("Loading watchlist data..."):
            df = load_watchlist_data()
        
        if df.empty:
            st.error("❌ Failed to load data. Check connection.")
            st.stop()
        
        # Data quality indicator
        null_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
        quality_color = "🟢" if null_pct < 10 else "🟡" if null_pct < 20 else "🔴"
        st.metric("Data Quality", f"{quality_color} {100-null_pct:.1f}%")
        st.metric("Total Stocks", len(df))
        
        # Run quantum analysis
        analyzed_df = run_quantum_analysis(df)
        
        st.success(f"✅ Quantum analysis complete!")
        
        # Filters section
        st.markdown("---")
        st.header("🔍 Quantum Filters")
        
        # Signal filter
        all_signals = list(CONFIG.SIGNAL_THRESHOLDS.keys())
        signal_counts = analyzed_df['signal'].value_counts()
        
        signal_labels = []
        for sig in all_signals:
            count = signal_counts.get(sig, 0)
            signal_labels.append(f"{sig} ({count})")
        
        selected_signal_labels = st.multiselect(
            "📊 Signal Types",
            options=signal_labels,
            default=[f"QUANTUM_BUY ({signal_counts.get('QUANTUM_BUY', 0)})", 
                    f"STRONG_BUY ({signal_counts.get('STRONG_BUY', 0)})",
                    f"BUY ({signal_counts.get('BUY', 0)})"],
            help="Filter by quantum signal strength"
        )
        
        # Extract signal names from labels
        selected_signals = [label.split(' (')[0] for label in selected_signal_labels]
        
        # Sector filter
        if 'sector' in analyzed_df.columns:
            sectors = sorted(analyzed_df['sector'].dropna().unique())
            selected_sectors = st.multiselect(
                "🏭 Sectors",
                options=sectors,
                default=sectors[:5] if len(sectors) > 5 else sectors,
                help="Filter by sector"
            )
        else:
            selected_sectors = []
        
        # Category filter
        if 'category' in analyzed_df.columns:
            categories = sorted(analyzed_df['category'].dropna().unique())
            selected_categories = st.multiselect(
                "📈 Market Cap Category",
                options=categories,
                default=categories,
                help="Small/Mid/Large Cap"
            )
        else:
            selected_categories = []
        
        # Price tier filter
        if 'price_tier' in analyzed_df.columns:
            price_tiers = sorted(analyzed_df['price_tier'].dropna().unique())
            selected_price_tiers = st.multiselect(
                "💰 Price Range",
                options=price_tiers,
                default=price_tiers,
                help="Filter by price tier"
            )
        else:
            selected_price_tiers = []
        
        # Special setups
        if 'special_setup' in analyzed_df.columns:
            special_count = len(analyzed_df[analyzed_df['special_setup'] != 'NONE'])
            show_special_only = st.checkbox(
                f"🎯 Special Setups Only ({special_count})",
                value=False,
                help="Show only rare high-probability setups"
            )
        else:
            show_special_only = False
        
        # Advanced filters
        with st.expander("⚡ Advanced Filters"):
            # Quantum score range
            quantum_range = st.slider(
                "Quantum Score Range",
                min_value=0.0,
                max_value=1.0,
                value=(0.5, 1.0),
                step=0.05,
                format="%.2f"
            )
            
            # Volume filter
            if 'rvol' in analyzed_df.columns:
                min_rvol = st.number_input(
                    "Min Relative Volume",
                    min_value=0.0,
                    value=0.0,
                    step=0.5,
                    format="%.1f"
                )
            else:
                min_rvol = 0
            
            # EPS growth filter
            if 'eps_change_pct' in analyzed_df.columns:
                min_eps_growth = st.number_input(
                    "Min EPS Growth %",
                    min_value=-100.0,
                    value=0.0,
                    step=10.0,
                    format="%.0f"
                )
            else:
                min_eps_growth = -100
            
            # Returns filter
            if 'ret_30d' in analyzed_df.columns:
                min_30d_return = st.number_input(
                    "Min 30-Day Return %",
                    min_value=-100.0,
                    value=-50.0,
                    step=10.0,
                    format="%.0f"
                )
            else:
                min_30d_return = -100
    
    # Apply all filters
    filtered_df = analyzed_df.copy()
    
    # Basic filters
    filtered_df = filtered_df[filtered_df['signal'].isin(selected_signals)]
    
    if selected_sectors and 'sector' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['sector'].isin(selected_sectors)]
    
    if selected_categories and 'category' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
    
    if selected_price_tiers and 'price_tier' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['price_tier'].isin(selected_price_tiers)]
    
    if show_special_only and 'special_setup' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['special_setup'] != 'NONE']
    
    # Advanced filters
    filtered_df = filtered_df[
        (filtered_df['quantum_score'] >= quantum_range[0]) &
        (filtered_df['quantum_score'] <= quantum_range[1])
    ]
    
    if 'rvol' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['rvol'] >= min_rvol]
    
    if 'eps_change_pct' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['eps_change_pct'] >= min_eps_growth]
    
    if 'ret_30d' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['ret_30d'] >= min_30d_return]
    
    # Show help if requested
    if st.session_state.get('show_help', False):
        with st.container():
            st.info("""
            ### 🚀 Quick Guide
            
            **Signals:**
            - 🟢 QUANTUM_BUY: Ultra-rare top 2-3%
            - 🟢 STRONG_BUY: Top 8-10%
            - 🟡 BUY: Top 20%
            - ⚪ WATCH: Monitor for entry
            
            **Special Setups:**
            - 🎯 Quantum Convergence: All factors align
            - 🏛️ Institutional Accumulation: Smart money
            - 💎 Value Reversal: Quality at lows
            - 🚀 Momentum Explosion: Breakout imminent
            
            **Tips:**
            - Use filters to narrow down opportunities
            - Check special setups daily
            - Export top picks for detailed research
            """)
    
    # Main content area
    if len(filtered_df) == 0:
        st.warning("😕 No stocks match your filters. Try relaxing the criteria.")
        st.stop()
    
    # Key metrics row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        quantum_buys = len(filtered_df[filtered_df['signal'] == 'QUANTUM_BUY'])
        st.metric("🎯 Quantum", quantum_buys)
    
    with col2:
        strong_buys = len(filtered_df[filtered_df['signal'] == 'STRONG_BUY'])
        st.metric("💪 Strong", strong_buys)
    
    with col3:
        buys = len(filtered_df[filtered_df['signal'] == 'BUY'])
        st.metric("✅ Buy", buys)
    
    with col4:
        if 'special_setup' in filtered_df.columns:
            special = len(filtered_df[filtered_df['special_setup'] != 'NONE'])
            st.metric("🌟 Special", special)
        else:
            st.metric("📊 Total", len(filtered_df))
    
    with col5:
        avg_quantum = filtered_df['quantum_score'].mean()
        st.metric("⚡ Avg Score", f"{avg_quantum:.3f}")
    
    with col6:
        if 'ret_30d' in filtered_df.columns:
            avg_ret = filtered_df['ret_30d'].mean()
            st.metric("📈 Avg 30D", f"{avg_ret:.1f}%")
        else:
            st.metric("✨ Ready", "100%")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Quantum Signals",
        "🌌 3D Landscape", 
        "🔥 Special Setups",
        "📊 Analytics",
        "💾 Export"
    ])
    
    # Tab 1: Main signals table
    with tab1:
        st.subheader("Top Quantum Trading Signals")
        
        # Search and display options
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_ticker = st.text_input("🔍 Search ticker", placeholder="e.g., RELIANCE")
        with col2:
            max_rows = st.selectbox("Show rows", [25, 50, 100, 200], index=1)
        with col3:
            sort_by = st.selectbox("Sort by", ['quantum_score', 'ret_30d', 'volume_intelligence'])
        
        # Apply search filter
        display_df = filtered_df.copy()
        if search_ticker:
            display_df = display_df[
                display_df['ticker'].str.contains(search_ticker.upper(), na=False)
            ]
        
        # Sort
        if sort_by in display_df.columns:
            display_df = display_df.sort_values(sort_by, ascending=False)
        
        # Select columns to display
        base_cols = ['ticker', 'company_name', 'signal', 'quantum_score']
        
        optional_cols = {
            'special_setup': 'Special Setup',
            'price': 'Price',
            'ret_30d': '30D Return',
            'volume_intelligence': 'Vol Intel',
            'critical_score': 'Critical',
            'temporal_energy': 'Temporal',
            'eps_tier': 'EPS Tier',
            'pe': 'P/E',
            'sector': 'Sector',
            'category': 'Category',
            'momentum_quantum': 'Momentum'
        }
        
        display_cols = base_cols + [col for col in optional_cols.keys() if col in display_df.columns]
        
        # Configure column display
        column_config = {
            'quantum_score': st.column_config.ProgressColumn(
                'Quantum Score',
                help='Overall quantum signal strength (0-1)',
                format='%.3f',
                min_value=0,
                max_value=1
            ),
            'price': st.column_config.NumberColumn('Price', format='₹%.2f'),
            'ret_30d': st.column_config.NumberColumn('30D %', format='%.1f%%'),
            'pe': st.column_config.NumberColumn('P/E', format='%.1f'),
            'volume_intelligence': st.column_config.NumberColumn('Vol Intel', format='%.2f'),
            'critical_score': st.column_config.ProgressColumn('Critical', format='%.2f', max_value=1),
            'temporal_energy': st.column_config.ProgressColumn('Temporal', format='%.2f', max_value=1)
        }
        
        # Display the data
        st.dataframe(
            display_df[display_cols].head(max_rows),
            use_container_width=True,
            height=600,
            column_config=column_config
        )
        
        # Quick insights
        if len(display_df) > 0:
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                top_pick = display_df.iloc[0]
                st.success(f"""
                **🏆 Top Pick: {top_pick['ticker']}**  
                Signal: {top_pick['signal']}  
                Quantum Score: {top_pick['quantum_score']:.3f}
                """)
            
            with col2:
                if 'sector' in display_df.columns:
                    top_sector = display_df.groupby('sector')['quantum_score'].mean().idxmax()
                    sector_score = display_df.groupby('sector')['quantum_score'].mean().max()
                    st.info(f"""
                    **🏭 Best Sector: {top_sector}**  
                    Avg Quantum: {sector_score:.3f}  
                    Stocks: {len(display_df[display_df['sector'] == top_sector])}
                    """)
            
            with col3:
                if 'special_setup' in display_df.columns:
                    special_types = display_df[display_df['special_setup'] != 'NONE']['special_setup'].value_counts()
                    if len(special_types) > 0:
                        st.warning(f"""
                        **🎯 Special Setups Found:**  
                        {', '.join([f"{k}: {v}" for k, v in special_types.items()])}
                        """)
    
    # Tab 2: 3D Visualization
    with tab2:
        st.subheader("Quantum Trading Landscape")
        
        # Create and display 3D plot
        landscape_fig = create_quantum_landscape(filtered_df)
        st.plotly_chart(landscape_fig, use_container_width=True)
        
        # Additional visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Signal distribution pie chart
            signal_dist = filtered_df['signal'].value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=signal_dist.index,
                values=signal_dist.values,
                hole=0.4,
                marker_colors=[CONFIG.SIGNAL_COLORS.get(s, '#gray') for s in signal_dist.index]
            )])
            
            fig.update_layout(
                title="Signal Distribution",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Market quantum gauge
            gauge_fig = create_signal_gauge(filtered_df)
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Sector heatmap below gauge
            if 'sector' in filtered_df.columns:
                heatmap_fig = create_sector_heatmap(filtered_df)
                st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # Tab 3: Special Setups
    with tab3:
        st.subheader("🔥 Quantum Special Setups")
        
        if 'special_setup' not in filtered_df.columns:
            st.info("Special setup detection requires complete data columns.")
        else:
            special_df = filtered_df[filtered_df['special_setup'] != 'NONE']
            
            if len(special_df) == 0:
                st.warning("No special setups in current filter. These are rare - check daily!")
            else:
                # Setup descriptions
                setup_info = {
                    'QUANTUM_CONVERGENCE': {
                        'icon': '🎯',
                        'color': 'success',
                        'desc': 'Ultra-rare: All quantum dimensions perfectly aligned',
                        'action': 'Immediate action required - highest probability setup'
                    },
                    'INSTITUTIONAL_ACCUMULATION': {
                        'icon': '🏛️',
                        'color': 'info',
                        'desc': 'Smart money quietly building positions',
                        'action': 'Follow the institutions - scale in gradually'
                    },
                    'VALUE_REVERSAL': {
                        'icon': '💎',
                        'color': 'warning',
                        'desc': 'Quality stock at major inflection point',
                        'action': 'High risk/reward - use tight stops'
                    },
                    'MOMENTUM_EXPLOSION': {
                        'icon': '🚀',
                        'color': 'info',
                        'desc': 'Momentum accelerating across all timeframes',
                        'action': 'Breakout imminent - position for explosive move'
                    }
                }
                
                # Group by setup type
                for setup_type in special_df['special_setup'].unique():
                    if setup_type in setup_info:
                        info = setup_info[setup_type]
                        setup_stocks = special_df[special_df['special_setup'] == setup_type]
                        
                        st.markdown(f"""
                        ### {info['icon']} {setup_type.replace('_', ' ').title()}
                        """)
                        
                        if info['color'] == 'success':
                            st.success(f"**{info['desc']}**")
                        elif info['color'] == 'info':
                            st.info(f"**{info['desc']}**")
                        else:
                            st.warning(f"**{info['desc']}**")
                        
                        st.write(f"*{info['action']}*")
                        
                        # Display stocks in this setup
                        for idx, (_, stock) in enumerate(setup_stocks.iterrows()):
                            if idx >= 10:  # Limit display
                                st.write(f"*...and {len(setup_stocks) - 10} more*")
                                break
                            
                            with st.expander(f"{stock['ticker']} - {stock.get('company_name', 'N/A')}"):
                                # Create 4 columns for metrics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("💰 Price", f"₹{stock.get('price', 0):.2f}")
                                    if 'from_low_pct' in stock:
                                        st.metric("📊 From 52W Low", f"{stock['from_low_pct']:.1f}%")
                                
                                with col2:
                                    st.metric("⚡ Quantum Score", f"{stock['quantum_score']:.3f}")
                                    if 'critical_score' in stock:
                                        st.metric("🎯 Critical Score", f"{stock['critical_score']:.3f}")
                                
                                with col3:
                                    if 'ret_30d' in stock:
                                        st.metric("📈 30D Return", f"{stock['ret_30d']:.1f}%")
                                    if 'volume_intelligence' in stock:
                                        st.metric("🔊 Volume Intel", f"{stock['volume_intelligence']:.2f}")
                                
                                with col4:
                                    if 'eps_tier' in stock:
                                        st.metric("💹 EPS Tier", stock['eps_tier'])
                                    if 'momentum_quantum' in stock:
                                        st.metric("🌀 Momentum", stock['momentum_quantum'])
                                
                                # Action button
                                st.markdown(f"""
                                **Suggested Action:** {info['action']}  
                                **Entry:** ₹{stock.get('price', 0):.2f} | 
                                **Stop:** ₹{stock.get('sma_20d', stock.get('price', 0) * 0.95):.2f} | 
                                **Target:** ₹{stock.get('price', 0) * 1.15:.2f}
                                """)
    
    # Tab 4: Analytics
    with tab4:
        st.subheader("Quantum Analytics Dashboard")
        
        # Quantum state distribution
        if 'momentum_quantum' in filtered_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                momentum_dist = filtered_df['momentum_quantum'].value_counts()
                
                fig = go.Figure(data=[go.Bar(
                    x=momentum_dist.values,
                    y=momentum_dist.index,
                    orientation='h',
                    marker_color=['#00ff00', '#ffd43b', '#fa5252', '#868e96']
                )])
                
                fig.update_layout(
                    title="Momentum Quantum States",
                    xaxis_title="Count",
                    yaxis_title="State",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'energy_flow' in filtered_df.columns:
                    energy_dist = filtered_df['energy_flow'].value_counts()
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=energy_dist.index,
                        values=energy_dist.values,
                        hole=0.3,
                        marker_colors=['#00ff00', '#fa5252', '#868e96']
                    )])
                    
                    fig.update_layout(
                        title="Energy Flow Distribution",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.subheader("Quantum Metrics Correlation")
        
        corr_cols = ['quantum_score', 'temporal_energy', 'collapse_probability',
                    'field_strength', 'critical_score', 'volume_intelligence']
        available_corr = [col for col in corr_cols if col in filtered_df.columns]
        
        if len(available_corr) >= 3:
            corr_data = filtered_df[available_corr].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_data.values,
                x=corr_data.columns,
                y=corr_data.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_data.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 12}
            ))
            
            fig.update_layout(
                title="Quantum Metrics Correlation Matrix",
                height=500,
                xaxis={'tickangle': -45}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("Statistical Summary")
        
        summary_cols = ['quantum_score', 'ret_30d', 'volume_intelligence', 'pe']
        summary_data = {}
        
        for col in summary_cols:
            if col in filtered_df.columns:
                summary_data[col] = {
                    'Mean': filtered_df[col].mean(),
                    'Median': filtered_df[col].median(),
                    'Std Dev': filtered_df[col].std(),
                    'Min': filtered_df[col].min(),
                    'Max': filtered_df[col].max()
                }
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data).T
            st.dataframe(summary_df.round(2), use_container_width=True)
    
    # Tab 5: Export
    with tab5:
        st.subheader("Export & Documentation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📥 Export Options")
            
            # Prepare export data
            export_cols = [col for col in filtered_df.columns 
                          if not col.startswith('_') and col != 'index']
            
            export_df = filtered_df[export_cols]
            
            # CSV export
            csv = export_df.to_csv(index=False)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            st.download_button(
                label="📄 Download Full Results (CSV)",
                data=csv,
                file_name=f"quantum_mantra_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Top picks export
            top_picks = export_df.head(20)
            top_csv = top_picks.to_csv(index=False)
            
            st.download_button(
                label="🏆 Download Top 20 Picks",
                data=top_csv,
                file_name=f"quantum_top20_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Summary stats
            st.markdown("### 📊 Session Summary")
            
            summary = {
                "Analysis Timestamp": timestamp,
                "Total Stocks Analyzed": len(analyzed_df),
                "Filtered Results": len(filtered_df),
                "Quantum Buys": quantum_buys,
                "Strong Buys": strong_buys,
                "Average Quantum Score": f"{avg_quantum:.3f}",
                "Special Setups": special if 'special_setup' in filtered_df.columns else "N/A",
                "Top Signal": filtered_df['signal'].value_counts().index[0] if len(filtered_df) > 0 else "N/A"
            }
            
            for key, value in summary.items():
                st.write(f"**{key}:** {value}")
        
        with col2:
            st.markdown("### 📚 Quantum Dimensions Explained")
            
            with st.expander("Layer 1: Quantum States"):
                st.write("""
                **Momentum Quantum States:**
                - ACCELERATING: Momentum increasing (bullish)
                - DECELERATING: Momentum slowing (caution)
                - REVERSING: Direction changing (opportunity)
                - CONSOLIDATING: Sideways action (wait)
                
                **Volume Intelligence:** Reveals WHO is trading
                - Positive: Smart money accumulating
                - Negative: Distribution happening
                """)
            
            with st.expander("Layer 2: Temporal Harmonics"):
                st.write("""
                Uses Fibonacci sequence to weight returns across timeframes.
                When all timeframes align (high phase alignment), 
                explosive moves are more likely.
                """)
            
            with st.expander("Layer 3: Wave Functions"):
                st.write("""
                **Three probability waves:**
                - Technical: MA alignment & price position
                - Fundamental: EPS growth & valuation
                - Structure: Sector & category relative strength
                
                When waves align, probability "collapses" into certainty.
                """)
            
            with st.expander("Layer 4: Energy Fields"):
                st.write("""
                **Field Strength:** How much capital the stock attracts
                **Field Coherence:** Trend consistency
                **Energy Flow:** Direction of capital movement
                """)
            
            with st.expander("Layer 5: Critical Points"):
                st.write("""
                Detects compression patterns that precede big moves:
                - Price compression (trading range)
                - MA compression (converging averages)
                - Volatility compression (calm before storm)
                """)
            
            with st.expander("Layer 6: Signal Synthesis"):
                st.write("""
                Combines all quantum dimensions into final signals.
                Risk-adjusted for position sizing.
                Special setups flagged for rare opportunities.
                """)
        
        # Footer
        st.markdown("---")
        st.markdown(f"""
        <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Quantum M.A.N.T.R.A. v{CONFIG.APP_VERSION} | Single-Sheet Edition</p>
        <p>Data Source: Google Sheets (GID: {CONFIG.WATCHLIST_GID})</p>
        <p><em>Remember: This is analysis, not advice. Always do your own research.</em></p>
        </div>
        """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
