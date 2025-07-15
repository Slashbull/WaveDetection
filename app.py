# mantra_ultimate.py - PRODUCTION READY FINAL VERSION
"""
M.A.N.T.R.A. Ultimate - The Most Powerful Stock Analysis System
==============================================================
Built with deep algorithmic intelligence to extract every bit of alpha
from your 41-column dataset. Every decision is data-driven.
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
        
        # Ultra-conservative thresholds as requested
        'SIGNAL_THRESHOLDS': {
            "ALPHA_EXTREME": 0.95,   # Top 5% - Extreme conviction
            "ALPHA_HIGH": 0.85,      # Top 15% - High conviction
            "ALPHA_MODERATE": 0.70,  # Top 30% - Moderate conviction
            "MONITOR": 0.50,         # Monitor zone
            "NEUTRAL": 0.35,         # No action
            "CAUTION": 0.20,         # Risk present
            "AVOID": 0.0            # High risk
        },
        
        # Signal colors
        'SIGNAL_COLORS': {
            "ALPHA_EXTREME": "#00ff41",   # Bright green
            "ALPHA_HIGH": "#28a745",      # Green
            "ALPHA_MODERATE": "#40c057",  # Light green
            "MONITOR": "#ffd43b",         # Yellow
            "NEUTRAL": "#868e96",         # Gray
            "CAUTION": "#ff6b6b",         # Light red
            "AVOID": "#c92a2a"           # Dark red
        },
        
        # Core column groups
        'COLUMN_GROUPS': {
            'identity': ['ticker', 'exchange', 'company_name', 'year'],
            'classification': ['market_cap', 'category', 'sector', 'price_tier'],
            'price': ['price', 'prev_close', 'low_52w', 'high_52w', 'from_low_pct', 'from_high_pct'],
            'returns': ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y'],
            'moving_averages': ['sma_20d', 'sma_50d', 'sma_200d', 'trading_under'],
            'volume': ['volume_1d', 'volume_7d', 'volume_30d', 'volume_3m', 
                      'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d', 'rvol'],
            'fundamentals': ['pe', 'eps_tier', 'eps_current', 'eps_last_qtr', 'eps_duplicate', 'eps_change_pct']
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

# Custom CSS for better UI
st.markdown("""
<style>
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING WITH ROBUST ERROR HANDLING
# ============================================================================
@st.cache_data(ttl=CONFIG['CACHE_TTL'], show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load data with comprehensive error handling and cleaning"""
    try:
        response = requests.get(CONFIG['SHEET_URL'], timeout=CONFIG['REQUEST_TIMEOUT'])
        response.raise_for_status()
        
        df = pd.read_csv(io.StringIO(response.text))
        
        # Remove empty columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed|^_|^$', regex=True)]
        
        # Clean column names
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        
        # Clean numeric columns with smart parsing
        numeric_patterns = {
            'price|close|low|high|sma|dma': lambda x: pd.to_numeric(
                x.astype(str).str.replace('[₹$,€£CrLKMB]', '', regex=True),
                errors='coerce'
            ),
            'ret_|from_.*_pct|eps_change_pct|vol_ratio': lambda x: pd.to_numeric(
                x.astype(str).str.replace('[%,]', '', regex=True),
                errors='coerce'
            ),
            'volume': lambda x: pd.to_numeric(
                x.astype(str).str.replace('[,]', '', regex=True),
                errors='coerce'
            ),
            'pe|eps|rvol': lambda x: pd.to_numeric(
                x.astype(str).str.replace('[,]', '', regex=True),
                errors='coerce'
            ),
            'market_cap': lambda x: pd.to_numeric(
                x.astype(str).str.replace('[₹$,€£CrLKMB]', '', regex=True)
                .str.extract(r'([\d.]+)', expand=False),
                errors='coerce'
            ) * x.astype(str).str.extract(r'([CrLKMB])', expand=False).map(
                {'Cr': 1, 'L': 0.01, 'K': 0.0001, 'M': 0.1, 'B': 10}
            ).fillna(1)
        }
        
        for pattern, converter in numeric_patterns.items():
            matching_cols = [col for col in df.columns if re.search(pattern, col)]
            for col in matching_cols:
                if col in df.columns:
                    df[col] = converter(df[col])
        
        # Ensure ticker is clean
        if 'ticker' in df.columns:
            df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
            df = df[df['ticker'].notna() & (df['ticker'] != 'NAN') & (df['ticker'] != '')]
        
        # Add timestamp
        df['data_timestamp'] = datetime.now()
        
        return df
        
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return pd.DataFrame()

# ============================================================================
# ADVANCED ALGORITHMIC ENGINES
# ============================================================================

class MomentumEngine:
    """Multi-timeframe momentum analysis with pattern recognition"""
    
    @staticmethod
    def calculate_momentum_signature(df: pd.DataFrame) -> pd.DataFrame:
        """Create unique momentum fingerprint for each stock"""
        
        # Momentum acceleration (is momentum increasing?)
        if all(col in df.columns for col in ['ret_1d', 'ret_3d', 'ret_7d']):
            df['momentum_accel_short'] = (
                df['ret_1d'].fillna(0) * 0.5 +
                (df['ret_1d'].fillna(0) - df['ret_3d'].fillna(0)/3) * 0.3 +
                (df['ret_3d'].fillna(0)/3 - df['ret_7d'].fillna(0)/7) * 0.2
            )
        
        # Momentum quality (consistency across timeframes)
        momentum_cols = [col for col in ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m'] 
                        if col in df.columns]
        if len(momentum_cols) >= 3:
            # Count positive periods
            positive_count = sum((df[col] > 0).astype(int) for col in momentum_cols)
            df['momentum_quality'] = positive_count / len(momentum_cols)
            
            # Momentum strength (weighted by recency)
            weights = [0.4, 0.3, 0.2, 0.1, 0.0][:len(momentum_cols)]
            df['momentum_strength'] = sum(
                df[col].fillna(0) * w for col, w in zip(momentum_cols, weights)
            ) / sum(weights)
        
        # Long-term trend strength
        if all(col in df.columns for col in ['ret_6m', 'ret_1y']):
            df['trend_strength'] = (
                df['ret_6m'].fillna(0) * 0.6 + 
                df['ret_1y'].fillna(0) * 0.4
            ) / 100
        
        # Momentum divergence detection
        if all(col in df.columns for col in ['price', 'sma_20d', 'ret_30d']):
            # Price above MA but negative returns = bearish divergence
            df['momentum_divergence'] = np.where(
                (df['price'] > df['sma_20d']) & (df['ret_30d'] < -5), -1,
                np.where(
                    (df['price'] < df['sma_20d']) & (df['ret_30d'] > 5), 1,
                    0
                )
            )
        
        return df

class VolumeIntelligence:
    """Smart money flow and volume pattern analysis"""
    
    @staticmethod
    def analyze_volume_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Detect institutional activity and volume anomalies"""
        
        # Volume surge detection (all ratios are percentages)
        if all(col in df.columns for col in ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d']):
            # Immediate surge
            df['volume_surge_1d'] = (df['vol_ratio_1d_90d'] > 100).astype(int)
            
            # Sustained surge
            df['volume_surge_sustained'] = (
                (df['vol_ratio_7d_90d'] > 50) & 
                (df['vol_ratio_30d_90d'] > 20)
            ).astype(int)
            
            # Smart money detection: High 30d volume but low 1d (accumulation)
            df['smart_money_accumulation'] = (
                (df['vol_ratio_30d_90d'] > 30) & 
                (df['vol_ratio_1d_90d'] < 0) &
                (df['vol_ratio_7d_90d'] > 10)
            ).astype(float)
            
            # Volume exhaustion: Declining volume after price move
            if 'ret_30d' in df.columns:
                df['volume_exhaustion'] = (
                    (df['ret_30d'] > 20) & 
                    (df['vol_ratio_7d_90d'] < df['vol_ratio_30d_90d'] - 20)
                ).astype(float) * -1  # Negative signal
            
            # Volume quality score
            df['volume_quality'] = (
                df['volume_surge_sustained'] * 0.4 +
                df['smart_money_accumulation'] * 0.4 +
                (df['volume_exhaustion'] + 1) * 0.2  # Inverted
            )
        
        # Relative volume analysis
        if 'rvol' in df.columns:
            df['rvol_category'] = pd.cut(
                df['rvol'].fillna(1),
                bins=[0, 0.5, 1, 2, 5, np.inf],
                labels=['Very Low', 'Low', 'Normal', 'High', 'Extreme']
            )
            
            # Volume-price confirmation
            if 'ret_1d' in df.columns:
                df['volume_price_confirm'] = np.where(
                    (df['rvol'] > 2) & (df['ret_1d'] > 2), 1,  # Bullish
                    np.where(
                        (df['rvol'] > 2) & (df['ret_1d'] < -2), -1,  # Bearish
                        0
                    )
                )
        
        return df

class TechnicalSetupDetector:
    """Advanced technical pattern recognition"""
    
    @staticmethod
    def detect_setups(df: pd.DataFrame) -> pd.DataFrame:
        """Identify high-probability technical setups"""
        
        # Moving average analysis
        if all(col in df.columns for col in ['price', 'sma_20d', 'sma_50d', 'sma_200d']):
            # Perfect alignment: Price > 20 > 50 > 200
            df['ma_perfect_align'] = (
                (df['price'] > df['sma_20d']) & 
                (df['sma_20d'] > df['sma_50d']) & 
                (df['sma_50d'] > df['sma_200d'])
            ).astype(int)
            
            # Golden cross proximity
            df['golden_cross_near'] = (
                (df['sma_50d'] / df['sma_200d'] > 0.95) & 
                (df['sma_50d'] / df['sma_200d'] < 1.05) &
                (df['sma_50d'] > df['sma_50d'].shift(5))  # 50MA rising
            ).astype(int)
            
            # Compression: All MAs converging
            ma_spread = (
                df[['sma_20d', 'sma_50d', 'sma_200d']].max(axis=1) - 
                df[['sma_20d', 'sma_50d', 'sma_200d']].min(axis=1)
            ) / df['price']
            df['ma_compression'] = (ma_spread < 0.1).astype(int)
        
        # 52-week range analysis
        if all(col in df.columns for col in ['from_low_pct', 'from_high_pct']):
            # Near 52w high with momentum
            df['breakout_candidate'] = (
                (df['from_high_pct'] > -10) & 
                (df['from_high_pct'] < 0) &
                (df.get('momentum_strength', 0) > 5)
            ).astype(int)
            
            # Oversold bounce setup
            df['oversold_bounce'] = (
                (df['from_low_pct'] < 20) & 
                (df.get('ret_7d', 0) > 2) &
                (df.get('volume_surge_sustained', 0) == 1)
            ).astype(int)
            
            # Range position score (0-1)
            df['range_position'] = df['from_low_pct'] / (
                df['from_low_pct'] - df['from_high_pct']
            ).replace(0, 1)
        
        # Consolidation breakout
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            df['consolidation_breakout'] = (
                (df['ret_30d'].abs() < 5) &  # Tight 30d range
                (df['ret_7d'] > 3) &  # Recent breakout
                (df.get('volume_surge_1d', 0) == 1)  # With volume
            ).astype(int)
        
        # Technical score
        setup_cols = ['ma_perfect_align', 'golden_cross_near', 'ma_compression',
                      'breakout_candidate', 'oversold_bounce', 'consolidation_breakout']
        available_cols = [col for col in setup_cols if col in df.columns]
        
        if available_cols:
            df['technical_score'] = df[available_cols].sum(axis=1) / len(available_cols)
        
        return df

class FundamentalAnalyzer:
    """Deep fundamental analysis with growth detection"""
    
    @staticmethod
    def analyze_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive fundamental scoring"""
        
        # EPS Analysis
        if 'eps_tier' in df.columns:
            tier_scores = {
                '5↓': -1, '5↑': 0.2, '15↑': 0.4, '35↑': 0.6,
                '55↑': 0.8, '75↑': 0.9, '95↑': 1.0
            }
            df['eps_tier_score'] = df['eps_tier'].map(tier_scores).fillna(0)
        
        # EPS Growth Quality
        if 'eps_change_pct' in df.columns:
            # Normalize EPS change to 0-1 scale
            df['eps_growth_score'] = df['eps_change_pct'].clip(-50, 100)
            df['eps_growth_score'] = (df['eps_growth_score'] + 50) / 150
            
            # EPS acceleration
            if 'eps_current' in df.columns and 'eps_last_qtr' in df.columns:
                df['eps_accelerating'] = (
                    (df['eps_current'] > df['eps_last_qtr']) & 
                    (df['eps_change_pct'] > 20)
                ).astype(int)
        
        # Valuation Analysis
        if 'pe' in df.columns:
            # PE reasonableness (lower is better, but not negative)
            df['pe_score'] = np.where(
                df['pe'] <= 0, 0,  # Negative PE = bad
                np.where(
                    df['pe'] > 50, 0,  # Too expensive
                    1 - (df['pe'] / 50)  # Linear scale 0-50
                )
            )
            
            # GARP Score (Growth at Reasonable Price)
            if 'eps_change_pct' in df.columns:
                # PEG-like ratio
                df['garp_score'] = np.where(
                    (df['pe'] > 0) & (df['eps_change_pct'] > 0),
                    df['eps_change_pct'] / df['pe'],
                    0
                ).clip(0, 2) / 2  # Normalize to 0-1
        
        # Fundamental Quality Composite
        fundamental_components = []
        weights = []
        
        component_weights = {
            'eps_tier_score': 0.3,
            'eps_growth_score': 0.3,
            'pe_score': 0.2,
            'garp_score': 0.2
        }
        
        for comp, weight in component_weights.items():
            if comp in df.columns:
                fundamental_components.append(df[comp] * weight)
                weights.append(weight)
        
        if fundamental_components:
            df['fundamental_score'] = sum(fundamental_components) / sum(weights)
        else:
            df['fundamental_score'] = 0.5
        
        return df

class MarketRegimeDetector:
    """Identify market regimes and conditions"""
    
    @staticmethod
    def detect_regime(df: pd.DataFrame) -> pd.DataFrame:
        """Classify stocks into market regimes"""
        
        # Trend regime
        if all(col in df.columns for col in ['ret_30d', 'ret_3m']):
            df['trend_regime'] = np.where(
                (df['ret_30d'] > 10) & (df['ret_3m'] > 20), 'Strong Uptrend',
                np.where(
                    (df['ret_30d'] > 0) & (df['ret_3m'] > 0), 'Uptrend',
                    np.where(
                        (df['ret_30d'] < -10) & (df['ret_3m'] < -20), 'Strong Downtrend',
                        np.where(
                            (df['ret_30d'] < 0) & (df['ret_3m'] < 0), 'Downtrend',
                            'Sideways'
                        )
                    )
                )
            )
        
        # Volatility regime
        if all(col in df.columns for col in ['ret_1d', 'ret_3d', 'ret_7d']):
            # Calculate short-term volatility
            returns_df = df[['ret_1d', 'ret_3d', 'ret_7d']]
            df['volatility'] = returns_df.std(axis=1)
            
            df['volatility_regime'] = pd.cut(
                df['volatility'],
                bins=[0, 2, 5, 10, np.inf],
                labels=['Low Vol', 'Normal Vol', 'High Vol', 'Extreme Vol']
            )
        
        # Relative strength vs market
        if 'ret_30d' in df.columns:
            market_return = df['ret_30d'].median()
            df['relative_strength'] = df['ret_30d'] - market_return
            
            df['rs_regime'] = pd.cut(
                df['relative_strength'],
                bins=[-np.inf, -10, -5, 5, 10, np.inf],
                labels=['Very Weak', 'Weak', 'Neutral', 'Strong', 'Very Strong']
            )
        
        return df

class SpecialSituationDetector:
    """Identify special trading situations"""
    
    @staticmethod
    def detect_special_situations(df: pd.DataFrame) -> pd.DataFrame:
        """Find unique high-probability setups"""
        
        df['special_situation'] = 'None'
        df['situation_score'] = 0
        
        # 1. Stealth Breakout: Low volume price appreciation
        if all(col in df.columns for col in ['ret_30d', 'vol_ratio_30d_90d', 'from_high_pct']):
            mask = (
                (df['ret_30d'] > 15) & 
                (df['vol_ratio_30d_90d'] < 0) & 
                (df['from_high_pct'] > -20)
            )
            df.loc[mask, 'special_situation'] = 'Stealth Breakout'
            df.loc[mask, 'situation_score'] = 0.85
        
        # 2. Momentum Explosion: Everything aligning
        if all(col in df.columns for col in ['momentum_quality', 'volume_quality', 'technical_score']):
            mask = (
                (df['momentum_quality'] > 0.8) & 
                (df['volume_quality'] > 0.7) & 
                (df['technical_score'] > 0.6) &
                (df['special_situation'] == 'None')
            )
            df.loc[mask, 'special_situation'] = 'Momentum Explosion'
            df.loc[mask, 'situation_score'] = 0.95
        
        # 3. Value Emergence: Cheap stocks starting to move
        if all(col in df.columns for col in ['pe', 'ret_30d', 'eps_change_pct', 'volume_surge_sustained']):
            mask = (
                (df['pe'] > 0) & (df['pe'] < 15) & 
                (df['ret_30d'] > 5) & 
                (df['eps_change_pct'] > 20) &
                (df['volume_surge_sustained'] == 1) &
                (df['special_situation'] == 'None')
            )
            df.loc[mask, 'special_situation'] = 'Value Emergence'
            df.loc[mask, 'situation_score'] = 0.80
        
        # 4. Institutional Accumulation
        if 'smart_money_accumulation' in df.columns:
            mask = (
                (df['smart_money_accumulation'] == 1) & 
                (df.get('ma_perfect_align', 0) == 1) &
                (df['special_situation'] == 'None')
            )
            df.loc[mask, 'special_situation'] = 'Institutional Accumulation'
            df.loc[mask, 'situation_score'] = 0.88
        
        # 5. Earnings Catalyst
        if all(col in df.columns for col in ['eps_accelerating', 'volume_surge_1d', 'ret_1d']):
            mask = (
                (df.get('eps_accelerating', 0) == 1) & 
                (df['volume_surge_1d'] == 1) & 
                (df['ret_1d'] > 3) &
                (df['special_situation'] == 'None')
            )
            df.loc[mask, 'special_situation'] = 'Earnings Catalyst'
            df.loc[mask, 'situation_score'] = 0.90
        
        return df

# ============================================================================
# MASTER SCORING ALGORITHM
# ============================================================================
class AlphaScoreEngine:
    """The ultimate scoring algorithm combining all signals"""
    
    @staticmethod
    def calculate_alpha_score(df: pd.DataFrame) -> pd.DataFrame:
        """Generate the final alpha score using non-linear combinations"""
        
        # Initialize all engines
        df = MomentumEngine.calculate_momentum_signature(df)
        df = VolumeIntelligence.analyze_volume_patterns(df)
        df = TechnicalSetupDetector.detect_setups(df)
        df = FundamentalAnalyzer.analyze_fundamentals(df)
        df = MarketRegimeDetector.detect_regime(df)
        df = SpecialSituationDetector.detect_special_situations(df)
        
        # Component scores with smart defaults
        components = {
            'momentum': ['momentum_quality', 'momentum_strength', 'trend_strength'],
            'volume': ['volume_quality', 'smart_money_accumulation'],
            'technical': ['technical_score', 'ma_perfect_align', 'range_position'],
            'fundamental': ['fundamental_score', 'eps_growth_score'],
            'special': ['situation_score']
        }
        
        # Calculate component scores
        component_scores = {}
        for comp_name, cols in components.items():
            available_cols = [col for col in cols if col in df.columns]
            if available_cols:
                # Use geometric mean for better balance
                comp_values = df[available_cols].fillna(0.5).clip(0.01, 1)
                component_scores[comp_name] = np.power(
                    comp_values.prod(axis=1), 
                    1/len(available_cols)
                )
            else:
                component_scores[comp_name] = 0.5
        
        # Non-linear combination with interaction terms
        df['alpha_score'] = (
            component_scores['momentum'] ** 0.3 * 0.25 +
            component_scores['volume'] ** 0.5 * 0.20 +
            component_scores['technical'] ** 0.4 * 0.20 +
            component_scores['fundamental'] ** 0.6 * 0.25 +
            component_scores['special'] * 0.10
        )
        
        # Regime adjustments
        if 'trend_regime' in df.columns:
            regime_multipliers = {
                'Strong Uptrend': 1.1,
                'Uptrend': 1.05,
                'Sideways': 1.0,
                'Downtrend': 0.9,
                'Strong Downtrend': 0.8
            }
            df['regime_multiplier'] = df['trend_regime'].map(regime_multipliers).fillna(1)
            df['alpha_score'] *= df['regime_multiplier']
        
        # Risk adjustments
        if 'volatility' in df.columns:
            # Lower score for extreme volatility
            risk_adjustment = 1 - (df['volatility'].clip(0, 20) / 100)
            df['alpha_score'] *= (0.7 + 0.3 * risk_adjustment)
        
        # Ensure valid range
        df['alpha_score'] = df['alpha_score'].fillna(0.5).clip(0, 1)
        
        # Generate signals
        for signal, threshold in sorted(CONFIG['SIGNAL_THRESHOLDS'].items(), 
                                      key=lambda x: x[1], reverse=True):
            df.loc[df['alpha_score'] >= threshold, 'signal'] = signal
        
        # Add confidence level
        df['confidence'] = pd.cut(
            df['alpha_score'],
            bins=[0, 0.3, 0.5, 0.7, 0.85, 1.0],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Extreme']
        )
        
        # Sort by alpha score
        df = df.sort_values('alpha_score', ascending=False)
        
        return df

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def create_alpha_landscape(df: pd.DataFrame, limit: int = 100) -> go.Figure:
    """Create 3D visualization of alpha landscape"""
    
    plot_df = df.head(limit)
    
    # Create 3D scatter
    fig = go.Figure(data=[go.Scatter3d(
        x=plot_df.get('momentum_strength', plot_df.index).fillna(0),
        y=plot_df.get('volume_quality', plot_df.index).fillna(0),
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
            xaxis_title="Momentum Strength",
            yaxis_title="Volume Quality",
            zaxis_title="Alpha Score",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=600,
        margin=dict(r=20, b=10, l=10, t=40)
    )
    
    return fig

def create_factor_heatmap(df: pd.DataFrame, limit: int = 50) -> go.Figure:
    """Create heatmap of factor scores"""
    
    factors = ['momentum_quality', 'volume_quality', 'technical_score', 
               'fundamental_score', 'alpha_score']
    available_factors = [f for f in factors if f in df.columns]
    
    if not available_factors:
        return go.Figure()
    
    plot_df = df.head(limit)
    
    # Create heatmap data
    heatmap_data = plot_df[available_factors].fillna(0).T
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=plot_df['ticker'].tolist(),
        y=[f.replace('_', ' ').title() for f in available_factors],
        colorscale='RdYlGn',
        zmid=0.5
    ))
    
    fig.update_layout(
        title=f"Factor Analysis Heatmap (Top {limit})",
        xaxis_title="Stocks",
        yaxis_title="Factors",
        height=400
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

def create_sector_performance_matrix(df: pd.DataFrame) -> go.Figure:
    """Create sector performance matrix"""
    
    if 'sector' not in df.columns or 'alpha_score' not in df.columns:
        return go.Figure()
    
    # Calculate sector metrics
    sector_stats = df.groupby('sector').agg({
        'alpha_score': ['mean', 'count'],
        'ret_30d': 'mean' if 'ret_30d' in df.columns else lambda x: 0
    }).round(3)
    
    sector_stats.columns = ['avg_alpha', 'count', 'avg_return']
    sector_stats = sector_stats.sort_values('avg_alpha', ascending=False).head(15)
    
    # Create bubble chart
    fig = go.Figure(data=[go.Scatter(
        x=sector_stats['avg_return'],
        y=sector_stats['avg_alpha'],
        mode='markers+text',
        marker=dict(
            size=sector_stats['count'] * 3,
            color=sector_stats['avg_alpha'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Avg Alpha")
        ),
        text=sector_stats.index,
        textposition="top center"
    )])
    
    fig.update_layout(
        title="Sector Performance Matrix",
        xaxis_title="Average 30D Return %",
        yaxis_title="Average Alpha Score",
        height=500
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
    Multi-Asset Neural Trading Research Assistant<br>
    <i>Every data point. Every pattern. Every opportunity.</i>
    </p>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading market data..."):
        df = load_data()
    
    if df.empty:
        st.error("❌ Failed to load data. Please check your connection and try again.")
        st.stop()
    
    # Run alpha analysis
    with st.spinner("Running alpha analysis..."):
        analyzed_df = AlphaScoreEngine.calculate_alpha_score(df)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Center")
        
        # Data info
        st.info(f"""
        📊 **Data Status**
        - Stocks: {len(analyzed_df):,}
        - Columns: {len(analyzed_df.columns)}
        - Updated: {analyzed_df['data_timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M')}
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
            selected_sectors = st.multiselect("Sectors", sectors, default=sectors[:5])
        else:
            selected_sectors = []
        
        # Special situations
        if 'special_situation' in analyzed_df.columns:
            situations = analyzed_df['special_situation'].unique()
            selected_situations = st.multiselect(
                "Special Situations",
                situations,
                default=['None']
            )
        else:
            selected_situations = ['None']
        
        # Advanced filters
        with st.expander("Advanced Filters"):
            min_alpha = st.slider("Min Alpha Score", 0.0, 1.0, 0.5, 0.05)
            
            if 'market_cap' in analyzed_df.columns:
                market_cap_range = st.select_slider(
                    "Market Cap Range",
                    options=['All', 'Large Cap', 'Mid Cap', 'Small Cap'],
                    value='All'
                )
            else:
                market_cap_range = 'All'
            
            if 'pe' in analyzed_df.columns:
                max_pe = st.number_input("Max P/E Ratio", value=50.0, min_value=0.0)
            else:
                max_pe = 9999
        
        # Refresh button
        if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Apply filters
    filtered_df = analyzed_df[analyzed_df['signal'].isin(selected_signals)]
    
    if selected_sectors and 'sector' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['sector'].isin(selected_sectors)]
    
    if 'special_situation' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['special_situation'].isin(selected_situations)]
    
    filtered_df = filtered_df[filtered_df['alpha_score'] >= min_alpha]
    
    if market_cap_range != 'All' and 'category' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['category'] == market_cap_range]
    
    if 'pe' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['pe'] > 0) & (filtered_df['pe'] <= max_pe)]
    
    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        extreme_alpha = len(filtered_df[filtered_df['signal'] == 'ALPHA_EXTREME'])
        st.metric("🎯 Extreme Alpha", extreme_alpha, 
                 delta=f"Top {extreme_alpha/len(analyzed_df)*100:.1f}%")
    
    with col2:
        high_alpha = len(filtered_df[filtered_df['signal'] == 'ALPHA_HIGH'])
        st.metric("💎 High Alpha", high_alpha)
    
    with col3:
        special_count = len(filtered_df[filtered_df['special_situation'] != 'None'])
        st.metric("🌟 Special Setups", special_count)
    
    with col4:
        avg_alpha = filtered_df['alpha_score'].mean()
        st.metric("⚡ Avg Alpha", f"{avg_alpha:.3f}")
    
    with col5:
        if 'ret_30d' in filtered_df.columns:
            avg_return = filtered_df['ret_30d'].mean()
            st.metric("📈 Avg 30D Return", f"{avg_return:.1f}%")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Alpha Signals", "📊 Analytics", 
                                      "🔬 Deep Dive", "💾 Export"])
    
    with tab1:
        # Search box
        search_term = st.text_input("🔍 Search stocks", placeholder="Enter ticker or company name...")
        
        display_df = filtered_df.copy()
        if search_term:
            mask = (
                display_df['ticker'].str.contains(search_term.upper(), na=False) |
                display_df.get('company_name', '').str.contains(search_term, case=False, na=False)
            )
            display_df = display_df[mask]
        
        # Display columns based on availability
        base_cols = ['ticker', 'company_name', 'signal', 'alpha_score', 'confidence']
        data_cols = ['price', 'ret_30d', 'pe', 'eps_change_pct', 'volume_quality', 
                    'momentum_strength', 'special_situation']
        
        display_cols = base_cols + [col for col in data_cols if col in display_df.columns]
        
        # Styled dataframe
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
                'eps_change_pct': st.column_config.NumberColumn('EPS Chg%', format='%.1f%%'),
                'volume_quality': st.column_config.ProgressColumn('Vol Quality', format='%.2f'),
                'momentum_strength': st.column_config.NumberColumn('Momentum', format='%.1f')
            }
        )
        
        # Top picks summary
        if len(display_df) > 0:
            st.success(f"""
            **🏆 Top Alpha Pick: {display_df.iloc[0]['ticker']}**  
            Signal: {display_df.iloc[0]['signal']} | 
            Alpha: {display_df.iloc[0]['alpha_score']:.3f} | 
            Confidence: {display_df.iloc[0]['confidence']}
            """)
    
    with tab2:
        # Analytics visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Alpha landscape
            fig_3d = create_alpha_landscape(filtered_df, 75)
            st.plotly_chart(fig_3d, use_container_width=True)
        
        with col2:
            # Factor heatmap
            fig_heatmap = create_factor_heatmap(filtered_df, 30)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Signal distribution
        fig_signals = create_signal_distribution(filtered_df)
        st.plotly_chart(fig_signals, use_container_width=True)
        
        # Sector performance
        if 'sector' in filtered_df.columns:
            fig_sector = create_sector_performance_matrix(filtered_df)
            st.plotly_chart(fig_sector, use_container_width=True)
    
    with tab3:
        # Deep dive analysis
        st.header("🔬 Deep Dive Analysis")
        
        # Select stock for deep dive
        stock_ticker = st.selectbox(
            "Select stock for analysis",
            filtered_df['ticker'].head(50).tolist()
        )
        
        if stock_ticker:
            stock_data = filtered_df[filtered_df['ticker'] == stock_ticker].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 📊 Scores")
                scores_to_show = [
                    ('Alpha Score', 'alpha_score'),
                    ('Momentum Quality', 'momentum_quality'),
                    ('Volume Quality', 'volume_quality'),
                    ('Technical Score', 'technical_score'),
                    ('Fundamental Score', 'fundamental_score')
                ]
                
                for label, col in scores_to_show:
                    if col in stock_data:
                        value = stock_data[col]
                        if pd.notna(value):
                            st.metric(label, f"{value:.3f}")
            
            with col2:
                st.markdown("### 💰 Fundamentals")
                if 'pe' in stock_data and pd.notna(stock_data['pe']):
                    st.metric("P/E Ratio", f"{stock_data['pe']:.1f}")
                if 'eps_change_pct' in stock_data and pd.notna(stock_data['eps_change_pct']):
                    st.metric("EPS Change %", f"{stock_data['eps_change_pct']:.1f}%")
                if 'eps_tier' in stock_data:
                    st.metric("EPS Tier", stock_data['eps_tier'])
            
            with col3:
                st.markdown("### 📈 Performance")
                perf_metrics = [
                    ('30D Return', 'ret_30d'),
                    ('3M Return', 'ret_3m'),
                    ('1Y Return', 'ret_1y')
                ]
                
                for label, col in perf_metrics:
                    if col in stock_data and pd.notna(stock_data[col]):
                        st.metric(label, f"{stock_data[col]:.1f}%")
            
            # Special insights
            if stock_data.get('special_situation', 'None') != 'None':
                st.info(f"🌟 **Special Situation Detected:** {stock_data['special_situation']}")
            
            # Technical insights
            tech_insights = []
            if stock_data.get('ma_perfect_align', 0) == 1:
                tech_insights.append("✅ Perfect MA alignment")
            if stock_data.get('golden_cross_near', 0) == 1:
                tech_insights.append("✅ Golden cross nearby")
            if stock_data.get('breakout_candidate', 0) == 1:
                tech_insights.append("✅ Breakout candidate")
            
            if tech_insights:
                st.success("**Technical Insights:** " + " | ".join(tech_insights))
    
    with tab4:
        # Export options
        st.header("💾 Export Data")
        
        # Prepare export data
        export_df = filtered_df.copy()
        
        # CSV export
        csv = export_df.to_csv(index=False)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                "📥 Download Full Results (CSV)",
                csv,
                f"mantra_alpha_{timestamp}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            # Top 20 export
            top20_csv = export_df.head(20).to_csv(index=False)
            st.download_button(
                "🏆 Download Top 20 (CSV)",
                top20_csv,
                f"mantra_top20_{timestamp}.csv",
                "text/csv",
                use_container_width=True
            )
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        
        summary = {
            "Analysis Timestamp": timestamp,
            "Total Stocks Analyzed": len(analyzed_df),
            "Filtered Results": len(filtered_df),
            "Extreme Alpha Signals": extreme_alpha,
            "High Alpha Signals": high_alpha,
            "Special Situations": special_count,
            "Average Alpha Score": float(avg_alpha.round(4)),
            "Top Sectors": filtered_df['sector'].value_counts().head(3).to_dict() 
                          if 'sector' in filtered_df.columns else {},
            "Signal Distribution": filtered_df['signal'].value_counts().to_dict()
        }
        
        st.json(summary)
        
        # Strategy recommendations
        st.subheader("🎯 Strategy Recommendations")
        
        if extreme_alpha > 0:
            st.success(f"""
            **Extreme Alpha Strategy:** {extreme_alpha} stocks showing exceptional signals.
            These represent the highest conviction opportunities with multiple factors aligning.
            Consider for core positions with appropriate position sizing.
            """)
        
        if special_count > 5:
            st.info(f"""
            **Special Situations Alert:** {special_count} stocks in special setups.
            These unique situations often precede significant moves. 
            Monitor closely for entry opportunities.
            """)
        
        # Footer
        st.markdown("---")
        st.caption("""
        *M.A.N.T.R.A. Ultimate uses advanced algorithmic analysis across momentum, volume, 
        technical, and fundamental factors. All signals are generated through systematic 
        analysis of market data. Always conduct your own research before making investment decisions.*
        """)

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
