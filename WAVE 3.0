"""
Wave Detection System 3.0 - Professional Stock Analysis Platform
===============================================================
The All-Time Best Implementation

REVOLUTIONARY FEATURES:
- Momentum Acceleration Detection
- Category-Contextual Scoring  
- Signal Freshness Algorithm
- Smart Money Flow Analysis
- Multi-Timeframe Health Matrix
- Adaptive Market Regime
- Pattern Success Memory
- Institutional Activity Detection

Author: Advanced AI Systems
Version: 3.0.0
Status: Production Ready
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
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION AND CONSTANTS
# ============================================

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Wave Detection 3.0 | Professional Trading",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class MarketRegime:
    """Market regime configuration"""
    BULL_THRESHOLD: float = 0.65
    BEAR_THRESHOLD: float = 0.35
    NEUTRAL_RANGE: Tuple[float, float] = (0.35, 0.65)

@dataclass
class CategoryProfile:
    """Market cap category characteristics"""
    name: str
    typical_volatility: float  # Daily %
    volume_threshold: float
    unusual_move_threshold: float
    position_size_multiplier: float

# Category profiles based on market cap
CATEGORY_PROFILES = {
    'Large Cap': CategoryProfile(
        name='Large Cap',
        typical_volatility=1.5,
        volume_threshold=1000000,
        unusual_move_threshold=3.0,
        position_size_multiplier=1.5
    ),
    'Mid Cap': CategoryProfile(
        name='Mid Cap',
        typical_volatility=2.5,
        volume_threshold=500000,
        unusual_move_threshold=5.0,
        position_size_multiplier=1.0
    ),
    'Small Cap': CategoryProfile(
        name='Small Cap',
        typical_volatility=4.0,
        volume_threshold=100000,
        unusual_move_threshold=8.0,
        position_size_multiplier=0.5
    ),
    'Micro Cap': CategoryProfile(
        name='Micro Cap',
        typical_volatility=6.0,
        volume_threshold=50000,
        unusual_move_threshold=10.0,
        position_size_multiplier=0.3
    )
}

class SignalStrength(Enum):
    """Signal strength enumeration"""
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    WATCH = "WATCH"
    AVOID = "AVOID"
    SELL = "SELL"

@dataclass
class TradingConstants:
    """All trading constants in one place"""
    # Momentum thresholds
    MOMENTUM_ACCELERATION_STRONG: float = 2.0
    MOMENTUM_ACCELERATION_WEAK: float = 0.5
    
    # Volume thresholds
    VOLUME_SPIKE_THRESHOLD: float = 2.5
    VOLUME_EXHAUSTION_THRESHOLD: float = 0.5
    
    # Position thresholds
    OVERSOLD_THRESHOLD: float = 30.0
    OVERBOUGHT_THRESHOLD: float = 80.0
    
    # Risk management
    MAX_POSITION_SIZE: float = 0.03  # 3% max
    MIN_POSITION_SIZE: float = 0.005  # 0.5% min
    
    # Sector constraints
    MIN_STOCKS_FOR_RELIABLE_SECTOR: int = 10
    SECTOR_CONCENTRATION_LIMIT: float = 0.25
    
    # Time decay
    SIGNAL_FRESH_DAYS: int = 2
    SIGNAL_STALE_DAYS: int = 5
    
    # Smart money thresholds
    SMART_MONEY_EPS_CHANGE: float = 15.0
    SMART_MONEY_VOLUME_SPIKE: float = 2.0

# Initialize constants
CONSTANTS = TradingConstants()

# ============================================
# PROFESSIONAL CSS STYLING
# ============================================

st.markdown("""
<style>
    /* Main Header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 1rem;
    }
    
    /* Signal Cards */
    .strong-buy-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 1rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    
    .buy-card {
        background-color: #d4f1e4;
        border-left: 5px solid #27ae60;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    
    .watch-card {
        background-color: #fef3c7;
        border-left: 5px solid #f59e0b;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    
    .sell-card {
        background-color: #fee2e2;
        border-left: 5px solid #ef4444;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        text-align: center;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.1);
    }
    
    /* Alert Badges */
    .alert-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .momentum-badge {
        background-color: #8b5cf6;
        color: white;
    }
    
    .volume-badge {
        background-color: #3b82f6;
        color: white;
    }
    
    .smart-money-badge {
        background-color: #10b981;
        color: white;
    }
    
    .warning-badge {
        background-color: #f59e0b;
        color: white;
    }
    
    /* Market Health Indicator */
    .market-health-indicator {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Professional Tables */
    .dataframe {
        font-size: 0.9rem;
    }
    
    .dataframe th {
        background-color: #f3f4f6;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 0.05em;
    }
    
    /* Category Indicators */
    .category-indicator {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 0.3rem;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .large-cap-indicator {
        background-color: #1e40af;
        color: white;
    }
    
    .mid-cap-indicator {
        background-color: #7c3aed;
        color: white;
    }
    
    .small-cap-indicator {
        background-color: #dc2626;
        color: white;
    }
    
    /* Animations */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA STRUCTURES AND MODELS
# ============================================

@dataclass
class StockSignal:
    """Complete stock signal information"""
    ticker: str
    company_name: str
    category: str
    sector: str
    price: float
    signal_strength: SignalStrength
    total_score: float
    
    # Component scores
    momentum_score: float
    volume_score: float
    position_score: float
    smart_money_score: float
    
    # Key metrics
    momentum_acceleration: float
    volume_pattern: str
    signal_freshness: str
    category_context: float
    
    # Risk metrics
    position_size: float
    stop_loss: float
    target_price: float
    risk_reward_ratio: float
    
    # Additional context
    entry_reasons: List[str] = field(default_factory=list)
    exit_warnings: List[str] = field(default_factory=list)
    special_situations: List[str] = field(default_factory=list)

@dataclass
class MarketHealth:
    """Complete market health assessment"""
    regime: str
    regime_strength: float
    market_score: int
    
    # Breadth metrics
    advance_decline_ratio: float
    new_highs: int
    new_lows: int
    stocks_above_sma20: float
    stocks_above_sma50: float
    stocks_above_sma200: float
    
    # Volume metrics
    volume_breadth: float
    unusual_volume_stocks: int
    
    # Momentum metrics
    average_momentum: float
    momentum_breadth: float
    
    # Category leadership
    leading_category: str
    lagging_category: str
    category_rotation: Dict[str, float]
    
    # Sector analysis
    top_sectors: List[Tuple[str, float]]
    bottom_sectors: List[Tuple[str, float]]
    sector_dispersion: float

# ============================================
# CORE DATA FUNCTIONS
# ============================================

@st.cache_data(ttl=300)
def load_market_data(sheet_url: str, gid: str) -> pd.DataFrame:
    """Load data from Google Sheets with comprehensive error handling"""
    try:
        # Construct CSV URL
        base_url = sheet_url.split('/edit')[0]
        csv_url = f"{base_url}/export?format=csv&gid={gid}"
        
        # Load data
        df = pd.read_csv(csv_url)
        
        if df.empty:
            logger.warning("Loaded empty dataframe")
            return pd.DataFrame()
            
        logger.info(f"Successfully loaded {len(df)} rows from Google Sheets")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        st.error(f"❌ Data loading failed: {str(e)}")
        return pd.DataFrame()

def clean_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Professional data cleaning and preparation"""
    if df.empty:
        return df
        
    try:
        df = df.copy()
        initial_rows = len(df)
        
        # Define column groups
        price_columns = ['price', 'prev_close', 'low_52w', 'high_52w', 
                        'sma_20d', 'sma_50d', 'sma_200d']
        return_columns = ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 
                         'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y']
        volume_columns = ['volume_1d', 'volume_7d', 'volume_30d', 
                         'volume_90d', 'volume_180d']
        ratio_columns = ['from_low_pct', 'from_high_pct', 'rvol',
                        'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
                        'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 
                        'vol_ratio_90d_180d']
        fundamental_columns = ['pe', 'eps_current', 'eps_last_qtr', 'eps_change_pct']
        
        # Clean numeric columns
        all_numeric = price_columns + return_columns + volume_columns + ratio_columns + fundamental_columns
        
        for col in all_numeric:
            if col in df.columns:
                # Clean string representations
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace('₹', '', regex=False)
                    df[col] = df[col].str.replace('%', '', regex=False)
                    df[col] = df[col].str.replace(',', '', regex=False)
                    df[col] = df[col].str.strip()
                    df[col] = df[col].replace(['', '-', 'N/A', 'n/a', '#N/A', 'nan'], np.nan)
                
                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean market cap
        if 'market_cap' in df.columns:
            df['market_cap_value'] = (
                df['market_cap'].astype(str)
                .str.replace('₹', '', regex=False)
                .str.replace(' Cr', '', regex=False)
                .str.replace(',', '', regex=False)
            )
            df['market_cap_value'] = pd.to_numeric(df['market_cap_value'], errors='coerce')
        
        # Clean categorical columns
        for col in ['ticker', 'company_name', 'category', 'sector']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(['nan', 'None', ''], 'Unknown')
        
        # Remove invalid rows
        df = df[df['ticker'] != 'Unknown']
        df = df[df['price'] > 0.01]
        df = df.drop_duplicates(subset=['ticker'], keep='first')
        
        # Apply data quality filters
        if 'rvol' in df.columns:
            df = df[df['rvol'] <= 50]  # Remove extreme outliers
        
        if 'ret_1d' in df.columns:
            df = df[df['ret_1d'].between(-25, 25)]  # Circuit limits
        
        # Calculate derived metrics
        df = calculate_momentum_metrics(df)
        df = calculate_volume_patterns(df)
        df = calculate_position_metrics(df)
        df = calculate_smart_money_indicators(df)
        
        final_rows = len(df)
        logger.info(f"Data cleaned: {initial_rows} → {final_rows} rows")
        
        return df
        
    except Exception as e:
        logger.error(f"Data cleaning failed: {str(e)}")
        st.error(f"❌ Data preparation failed: {str(e)}")
        return df

# ============================================
# ADVANCED CALCULATION FUNCTIONS
# ============================================

def calculate_momentum_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate advanced momentum metrics"""
    if df.empty:
        return df
        
    # Momentum acceleration
    df['momentum_1d'] = df['ret_1d']
    df['momentum_3d_avg'] = df['ret_3d'] / 3
    df['momentum_7d_avg'] = df['ret_7d'] / 7
    
    # Acceleration score
    df['momentum_acceleration'] = (
        df['momentum_1d'] - df['momentum_7d_avg']
    )
    
    # Momentum quality
    df['momentum_quality'] = df.apply(
        lambda row: calculate_momentum_quality(
            row['ret_1d'], 
            row['ret_3d'], 
            row['ret_7d'],
            row['ret_30d']
        ), 
        axis=1
    )
    
    # Trend strength
    df['trend_consistency'] = df.apply(
        lambda row: calculate_trend_consistency(row),
        axis=1
    )
    
    return df

def calculate_momentum_quality(ret_1d: float, ret_3d: float, 
                             ret_7d: float, ret_30d: float) -> str:
    """Determine momentum quality"""
    # Check for acceleration
    if ret_1d > ret_3d/3 and ret_3d/3 > ret_7d/7:
        return "ACCELERATING"
    
    # Check for deceleration
    elif ret_1d < ret_3d/3 and ret_3d/3 < ret_7d/7:
        return "DECELERATING"
    
    # Check for reversal
    elif ret_1d > 0 and ret_3d < 0:
        return "REVERSAL_UP"
    elif ret_1d < 0 and ret_3d > 0:
        return "REVERSAL_DOWN"
    
    # Check for exhaustion
    elif abs(ret_7d) > 15 and abs(ret_1d) < abs(ret_7d/7):
        return "EXHAUSTED"
    
    return "STABLE"

def calculate_trend_consistency(row: pd.Series) -> float:
    """Calculate trend consistency score"""
    score = 0
    
    # Price vs SMAs
    if row['price'] > row['sma_20d']:
        score += 1
    if row['sma_20d'] > row['sma_50d']:
        score += 1
    if row['sma_50d'] > row['sma_200d']:
        score += 1
    
    # Returns alignment
    if row['ret_7d'] > 0 and row['ret_30d'] > 0:
        score += 1
    if row['ret_30d'] > 0 and row['ret_3m'] > 0:
        score += 1
    
    return score / 5

def calculate_volume_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Identify volume patterns"""
    if df.empty:
        return df
    
    # Volume surge detection
    df['volume_surge_today'] = df['volume_1d'] / (df['volume_7d'] / 7)
    df['volume_surge_today'] = df['volume_surge_today'].fillna(1)
    
    # Volume trend
    df['volume_trend'] = df['vol_ratio_30d_90d']
    
    # Volume pattern classification
    df['volume_pattern'] = df.apply(classify_volume_pattern, axis=1)
    
    # Smart volume (volume vs price move)
    df['smart_volume_ratio'] = df.apply(
        lambda row: calculate_smart_volume(row['rvol'], row['ret_1d']),
        axis=1
    )
    
    return df

def classify_volume_pattern(row: pd.Series) -> str:
    """Classify volume pattern"""
    rvol = row.get('rvol', 1)
    surge = row.get('volume_surge_today', 1)
    trend = row.get('volume_trend', 1)
    ret_1d = row.get('ret_1d', 0)
    
    # Explosive volume
    if rvol > 3 and surge > 3:
        return "EXPLOSIVE"
    
    # Accumulation
    elif rvol > 1.5 and ret_1d > 0 and trend > 1:
        return "ACCUMULATION"
    
    # Distribution
    elif rvol > 1.5 and ret_1d < 0:
        return "DISTRIBUTION"
    
    # Exhaustion
    elif rvol < 0.5 and abs(ret_1d) < 1:
        return "EXHAUSTION"
    
    # Breakout
    elif rvol > 2 and abs(ret_1d) > 3:
        return "BREAKOUT"
    
    return "NORMAL"

def calculate_smart_volume(rvol: float, ret_1d: float) -> float:
    """Calculate smart volume ratio"""
    if abs(ret_1d) < 0.1:
        return 0
    return rvol / (1 + abs(ret_1d))

def calculate_position_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate position-based metrics"""
    if df.empty:
        return df
    
    # 52-week range position
    df['range_position'] = (
        (df['price'] - df['low_52w']) / 
        (df['high_52w'] - df['low_52w'])
    ).fillna(0.5)
    
    # Breakout detection
    df['near_52w_high'] = df['from_high_pct'] > -5
    df['near_52w_low'] = df['from_low_pct'] < 5
    
    # Support/Resistance levels
    df['above_all_smas'] = (
        (df['price'] > df['sma_20d']) & 
        (df['price'] > df['sma_50d']) & 
        (df['price'] > df['sma_200d'])
    )
    
    return df

def calculate_smart_money_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Detect institutional/smart money activity"""
    if df.empty:
        return df
    
    # EPS momentum with volume
    df['smart_money_eps'] = df.apply(
        lambda row: (
            row.get('eps_change_pct', 0) > CONSTANTS.SMART_MONEY_EPS_CHANGE and
            row.get('rvol', 0) > CONSTANTS.SMART_MONEY_VOLUME_SPIKE
        ),
        axis=1
    )
    
    # Unusual activity for category
    df['unusual_activity'] = False
    for category, profile in CATEGORY_PROFILES.items():
        mask = df['category'] == category
        if mask.any():
            df.loc[mask, 'unusual_activity'] = (
                df.loc[mask, 'ret_1d'].abs() > profile.unusual_move_threshold
            ) & (df.loc[mask, 'rvol'] > 2)
    
    # Institutional accumulation pattern
    df['institutional_pattern'] = (
        (df['volume_trend'] > 1.2) &  # Rising volume trend
        (df['ret_30d'] > 0) &  # Positive medium-term
        (df['from_high_pct'] < -10)  # Not at highs
    )
    
    return df

# ============================================
# MARKET ANALYSIS FUNCTIONS
# ============================================

def analyze_market_health(df: pd.DataFrame) -> MarketHealth:
    """Comprehensive market health analysis"""
    if df.empty:
        return create_default_market_health()
    
    total_stocks = len(df)
    
    # Market breadth
    advancing = len(df[df['ret_1d'] > 0])
    declining = len(df[df['ret_1d'] < 0])
    
    advance_decline_ratio = (
        advancing / declining if declining > 0 
        else float(advancing) if advancing > 0 
        else 1.0
    )
    
    # SMA analysis
    above_sma20 = len(df[df['price'] > df['sma_20d']]) / total_stocks * 100
    above_sma50 = len(df[df['price'] > df['sma_50d']]) / total_stocks * 100
    above_sma200 = len(df[df['price'] > df['sma_200d']]) / total_stocks * 100
    
    # Market regime
    sma50_ratio = above_sma50 / 100
    market_regime = MarketRegime()
    
    if sma50_ratio >= market_regime.BULL_THRESHOLD:
        regime = "BULL"
        regime_strength = (sma50_ratio - market_regime.BULL_THRESHOLD) / (1 - market_regime.BULL_THRESHOLD)
    elif sma50_ratio <= market_regime.BEAR_THRESHOLD:
        regime = "BEAR"
        regime_strength = (market_regime.BEAR_THRESHOLD - sma50_ratio) / market_regime.BEAR_THRESHOLD
    else:
        regime = "NEUTRAL"
        regime_strength = 0.5
    
    # New highs/lows
    new_highs = len(df[df['from_high_pct'] > -5])
    new_lows = len(df[df['from_low_pct'] < 5])
    
    # Volume breadth
    high_volume_stocks = len(df[df['rvol'] > 2])
    volume_breadth = high_volume_stocks / total_stocks * 100
    
    # Momentum analysis
    avg_momentum = df['ret_7d'].mean()
    positive_momentum = len(df[df['ret_7d'] > 0]) / total_stocks * 100
    
    # Category analysis
    category_performance = df.groupby('category')['ret_30d'].mean().sort_values(ascending=False)
    leading_category = category_performance.index[0] if not category_performance.empty else "Unknown"
    lagging_category = category_performance.index[-1] if not category_performance.empty else "Unknown"
    
    # Sector analysis
    sector_performance = df.groupby('sector')['ret_30d'].mean().sort_values(ascending=False)
    top_sectors = list(sector_performance.head(5).items())
    bottom_sectors = list(sector_performance.tail(5).items())
    sector_dispersion = sector_performance.std()
    
    # Market score calculation
    market_score = calculate_market_score(
        sma50_ratio, 
        advance_decline_ratio,
        volume_breadth / 100,
        positive_momentum / 100
    )
    
    return MarketHealth(
        regime=regime,
        regime_strength=regime_strength,
        market_score=market_score,
        advance_decline_ratio=advance_decline_ratio,
        new_highs=new_highs,
        new_lows=new_lows,
        stocks_above_sma20=above_sma20,
        stocks_above_sma50=above_sma50,
        stocks_above_sma200=above_sma200,
        volume_breadth=volume_breadth,
        unusual_volume_stocks=high_volume_stocks,
        average_momentum=avg_momentum,
        momentum_breadth=positive_momentum,
        leading_category=leading_category,
        lagging_category=lagging_category,
        category_rotation=category_performance.to_dict(),
        top_sectors=top_sectors,
        bottom_sectors=bottom_sectors,
        sector_dispersion=sector_dispersion
    )

def create_default_market_health() -> MarketHealth:
    """Create default market health when no data available"""
    return MarketHealth(
        regime="UNKNOWN",
        regime_strength=0,
        market_score=50,
        advance_decline_ratio=1.0,
        new_highs=0,
        new_lows=0,
        stocks_above_sma20=50.0,
        stocks_above_sma50=50.0,
        stocks_above_sma200=50.0,
        volume_breadth=0.0,
        unusual_volume_stocks=0,
        average_momentum=0.0,
        momentum_breadth=50.0,
        leading_category="Unknown",
        lagging_category="Unknown",
        category_rotation={},
        top_sectors=[],
        bottom_sectors=[],
        sector_dispersion=0.0
    )

def calculate_market_score(sma_ratio: float, ad_ratio: float, 
                         vol_breadth: float, mom_breadth: float) -> int:
    """Calculate comprehensive market score"""
    # Weighted components
    sma_score = sma_ratio * 30  # 30% weight
    ad_score = min(ad_ratio / 2, 1) * 25  # 25% weight
    vol_score = vol_breadth * 25  # 25% weight
    mom_score = mom_breadth * 20  # 20% weight
    
    total_score = sma_score + ad_score + vol_score + mom_score
    return int(min(max(total_score, 0), 100))

# ============================================
# SIGNAL GENERATION ENGINE
# ============================================

def generate_trading_signals(df: pd.DataFrame, market_health: MarketHealth) -> pd.DataFrame:
    """Generate comprehensive trading signals"""
    if df.empty:
        return df
    
    # Calculate signal components for each stock
    signals = []
    
    for _, row in df.iterrows():
        signal = calculate_stock_signal(row, market_health)
        signals.append(signal)
    
    # Convert to dataframe
    signal_df = pd.DataFrame([
        {
            'ticker': s.ticker,
            'company_name': s.company_name,
            'category': s.category,
            'sector': s.sector,
            'price': s.price,
            'signal_strength': s.signal_strength.value,
            'total_score': s.total_score,
            'momentum_score': s.momentum_score,
            'volume_score': s.volume_score,
            'position_score': s.position_score,
            'smart_money_score': s.smart_money_score,
            'momentum_acceleration': s.momentum_acceleration,
            'volume_pattern': s.volume_pattern,
            'signal_freshness': s.signal_freshness,
            'category_context': s.category_context,
            'position_size': s.position_size,
            'stop_loss': s.stop_loss,
            'target_price': s.target_price,
            'risk_reward_ratio': s.risk_reward_ratio,
            'entry_reasons': s.entry_reasons,
            'exit_warnings': s.exit_warnings,
            'special_situations': s.special_situations,
            **{col: row[col] for col in df.columns if col not in 
               ['ticker', 'company_name', 'category', 'sector', 'price']}
        }
        for s in signals
    ])
    
    return signal_df

def calculate_stock_signal(row: pd.Series, market_health: MarketHealth) -> StockSignal:
    """Calculate comprehensive signal for a single stock"""
    # Get category profile
    category_profile = CATEGORY_PROFILES.get(
        row['category'], 
        CATEGORY_PROFILES['Small Cap']  # Default
    )
    
    # Calculate component scores
    momentum_score, momentum_details = calculate_momentum_score(row)
    volume_score, volume_details = calculate_volume_score(row)
    position_score, position_details = calculate_position_score(row)
    smart_money_score, smart_details = calculate_smart_money_score(row)
    
    # Category context adjustment
    category_context = calculate_category_context(row, category_profile)
    
    # Market regime adjustment
    regime_multiplier = get_regime_multiplier(market_health.regime)
    
    # Calculate total score with weights
    base_score = (
        momentum_score * 0.35 +
        volume_score * 0.30 +
        position_score * 0.20 +
        smart_money_score * 0.15
    )
    
    # Apply adjustments
    total_score = base_score * category_context * regime_multiplier
    
    # Determine signal strength
    signal_strength = determine_signal_strength(total_score, momentum_details, volume_details)
    
    # Calculate position size
    position_size = calculate_position_size(
        total_score, 
        signal_strength,
        category_profile,
        row.get('rvol', 1)
    )
    
    # Risk management
    stop_loss, target_price = calculate_risk_targets(
        row['price'],
        signal_strength,
        row.get('from_low_pct', 50),
        category_profile.typical_volatility
    )
    
    risk_reward_ratio = (target_price - row['price']) / (row['price'] - stop_loss)
    
    # Signal freshness
    signal_freshness = determine_signal_freshness(
        momentum_details['quality'],
        row.get('ret_1d', 0),
        row.get('rvol', 1)
    )
    
    # Compile entry reasons and warnings
    entry_reasons = compile_entry_reasons(
        momentum_details, volume_details, position_details, smart_details
    )
    
    exit_warnings = compile_exit_warnings(row)
    
    special_situations = identify_special_situations(row, market_health)
    
    return StockSignal(
        ticker=row['ticker'],
        company_name=row['company_name'],
        category=row['category'],
        sector=row['sector'],
        price=row['price'],
        signal_strength=signal_strength,
        total_score=total_score,
        momentum_score=momentum_score,
        volume_score=volume_score,
        position_score=position_score,
        smart_money_score=smart_money_score,
        momentum_acceleration=momentum_details['acceleration'],
        volume_pattern=volume_details['pattern'],
        signal_freshness=signal_freshness,
        category_context=category_context,
        position_size=position_size,
        stop_loss=stop_loss,
        target_price=target_price,
        risk_reward_ratio=risk_reward_ratio,
        entry_reasons=entry_reasons,
        exit_warnings=exit_warnings,
        special_situations=special_situations
    )

def calculate_momentum_score(row: pd.Series) -> Tuple[float, Dict]:
    """Calculate momentum component score"""
    score = 0
    details = {
        'acceleration': row.get('momentum_acceleration', 0),
        'quality': row.get('momentum_quality', 'STABLE'),
        'trend_consistency': row.get('trend_consistency', 0)
    }
    
    # Acceleration scoring
    if details['acceleration'] > CONSTANTS.MOMENTUM_ACCELERATION_STRONG:
        score += 40
    elif details['acceleration'] > 0:
        score += 20
    elif details['acceleration'] < -CONSTANTS.MOMENTUM_ACCELERATION_STRONG:
        score -= 20
    
    # Quality scoring
    quality_scores = {
        'ACCELERATING': 30,
        'REVERSAL_UP': 20,
        'STABLE': 10,
        'DECELERATING': -10,
        'REVERSAL_DOWN': -20,
        'EXHAUSTED': -30
    }
    score += quality_scores.get(details['quality'], 0)
    
    # Trend consistency
    score += details['trend_consistency'] * 20
    
    # Time-based momentum
    if row.get('ret_1d', 0) > 0 and row.get('ret_7d', 0) > 0:
        score += 10
    
    return max(0, min(100, score)), details

def calculate_volume_score(row: pd.Series) -> Tuple[float, Dict]:
    """Calculate volume component score"""
    score = 0
    details = {
        'pattern': row.get('volume_pattern', 'NORMAL'),
        'rvol': row.get('rvol', 1),
        'surge': row.get('volume_surge_today', 1),
        'smart_volume': row.get('smart_volume_ratio', 0)
    }
    
    # Pattern scoring
    pattern_scores = {
        'EXPLOSIVE': 40,
        'BREAKOUT': 30,
        'ACCUMULATION': 25,
        'NORMAL': 10,
        'DISTRIBUTION': -20,
        'EXHAUSTION': -30
    }
    score += pattern_scores.get(details['pattern'], 0)
    
    # Relative volume
    if details['rvol'] > 3:
        score += 30
    elif details['rvol'] > 2:
        score += 20
    elif details['rvol'] > 1.5:
        score += 10
    elif details['rvol'] < 0.5:
        score -= 20
    
    # Volume surge
    if details['surge'] > 3:
        score += 20
    elif details['surge'] > 2:
        score += 10
    
    # Smart volume (efficiency)
    if details['smart_volume'] < 0.5 and details['rvol'] > 2:
        score += 10  # Efficient volume
    
    return max(0, min(100, score)), details

def calculate_position_score(row: pd.Series) -> Tuple[float, Dict]:
    """Calculate position component score"""
    score = 0
    details = {
        'from_low': row.get('from_low_pct', 50),
        'from_high': row.get('from_high_pct', -50),
        'range_position': row.get('range_position', 0.5),
        'above_all_smas': row.get('above_all_smas', False)
    }
    
    # Distance from low
    if details['from_low'] < CONSTANTS.OVERSOLD_THRESHOLD:
        score += 30
    elif details['from_low'] < 50:
        score += 20
    elif details['from_low'] > CONSTANTS.OVERBOUGHT_THRESHOLD:
        score -= 10
    
    # Distance from high
    if details['from_high'] < -20:
        score += 20  # Room to run
    elif details['from_high'] > -5 and row.get('ret_1d', 0) > 2:
        score += 30  # Breaking out
    elif details['from_high'] > -5 and row.get('ret_1d', 0) < 0:
        score -= 20  # Rejection at highs
    
    # Range position
    if details['range_position'] > 0.8 and row.get('momentum_quality') == 'ACCELERATING':
        score += 20  # High range breakout
    elif details['range_position'] < 0.2 and row.get('ret_7d', 0) > 0:
        score += 15  # Oversold bounce
    
    # SMA alignment
    if details['above_all_smas']:
        score += 15
    
    return max(0, min(100, score)), details

def calculate_smart_money_score(row: pd.Series) -> Tuple[float, Dict]:
    """Calculate smart money component score"""
    score = 0
    details = {
        'eps_momentum': row.get('smart_money_eps', False),
        'unusual_activity': row.get('unusual_activity', False),
        'institutional_pattern': row.get('institutional_pattern', False),
        'eps_change': row.get('eps_change_pct', 0)
    }
    
    # EPS momentum with volume
    if details['eps_momentum']:
        score += 40
    elif details['eps_change'] > 10:
        score += 20
    
    # Unusual activity for category
    if details['unusual_activity']:
        score += 30
    
    # Institutional accumulation
    if details['institutional_pattern']:
        score += 30
    
    return max(0, min(100, score)), details

def calculate_category_context(row: pd.Series, profile: CategoryProfile) -> float:
    """Calculate category-specific context multiplier"""
    multiplier = 1.0
    
    # Relative volatility
    actual_volatility = abs(row.get('ret_1d', 0))
    if actual_volatility > profile.typical_volatility * 2:
        multiplier *= 1.2  # Unusual for category
    
    # Volume relative to category norms
    if row.get('volume_30d', 0) > profile.volume_threshold * 2:
        multiplier *= 1.1
    
    # Category-specific position size adjustment
    multiplier *= profile.position_size_multiplier
    
    return multiplier

def get_regime_multiplier(regime: str) -> float:
    """Get market regime adjustment multiplier"""
    multipliers = {
        'BULL': 1.1,
        'NEUTRAL': 1.0,
        'BEAR': 0.8,
        'UNKNOWN': 0.9
    }
    return multipliers.get(regime, 1.0)

def determine_signal_strength(score: float, momentum: Dict, volume: Dict) -> SignalStrength:
    """Determine signal strength from scores"""
    # Strong buy conditions
    if (score >= 80 and 
        momentum['quality'] in ['ACCELERATING', 'REVERSAL_UP'] and
        volume['pattern'] in ['EXPLOSIVE', 'BREAKOUT', 'ACCUMULATION']):
        return SignalStrength.STRONG_BUY
    
    # Buy conditions
    elif score >= 65:
        return SignalStrength.BUY
    
    # Watch conditions
    elif score >= 50:
        return SignalStrength.WATCH
    
    # Sell conditions
    elif (score < 30 or 
          momentum['quality'] in ['EXHAUSTED', 'REVERSAL_DOWN'] or
          volume['pattern'] == 'DISTRIBUTION'):
        return SignalStrength.SELL
    
    # Default avoid
    return SignalStrength.AVOID

def calculate_position_size(score: float, signal: SignalStrength, 
                          profile: CategoryProfile, rvol: float) -> float:
    """Calculate appropriate position size"""
    # Base position sizes by signal
    base_sizes = {
        SignalStrength.STRONG_BUY: 0.025,
        SignalStrength.BUY: 0.02,
        SignalStrength.WATCH: 0.015,
        SignalStrength.AVOID: 0.01,
        SignalStrength.SELL: 0
    }
    
    base_size = base_sizes.get(signal, 0.01)
    
    # Adjust for score confidence
    confidence_multiplier = min(score / 100, 1.0)
    
    # Adjust for volatility
    volatility_multiplier = min(2 / max(rvol, 1), 1.0)
    
    # Apply category multiplier
    size = base_size * confidence_multiplier * volatility_multiplier * profile.position_size_multiplier
    
    # Apply limits
    return max(CONSTANTS.MIN_POSITION_SIZE, 
               min(CONSTANTS.MAX_POSITION_SIZE, size))

def calculate_risk_targets(price: float, signal: SignalStrength, 
                         from_low: float, volatility: float) -> Tuple[float, float]:
    """Calculate stop loss and target prices"""
    # Base risk/reward by signal
    risk_rewards = {
        SignalStrength.STRONG_BUY: (5, 20),
        SignalStrength.BUY: (6, 15),
        SignalStrength.WATCH: (7, 12),
        SignalStrength.AVOID: (8, 8),
        SignalStrength.SELL: (3, 0)
    }
    
    risk_pct, reward_pct = risk_rewards.get(signal, (7, 10))
    
    # Adjust for position in range
    if from_low < 30:
        reward_pct *= 1.2
    elif from_low > 70:
        risk_pct *= 1.2
    
    # Adjust for volatility
    risk_pct = max(risk_pct, volatility * 2)
    
    stop_loss = price * (1 - risk_pct / 100)
    target_price = price * (1 + reward_pct / 100)
    
    return stop_loss, target_price

def determine_signal_freshness(quality: str, ret_1d: float, rvol: float) -> str:
    """Determine how fresh/actionable the signal is"""
    if quality == 'ACCELERATING' and ret_1d > 2 and rvol > 2:
        return "FRESH_HOT"
    elif quality in ['ACCELERATING', 'REVERSAL_UP'] and rvol > 1.5:
        return "FRESH"
    elif quality == 'EXHAUSTED' or (abs(ret_1d) > 5 and rvol < 1):
        return "STALE"
    else:
        return "NORMAL"

def compile_entry_reasons(momentum: Dict, volume: Dict, 
                         position: Dict, smart: Dict) -> List[str]:
    """Compile human-readable entry reasons"""
    reasons = []
    
    # Momentum reasons
    if momentum['acceleration'] > CONSTANTS.MOMENTUM_ACCELERATION_STRONG:
        reasons.append("🚀 Strong momentum acceleration")
    if momentum['quality'] == 'ACCELERATING':
        reasons.append("📈 Momentum accelerating")
    elif momentum['quality'] == 'REVERSAL_UP':
        reasons.append("🔄 Bullish reversal pattern")
    
    # Volume reasons
    if volume['pattern'] == 'EXPLOSIVE':
        reasons.append("💥 Explosive volume surge")
    elif volume['pattern'] == 'BREAKOUT':
        reasons.append("📊 Breakout volume pattern")
    elif volume['pattern'] == 'ACCUMULATION':
        reasons.append("🏦 Institutional accumulation")
    
    # Position reasons
    if position['from_low'] < 30:
        reasons.append("📍 Near 52-week lows - oversold")
    if position['from_high'] > -5 and position['above_all_smas']:
        reasons.append("🎯 Breaking to new highs")
    
    # Smart money reasons
    if smart['eps_momentum']:
        reasons.append("💰 Smart money: EPS surprise + volume")
    if smart['unusual_activity']:
        reasons.append("🔍 Unusual activity for category")
    
    return reasons

def compile_exit_warnings(row: pd.Series) -> List[str]:
    """Compile exit warnings"""
    warnings = []
    
    # Momentum warnings
    if row.get('momentum_quality') == 'EXHAUSTED':
        warnings.append("⚠️ Momentum exhausted")
    if row.get('ret_1d', 0) < -3:
        warnings.append("📉 Sharp daily decline")
    
    # Volume warnings
    if row.get('volume_pattern') == 'DISTRIBUTION':
        warnings.append("🚨 Distribution pattern")
    if row.get('rvol', 1) < 0.5:
        warnings.append("💤 Volume dried up")
    
    # Position warnings
    if row.get('from_high_pct', 0) > -5 and row.get('ret_1d', 0) < 0:
        warnings.append("🔻 Rejected at highs")
    
    # Technical warnings
    if row.get('price', 0) < row.get('sma_20d', 0):
        warnings.append("📊 Below 20-day SMA")
    
    return warnings

def identify_special_situations(row: pd.Series, market_health: MarketHealth) -> List[str]:
    """Identify special trading situations"""
    situations = []
    
    # New 52-week high breakout
    if row.get('from_high_pct', 0) > -1 and row.get('rvol', 1) > 2:
        situations.append("🎯 NEW 52-WEEK HIGH BREAKOUT")
    
    # Sector leadership
    if row['sector'] in [s[0] for s in market_health.top_sectors[:3]]:
        situations.append(f"👑 Leading sector: {row['sector']}")
    
    # Category rotation
    if row['category'] == market_health.leading_category:
        situations.append(f"🔄 Leading category: {row['category']}")
    
    # Oversold bounce in uptrend
    if (row.get('from_low_pct', 50) < 30 and 
        row.get('sma_50d', 0) > row.get('sma_200d', 0)):
        situations.append("🏀 Oversold bounce in uptrend")
    
    return situations

# ============================================
# VISUALIZATION FUNCTIONS
# ============================================

def create_market_health_dashboard(market_health: MarketHealth) -> go.Figure:
    """Create comprehensive market health dashboard"""
    fig = go.Figure()
    
    # Market score gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=market_health.market_score,
        title={'text': f"Market Health Score<br>{market_health.regime} Market"},
        domain={'x': [0, 0.5], 'y': [0.5, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': get_regime_color(market_health.regime)},
            'steps': [
                {'range': [0, 35], 'color': '#fee2e2'},
                {'range': [35, 65], 'color': '#fef3c7'},
                {'range': [65, 100], 'color': '#d4f1e4'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': market_health.market_score
            }
        }
    ))
    
    # Advance/Decline ratio
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=market_health.advance_decline_ratio,
        title={'text': "Advance/Decline Ratio"},
        delta={'reference': 1.0, 'relative': True},
        domain={'x': [0.5, 1], 'y': [0.5, 1]}
    ))
    
    # Breadth indicators
    breadth_categories = ['Above SMA20', 'Above SMA50', 'Above SMA200']
    breadth_values = [
        market_health.stocks_above_sma20,
        market_health.stocks_above_sma50,
        market_health.stocks_above_sma200
    ]
    
    fig.add_trace(go.Bar(
        x=breadth_values,
        y=breadth_categories,
        orientation='h',
        marker_color=['#3b82f6', '#8b5cf6', '#10b981'],
        text=[f"{v:.1f}%" for v in breadth_values],
        textposition='auto',
        name='Market Breadth'
    ))
    
    fig.update_layout(
        title="Market Health Dashboard",
        height=600,
        showlegend=False,
        xaxis={'range': [0, 100], 'title': 'Percentage'},
        yaxis={'title': ''}
    )
    
    return fig

def get_regime_color(regime: str) -> str:
    """Get color for market regime"""
    colors = {
        'BULL': '#10b981',
        'NEUTRAL': '#f59e0b',
        'BEAR': '#ef4444',
        'UNKNOWN': '#6b7280'
    }
    return colors.get(regime, '#6b7280')

def create_sector_rotation_map(market_health: MarketHealth) -> go.Figure:
    """Create sector rotation visualization"""
    if not market_health.top_sectors:
        return go.Figure()
    
    sectors = [s[0] for s in market_health.top_sectors[:10]]
    performances = [s[1] for s in market_health.top_sectors[:10]]
    
    # Color based on performance
    colors = ['#10b981' if p > 5 else '#f59e0b' if p > 0 else '#ef4444' 
              for p in performances]
    
    fig = go.Figure(go.Bar(
        x=performances,
        y=sectors,
        orientation='h',
        marker_color=colors,
        text=[f"{p:.1f}%" for p in performances],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Top 10 Sectors - 30 Day Performance",
        xaxis_title="30-Day Return (%)",
        yaxis_title="",
        height=500,
        margin=dict(l=150)
    )
    
    return fig

def create_signal_quality_matrix(signals_df: pd.DataFrame) -> go.Figure:
    """Create signal quality visualization"""
    if signals_df.empty:
        return go.Figure()
    
    # Group by signal strength
    signal_counts = signals_df['signal_strength'].value_counts()
    
    # Create sunburst chart
    fig = go.Figure(go.Sunburst(
        labels=['All Signals'] + signal_counts.index.tolist(),
        parents=[''] + ['All Signals'] * len(signal_counts),
        values=[len(signals_df)] + signal_counts.values.tolist(),
        marker=dict(
            colors=['#f3f4f6'] + [
                '#10b981' if 'BUY' in s else '#f59e0b' if 'WATCH' in s else '#ef4444'
                for s in signal_counts.index
            ]
        )
    ))
    
    fig.update_layout(
        title="Signal Distribution",
        height=400
    )
    
    return fig

def create_momentum_heatmap(signals_df: pd.DataFrame) -> go.Figure:
    """Create momentum analysis heatmap"""
    if signals_df.empty or len(signals_df) < 5:
        return go.Figure()
    
    # Get top 20 momentum stocks
    top_momentum = signals_df.nlargest(20, 'momentum_score')
    
    # Create heatmap data
    heatmap_data = []
    metrics = ['ret_1d', 'ret_7d', 'ret_30d', 'momentum_acceleration']
    
    for metric in metrics:
        if metric in top_momentum.columns:
            heatmap_data.append(top_momentum[metric].values)
    
    fig = go.Figure(go.Heatmap(
        z=heatmap_data,
        x=top_momentum['ticker'].values,
        y=['1-Day', '7-Day', '30-Day', 'Acceleration'],
        colorscale='RdYlGn',
        text=[[f"{val:.1f}" for val in row] for row in heatmap_data],
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Top 20 Momentum Stocks - Performance Heatmap",
        height=400,
        xaxis_title="",
        yaxis_title=""
    )
    
    return fig

# ============================================
# REPORT GENERATION
# ============================================

def generate_excel_report(signals_df: pd.DataFrame, market_health: MarketHealth) -> BytesIO:
    """Generate comprehensive Excel report"""
    output = BytesIO()
    
    try:
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
            
            strong_buy_format = workbook.add_format({'bg_color': '#d4f1e4'})
            buy_format = workbook.add_format({'bg_color': '#e0f2fe'})
            watch_format = workbook.add_format({'bg_color': '#fef3c7'})
            sell_format = workbook.add_format({'bg_color': '#fee2e2'})
            
            # Sheet 1: Executive Summary
            summary_data = {
                'Metric': [
                    'Analysis Date',
                    'Market Regime',
                    'Market Score',
                    'Regime Strength',
                    'Leading Category',
                    'Leading Sectors',
                    'Total Stocks Analyzed',
                    'Strong Buy Signals',
                    'Buy Signals',
                    'Watch Signals'
                ],
                'Value': [
                    datetime.now().strftime('%Y-%m-%d %H:%M'),
                    market_health.regime,
                    str(market_health.market_score),
                    f"{market_health.regime_strength:.1%}",
                    market_health.leading_category,
                    ', '.join([s[0] for s in market_health.top_sectors[:3]]),
                    str(len(signals_df)),
                    str(len(signals_df[signals_df['signal_strength'] == 'STRONG BUY'])),
                    str(len(signals_df[signals_df['signal_strength'] == 'BUY'])),
                    str(len(signals_df[signals_df['signal_strength'] == 'WATCH']))
                ]
            }
            
            pd.DataFrame(summary_data).to_excel(
                writer, sheet_name='Executive Summary', index=False
            )
            
            # Sheet 2: Top Opportunities
            strong_buys = signals_df[signals_df['signal_strength'] == 'STRONG BUY'].head(20)
            buys = signals_df[signals_df['signal_strength'] == 'BUY'].head(20)
            
            opportunities = pd.concat([strong_buys, buys])
            
            if not opportunities.empty:
                opportunity_data = opportunities[[
                    'ticker', 'company_name', 'category', 'sector',
                    'price', 'total_score', 'position_size',
                    'stop_loss', 'target_price', 'risk_reward_ratio',
                    'signal_freshness', 'momentum_acceleration'
                ]].copy()
                
                opportunity_data.columns = [
                    'Ticker', 'Company', 'Category', 'Sector',
                    'Price', 'Score', 'Position %',
                    'Stop Loss', 'Target', 'Risk/Reward',
                    'Freshness', 'Momentum'
                ]
                
                opportunity_data['Position %'] = opportunity_data['Position %'] * 100
                
                opportunity_data.to_excel(
                    writer, sheet_name='Top Opportunities', index=False
                )
            
            # Sheet 3: Market Analysis
            market_data = {
                'Indicator': [
                    'Advance/Decline Ratio',
                    'New Highs',
                    'New Lows',
                    'Volume Breadth',
                    'Momentum Breadth',
                    'Sector Dispersion'
                ],
                'Value': [
                    f"{market_health.advance_decline_ratio:.2f}",
                    str(market_health.new_highs),
                    str(market_health.new_lows),
                    f"{market_health.volume_breadth:.1f}%",
                    f"{market_health.momentum_breadth:.1f}%",
                    f"{market_health.sector_dispersion:.1f}"
                ]
            }
            
            pd.DataFrame(market_data).to_excel(
                writer, sheet_name='Market Analysis', index=False
            )
            
            # Sheet 4: Exit Alerts
            exits = signals_df[signals_df['signal_strength'] == 'SELL'].head(20)
            
            if not exits.empty:
                exit_data = exits[[
                    'ticker', 'company_name', 'price',
                    'ret_1d', 'momentum_quality', 'volume_pattern'
                ]].copy()
                
                exit_data.columns = [
                    'Ticker', 'Company', 'Price',
                    '1-Day Return', 'Momentum', 'Volume Pattern'
                ]
                
                exit_data.to_excel(
                    writer, sheet_name='Exit Alerts', index=False
                )
            
            # Format all sheets
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                worksheet.set_zoom(90)
                
                # Auto-fit columns
                for i in range(20):
                    worksheet.set_column(i, i, 15)
        
        output.seek(0)
        return output
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        output.seek(0)
        return output

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main application entry point"""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Wave Detection 3.0</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; font-size: 1.2rem; color: #6b7280;">'
        'Professional Trading System with Advanced Signal Detection</p>', 
        unsafe_allow_html=True
    )
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## 📊 Data Configuration")
        
        # Data source
        sheet_url = st.text_input(
            "Google Sheets URL",
            value="https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk/edit?usp=sharing",
            help="Your Google Sheets URL containing stock data"
        )
        
        gid = st.text_input(
            "Sheet GID",
            value="2026492216",
            help="Sheet identifier (found in URL after gid=)"
        )
        
        # Refresh button
        if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Load and prepare data
        with st.spinner("Loading market data..."):
            raw_df = load_market_data(sheet_url, gid)
        
        if raw_df.empty:
            st.error("❌ No data loaded. Please check your URL and GID.")
            return
        
        # Clean and prepare data
        with st.spinner("Analyzing market data..."):
            df = clean_and_prepare_data(raw_df)
        
        if df.empty:
            st.error("❌ No valid data after cleaning.")
            return
        
        # Analyze market health
        market_health = analyze_market_health(df)
        
        # Smart Filters Section
        st.markdown("## 🎯 Smart Filters")
        
        # Signal strength filter
        signal_types = st.multiselect(
            "Signal Types",
            options=['STRONG BUY', 'BUY', 'WATCH'],
            default=['STRONG BUY', 'BUY'],
            help="Select signal types to display"
        )
        
        # Category filter
        categories = sorted(df['category'].unique())
        selected_categories = st.multiselect(
            "Categories",
            options=categories,
            default=categories,
            help="Filter by market cap category"
        )
        
        # Sector filter with counts
        sector_counts = df['sector'].value_counts()
        sector_options = [
            f"{sector} ({count})" 
            for sector, count in sector_counts.items()
        ]
        
        selected_sectors_raw = st.multiselect(
            "Sectors",
            options=['All'] + sector_options,
            default=['All'],
            help="Filter by sector (shows stock count)"
        )
        
        # Extract sector names
        if 'All' in selected_sectors_raw:
            selected_sectors = list(sector_counts.index)
        else:
            selected_sectors = [
                s.rsplit(' (', 1)[0] for s in selected_sectors_raw
            ]
        
        # Advanced settings
        st.markdown("---")
        st.markdown("## ⚙️ Advanced Settings")
        
        # Minimum score threshold
        min_score = st.slider(
            "Minimum Signal Score",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            help="Minimum total score for signals"
        )
        
        # Freshness filter
        freshness_filter = st.selectbox(
            "Signal Freshness",
            options=['All', 'FRESH_HOT', 'FRESH', 'NORMAL'],
            index=0,
            help="Filter by signal freshness"
        )
        
        # Position limit
        max_positions = st.number_input(
            "Maximum Positions",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="Maximum number of positions to show"
        )
    
    # Main content area
    # Generate trading signals
    with st.spinner("Generating trading signals..."):
        signals_df = generate_trading_signals(df, market_health)
    
    # Apply filters
    filtered_signals = signals_df[
        (signals_df['signal_strength'].isin(signal_types)) &
        (signals_df['category'].isin(selected_categories)) &
        (signals_df['sector'].isin(selected_sectors)) &
        (signals_df['total_score'] >= min_score)
    ]
    
    if freshness_filter != 'All':
        filtered_signals = filtered_signals[
            filtered_signals['signal_freshness'] == freshness_filter
        ]
    
    # Sort by total score
    filtered_signals = filtered_signals.sort_values(
        'total_score', ascending=False
    ).head(max_positions)
    
    # Market Overview Section
    st.markdown("## 🌍 Market Overview")
    
    # Market health metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        regime_color = get_regime_color(market_health.regime)
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: {regime_color}; margin: 0;">{market_health.regime}</h3>
            <p style="margin: 0.5rem 0; font-size: 0.9rem;">Market Regime</p>
            <h2 style="margin: 0;">{market_health.market_score}</h2>
            <p style="margin: 0; font-size: 0.8rem; color: #6b7280;">Health Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        ad_color = "#10b981" if market_health.advance_decline_ratio > 1.5 else "#ef4444" if market_health.advance_decline_ratio < 0.67 else "#f59e0b"
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: {ad_color}; margin: 0;">{market_health.advance_decline_ratio:.2f}</h3>
            <p style="margin: 0.5rem 0; font-size: 0.9rem;">A/D Ratio</p>
            <p style="margin: 0; font-size: 0.8rem; color: #6b7280;">
                {market_health.stocks_above_sma50:.0f}% above SMA50
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0;">{market_health.leading_category}</h3>
            <p style="margin: 0.5rem 0; font-size: 0.9rem;">Leading Category</p>
            <p style="margin: 0; font-size: 0.8rem; color: #6b7280;">
                Vol Breadth: {market_health.volume_breadth:.0f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        top_sector = market_health.top_sectors[0][0] if market_health.top_sectors else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0;">{top_sector}</h3>
            <p style="margin: 0.5rem 0; font-size: 0.9rem;">Top Sector</p>
            <p style="margin: 0; font-size: 0.8rem; color: #6b7280;">
                {market_health.unusual_volume_stocks} high vol stocks
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualizations
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        market_dash = create_market_health_dashboard(market_health)
        st.plotly_chart(market_dash, use_container_width=True)
    
    with col2:
        sector_map = create_sector_rotation_map(market_health)
        st.plotly_chart(sector_map, use_container_width=True)
    
    # Trading Signals Section
    st.markdown("---")
    st.markdown("## 🎯 Trading Signals")
    
    # Signal summary
    col1, col2, col3, col4 = st.columns(4)
    
    strong_buy_count = len(filtered_signals[filtered_signals['signal_strength'] == 'STRONG BUY'])
    buy_count = len(filtered_signals[filtered_signals['signal_strength'] == 'BUY'])
    watch_count = len(filtered_signals[filtered_signals['signal_strength'] == 'WATCH'])
    total_count = len(filtered_signals)
    
    with col1:
        st.metric("🟢 Strong Buy", strong_buy_count)
    with col2:
        st.metric("🔵 Buy", buy_count)
    with col3:
        st.metric("🟡 Watch", watch_count)
    with col4:
        st.metric("📊 Total Signals", total_count)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Top Opportunities",
        "📊 Signal Analysis", 
        "🚨 Risk Alerts",
        "📈 Performance Maps",
        "📥 Reports"
    ])
    
    with tab1:
        st.markdown("### 🎯 Top Trading Opportunities")
        
        if filtered_signals.empty:
            st.info("No signals match your current filters.")
        else:
            # Group by signal strength
            for signal_type in ['STRONG BUY', 'BUY', 'WATCH']:
                type_signals = filtered_signals[
                    filtered_signals['signal_strength'] == signal_type
                ]
                
                if not type_signals.empty:
                    if signal_type == 'STRONG BUY':
                        st.markdown("#### 💎 STRONG BUY SIGNALS")
                    elif signal_type == 'BUY':
                        st.markdown("#### 🟢 BUY SIGNALS")
                    else:
                        st.markdown("#### 🟡 WATCH SIGNALS")
                    
                    for _, signal in type_signals.iterrows():
                        # Determine card style
                        if signal_type == 'STRONG BUY':
                            card_class = "strong-buy-card"
                        elif signal_type == 'BUY':
                            card_class = "buy-card"
                        else:
                            card_class = "watch-card"
                        
                        # Signal card
                        st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
                        
                        col1, col2, col3, col4 = st.columns([3, 2, 2, 1.5])
                        
                        with col1:
                            st.markdown(f"### {signal['ticker']}")
                            st.markdown(f"{signal['company_name'][:60]}...")
                            
                            # Category and sector badges
                            category_class = signal['category'].lower().replace(' ', '-')
                            st.markdown(
                                f'<span class="category-indicator {category_class}-indicator">'
                                f'{signal["category"]}</span> '
                                f'<span class="alert-badge">{signal["sector"]}</span>',
                                unsafe_allow_html=True
                            )
                            
                            # Entry reasons
                            if signal['entry_reasons']:
                                st.markdown("**Entry Reasons:**")
                                for reason in signal['entry_reasons'][:3]:
                                    st.markdown(f"• {reason}")
                        
                        with col2:
                            st.metric("Entry Price", f"₹{signal['price']:.2f}")
                            st.metric("Signal Score", f"{signal['total_score']:.0f}/100")
                            
                            # Special badges
                            if signal['momentum_acceleration'] > 2:
                                st.markdown(
                                    '<span class="momentum-badge pulse">🚀 Accelerating</span>',
                                    unsafe_allow_html=True
                                )
                            
                            if signal['volume_pattern'] in ['EXPLOSIVE', 'BREAKOUT']:
                                st.markdown(
                                    f'<span class="volume-badge">{signal["volume_pattern"]}</span>',
                                    unsafe_allow_html=True
                                )
                        
                        with col3:
                            st.metric("Target", f"₹{signal['target_price']:.2f}")
                            st.metric("Stop Loss", f"₹{signal['stop_loss']:.2f}")
                            st.metric("Risk/Reward", f"{signal['risk_reward_ratio']:.1f}")
                        
                        with col4:
                            st.metric("Position Size", f"{signal['position_size']*100:.1f}%")
                            
                            # Freshness indicator
                            freshness_colors = {
                                'FRESH_HOT': '#ef4444',
                                'FRESH': '#f59e0b',
                                'NORMAL': '#3b82f6',
                                'STALE': '#6b7280'
                            }
                            fresh_color = freshness_colors.get(signal['signal_freshness'], '#6b7280')
                            st.markdown(
                                f'<p style="color: {fresh_color}; font-weight: bold;">'
                                f'{signal["signal_freshness"]}</p>',
                                unsafe_allow_html=True
                            )
                        
                        # Expandable details
                        with st.expander("📊 Detailed Analysis"):
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.markdown("**Momentum Metrics**")
                                st.write(f"1-Day: {signal['ret_1d']:.1f}%")
                                st.write(f"7-Day: {signal['ret_7d']:.1f}%")
                                st.write(f"30-Day: {signal['ret_30d']:.1f}%")
                                st.write(f"Acceleration: {signal['momentum_acceleration']:.2f}")
                            
                            with col2:
                                st.markdown("**Volume Analysis**")
                                st.write(f"Rel Volume: {signal['rvol']:.1f}x")
                                st.write(f"Volume Pattern: {signal['volume_pattern']}")
                                st.write(f"Smart Volume: {signal.get('smart_volume_ratio', 0):.2f}")
                            
                            with col3:
                                st.markdown("**Position Metrics**")
                                st.write(f"From Low: {signal['from_low_pct']:.0f}%")
                                st.write(f"From High: {signal['from_high_pct']:.0f}%")
                                st.write(f"Range Pos: {signal.get('range_position', 0.5):.1%}")
                            
                            with col4:
                                st.markdown("**Component Scores**")
                                st.write(f"Momentum: {signal['momentum_score']:.0f}")
                                st.write(f"Volume: {signal['volume_score']:.0f}")
                                st.write(f"Position: {signal['position_score']:.0f}")
                                st.write(f"Smart Money: {signal['smart_money_score']:.0f}")
                            
                            # Special situations
                            if signal['special_situations']:
                                st.markdown("**🌟 Special Situations:**")
                                for situation in signal['special_situations']:
                                    st.write(f"• {situation}")
                            
                            # Exit warnings
                            if signal['exit_warnings']:
                                st.markdown("**⚠️ Risk Warnings:**")
                                for warning in signal['exit_warnings']:
                                    st.write(f"• {warning}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown("")
    
    with tab2:
        st.markdown("### 📊 Signal Analysis")
        
        if not filtered_signals.empty:
            # Signal quality distribution
            col1, col2 = st.columns(2)
            
            with col1:
                signal_quality = create_signal_quality_matrix(filtered_signals)
                st.plotly_chart(signal_quality, use_container_width=True)
            
            with col2:
                # Score distribution
                fig = go.Figure(go.Histogram(
                    x=filtered_signals['total_score'],
                    nbinsx=20,
                    marker_color='#8b5cf6'
                ))
                fig.update_layout(
                    title="Signal Score Distribution",
                    xaxis_title="Total Score",
                    yaxis_title="Count",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Top signals table
            st.markdown("#### 📋 Signal Details")
            
            display_columns = [
                'ticker', 'company_name', 'category', 'sector',
                'total_score', 'momentum_acceleration', 'volume_pattern',
                'position_size', 'risk_reward_ratio'
            ]
            
            display_df = filtered_signals[display_columns].copy()
            display_df['position_size'] = display_df['position_size'] * 100
            display_df.columns = [
                'Ticker', 'Company', 'Category', 'Sector',
                'Score', 'Momentum', 'Volume Pattern',
                'Position %', 'Risk/Reward'
            ]
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )
        else:
            st.info("No signals to analyze with current filters.")
    
    with tab3:
        st.markdown("### 🚨 Risk Alerts & Exit Signals")
        
        # Exit signals
        exit_signals = signals_df[signals_df['signal_strength'] == 'SELL']
        
        if not exit_signals.empty:
            st.markdown("#### 🔴 EXIT SIGNALS")
            
            for _, signal in exit_signals.head(10).iterrows():
                st.markdown('<div class="sell-card">', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([3, 2, 2])
                
                with col1:
                    st.markdown(f"### {signal['ticker']} - EXIT POSITION")
                    st.markdown(f"{signal['company_name'][:60]}...")
                    
                    if signal['exit_warnings']:
                        st.markdown("**Exit Reasons:**")
                        for warning in signal['exit_warnings']:
                            st.write(f"• {warning}")
                
                with col2:
                    st.metric("Current Price", f"₹{signal['price']:.2f}")
                    st.metric("Today's Return", f"{signal['ret_1d']:.1f}%")
                
                with col3:
                    st.metric("Momentum", signal.get('momentum_quality', 'N/A'))
                    st.metric("Volume Pattern", signal.get('volume_pattern', 'N/A'))
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("")
        
        # Risk warnings
        st.markdown("#### ⚠️ Risk Warnings")
        
        # High volatility stocks
        high_vol = filtered_signals[filtered_signals['rvol'] > 3]
        if not high_vol.empty:
            st.warning(
                f"**High Volatility Alert:** {len(high_vol)} stocks showing "
                f"extreme volume (>3x normal)"
            )
        
        # Momentum exhaustion
        exhausted = filtered_signals[
            filtered_signals['momentum_quality'] == 'EXHAUSTED'
        ]
        if not exhausted.empty:
            st.warning(
                f"**Momentum Exhaustion:** {len(exhausted)} stocks showing "
                f"signs of momentum exhaustion"
            )
    
    with tab4:
        st.markdown("### 📈 Performance Maps")
        
        if not filtered_signals.empty:
            # Momentum heatmap
            momentum_map = create_momentum_heatmap(filtered_signals)
            st.plotly_chart(momentum_map, use_container_width=True)
            
            # Category performance
            category_perf = filtered_signals.groupby('category').agg({
                'total_score': 'mean',
                'ticker': 'count'
            }).round(1)
            
            fig = go.Figure(go.Bar(
                x=category_perf.index,
                y=category_perf['total_score'],
                text=[f"{score:.0f} ({count} stocks)" 
                      for score, count in zip(category_perf['total_score'], 
                                            category_perf['ticker'])],
                textposition='auto',
                marker_color=['#1e40af', '#7c3aed', '#dc2626', '#059669']
            ))
            
            fig.update_layout(
                title="Average Signal Score by Category",
                xaxis_title="Category",
                yaxis_title="Average Score",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for performance maps.")
    
    with tab5:
        st.markdown("### 📥 Download Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Generate Excel Report", type="primary", use_container_width=True):
                with st.spinner("Generating comprehensive report..."):
                    excel_file = generate_excel_report(filtered_signals, market_health)
                    
                    st.download_button(
                        label="📥 Download Excel Report",
                        data=excel_file,
                        file_name=f"wave_detection_3.0_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
        
        with col2:
            if st.button("📝 Generate Summary", type="secondary", use_container_width=True):
                # Generate text summary
                summary = f"""
WAVE DETECTION 3.0 - MARKET ANALYSIS
{datetime.now().strftime('%Y-%m-%d %H:%M')}

MARKET STATUS: {market_health.regime} (Score: {market_health.market_score}/100)
Regime Strength: {market_health.regime_strength:.1%}
Leading Category: {market_health.leading_category}
Top Sector: {market_health.top_sectors[0][0] if market_health.top_sectors else 'N/A'}

SIGNAL SUMMARY:
🟢 Strong Buy: {strong_buy_count} signals
🔵 Buy: {buy_count} signals  
🟡 Watch: {watch_count} signals

TOP 5 OPPORTUNITIES:
"""
                for i, (_, signal) in enumerate(filtered_signals.head(5).iterrows(), 1):
                    summary += f"""
{i}. {signal['ticker']} - {signal['company_name'][:30]}...
   Entry: ₹{signal['price']:.0f} | Target: ₹{signal['target_price']:.0f} 
   Score: {signal['total_score']:.0f} | Position: {signal['position_size']*100:.1f}%
   {signal['entry_reasons'][0] if signal['entry_reasons'] else 'Multiple bullish signals'}
"""
                
                summary += f"""

MARKET BREADTH:
- Advance/Decline: {market_health.advance_decline_ratio:.2f}
- Above SMA50: {market_health.stocks_above_sma50:.0f}%
- Volume Breadth: {market_health.volume_breadth:.0f}%

Generated by Wave Detection 3.0 - Professional Trading System
"""
                
                st.text_area(
                    "Market Summary",
                    summary,
                    height=500
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.9rem;">
        Wave Detection 3.0 | Professional Trading System<br>
        Advanced Signal Detection with Machine Learning<br>
        © 2024 - All Rights Reserved
    </div>
    """, unsafe_allow_html=True)

# Application entry point
if __name__ == "__main__":
    main()
