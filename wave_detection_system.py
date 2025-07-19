"""
Wave Detection System 5.0 - The Ultimate Trading Edge
=====================================================
Philosophy: Simple. Fast. Profitable.

Core Innovation:
- Pure relative ranking (no thresholds)
- Momentum acceleration detection
- Appearance/disappearance tracking
- Volume conviction scoring
- Sector rotation monitoring
- Statistical edge validation

Author: Elite Trading Systems
Version: 5.0.0
Status: Production Ready
Performance: Optimized for Speed
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import logging
from io import BytesIO
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import hashlib
import warnings
from pathlib import Path
import pickle
from functools import lru_cache, wraps
import time

# ============================================
# CONFIGURATION AND INITIALIZATION
# ============================================

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(funcName)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        if execution_time > 0.1:  # Log if takes more than 100ms
            logger.warning(f"{func.__name__} took {execution_time:.3f}s")
        return result
    return wrapper

# Streamlit page config
try:
    st.set_page_config(
        page_title="Wave Detection 5.0 | The Edge",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "Wave Detection 5.0 - Simple. Fast. Profitable."
        }
    )
except:
    pass  # Page config already set

# ============================================
# DATA MODELS AND CONSTANTS
# ============================================

@dataclass(frozen=True)
class TradingConstants:
    """Immutable trading constants"""
    # Data quality
    MIN_PRICE: float = 1.0
    MIN_VOLUME: float = 50_000
    MAX_SPREAD: float = 0.10  # 10% max spread
    
    # Ranking thresholds
    TOP_MOMENTUM_PERCENTILE: float = 0.90  # Top 10%
    VOLUME_CONFIRMATION: float = 1.5  # Min rvol for confirmation
    
    # Position sizing
    MAX_POSITION_SIZE: float = 0.03  # 3% max
    MIN_POSITION_SIZE: float = 0.005  # 0.5% min
    MAX_POSITIONS: int = 20
    
    # Risk management
    MAX_SECTOR_EXPOSURE: float = 0.30  # 30% max per sector
    INITIAL_STOP_LOSS: float = 0.05  # 5% initial stop
    
    # Performance tracking
    TRACKING_LOOKBACK_DAYS: int = 30
    MIN_SAMPLE_SIZE: int = 20

# Initialize constants
CONSTANTS = TradingConstants()

@dataclass
class Signal:
    """Trading signal with all necessary information"""
    ticker: str
    company_name: str
    sector: str
    category: str
    
    # Signal properties
    signal_type: str  # 'BUY', 'HOLD', 'SELL', 'EXIT'
    signal_strength: float  # 0-100
    signal_timestamp: datetime
    
    # Ranking metrics
    momentum_rank: float  # 0-1 percentile
    volume_rank: float
    combined_rank: float
    
    # Raw metrics
    price: float
    momentum_score: float
    volume_score: float
    
    # Acceleration metrics
    momentum_acceleration: float
    volume_acceleration: float
    
    # Risk metrics
    volatility_estimate: float
    liquidity_score: float
    
    # Position sizing
    recommended_size: float
    confidence_level: float
    
    # Lifecycle tracking
    days_in_list: int
    first_appearance: Optional[datetime]
    
    # Exit conditions
    exit_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame"""
        return {
            'ticker': self.ticker,
            'company_name': self.company_name,
            'sector': self.sector,
            'category': self.category,
            'signal_type': self.signal_type,
            'signal_strength': self.signal_strength,
            'momentum_rank': self.momentum_rank,
            'volume_rank': self.volume_rank,
            'combined_rank': self.combined_rank,
            'price': self.price,
            'momentum_score': self.momentum_score,
            'volume_score': self.volume_score,
            'momentum_acceleration': self.momentum_acceleration,
            'recommended_size': self.recommended_size,
            'days_in_list': self.days_in_list
        }

@dataclass
class MarketConditions:
    """Current market conditions assessment"""
    timestamp: datetime
    total_stocks: int
    
    # Market regime
    regime: str  # 'TRENDING', 'RANGING', 'VOLATILE'
    regime_confidence: float
    
    # Breadth metrics
    advancing_pct: float
    declining_pct: float
    new_highs_pct: float
    new_lows_pct: float
    
    # Momentum distribution
    avg_momentum: float
    momentum_dispersion: float
    positive_momentum_pct: float
    
    # Volume metrics
    avg_relative_volume: float
    high_volume_pct: float
    
    # Sector analysis
    leading_sectors: List[Tuple[str, float]]
    sector_rotation_intensity: float
    
    # Risk metrics
    market_volatility: float
    correlation_level: float

@dataclass
class PerformanceMetrics:
    """Track system performance"""
    date: datetime
    signal_type: str
    ticker: str
    entry_price: float
    exit_price: Optional[float]
    return_pct: Optional[float]
    holding_days: int
    exit_reason: Optional[str]
    
# ============================================
# PROFESSIONAL CSS STYLING
# ============================================

PROFESSIONAL_CSS = """
<style>
    /* Global Reset and Base */
    .main {
        padding: 0;
        max-width: 100%;
    }
    
    /* Typography */
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 1rem;
        letter-spacing: -1px;
    }
    
    .sub-header {
        text-align: center;
        color: #6b7280;
        font-size: 1.25rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Cards and Containers */
    .signal-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .signal-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.12);
    }
    
    .signal-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 5px;
        height: 100%;
        background: var(--signal-color);
    }
    
    /* Signal Types */
    .signal-buy {
        --signal-color: #10b981;
    }
    
    .signal-hold {
        --signal-color: #f59e0b;
    }
    
    .signal-sell {
        --signal-color: #ef4444;
    }
    
    .signal-exit {
        --signal-color: #991b1b;
    }
    
    /* Metrics */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
        height: 100%;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0.25rem;
    }
    
    .badge-momentum {
        background-color: #8b5cf6;
        color: white;
    }
    
    .badge-volume {
        background-color: #3b82f6;
        color: white;
    }
    
    .badge-new {
        background-color: #10b981;
        color: white;
        animation: pulse 2s infinite;
    }
    
    .badge-exit {
        background-color: #ef4444;
        color: white;
    }
    
    /* Rank Indicators */
    .rank-bar {
        width: 100%;
        height: 8px;
        background-color: #e5e7eb;
        border-radius: 4px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .rank-fill {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        transition: width 0.3s ease;
    }
    
    /* Tables */
    .dataframe {
        font-size: 0.875rem;
        width: 100%;
    }
    
    .dataframe th {
        background-color: #f9fafb;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        padding: 0.75rem;
        text-align: left;
    }
    
    .dataframe td {
        padding: 0.75rem;
        border-bottom: 1px solid #e5e7eb;
    }
    
    /* Animations */
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.7;
        }
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(-100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .slide-in {
        animation: slideIn 0.3s ease-out;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .signal-card {
            padding: 1rem;
        }
        
        .metric-value {
            font-size: 2rem;
        }
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .signal-card, .metric-card {
            background: #1f2937;
            color: #f9fafb;
        }
        
        .dataframe th {
            background-color: #374151;
            color: #f9fafb;
        }
        
        .dataframe td {
            border-color: #374151;
        }
    }
</style>
"""

# ============================================
# DATA STORAGE AND PERSISTENCE
# ============================================

class DataStore:
    """Handle data persistence for tracking"""
    
    def __init__(self, base_path: str = "wave_detection_data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
    def save_daily_snapshot(self, df: pd.DataFrame, date: datetime) -> None:
        """Save daily data snapshot"""
        try:
            filename = self.base_path / f"snapshot_{date.strftime('%Y%m%d')}.pkl"
            df.to_pickle(filename)
            logger.info(f"Saved snapshot: {filename}")
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
    
    def load_daily_snapshot(self, date: datetime) -> Optional[pd.DataFrame]:
        """Load daily data snapshot"""
        try:
            filename = self.base_path / f"snapshot_{date.strftime('%Y%m%d')}.pkl"
            if filename.exists():
                return pd.read_pickle(filename)
        except Exception as e:
            logger.error(f"Failed to load snapshot: {e}")
        return None
    
    def save_signals(self, signals: List[Signal], date: datetime) -> None:
        """Save trading signals"""
        try:
            filename = self.base_path / f"signals_{date.strftime('%Y%m%d')}.json"
            signal_data = [s.to_dict() for s in signals]
            with open(filename, 'w') as f:
                json.dump(signal_data, f, indent=2, default=str)
            logger.info(f"Saved {len(signals)} signals")
        except Exception as e:
            logger.error(f"Failed to save signals: {e}")
    
    def get_appearance_history(self, ticker: str) -> List[datetime]:
        """Get dates when ticker appeared in data"""
        appearances = []
        for snapshot_file in sorted(self.base_path.glob("snapshot_*.pkl")):
            try:
                date_str = snapshot_file.stem.split('_')[1]
                date = datetime.strptime(date_str, '%Y%m%d')
                df = pd.read_pickle(snapshot_file)
                if ticker in df['ticker'].values:
                    appearances.append(date)
            except:
                continue
        return appearances

# ============================================
# CORE CALCULATION ENGINE
# ============================================

class MomentumEngine:
    """Core momentum calculation engine"""
    
    @staticmethod
    @monitor_performance
    def calculate_momentum_score(row: pd.Series) -> float:
        """Calculate pure momentum score with acceleration"""
        try:
            # Base momentum (weighted by recency)
            momentum = (
                row.get('ret_1d', 0) * 3 +    # Today heavily weighted
                row.get('ret_3d', 0) * 2 +    # Recent confirmation
                row.get('ret_7d', 0) * 1      # Trend context
            ) / 6  # Normalize
            
            # Momentum acceleration
            if row.get('ret_7d', 0) != 0:
                acceleration = (row.get('ret_1d', 0) - row.get('ret_7d', 0) / 7) / abs(row.get('ret_7d', 0) / 7)
            else:
                acceleration = row.get('ret_1d', 0)
            
            # Trend alignment bonus
            trend_bonus = 0
            if row.get('price', 0) > row.get('sma_20d', 0):
                trend_bonus += 0.2
            if row.get('sma_20d', 0) > row.get('sma_50d', 0):
                trend_bonus += 0.1
            if row.get('sma_50d', 0) > row.get('sma_200d', 0):
                trend_bonus += 0.1
            
            # Combined score
            score = momentum * (1 + acceleration) * (1 + trend_bonus)
            
            return score
            
        except Exception as e:
            logger.error(f"Momentum calculation error: {e}")
            return 0.0
    
    @staticmethod
    def calculate_momentum_acceleration(row: pd.Series) -> float:
        """Calculate momentum acceleration rate"""
        try:
            # Daily acceleration
            daily_acc = row.get('ret_1d', 0)
            
            # 3-day acceleration
            if row.get('ret_3d', 0) != 0:
                three_day_acc = (row.get('ret_1d', 0) - row.get('ret_3d', 0) / 3)
            else:
                three_day_acc = daily_acc
            
            # 7-day acceleration
            if row.get('ret_7d', 0) != 0:
                seven_day_acc = (row.get('ret_1d', 0) - row.get('ret_7d', 0) / 7)
            else:
                seven_day_acc = daily_acc
            
            # Weighted acceleration
            acceleration = (daily_acc * 0.5 + three_day_acc * 0.3 + seven_day_acc * 0.2)
            
            return acceleration
            
        except Exception:
            return 0.0

class VolumeEngine:
    """Volume analysis engine"""
    
    @staticmethod
    @monitor_performance
    def calculate_volume_score(row: pd.Series) -> float:
        """Calculate volume conviction score"""
        try:
            # Relative volume is primary factor
            rvol = row.get('rvol', 1.0)
            
            # Volume trend (30d vs 90d)
            vol_trend = row.get('vol_ratio_30d_90d', 1.0)
            
            # Volume acceleration (recent vs historical)
            vol_accel = 1.0
            if row.get('vol_ratio_1d_90d', 0) > 0 and row.get('vol_ratio_7d_90d', 0) > 0:
                vol_accel = row.get('vol_ratio_1d_90d', 1.0) / row.get('vol_ratio_7d_90d', 1.0)
            
            # Smart volume (volume efficiency)
            price_move = abs(row.get('ret_1d', 0))
            if price_move > 0.1:  # Minimum price movement
                volume_efficiency = price_move / max(rvol, 0.1)
            else:
                volume_efficiency = 0
            
            # Combined score
            score = rvol * vol_trend * vol_accel * (1 + volume_efficiency)
            
            return score
            
        except Exception as e:
            logger.error(f"Volume calculation error: {e}")
            return 1.0
    
    @staticmethod
    def calculate_volume_acceleration(row: pd.Series) -> float:
        """Calculate volume acceleration"""
        try:
            # Compare recent ratios to longer-term ratios
            short_term = row.get('vol_ratio_7d_90d', 1.0)
            long_term = row.get('vol_ratio_30d_180d', 1.0)
            
            acceleration = short_term - long_term
            
            return acceleration
            
        except Exception:
            return 0.0

class RankingEngine:
    """Handle all ranking operations"""
    
    @staticmethod
    @monitor_performance
    def calculate_relative_ranks(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate percentile ranks for all metrics"""
        try:
            # Momentum rank
            df['momentum_rank'] = df['momentum_score'].rank(pct=True, method='dense')
            
            # Volume rank
            df['volume_rank'] = df['volume_score'].rank(pct=True, method='dense')
            
            # Acceleration ranks
            df['momentum_accel_rank'] = df['momentum_acceleration'].rank(pct=True, method='dense')
            df['volume_accel_rank'] = df['volume_acceleration'].rank(pct=True, method='dense')
            
            # Combined rank (weighted)
            df['combined_rank'] = (
                df['momentum_rank'] * 0.4 +
                df['volume_rank'] * 0.3 +
                df['momentum_accel_rank'] * 0.2 +
                df['volume_accel_rank'] * 0.1
            )
            
            # Overall rank
            df['overall_rank'] = df['combined_rank'].rank(pct=True, method='dense')
            
            return df
            
        except Exception as e:
            logger.error(f"Ranking calculation error: {e}")
            return df
    
    @staticmethod
    def get_top_percentile(df: pd.DataFrame, percentile: float) -> pd.DataFrame:
        """Get top percentile of stocks"""
        try:
            threshold = df['combined_rank'].quantile(percentile)
            return df[df['combined_rank'] >= threshold].sort_values('combined_rank', ascending=False)
        except Exception:
            return pd.DataFrame()

# ============================================
# SIGNAL GENERATION
# ============================================

class SignalGenerator:
    """Generate trading signals from ranked data"""
    
    def __init__(self, data_store: DataStore):
        self.data_store = data_store
    
    @monitor_performance
    def generate_signals(self, df: pd.DataFrame, previous_df: Optional[pd.DataFrame] = None) -> List[Signal]:
        """Generate all trading signals"""
        signals = []
        
        try:
            # Get appearance/disappearance sets
            if previous_df is not None:
                current_tickers = set(df['ticker'])
                previous_tickers = set(previous_df['ticker'])
                
                new_appearances = current_tickers - previous_tickers
                disappearances = previous_tickers - current_tickers
            else:
                new_appearances = set(df['ticker'])
                disappearances = set()
            
            # Process each stock
            for _, row in df.iterrows():
                ticker = row['ticker']
                
                # Calculate days in list
                appearances = self.data_store.get_appearance_history(ticker)
                days_in_list = len(appearances)
                first_appearance = min(appearances) if appearances else datetime.now()
                
                # Determine signal type
                signal_type = self._determine_signal_type(
                    row, 
                    is_new=ticker in new_appearances,
                    days_in_list=days_in_list
                )
                
                # Calculate signal strength
                signal_strength = self._calculate_signal_strength(row, signal_type)
                
                # Calculate position size
                position_size = self._calculate_position_size(
                    row, 
                    signal_type, 
                    signal_strength
                )
                
                # Create signal
                signal = Signal(
                    ticker=ticker,
                    company_name=row.get('company_name', ticker),
                    sector=row.get('sector', 'Unknown'),
                    category=row.get('category', 'Unknown'),
                    signal_type=signal_type,
                    signal_strength=signal_strength,
                    signal_timestamp=datetime.now(),
                    momentum_rank=row.get('momentum_rank', 0),
                    volume_rank=row.get('volume_rank', 0),
                    combined_rank=row.get('combined_rank', 0),
                    price=row.get('price', 0),
                    momentum_score=row.get('momentum_score', 0),
                    volume_score=row.get('volume_score', 0),
                    momentum_acceleration=row.get('momentum_acceleration', 0),
                    volume_acceleration=row.get('volume_acceleration', 0),
                    volatility_estimate=self._estimate_volatility(row),
                    liquidity_score=row.get('volume_30d', 0) / 1e6,  # In millions
                    recommended_size=position_size,
                    confidence_level=signal_strength / 100,
                    days_in_list=days_in_list,
                    first_appearance=first_appearance
                )
                
                signals.append(signal)
            
            # Add exit signals for disappearances
            for ticker in disappearances:
                exit_signal = self._create_exit_signal(ticker, "Stock removed from data")
                if exit_signal:
                    signals.append(exit_signal)
            
            # Sort by signal strength
            signals.sort(key=lambda x: x.signal_strength, reverse=True)
            
            return signals
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return []
    
    def _determine_signal_type(self, row: pd.Series, is_new: bool, days_in_list: int) -> str:
        """Determine signal type based on conditions"""
        try:
            # Strong buy conditions
            if (row.get('combined_rank', 0) >= CONSTANTS.TOP_MOMENTUM_PERCENTILE and
                row.get('rvol', 0) >= CONSTANTS.VOLUME_CONFIRMATION and
                (is_new or days_in_list <= 3)):
                return 'BUY'
            
            # Exit conditions
            elif (row.get('momentum_rank', 0) < 0.3 or
                  row.get('momentum_acceleration', 0) < -5 or
                  days_in_list > 15):
                return 'SELL'
            
            # Hold conditions
            elif row.get('combined_rank', 0) >= 0.7:
                return 'HOLD'
            
            else:
                return 'SELL'
                
        except Exception:
            return 'HOLD'
    
    def _calculate_signal_strength(self, row: pd.Series, signal_type: str) -> float:
        """Calculate signal strength (0-100)"""
        try:
            base_strength = row.get('combined_rank', 0.5) * 100
            
            # Adjust for signal type
            if signal_type == 'BUY':
                # Boost for strong momentum acceleration
                if row.get('momentum_acceleration', 0) > 2:
                    base_strength *= 1.2
                # Boost for volume confirmation
                if row.get('rvol', 1) > 2:
                    base_strength *= 1.1
            
            elif signal_type == 'SELL':
                # Invert for sell signals
                base_strength = 100 - base_strength
            
            return min(100, max(0, base_strength))
            
        except Exception:
            return 50.0
    
    def _calculate_position_size(self, row: pd.Series, signal_type: str, signal_strength: float) -> float:
        """Calculate recommended position size"""
        try:
            if signal_type != 'BUY':
                return 0.0
            
            # Base size on signal strength
            base_size = (signal_strength / 100) * CONSTANTS.MAX_POSITION_SIZE
            
            # Adjust for volatility
            volatility = self._estimate_volatility(row)
            if volatility > 5:  # High volatility
                base_size *= 0.7
            elif volatility < 2:  # Low volatility
                base_size *= 1.2
            
            # Adjust for liquidity
            volume_30d = row.get('volume_30d', 0)
            if volume_30d < 100_000:
                base_size *= 0.5
            elif volume_30d > 1_000_000:
                base_size *= 1.1
            
            return max(CONSTANTS.MIN_POSITION_SIZE, min(base_size, CONSTANTS.MAX_POSITION_SIZE))
            
        except Exception:
            return CONSTANTS.MIN_POSITION_SIZE
    
    def _estimate_volatility(self, row: pd.Series) -> float:
        """Estimate stock volatility"""
        try:
            # Use return ranges as volatility proxy
            daily_vol = abs(row.get('ret_1d', 0))
            weekly_vol = abs(row.get('ret_7d', 0)) / 5  # Approximate daily from weekly
            monthly_vol = abs(row.get('ret_30d', 0)) / 22  # Approximate daily from monthly
            
            # Weight recent volatility more
            estimated_vol = (daily_vol * 0.5 + weekly_vol * 0.3 + monthly_vol * 0.2)
            
            return estimated_vol
            
        except Exception:
            return 3.0  # Default medium volatility
    
    def _create_exit_signal(self, ticker: str, reason: str) -> Optional[Signal]:
        """Create exit signal for disappeared stocks"""
        try:
            return Signal(
                ticker=ticker,
                company_name=ticker,
                sector='Unknown',
                category='Unknown',
                signal_type='EXIT',
                signal_strength=100,
                signal_timestamp=datetime.now(),
                momentum_rank=0,
                volume_rank=0,
                combined_rank=0,
                price=0,
                momentum_score=0,
                volume_score=0,
                momentum_acceleration=0,
                volume_acceleration=0,
                volatility_estimate=0,
                liquidity_score=0,
                recommended_size=0,
                confidence_level=1,
                days_in_list=0,
                first_appearance=None,
                exit_reasons=[reason]
            )
        except Exception:
            return None

# ============================================
# MARKET ANALYSIS
# ============================================

class MarketAnalyzer:
    """Analyze overall market conditions"""
    
    @staticmethod
    @monitor_performance
    def analyze_market(df: pd.DataFrame) -> MarketConditions:
        """Comprehensive market analysis"""
        try:
            total_stocks = len(df)
            
            # Market breadth
            advancing = len(df[df['ret_1d'] > 0])
            declining = len(df[df['ret_1d'] < 0])
            
            # Momentum distribution
            avg_momentum = df['momentum_score'].mean()
            momentum_dispersion = df['momentum_score'].std()
            positive_momentum = len(df[df['momentum_score'] > 0])
            
            # Volume analysis
            avg_rvol = df['rvol'].mean()
            high_volume = len(df[df['rvol'] > CONSTANTS.VOLUME_CONFIRMATION])
            
            # Sector analysis
            sector_performance = df.groupby('sector')['ret_7d'].mean().sort_values(ascending=False)
            leading_sectors = [(s, p) for s, p in sector_performance.head(5).items()]
            
            # Determine regime
            regime = MarketAnalyzer._determine_regime(df)
            
            # Calculate volatility
            market_volatility = df['ret_1d'].std()
            
            return MarketConditions(
                timestamp=datetime.now(),
                total_stocks=total_stocks,
                regime=regime['type'],
                regime_confidence=regime['confidence'],
                advancing_pct=(advancing / total_stocks * 100) if total_stocks > 0 else 0,
                declining_pct=(declining / total_stocks * 100) if total_stocks > 0 else 0,
                new_highs_pct=len(df[df['from_high_pct'] > -5]) / total_stocks * 100 if total_stocks > 0 else 0,
                new_lows_pct=len(df[df['from_low_pct'] < 5]) / total_stocks * 100 if total_stocks > 0 else 0,
                avg_momentum=avg_momentum,
                momentum_dispersion=momentum_dispersion,
                positive_momentum_pct=(positive_momentum / total_stocks * 100) if total_stocks > 0 else 0,
                avg_relative_volume=avg_rvol,
                high_volume_pct=(high_volume / total_stocks * 100) if total_stocks > 0 else 0,
                leading_sectors=leading_sectors,
                sector_rotation_intensity=sector_performance.std(),
                market_volatility=market_volatility,
                correlation_level=MarketAnalyzer._calculate_correlation_level(df)
            )
            
        except Exception as e:
            logger.error(f"Market analysis error: {e}")
            return MarketConditions(
                timestamp=datetime.now(),
                total_stocks=0,
                regime='UNKNOWN',
                regime_confidence=0,
                advancing_pct=0,
                declining_pct=0,
                new_highs_pct=0,
                new_lows_pct=0,
                avg_momentum=0,
                momentum_dispersion=0,
                positive_momentum_pct=0,
                avg_relative_volume=1,
                high_volume_pct=0,
                leading_sectors=[],
                sector_rotation_intensity=0,
                market_volatility=0,
                correlation_level=0
            )
    
    @staticmethod
    def _determine_regime(df: pd.DataFrame) -> Dict[str, Any]:
        """Determine market regime"""
        try:
            # Check trend consistency
            above_sma20 = len(df[df['price'] > df['sma_20d']]) / len(df)
            above_sma50 = len(df[df['price'] > df['sma_50d']]) / len(df)
            
            # Check momentum
            positive_momentum = len(df[df['ret_7d'] > 0]) / len(df)
            
            # Determine regime
            if above_sma50 > 0.65 and positive_momentum > 0.6:
                return {'type': 'TRENDING', 'confidence': min(above_sma50, positive_momentum)}
            elif above_sma50 < 0.35 and positive_momentum < 0.4:
                return {'type': 'DECLINING', 'confidence': 1 - max(above_sma50, positive_momentum)}
            elif df['ret_1d'].std() > df['ret_30d'].std() / 5:
                return {'type': 'VOLATILE', 'confidence': 0.7}
            else:
                return {'type': 'RANGING', 'confidence': 0.5}
                
        except Exception:
            return {'type': 'UNKNOWN', 'confidence': 0}
    
    @staticmethod
    def _calculate_correlation_level(df: pd.DataFrame) -> float:
        """Calculate market correlation level"""
        try:
            # Simple correlation proxy: how many stocks moving in same direction
            if len(df) == 0:
                return 0
                
            same_direction = len(df[
                (df['ret_1d'] > 0) == (df['ret_7d'] > 0)
            ])
            
            return same_direction / len(df)
            
        except Exception:
            return 0.5

# ============================================
# DATA PROCESSING PIPELINE
# ============================================

class DataPipeline:
    """Main data processing pipeline"""
    
    def __init__(self):
        self.data_store = DataStore()
        self.momentum_engine = MomentumEngine()
        self.volume_engine = VolumeEngine()
        self.ranking_engine = RankingEngine()
        self.signal_generator = SignalGenerator(self.data_store)
        self.market_analyzer = MarketAnalyzer()
    
    @monitor_performance
    def process_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal], MarketConditions]:
        """Complete data processing pipeline"""
        try:
            # Clean and validate data
            df = self._clean_data(df)
            
            if df.empty:
                logger.warning("No data after cleaning")
                return df, [], self.market_analyzer.analyze_market(df)
            
            # Calculate scores
            df = self._calculate_scores(df)
            
            # Calculate ranks
            df = self.ranking_engine.calculate_relative_ranks(df)
            
            # Load previous data for comparison
            yesterday = datetime.now() - timedelta(days=1)
            previous_df = self.data_store.load_daily_snapshot(yesterday)
            
            # Generate signals
            signals = self.signal_generator.generate_signals(df, previous_df)
            
            # Analyze market
            market_conditions = self.market_analyzer.analyze_market(df)
            
            # Save data
            self.data_store.save_daily_snapshot(df, datetime.now())
            self.data_store.save_signals(signals, datetime.now())
            
            return df, signals, market_conditions
            
        except Exception as e:
            logger.error(f"Pipeline processing error: {e}")
            return pd.DataFrame(), [], self.market_analyzer.analyze_market(pd.DataFrame())
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data"""
        try:
            # Remove invalid prices
            df = df[df['price'] >= CONSTANTS.MIN_PRICE]
            
            # Remove low volume stocks
            df = df[df['volume_30d'] >= CONSTANTS.MIN_VOLUME]
            
            # Remove extreme values
            df = df[df['ret_1d'].between(-30, 30)]  # Circuit limit protection
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['ticker'], keep='first')
            
            # Ensure required columns have valid data
            required_columns = ['price', 'ret_1d', 'ret_7d', 'rvol', 'volume_30d']
            for col in required_columns:
                if col in df.columns:
                    df = df[df[col].notna()]
            
            logger.info(f"Data cleaned: {len(df)} stocks remain")
            return df
            
        except Exception as e:
            logger.error(f"Data cleaning error: {e}")
            return pd.DataFrame()
    
    def _calculate_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all scores"""
        try:
            # Calculate momentum scores
            df['momentum_score'] = df.apply(self.momentum_engine.calculate_momentum_score, axis=1)
            df['momentum_acceleration'] = df.apply(self.momentum_engine.calculate_momentum_acceleration, axis=1)
            
            # Calculate volume scores
            df['volume_score'] = df.apply(self.volume_engine.calculate_volume_score, axis=1)
            df['volume_acceleration'] = df.apply(self.volume_engine.calculate_volume_acceleration, axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Score calculation error: {e}")
            return df

# ============================================
# VISUALIZATION COMPONENTS
# ============================================

class Visualizer:
    """Create all visualizations"""
    
    @staticmethod
    def create_market_overview(market_conditions: MarketConditions) -> go.Figure:
        """Create market overview dashboard"""
        fig = go.Figure()
        
        # Market regime indicator
        regime_colors = {
            'TRENDING': '#10b981',
            'DECLINING': '#ef4444',
            'VOLATILE': '#f59e0b',
            'RANGING': '#6b7280',
            'UNKNOWN': '#9ca3af'
        }
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=market_conditions.regime_confidence * 100,
            title={'text': f"{market_conditions.regime} Market"},
            domain={'x': [0, 0.5], 'y': [0.5, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': regime_colors.get(market_conditions.regime, '#6b7280')},
                'steps': [
                    {'range': [0, 30], 'color': '#fee2e2'},
                    {'range': [30, 70], 'color': '#fef3c7'},
                    {'range': [70, 100], 'color': '#d4f1e4'}
                ]
            }
        ))
        
        # Breadth indicator
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=market_conditions.advancing_pct,
            title={'text': "Advancing %"},
            delta={'reference': 50, 'relative': False},
            domain={'x': [0.5, 1], 'y': [0.5, 1]}
        ))
        
        # Volume indicator
        fig.add_trace(go.Indicator(
            mode="number",
            value=market_conditions.avg_relative_volume,
            title={'text': "Avg Relative Volume"},
            number={'suffix': "x"},
            domain={'x': [0, 0.5], 'y': [0, 0.5]}
        ))
        
        # Momentum indicator
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=market_conditions.positive_momentum_pct,
            title={'text': "Positive Momentum %"},
            delta={'reference': 50, 'relative': False},
            domain={'x': [0.5, 1], 'y': [0, 0.5]}
        ))
        
        fig.update_layout(
            title="Market Conditions Overview",
            height=400,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_signal_distribution(signals: List[Signal]) -> go.Figure:
        """Create signal distribution chart"""
        if not signals:
            return go.Figure()
        
        # Count by signal type
        signal_counts = {}
        for signal in signals:
            signal_counts[signal.signal_type] = signal_counts.get(signal.signal_type, 0) + 1
        
        # Create bar chart
        fig = go.Figure(go.Bar(
            x=list(signal_counts.keys()),
            y=list(signal_counts.values()),
            marker_color=['#10b981', '#f59e0b', '#ef4444', '#991b1b'],
            text=list(signal_counts.values()),
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Signal Distribution",
            xaxis_title="Signal Type",
            yaxis_title="Count",
            height=300,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_sector_momentum(df: pd.DataFrame) -> go.Figure:
        """Create sector momentum chart"""
        if df.empty:
            return go.Figure()
        
        # Calculate sector metrics
        sector_data = df.groupby('sector').agg({
            'momentum_score': 'mean',
            'volume_score': 'mean',
            'ticker': 'count'
        }).round(2)
        
        # Filter sectors with enough stocks
        sector_data = sector_data[sector_data['ticker'] >= 3]
        
        if sector_data.empty:
            return go.Figure()
        
        # Sort by momentum
        sector_data = sector_data.sort_values('momentum_score', ascending=True).tail(15)
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=sector_data.index,
            x=sector_data['momentum_score'],
            orientation='h',
            name='Momentum Score',
            marker_color='#8b5cf6',
            text=[f"{x:.2f}" for x in sector_data['momentum_score']],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Top 15 Sectors by Momentum",
            xaxis_title="Average Momentum Score",
            yaxis_title="",
            height=500,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_rank_scatter(df: pd.DataFrame) -> go.Figure:
        """Create momentum vs volume rank scatter"""
        if df.empty:
            return go.Figure()
        
        # Get top stocks
        top_stocks = df.nlargest(50, 'combined_rank')
        
        if top_stocks.empty:
            return go.Figure()
        
        # Create scatter plot
        fig = go.Figure()
        
        # Color by combined rank
        fig.add_trace(go.Scatter(
            x=top_stocks['momentum_rank'],
            y=top_stocks['volume_rank'],
            mode='markers+text',
            text=top_stocks['ticker'],
            textposition="top center",
            marker=dict(
                size=top_stocks['combined_rank'] * 20,
                color=top_stocks['combined_rank'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Combined Rank")
            ),
            hovertemplate="<b>%{text}</b><br>" +
                         "Momentum Rank: %{x:.2f}<br>" +
                         "Volume Rank: %{y:.2f}<br>" +
                         "<extra></extra>"
        ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            showlegend=False
        ))
        
        fig.update_layout(
            title="Top 50 Stocks: Momentum vs Volume Rank",
            xaxis_title="Momentum Rank (Percentile)",
            yaxis_title="Volume Rank (Percentile)",
            height=600,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        return fig

# ============================================
# REPORT GENERATION
# ============================================

class ReportGenerator:
    """Generate Excel and text reports"""
    
    @staticmethod
    def generate_excel_report(
        df: pd.DataFrame, 
        signals: List[Signal], 
        market_conditions: MarketConditions
    ) -> BytesIO:
        """Generate comprehensive Excel report"""
        output = BytesIO()
        
        try:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                workbook = writer.book
                
                # Define formats
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'vcenter',
                    'align': 'center',
                    'fg_color': '#4A5568',
                    'font_color': 'white',
                    'border': 1
                })
                
                number_format = workbook.add_format({'num_format': '#,##0.00'})
                percent_format = workbook.add_format({'num_format': '0.00%'})
                currency_format = workbook.add_format({'num_format': '₹#,##0.00'})
                
                # Sheet 1: Summary
                summary_data = pd.DataFrame({
                    'Metric': [
                        'Analysis Date',
                        'Market Regime',
                        'Total Stocks Analyzed',
                        'Buy Signals',
                        'Hold Signals',
                        'Sell Signals',
                        'Exit Signals',
                        'Market Volatility',
                        'Average Volume',
                        'Top Sector'
                    ],
                    'Value': [
                        market_conditions.timestamp.strftime('%Y-%m-%d %H:%M'),
                        f"{market_conditions.regime} ({market_conditions.regime_confidence:.1%})",
                        str(market_conditions.total_stocks),
                        str(sum(1 for s in signals if s.signal_type == 'BUY')),
                        str(sum(1 for s in signals if s.signal_type == 'HOLD')),
                        str(sum(1 for s in signals if s.signal_type == 'SELL')),
                        str(sum(1 for s in signals if s.signal_type == 'EXIT')),
                        f"{market_conditions.market_volatility:.2%}",
                        f"{market_conditions.avg_relative_volume:.2f}x",
                        market_conditions.leading_sectors[0][0] if market_conditions.leading_sectors else 'N/A'
                    ]
                })
                
                summary_data.to_excel(writer, sheet_name='Summary', index=False)
                
                # Sheet 2: Buy Signals
                buy_signals = [s for s in signals if s.signal_type == 'BUY']
                if buy_signals:
                    buy_df = pd.DataFrame([s.to_dict() for s in buy_signals])
                    buy_df = buy_df[[
                        'ticker', 'company_name', 'sector', 'price',
                        'signal_strength', 'momentum_rank', 'volume_rank',
                        'combined_rank', 'recommended_size', 'days_in_list'
                    ]]
                    buy_df.to_excel(writer, sheet_name='Buy Signals', index=False)
                
                # Sheet 3: All Signals
                if signals:
                    all_signals_df = pd.DataFrame([s.to_dict() for s in signals])
                    all_signals_df.to_excel(writer, sheet_name='All Signals', index=False)
                
                # Sheet 4: Market Analysis
                market_df = pd.DataFrame({
                    'Metric': [
                        'Advancing %',
                        'Declining %',
                        'New Highs %',
                        'New Lows %',
                        'Positive Momentum %',
                        'High Volume %'
                    ],
                    'Value': [
                        f"{market_conditions.advancing_pct:.1f}%",
                        f"{market_conditions.declining_pct:.1f}%",
                        f"{market_conditions.new_highs_pct:.1f}%",
                        f"{market_conditions.new_lows_pct:.1f}%",
                        f"{market_conditions.positive_momentum_pct:.1f}%",
                        f"{market_conditions.high_volume_pct:.1f}%"
                    ]
                })
                market_df.to_excel(writer, sheet_name='Market Analysis', index=False)
                
                # Format all sheets
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    worksheet.freeze_panes(1, 0)
                    
                    # Auto-fit columns
                    for i in range(20):
                        worksheet.set_column(i, i, 15)
            
            output.seek(0)
            return output
            
        except Exception as e:
            logger.error(f"Excel generation error: {e}")
            output.seek(0)
            return output
    
    @staticmethod
    def generate_text_summary(
        signals: List[Signal], 
        market_conditions: MarketConditions
    ) -> str:
        """Generate text summary"""
        buy_signals = [s for s in signals if s.signal_type == 'BUY']
        exit_signals = [s for s in signals if s.signal_type == 'EXIT']
        
        summary = f"""
WAVE DETECTION 5.0 - DAILY SUMMARY
{datetime.now().strftime('%Y-%m-%d %H:%M')}

MARKET CONDITIONS:
- Regime: {market_conditions.regime} (Confidence: {market_conditions.regime_confidence:.1%})
- Volatility: {market_conditions.market_volatility:.1%}
- Advancing: {market_conditions.advancing_pct:.1f}% | Declining: {market_conditions.declining_pct:.1f}%
- Average Volume: {market_conditions.avg_relative_volume:.2f}x

TOP SECTORS:
"""
        
        for sector, perf in market_conditions.leading_sectors[:5]:
            summary += f"- {sector}: {perf:.1f}%\n"
        
        summary += f"\nBUY SIGNALS ({len(buy_signals)}):\n"
        for signal in buy_signals[:10]:
            summary += f"""
{signal.ticker} - {signal.company_name[:30]}
  Signal Strength: {signal.signal_strength:.0f} | Rank: {signal.combined_rank:.2%}
  Price: ₹{signal.price:.2f} | Position: {signal.recommended_size:.1%}
  Momentum Rank: {signal.momentum_rank:.2%} | Volume Rank: {signal.volume_rank:.2%}
"""
        
        if exit_signals:
            summary += f"\nEXIT SIGNALS ({len(exit_signals)}):\n"
            for signal in exit_signals[:5]:
                summary += f"- {signal.ticker}: {', '.join(signal.exit_reasons)}\n"
        
        summary += f"""
TRADING RULES:
1. Buy top-ranked stocks at market open
2. Exit when stock disappears or momentum dies
3. Position size based on signal strength
4. Maximum {CONSTANTS.MAX_POSITIONS} positions
5. Review daily before market open

Generated by Wave Detection 5.0
"""
        
        return summary

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main application entry point"""
    
    # Apply custom CSS
    st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">⚡ Wave Detection 5.0</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">The Ultimate Trading Edge - Simple. Fast. Profitable.</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'data_pipeline' not in st.session_state:
        st.session_state.data_pipeline = DataPipeline()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Data Configuration")
        
        # Data source
        sheet_url = st.text_input(
            "Google Sheets URL",
            value="https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk/edit?usp=sharing",
            help="Your momentum watchlist Google Sheets URL"
        )
        
        gid = st.text_input(
            "Sheet GID",
            value="2026492216",
            help="Sheet identifier (found in URL after gid=)"
        )
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", type="primary", use_container_width=True):
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear Cache", type="secondary", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache cleared!")
        
        st.markdown("---")
        
        # Settings
        st.markdown("## ⚙️ Settings")
        
        # Percentile threshold
        top_percentile = st.slider(
            "Signal Percentile Threshold",
            min_value=0.80,
            max_value=0.99,
            value=0.90,
            step=0.01,
            help="Top X% of stocks to consider for signals"
        )
        
        # Volume confirmation
        min_rvol = st.number_input(
            "Minimum Volume Confirmation",
            min_value=1.0,
            max_value=5.0,
            value=1.5,
            step=0.1,
            help="Minimum relative volume for buy signals"
        )
        
        # Max positions
        max_positions = st.number_input(
            "Maximum Positions",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="Maximum number of concurrent positions"
        )
        
        # Update constants
        CONSTANTS = TradingConstants(
            MIN_PRICE=1.0,
            MIN_VOLUME=50_000,
            MAX_SPREAD=0.10,
            TOP_MOMENTUM_PERCENTILE=top_percentile,
            VOLUME_CONFIRMATION=min_rvol,
            MAX_POSITION_SIZE=0.03,
            MIN_POSITION_SIZE=0.005,
            MAX_POSITIONS=max_positions,
            MAX_SECTOR_EXPOSURE=0.30,
            INITIAL_STOP_LOSS=0.05,
            TRACKING_LOOKBACK_DAYS=30,
            MIN_SAMPLE_SIZE=20
        )
    
    # Main content
    try:
        # Load data
        @st.cache_data(ttl=300)
        def load_data(url: str, gid: str) -> pd.DataFrame:
            """Load data from Google Sheets"""
            try:
                csv_url = f"{url.split('/edit')[0]}/export?format=csv&gid={gid}"
                df = pd.read_csv(csv_url)
                logger.info(f"Loaded {len(df)} rows")
                return df
            except Exception as e:
                logger.error(f"Data loading failed: {e}")
                st.error(f"Failed to load data: {e}")
                return pd.DataFrame()
        
        with st.spinner("Loading market data..."):
            raw_df = load_data(sheet_url, gid)
        
        if raw_df.empty:
            st.error("No data loaded. Please check your URL and GID.")
            st.stop()
        
        # Process data
        with st.spinner("Processing signals..."):
            df, signals, market_conditions = st.session_state.data_pipeline.process_data(raw_df)
        
        # Market Overview
        st.markdown("## 📈 Market Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            regime_colors = {
                'TRENDING': '#10b981',
                'DECLINING': '#ef4444',
                'VOLATILE': '#f59e0b',
                'RANGING': '#6b7280'
            }
            color = regime_colors.get(market_conditions.regime, '#6b7280')
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Market Regime</div>
                <div class="metric-value" style="color: {color};">{market_conditions.regime}</div>
                <div class="metric-label">Confidence: {market_conditions.regime_confidence:.0%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Market Breadth</div>
                <div class="metric-value">{market_conditions.advancing_pct:.0f}%</div>
                <div class="metric-label">Advancing</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Volume</div>
                <div class="metric-value">{market_conditions.avg_relative_volume:.1f}x</div>
                <div class="metric-label">Relative to Normal</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Volatility</div>
                <div class="metric-value">{market_conditions.market_volatility:.1%}</div>
                <div class="metric-label">Daily</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            market_fig = Visualizer.create_market_overview(market_conditions)
            st.plotly_chart(market_fig, use_container_width=True)
        
        with col2:
            signal_dist = Visualizer.create_signal_distribution(signals)
            st.plotly_chart(signal_dist, use_container_width=True)
        
        # Signals Section
        st.markdown("## 🎯 Trading Signals")
        
        # Signal summary
        buy_signals = [s for s in signals if s.signal_type == 'BUY']
        hold_signals = [s for s in signals if s.signal_type == 'HOLD']
        sell_signals = [s for s in signals if s.signal_type == 'SELL']
        exit_signals = [s for s in signals if s.signal_type == 'EXIT']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🟢 Buy", len(buy_signals))
        with col2:
            st.metric("🟡 Hold", len(hold_signals))
        with col3:
            st.metric("🔴 Sell", len(sell_signals))
        with col4:
            st.metric("⚫ Exit", len(exit_signals))
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🟢 Buy Signals",
            "⚫ Exit Alerts",
            "📊 Analysis",
            "📈 Rankings",
            "📥 Reports"
        ])
        
        with tab1:
            st.markdown("### 🟢 Buy Signals - Act Now!")
            
            if buy_signals:
                for signal in buy_signals[:10]:  # Top 10
                    st.markdown(f'<div class="signal-card signal-buy slide-in">', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4, col5 = st.columns([2.5, 1.5, 1.5, 1.5, 1])
                    
                    with col1:
                        st.markdown(f"### {signal.ticker}")
                        st.markdown(f"{signal.company_name[:40]}...")
                        st.markdown(f"**Sector:** {signal.sector} | **Category:** {signal.category}")
                        
                        # New appearance badge
                        if signal.days_in_list <= 1:
                            st.markdown('<span class="badge badge-new">NEW</span>', unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Price", f"₹{signal.price:.2f}")
                        st.metric("Signal", f"{signal.signal_strength:.0f}/100")
                    
                    with col3:
                        st.metric("Momentum Rank", f"{signal.momentum_rank:.0%}")
                        st.metric("Volume Rank", f"{signal.volume_rank:.0%}")
                    
                    with col4:
                        st.metric("Combined Rank", f"{signal.combined_rank:.0%}")
                        st.metric("Position Size", f"{signal.recommended_size:.1%}")
                    
                    with col5:
                        st.metric("Days Listed", signal.days_in_list)
                        
                        # Rank visualization
                        st.markdown(f"""
                        <div class="rank-bar">
                            <div class="rank-fill" style="width: {signal.combined_rank*100}%"></div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("")
            else:
                st.info("No buy signals at current thresholds. Try adjusting settings.")
        
        with tab2:
            st.markdown("### ⚫ Exit Alerts - Immediate Action Required!")
            
            if exit_signals:
                for signal in exit_signals:
                    st.markdown(f'<div class="signal-card signal-exit">', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"### {signal.ticker} - EXIT NOW")
                        st.markdown(f"**Reasons:** {', '.join(signal.exit_reasons)}")
                    
                    with col2:
                        st.markdown('<span class="badge badge-exit">IMMEDIATE EXIT</span>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.success("No exit signals - all positions can be held.")
        
        with tab3:
            st.markdown("### 📊 Market Analysis")
            
            # Sector momentum
            sector_fig = Visualizer.create_sector_momentum(df)
            st.plotly_chart(sector_fig, use_container_width=True)
            
            # Leading sectors details
            if market_conditions.leading_sectors:
                st.markdown("#### 🏆 Leading Sectors")
                
                cols = st.columns(5)
                for i, (sector, perf) in enumerate(market_conditions.leading_sectors[:5]):
                    with cols[i]:
                        color = "#10b981" if perf > 0 else "#ef4444"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">{sector[:15]}</div>
                            <div class="metric-value" style="color: {color};">{perf:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        with tab4:
            st.markdown("### 📈 Stock Rankings")
            
            # Rank scatter plot
            rank_fig = Visualizer.create_rank_scatter(df)
            st.plotly_chart(rank_fig, use_container_width=True)
            
            # Top ranked stocks table
            st.markdown("#### 🏆 Top 20 Ranked Stocks")
            
            if not df.empty:
                top_20 = df.nlargest(20, 'combined_rank')[[
                    'ticker', 'company_name', 'sector', 'price',
                    'momentum_rank', 'volume_rank', 'combined_rank',
                    'momentum_score', 'volume_score'
                ]].round(3)
                
                st.dataframe(
                    top_20,
                    use_container_width=True,
                    height=400
                )
        
        with tab5:
            st.markdown("### 📥 Download Reports")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Generate Excel Report", type="primary", use_container_width=True):
                    excel_file = ReportGenerator.generate_excel_report(df, signals, market_conditions)
                    
                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_file,
                        file_name=f"wave_detection_5_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
            
            with col2:
                if st.button("📝 Generate Text Summary", type="secondary", use_container_width=True):
                    summary = ReportGenerator.generate_text_summary(signals, market_conditions)
                    
                    st.text_area(
                        "Daily Summary",
                        summary,
                        height=600
                    )
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"An error occurred: {e}")
        st.info("Please check your data and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.875rem; padding: 1rem;">
        Wave Detection 5.0 - The Ultimate Trading Edge<br>
        Simple. Fast. Profitable.<br>
        © 2024 - Professional Trading Systems
    </div>
    """, unsafe_allow_html=True)

# Entry point
if __name__ == "__main__":
    main()
