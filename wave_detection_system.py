"""
Wave Detection System 8.0 - Ultimate Professional Trading Analytics Platform
==========================================================================
The definitive implementation combining sophisticated market analysis with
professional software engineering practices.

Features:
- Advanced pattern detection with lifecycle analysis
- Vectorized calculations for 2000+ stocks in milliseconds
- Actionable trading signals with risk management
- Sector-relative scoring and peer comparison
- Professional error handling and logging
- Mobile-responsive UI with progressive enhancement

Architecture: Clean Architecture with Domain-Driven Design
Performance: Sub-second response for 2000+ stocks
Reliability: Enterprise-grade error handling and recovery

Version: 8.0.0 (Ultimate Edition)
Author: Professional Implementation
License: MIT
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import logging
from io import BytesIO
from typing import Dict, List, Tuple, Optional, Union, Any, Protocol
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import warnings
from functools import lru_cache, wraps
import time

# ============================================
# CONFIGURATION & SETUP
# ============================================

# Suppress warnings in production
warnings.filterwarnings('ignore')

# Configure logging with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Wave Detection 8.0 | Ultimate Trading Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "Wave Detection 8.0 - The Ultimate Trading Analytics Platform"
    }
)

# ============================================
# DOMAIN MODELS & ENUMS
# ============================================

class SignalType(Enum):
    """Trading signal classifications"""
    STRONG_BUY = "🚀 STRONG BUY"
    BUY = "✅ BUY"
    HOLD = "⏸️ HOLD"
    SELL = "📉 SELL"
    STRONG_SELL = "🔴 STRONG SELL"
    
    @property
    def is_bullish(self) -> bool:
        return self in [SignalType.STRONG_BUY, SignalType.BUY]
    
    @property
    def is_bearish(self) -> bool:
        return self in [SignalType.SELL, SignalType.STRONG_SELL]

class LifecycleStage(Enum):
    """Market lifecycle stages based on Wyckoff methodology"""
    ACCUMULATION = "ACCUMULATION"
    EARLY_MARKUP = "EARLY_MARKUP"
    LATE_MARKUP = "LATE_MARKUP"
    DISTRIBUTION = "DISTRIBUTION"
    MARKDOWN = "MARKDOWN"
    RECOVERY = "RECOVERY"
    UNKNOWN = "UNKNOWN"
    
    @property
    def color(self) -> str:
        """Get color for visualization"""
        colors = {
            "ACCUMULATION": "#27AE60",
            "EARLY_MARKUP": "#3498DB",
            "LATE_MARKUP": "#F39C12",
            "DISTRIBUTION": "#E74C3C",
            "MARKDOWN": "#95A5A6",
            "RECOVERY": "#9B59B6",
            "UNKNOWN": "#BDC3C7"
        }
        return colors.get(self.value, "#BDC3C7")

class VolumePattern(Enum):
    """Volume behavior patterns"""
    EXPANDING = "EXPANDING"
    CONTRACTING = "CONTRACTING"
    SPIKE = "SPIKE"
    DORMANT = "DORMANT"
    NEUTRAL = "NEUTRAL"

class PositionOpportunity(Enum):
    """Position-based trading opportunities"""
    BREAKOUT = "BREAKOUT"
    REVERSAL = "REVERSAL"
    MOMENTUM = "MOMENTUM"
    AVOID = "AVOID"
    NEUTRAL = "NEUTRAL"

# ============================================
# CONFIGURATION
# ============================================

@dataclass(frozen=True)
class Config:
    """Immutable application configuration"""
    # Data source
    DEFAULT_SHEET_URL: str = "https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk/edit?usp=sharing"
    DEFAULT_GID: str = "2026492216"
    
    # Performance settings
    CACHE_TTL: int = 300  # 5 minutes
    MIN_STOCKS_FOR_SECTOR_ANALYSIS: int = 3
    VECTORIZATION_CHUNK_SIZE: int = 500  # Process in chunks for memory efficiency
    
    # Filtering
    MIN_PRICE: float = 10.0
    MIN_VOLUME: float = 10000
    MIN_DATA_QUALITY_SCORE: float = 0.7  # Minimum % of non-null values
    
    # Scoring weights
    MOMENTUM_WEIGHT: float = 0.35
    POSITION_WEIGHT: float = 0.25
    VOLUME_WEIGHT: float = 0.25
    QUALITY_WEIGHT: float = 0.15
    
    # Advanced scoring weights
    PATTERN_WEIGHT: float = 0.30
    LIFECYCLE_WEIGHT: float = 0.30
    TECHNICAL_WEIGHT: float = 0.40
    
    # Trading thresholds
    STRONG_BUY_THRESHOLD: float = 85
    BUY_THRESHOLD: float = 70
    SELL_THRESHOLD: float = 30
    STRONG_SELL_THRESHOLD: float = 20
    
    # Risk management
    DEFAULT_STOP_LOSS_PCT: float = 5.0
    DEFAULT_TARGET_PCT: float = 10.0
    MAX_POSITION_SIZE_PCT: float = 5.0  # Max % of portfolio per position
    
    # Display settings
    DEFAULT_DISPLAY_COUNT: int = 20
    MAX_DISPLAY_COUNT: int = 100
    
    # Tier definitions
    EPS_TIERS: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "Below 5": (-float('inf'), 5),
        "5-15": (5, 15),
        "15-35": (15, 35),
        "35-55": (35, 55),
        "55-75": (55, 75),
        "75-95": (75, 95),
        "Above 95": (95, float('inf'))
    })
    
    PRICE_TIERS: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "Below 100": (-float('inf'), 100),
        "100-200": (100, 200),
        "200-500": (200, 500),
        "500-1000": (500, 1000),
        "1000-2000": (1000, 2000),
        "2000-5000": (2000, 5000),
        "Above 5000": (5000, float('inf'))
    })

# Create global config instance
config = Config()

# ============================================
# DATA MODELS
# ============================================

@dataclass
class StockMetrics:
    """Core metrics for a stock"""
    # Price metrics
    price: float
    price_change_1d: float
    price_change_7d: float
    price_change_30d: float
    
    # Volume metrics
    volume: float
    relative_volume: float
    volume_trend: float
    
    # Technical indicators
    distance_from_low: float
    distance_from_high: float
    ma_alignment_score: float
    
    # Fundamental metrics (optional)
    eps: Optional[float] = None
    pe_ratio: Optional[float] = None
    eps_growth: Optional[float] = None

@dataclass
class StockAnalysis:
    """Complete analysis results for a stock"""
    # Identity
    ticker: str
    company_name: str
    sector: str
    category: str
    
    # Core metrics
    metrics: StockMetrics
    
    # Scores
    power_score: float
    momentum_score: float
    position_score: float
    volume_score: float
    quality_score: float
    
    # Advanced analysis
    lifecycle_stage: LifecycleStage
    volume_pattern: VolumePattern
    position_opportunity: PositionOpportunity
    future_potential_score: float
    
    # Trading signal
    signal: SignalType
    confidence: float
    
    # Risk metrics
    stop_loss: float
    target: float
    risk_reward_ratio: float
    position_size_pct: float
    
    # Rankings
    overall_rank: int
    sector_rank: int
    peer_percentile: float

# ============================================
# INTERFACES (PROTOCOLS)
# ============================================

class DataProcessor(Protocol):
    """Interface for data processing operations"""
    def validate(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate dataframe structure"""
        ...
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data"""
        ...
    
    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated fields"""
        ...

class ScoringEngine(Protocol):
    """Interface for scoring operations"""
    def calculate_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all scores"""
        ...
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals"""
        ...

# ============================================
# UTILITY FUNCTIONS & DECORATORS
# ============================================

def performance_monitor(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = None
        
        try:
            import psutil
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
        except:
            pass
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log performance metrics
            log_msg = f"{func.__name__} completed in {execution_time:.2f}s"
            
            if start_memory:
                try:
                    end_memory = process.memory_info().rss / 1024 / 1024
                    memory_used = end_memory - start_memory
                    log_msg += f" | Memory: {memory_used:+.1f}MB"
                except:
                    pass
            
            if execution_time > 1.0:
                logger.warning(log_msg)
            else:
                logger.debug(log_msg)
            
            return result
            
        except Exception as e:
            logger.error(f"{func.__name__} failed after {time.time() - start_time:.2f}s: {str(e)}")
            raise
    
    return wrapper

def safe_divide(numerator: Union[float, np.ndarray], 
                denominator: Union[float, np.ndarray], 
                default: float = 0.0) -> Union[float, np.ndarray]:
    """Safely divide with proper handling of arrays and scalars"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(denominator != 0, numerator / denominator, default)
    return result

def validate_required_columns(df: pd.DataFrame, required: List[str]) -> Tuple[bool, List[str]]:
    """Validate dataframe has required columns"""
    missing = [col for col in required if col not in df.columns]
    return len(missing) == 0, missing

# ============================================
# DATA PIPELINE IMPLEMENTATION
# ============================================

class IndiaNSEDataProcessor:
    """Specialized processor for Indian NSE/BSE data"""
    
    REQUIRED_COLUMNS = [
        'ticker', 'company_name', 'price', 'volume_1d',
        'ret_1d', 'ret_7d', 'ret_30d', 'rvol',
        'from_low_pct', 'from_high_pct',
        'sma_20d', 'sma_50d', 'sma_200d',
        'category', 'sector'
    ]
    
    @staticmethod
    def clean_indian_number(value: Any) -> float:
        """Clean Indian formatted numbers (₹, commas, %)"""
        if pd.isna(value):
            return np.nan
            
        try:
            # Convert to string and clean
            cleaned = str(value).strip()
            
            # Remove currency and formatting
            for char in ['₹', '%', ',', ' ']:
                cleaned = cleaned.replace(char, '')
            
            # Handle special cases
            if cleaned in ['', '-', 'N/A', '#N/A', 'nan', 'None']:
                return np.nan
                
            return float(cleaned)
        except:
            return np.nan
    
    @performance_monitor
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete processing pipeline"""
        logger.info(f"Processing {len(df)} stocks")
        
        # Validate
        is_valid, missing = validate_required_columns(df, self.REQUIRED_COLUMNS)
        if not is_valid:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Clean numeric columns - vectorized operations
        numeric_cols = [
            'price', 'volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d',
            'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y',
            'from_low_pct', 'from_high_pct', 'rvol',
            'sma_20d', 'sma_50d', 'sma_200d',
            'low_52w', 'high_52w', 'pe', 'eps_current', 'eps_change_pct'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].apply(self.clean_indian_number), 
                    errors='coerce'
                )
        
        # Handle volume ratios - they come as percentage changes
        vol_ratio_cols = [col for col in df.columns if 'vol_ratio' in col]
        for col in vol_ratio_cols:
            if col in df.columns:
                # Convert percentage to ratio: -56.61% becomes 0.4339
                df[col] = (100 + pd.to_numeric(df[col].apply(self.clean_indian_number), errors='coerce')) / 100
                df[col].fillna(1.0, inplace=True)
        
        # Clean categorical columns
        categorical_cols = ['ticker', 'company_name', 'category', 'sector']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(['nan', 'None', '', 'N/A'], 'Unknown')
        
        # Calculate data quality score
        numeric_data = df[numeric_cols].select_dtypes(include=[np.number])
        df['data_quality_score'] = (numeric_data.notna().sum(axis=1) / len(numeric_cols))
        
        # Add derived fields
        df = self._add_derived_fields(df)
        
        # Filter out low quality data
        initial_count = len(df)
        df = df[
            (df['price'] > config.MIN_PRICE) &
            (df['volume_1d'] > config.MIN_VOLUME) &
            (df['data_quality_score'] > config.MIN_DATA_QUALITY_SCORE)
        ]
        
        logger.info(f"Filtered: {initial_count} → {len(df)} stocks")
        
        return df
    
    def _add_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated fields for analysis"""
        # Price momentum consistency
        df['returns_aligned'] = (
            (df['ret_1d'] > 0) & 
            (df['ret_7d'] > 0) & 
            (df['ret_30d'] > 0)
        ).astype(int)
        
        # MA alignment
        df['ma_aligned'] = (
            (df['price'] > df['sma_20d']) & 
            (df['sma_20d'] > df['sma_50d']) & 
            (df['sma_50d'] > df['sma_200d'])
        ).astype(int)
        
        # Volatility estimate (using return variance)
        df['volatility_estimate'] = df[['ret_1d', 'ret_7d', 'ret_30d']].std(axis=1)
        
        # Volume surge indicator
        df['volume_surge'] = (df['rvol'] > 2.0).astype(int)
        
        return df

# ============================================
# SCORING ENGINE IMPLEMENTATION
# ============================================

class AdvancedScoringEngine:
    """Vectorized scoring engine with pattern detection"""
    
    @performance_monitor
    def calculate_all_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all scores using vectorized operations"""
        logger.info("Calculating scores for all stocks")
        
        # Core scores
        df['momentum_score'] = self._calculate_momentum_score_vectorized(df)
        df['position_score'] = self._calculate_position_score_vectorized(df)
        df['volume_score'] = self._calculate_volume_score_vectorized(df)
        df['quality_score'] = self._calculate_quality_score_vectorized(df)
        
        # Master score
        df['power_score'] = (
            df['momentum_score'] * config.MOMENTUM_WEIGHT +
            df['position_score'] * config.POSITION_WEIGHT +
            df['volume_score'] * config.VOLUME_WEIGHT +
            df['quality_score'] * config.QUALITY_WEIGHT
        )
        
        # Advanced analysis
        df = self._detect_patterns_vectorized(df)
        df = self._calculate_future_potential_vectorized(df)
        
        # Generate signals
        df = self._generate_signals_vectorized(df)
        
        # Calculate rankings
        df = self._calculate_rankings(df)
        
        return df
    
    def _calculate_momentum_score_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """Calculate momentum score using vectorized operations"""
        # Direct period returns comparison (no daily averaging)
        momentum_components = pd.DataFrame()
        
        # Short-term momentum (40%)
        momentum_components['short'] = (
            np.where(df['ret_1d'] > 5, 100, 
            np.where(df['ret_1d'] > 2, 80,
            np.where(df['ret_1d'] > 0, 60,
            np.where(df['ret_1d'] > -2, 40, 20))))
        ) * 0.4
        
        # Medium-term momentum (35%)
        momentum_components['medium'] = (
            np.where(df['ret_30d'] > 20, 100,
            np.where(df['ret_30d'] > 10, 80,
            np.where(df['ret_30d'] > 0, 60,
            np.where(df['ret_30d'] > -10, 40, 20))))
        ) * 0.35
        
        # Consistency bonus (25%)
        momentum_components['consistency'] = (
            df['returns_aligned'] * 100 * 0.25
        )
        
        return momentum_components.sum(axis=1)
    
    def _calculate_position_score_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """Calculate position score using vectorized operations"""
        position_components = pd.DataFrame()
        
        # Distance from low - optimal zone 20-100%
        position_components['low_distance'] = np.where(
            df['from_low_pct'] <= 100,
            df['from_low_pct'] * 0.4,  # Linear up to 40 points
            np.maximum(0, 40 - (df['from_low_pct'] - 100) * 0.1)  # Penalty for overextension
        )
        
        # Distance from high - best zone -40% to -10%
        position_components['high_distance'] = np.where(
            df['from_high_pct'] > -10, 30,  # Near highs
            np.where(
                df['from_high_pct'] > -40,
                30 + df['from_high_pct'] * 0.5,  # Gradual decrease
                10  # Too far from highs
            )
        )
        
        # MA alignment bonus (30%)
        position_components['ma_bonus'] = df['ma_aligned'] * 30
        
        return np.clip(position_components.sum(axis=1), 0, 100)
    
    def _calculate_volume_score_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume score using vectorized operations"""
        # RVOL is primary indicator
        rvol_score = np.clip(df['rvol'] * 20, 0, 60)
        
        # Volume trend from ratios
        vol_trend_score = 0
        if 'vol_ratio_30d_90d' in df.columns:
            vol_trend_score = np.clip((df['vol_ratio_30d_90d'] - 0.5) * 40, 0, 40)
        
        return np.clip(rvol_score + vol_trend_score, 0, 100)
    
    def _calculate_quality_score_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """Calculate quality score (optional fundamentals)"""
        # If no fundamental data, use technical quality
        if 'eps_change_pct' not in df.columns:
            # Technical quality based on trend and volatility
            trend_quality = df['ma_aligned'] * 50
            volatility_penalty = np.clip(df['volatility_estimate'] * 10, 0, 50)
            return np.clip(trend_quality + (50 - volatility_penalty), 0, 100)
        
        # Fundamental quality
        quality_components = pd.DataFrame()
        
        # EPS growth
        quality_components['eps_growth'] = np.where(
            df['eps_change_pct'] > 50, 100,
            np.where(df['eps_change_pct'] > 20, 80,
            np.where(df['eps_change_pct'] > 0, 60,
            np.where(df['eps_change_pct'] > -20, 40, 20)))
        ) * 0.5
        
        # PE valuation
        if 'pe' in df.columns:
            quality_components['pe_score'] = np.where(
                df['pe'] < 0, 30,  # Negative PE
                np.where(df['pe'] < 15, 100,
                np.where(df['pe'] < 25, 80,
                np.where(df['pe'] < 35, 60,
                np.where(df['pe'] < 50, 40, 20))))
            ) * 0.5
        else:
            quality_components['pe_score'] = 50  # Neutral if no PE data
        
        return quality_components.sum(axis=1)
    
    def _detect_patterns_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect patterns using vectorized operations"""
        # Lifecycle Stage Detection
        conditions = [
            # ACCUMULATION: Deep pullback + volume returning + positive momentum
            (df['from_high_pct'] < -50) & 
            (df['vol_ratio_30d_90d'] > 1.0) & 
            (df['ret_7d'] > -5),
            
            # EARLY_MARKUP: Above MAs + momentum positive + volume expanding
            (df['price'] > df['sma_50d']) & 
            (df['ret_30d'] > 0) & 
            (df['vol_ratio_30d_90d'] > 1.1),
            
            # LATE_MARKUP: Near highs + momentum slowing
            (df['from_high_pct'] > -20) & 
            (df['ret_1d'] < df['ret_7d'] / 7),
            
            # DISTRIBUTION: Near highs + volume declining + momentum negative
            (df['from_high_pct'] > -10) & 
            (df['vol_ratio_30d_90d'] < 0.9) & 
            (df['ret_7d'] < 0),
            
            # MARKDOWN: Below MAs + negative momentum
            (df['price'] < df['sma_50d']) & 
            (df['price'] < df['sma_200d']) & 
            (df['ret_30d'] < -10),
            
            # RECOVERY: Was down, now turning up
            (df['from_high_pct'] < -30) & 
            (df['ret_7d'] > 0) & 
            (df['ret_1d'] > 0)
        ]
        
        choices = [
            LifecycleStage.ACCUMULATION.value,
            LifecycleStage.EARLY_MARKUP.value,
            LifecycleStage.LATE_MARKUP.value,
            LifecycleStage.DISTRIBUTION.value,
            LifecycleStage.MARKDOWN.value,
            LifecycleStage.RECOVERY.value
        ]
        
        df['lifecycle_stage'] = np.select(conditions, choices, default=LifecycleStage.UNKNOWN.value)
        
        # Volume Pattern Detection
        vol_conditions = [
            # EXPANDING: Increasing volume trend
            (df['rvol'] > 1.5) & 
            (df.get('vol_ratio_7d_90d', 1) > df.get('vol_ratio_30d_90d', 1)),
            
            # SPIKE: Sudden volume surge
            (df['rvol'] > 3.0),
            
            # CONTRACTING: Decreasing volume
            (df['rvol'] < 0.7) & 
            (df.get('vol_ratio_30d_90d', 1) < 0.8),
            
            # DORMANT: Very low volume
            (df['rvol'] < 0.5)
        ]
        
        vol_choices = [
            VolumePattern.EXPANDING.value,
            VolumePattern.SPIKE.value,
            VolumePattern.CONTRACTING.value,
            VolumePattern.DORMANT.value
        ]
        
        df['volume_pattern'] = np.select(vol_conditions, vol_choices, default=VolumePattern.NEUTRAL.value)
        
        # Position Opportunity Detection
        opp_conditions = [
            # BREAKOUT: Near highs + positive momentum + volume
            (df['from_high_pct'] > -10) & 
            (df['ret_30d'] > 0) & 
            (df['ma_aligned'] == 1),
            
            # REVERSAL: Deep pullback + turning positive
            (df['from_high_pct'] < -40) & 
            (df['ret_7d'] > 0) & 
            (df['volume_surge'] == 1),
            
            # MOMENTUM: Strong trend continuation
            (df['from_low_pct'] > 50) & 
            (df['from_high_pct'] > -30) & 
            (df['returns_aligned'] == 1),
            
            # AVOID: Weak structure
            (df['from_high_pct'] < -20) & 
            (df['ret_30d'] < 0) & 
            (df['ma_aligned'] == 0)
        ]
        
        opp_choices = [
            PositionOpportunity.BREAKOUT.value,
            PositionOpportunity.REVERSAL.value,
            PositionOpportunity.MOMENTUM.value,
            PositionOpportunity.AVOID.value
        ]
        
        df['position_opportunity'] = np.select(opp_conditions, opp_choices, default=PositionOpportunity.NEUTRAL.value)
        
        return df
    
    def _calculate_future_potential_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate future potential based on lifecycle positioning"""
        # Base potential by lifecycle stage
        stage_potential = {
            LifecycleStage.ACCUMULATION.value: 90,
            LifecycleStage.EARLY_MARKUP.value: 85,
            LifecycleStage.LATE_MARKUP.value: 40,
            LifecycleStage.DISTRIBUTION.value: 20,
            LifecycleStage.MARKDOWN.value: 10,
            LifecycleStage.RECOVERY.value: 75,
            LifecycleStage.UNKNOWN.value: 50
        }
        
        df['future_potential_score'] = df['lifecycle_stage'].map(stage_potential)
        
        # Adjust based on momentum acceleration
        momentum_accel = df['ret_1d'] > (df['ret_7d'] / 7)
        df.loc[momentum_accel, 'future_potential_score'] += 10
        
        # Adjust based on volume pattern
        vol_positive = df['volume_pattern'].isin([VolumePattern.EXPANDING.value, VolumePattern.SPIKE.value])
        df.loc[vol_positive, 'future_potential_score'] += 5
        
        # Clip to valid range
        df['future_potential_score'] = np.clip(df['future_potential_score'], 0, 100)
        
        return df
    
    def _generate_signals_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on scores"""
        # Signal based on power score
        conditions = [
            df['power_score'] >= config.STRONG_BUY_THRESHOLD,
            df['power_score'] >= config.BUY_THRESHOLD,
            df['power_score'] <= config.STRONG_SELL_THRESHOLD,
            df['power_score'] <= config.SELL_THRESHOLD
        ]
        
        choices = [
            SignalType.STRONG_BUY.value,
            SignalType.BUY.value,
            SignalType.STRONG_SELL.value,
            SignalType.SELL.value
        ]
        
        df['signal'] = np.select(conditions, choices, default=SignalType.HOLD.value)
        
        # Calculate confidence
        df['confidence'] = self._calculate_confidence_vectorized(df)
        
        # Risk management
        df['stop_loss'] = df['price'] * (1 - config.DEFAULT_STOP_LOSS_PCT / 100)
        df['target'] = df['price'] * (1 + config.DEFAULT_TARGET_PCT / 100)
        df['risk_reward_ratio'] = config.DEFAULT_TARGET_PCT / config.DEFAULT_STOP_LOSS_PCT
        
        # Position sizing based on volatility
        df['position_size_pct'] = np.where(
            df['volatility_estimate'] > 5, 2.0,  # High volatility = smaller position
            np.where(df['volatility_estimate'] > 3, 3.0,
            np.where(df['volatility_estimate'] > 2, 4.0, 5.0))
        )
        
        return df
    
    def _calculate_confidence_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """Calculate signal confidence"""
        confidence = pd.Series(50, index=df.index)  # Base confidence
        
        # Volume confirmation
        confidence += np.where(df['rvol'] > 2, 15, 
                     np.where(df['rvol'] > 1.5, 10, 0))
        
        # Momentum consistency
        confidence += np.where(df['returns_aligned'] == 1, 20, 0)
        
        # Position strength
        confidence += np.where(
            (df['from_high_pct'] > -20) & (df['from_low_pct'] > 30), 
            15, 0
        )
        
        return np.clip(confidence, 0, 100)
    
    def _calculate_rankings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various rankings"""
        # Overall ranking
        df['overall_rank'] = df['power_score'].rank(ascending=False, method='min').astype(int)
        
        # Sector ranking
        df['sector_rank'] = df.groupby('sector')['power_score'].rank(
            ascending=False, method='min'
        ).astype(int)
        
        # Percentile within sector
        df['peer_percentile'] = df.groupby('sector')['power_score'].rank(pct=True) * 100
        
        # Category ranking
        df['category_rank'] = df.groupby('category')['power_score'].rank(
            ascending=False, method='min'
        ).astype(int)
        
        return df

# ============================================
# VISUALIZATION ENGINE
# ============================================

class VisualizationEngine:
    """High-performance visualization engine"""
    
    @staticmethod
    @lru_cache(maxsize=32)
    def get_color_scheme() -> Dict[str, str]:
        """Get consistent color scheme"""
        return {
            'primary': '#1e3c72',
            'secondary': '#2a5298',
            'success': '#27ae60',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'info': '#3498db',
            'dark': '#2c3e50',
            'light': '#ecf0f1'
        }
    
    @staticmethod
    def create_signal_distribution_chart(df: pd.DataFrame) -> go.Figure:
        """Create signal distribution pie chart"""
        signal_counts = df['signal'].value_counts()
        
        colors = {
            SignalType.STRONG_BUY.value: '#27ae60',
            SignalType.BUY.value: '#2ecc71',
            SignalType.HOLD.value: '#95a5a6',
            SignalType.SELL.value: '#e67e22',
            SignalType.STRONG_SELL.value: '#e74c3c'
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=signal_counts.index,
            values=signal_counts.values,
            marker=dict(colors=[colors.get(x, '#95a5a6') for x in signal_counts.index]),
            textinfo='label+percent',
            hole=0.4
        )])
        
        fig.update_layout(
            title="Trading Signal Distribution",
            height=400,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_lifecycle_heatmap(df: pd.DataFrame) -> go.Figure:
        """Create lifecycle stage heatmap by sector"""
        if 'lifecycle_stage' not in df.columns:
            return go.Figure()
        
        # Create pivot table
        pivot = pd.crosstab(df['sector'], df['lifecycle_stage'])
        
        # Define stage order
        stage_order = [
            LifecycleStage.ACCUMULATION.value,
            LifecycleStage.EARLY_MARKUP.value,
            LifecycleStage.LATE_MARKUP.value,
            LifecycleStage.DISTRIBUTION.value,
            LifecycleStage.MARKDOWN.value,
            LifecycleStage.RECOVERY.value
        ]
        
        # Reorder columns
        pivot = pivot.reindex(columns=[s for s in stage_order if s in pivot.columns], fill_value=0)
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='Viridis',
            text=pivot.values,
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Lifecycle Stage Distribution by Sector",
            xaxis_title="Lifecycle Stage",
            yaxis_title="Sector",
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_sector_performance_bubble(df: pd.DataFrame) -> go.Figure:
        """Create sector performance bubble chart"""
        sector_stats = df.groupby('sector').agg({
            'power_score': 'mean',
            'momentum_score': 'mean',
            'volume_score': 'mean',
            'ticker': 'count'
        }).reset_index()
        
        # Filter sectors with minimum stocks
        sector_stats = sector_stats[sector_stats['ticker'] >= config.MIN_STOCKS_FOR_SECTOR_ANALYSIS]
        
        fig = px.scatter(
            sector_stats,
            x='momentum_score',
            y='power_score',
            size='ticker',
            color='volume_score',
            hover_data=['ticker'],
            text='sector',
            title='Sector Performance Analysis',
            labels={
                'momentum_score': 'Average Momentum Score',
                'power_score': 'Average Power Score',
                'volume_score': 'Average Volume Score',
                'ticker': 'Number of Stocks'
            },
            color_continuous_scale='Viridis'
        )
        
        fig.update_traces(textposition='top center')
        fig.update_layout(height=500)
        
        return fig
    
    @staticmethod
    def create_risk_reward_scatter(df: pd.DataFrame, limit: int = 100) -> go.Figure:
        """Create risk-reward scatter plot"""
        top_stocks = df.nlargest(limit, 'power_score')
        
        # Calculate potential return and risk
        top_stocks['potential_return'] = (top_stocks['target'] - top_stocks['price']) / top_stocks['price'] * 100
        top_stocks['downside_risk'] = (top_stocks['price'] - top_stocks['stop_loss']) / top_stocks['price'] * 100
        
        fig = px.scatter(
            top_stocks,
            x='downside_risk',
            y='potential_return',
            size='confidence',
            color='signal',
            hover_data=['ticker', 'company_name', 'power_score'],
            title='Risk-Reward Analysis (Top 100)',
            labels={
                'downside_risk': 'Downside Risk (%)',
                'potential_return': 'Potential Return (%)',
                'confidence': 'Confidence'
            },
            color_discrete_map={
                SignalType.STRONG_BUY.value: '#27ae60',
                SignalType.BUY.value: '#2ecc71',
                SignalType.HOLD.value: '#95a5a6',
                SignalType.SELL.value: '#e67e22',
                SignalType.STRONG_SELL.value: '#e74c3c'
            }
        )
        
        # Add diagonal line for 1:1 risk-reward
        max_val = max(top_stocks['downside_risk'].max(), top_stocks['potential_return'].max())
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                showlegend=False,
                name='1:1 Risk-Reward'
            )
        )
        
        fig.update_layout(height=500)
        
        return fig

# ============================================
# EXPORT ENGINE
# ============================================

class ExportEngine:
    """Handle all export operations"""
    
    @staticmethod
    def generate_excel_report(df: pd.DataFrame) -> BytesIO:
        """Generate comprehensive Excel report"""
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#1e3c72',
                'font_color': 'white',
                'border': 1,
                'align': 'center',
                'valign': 'vcenter'
            })
            
            number_format = workbook.add_format({'num_format': '#,##0.00'})
            percent_format = workbook.add_format({'num_format': '0.00%'})
            currency_format = workbook.add_format({'num_format': '₹#,##0.00'})
            
            # Sheet 1: Executive Summary
            summary_data = {
                'Metric': [
                    'Total Stocks Analyzed',
                    'Strong Buy Signals',
                    'Buy Signals',
                    'Average Power Score',
                    'Top Performing Sector',
                    'Market Breadth (% Positive)',
                    'Report Generated'
                ],
                'Value': [
                    len(df),
                    len(df[df['signal'] == SignalType.STRONG_BUY.value]),
                    len(df[df['signal'] == SignalType.BUY.value]),
                    f"{df['power_score'].mean():.2f}",
                    df.groupby('sector')['power_score'].mean().idxmax(),
                    f"{(df['ret_1d'] > 0).mean() * 100:.1f}%",
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
            
            # Sheet 2: Buy Signals
            buy_signals = df[df['signal'].isin([SignalType.BUY.value, SignalType.STRONG_BUY.value])].copy()
            
            if not buy_signals.empty:
                buy_export = pd.DataFrame({
                    'Rank': buy_signals['overall_rank'],
                    'Ticker': buy_signals['ticker'],
                    'Company': buy_signals['company_name'],
                    'Signal': buy_signals['signal'],
                    'Power Score': buy_signals['power_score'].round(1),
                    'Confidence %': buy_signals['confidence'].round(0),
                    'Price': buy_signals['price'],
                    'Stop Loss': buy_signals['stop_loss'],
                    'Target': buy_signals['target'],
                    'Risk/Reward': buy_signals['risk_reward_ratio'].round(2),
                    'Position Size %': buy_signals['position_size_pct'],
                    'Sector': buy_signals['sector'],
                    'Lifecycle': buy_signals.get('lifecycle_stage', 'N/A'),
                    'Volume Pattern': buy_signals.get('volume_pattern', 'N/A'),
                    'Opportunity': buy_signals.get('position_opportunity', 'N/A')
                })
                
                buy_export.to_excel(writer, sheet_name='Buy Signals', index=False)
            
            # Sheet 3: Sector Analysis
            sector_analysis = df.groupby('sector').agg({
                'power_score': ['mean', 'std', 'count'],
                'momentum_score': 'mean',
                'volume_score': 'mean',
                'quality_score': 'mean',
                'signal': lambda x: (x.isin([SignalType.BUY.value, SignalType.STRONG_BUY.value])).sum()
            }).round(2)
            
            sector_analysis.columns = ['Avg Score', 'Std Dev', 'Count', 'Avg Momentum', 
                                      'Avg Volume', 'Avg Quality', 'Buy Signals']
            sector_analysis.to_excel(writer, sheet_name='Sector Analysis')
            
            # Sheet 4: Lifecycle Analysis
            if 'lifecycle_stage' in df.columns:
                lifecycle_analysis = df.pivot_table(
                    index='lifecycle_stage',
                    values=['power_score', 'future_potential_score'],
                    aggfunc=['count', 'mean']
                ).round(2)
                
                lifecycle_analysis.to_excel(writer, sheet_name='Lifecycle Analysis')
            
            # Sheet 5: All Stocks
            all_stocks_export = pd.DataFrame({
                'Rank': df['overall_rank'],
                'Ticker': df['ticker'],
                'Company': df['company_name'],
                'Power Score': df['power_score'].round(1),
                'Signal': df['signal'],
                'Price': df['price'],
                'Change %': df['ret_1d'],
                'Sector': df['sector'],
                'Category': df['category']
            })
            
            all_stocks_export.to_excel(writer, sheet_name='All Stocks', index=False)
            
            # Format all sheets
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                worksheet.freeze_panes(1, 0)
                
                # Auto-adjust column widths
                for i, col in enumerate(pd.read_excel(output, sheet_name=sheet_name).columns):
                    worksheet.set_column(i, i, max(len(str(col)), 12))
        
        output.seek(0)
        return output
    
    @staticmethod
    def generate_trading_signals_csv(df: pd.DataFrame) -> str:
        """Generate CSV for trading signals"""
        signals = df[df['signal'].isin([SignalType.BUY.value, SignalType.STRONG_BUY.value])].copy()
        
        export_df = pd.DataFrame({
            'ticker': signals['ticker'],
            'signal': signals['signal'],
            'price': signals['price'],
            'stop_loss': signals['stop_loss'],
            'target': signals['target'],
            'position_size_pct': signals['position_size_pct'],
            'confidence': signals['confidence']
        })
        
        return export_df.to_csv(index=False)

# ============================================
# UI COMPONENTS
# ============================================

class UIComponents:
    """Reusable UI components with consistent styling"""
    
    @staticmethod
    def render_custom_css():
        """Inject custom CSS for professional UI"""
        st.markdown("""
        <style>
        /* Main container */
        .main {
            padding: 0;
            max-width: 100%;
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 2.5rem;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .main-header h1 {
            margin: 0;
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: -1px;
        }
        
        .main-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1.25rem;
            opacity: 0.9;
            font-weight: 300;
        }
        
        /* Metric cards */
        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            text-align: center;
            height: 100%;
            transition: transform 0.2s;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.12);
        }
        
        .metric-card h3 {
            margin: 0 0 0.75rem 0;
            color: #2c3e50;
            font-size: 0.875rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .metric-card .value {
            font-size: 2.25rem;
            font-weight: 700;
            color: #1e3c72;
            line-height: 1;
        }
        
        .metric-card .delta {
            font-size: 0.875rem;
            margin-top: 0.5rem;
            font-weight: 500;
        }
        
        .delta-positive {
            color: #27ae60;
        }
        
        .delta-negative {
            color: #e74c3c;
        }
        
        /* Signal badges */
        .signal-badge {
            display: inline-block;
            padding: 0.375rem 0.875rem;
            border-radius: 24px;
            font-size: 0.875rem;
            font-weight: 600;
            letter-spacing: 0.25px;
        }
        
        .signal-strong-buy {
            background: #27ae60;
            color: white;
        }
        
        .signal-buy {
            background: #2ecc71;
            color: white;
        }
        
        .signal-hold {
            background: #95a5a6;
            color: white;
        }
        
        .signal-sell {
            background: #e67e22;
            color: white;
        }
        
        .signal-strong-sell {
            background: #e74c3c;
            color: white;
        }
        
        /* Tables */
        .dataframe {
            font-size: 0.875rem;
            border-radius: 8px;
            overflow: hidden;
        }
        
        .dataframe th {
            background: #1e3c72 !important;
            color: white !important;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.5px;
            padding: 0.75rem 1rem !important;
        }
        
        .dataframe td {
            padding: 0.625rem 1rem !important;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            font-weight: 600;
            border-radius: 8px;
            letter-spacing: 0.25px;
            transition: all 0.3s;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(30, 60, 114, 0.3);
        }
        
        /* Sidebar */
        .css-1d391kg {
            background: #f8f9fa;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            background: #f8f9fa;
            padding: 0.5rem;
            border-radius: 12px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            white-space: pre-wrap;
            background: transparent;
            border-radius: 8px;
            color: #2c3e50;
            font-weight: 600;
            padding: 0 1.5rem;
        }
        
        .stTabs [aria-selected="true"] {
            background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        
        /* Info boxes */
        .info-box {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        .success-box {
            background: #e8f5e9;
            border-left: 4px solid #4caf50;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        .warning-box {
            background: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .main-header h1 {
                font-size: 2rem;
            }
            
            .metric-card .value {
                font-size: 1.75rem;
            }
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_header():
        """Render application header"""
        st.markdown("""
        <div class="main-header">
            <h1>🚀 Wave Detection 8.0</h1>
            <p>Ultimate Professional Trading Analytics Platform</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_metrics_row(df: pd.DataFrame):
        """Render key metrics in cards"""
        cols = st.columns(5)
        
        # Total stocks
        with cols[0]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Stocks</h3>
                <div class="value">{len(df):,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Buy signals
        buy_count = len(df[df['signal'].isin([SignalType.BUY.value, SignalType.STRONG_BUY.value])])
        with cols[1]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Buy Signals</h3>
                <div class="value">{buy_count}</div>
                <div class="delta delta-positive">
                    {buy_count / len(df) * 100:.1f}% of market
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Average score
        avg_score = df['power_score'].mean()
        with cols[2]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Avg Power Score</h3>
                <div class="value">{avg_score:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Market breadth
        breadth = (df['ret_1d'] > 0).mean() * 100
        with cols[3]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Market Breadth</h3>
                <div class="value">{breadth:.0f}%</div>
                <div class="delta {'delta-positive' if breadth > 50 else 'delta-negative'}">
                    {'Bullish' if breadth > 50 else 'Bearish'}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Top sector
        if len(df) > 0:
            top_sector = df.groupby('sector')['power_score'].mean().idxmax()
            with cols[4]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Top Sector</h3>
                    <div class="value" style="font-size: 1.5rem;">{top_sector[:12]}</div>
                </div>
                """, unsafe_allow_html=True)
    
    @staticmethod
    def render_signal_table(df: pd.DataFrame, limit: int = 20):
        """Render formatted signal table"""
        if df.empty:
            st.warning("No stocks to display")
            return
        
        # Get top stocks
        display_df = df.nlargest(min(limit, len(df)), 'power_score').copy()
        
        # Format for display
        formatted_df = pd.DataFrame({
            '🏆 Rank': display_df['overall_rank'],
            '📊 Ticker': display_df['ticker'],
            '🏢 Company': display_df['company_name'],
            '📈 Signal': display_df['signal'],
            '💯 Score': display_df['power_score'].round(1),
            '🎯 Confidence': display_df['confidence'].apply(lambda x: f"{x:.0f}%"),
            '💰 Price': display_df['price'].apply(lambda x: f"₹{x:,.2f}"),
            '📊 Change': display_df['ret_1d'].apply(
                lambda x: f"<span style='color: {'green' if x > 0 else 'red'}'>{x:+.2f}%</span>"
            ),
            '🔊 Volume': display_df['rvol'].apply(lambda x: f"{x:.1f}x"),
            '🛡️ Stop Loss': display_df['stop_loss'].apply(lambda x: f"₹{x:,.2f}"),
            '🎯 Target': display_df['target'].apply(lambda x: f"₹{x:,.2f}"),
            '🏭 Sector': display_df['sector']
        })
        
        # Display with custom styling
        st.markdown(
            formatted_df.to_html(escape=False, index=False),
            unsafe_allow_html=True
        )

# ============================================
# MAIN APPLICATION
# ============================================

class WaveDetectionApp:
    """Main application controller"""
    
    def __init__(self):
        self.config = config
        self.data_processor = IndiaNSEDataProcessor()
        self.scoring_engine = AdvancedScoringEngine()
        self.viz_engine = VisualizationEngine()
        self.export_engine = ExportEngine()
        self.ui = UIComponents()
    
    @st.cache_data(ttl=config.CACHE_TTL)
    def load_and_process_data(_self, sheet_url: str, gid: str) -> pd.DataFrame:
        """Load and process data with caching"""
        try:
            # Construct CSV URL
            base_url = sheet_url.split('/edit')[0]
            csv_url = f"{base_url}/export?format=csv&gid={gid}"
            
            logger.info(f"Loading data from: {csv_url}")
            
            # Load data
            df = pd.read_csv(csv_url)
            
            if df.empty:
                raise ValueError("Empty dataframe loaded")
            
            # Process data
            df = _self.data_processor.process(df)
            
            # Calculate scores
            df = _self.scoring_engine.calculate_all_scores(df)
            
            logger.info(f"Successfully processed {len(df)} stocks")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load/process data: {str(e)}")
            raise
    
    def render_sidebar(self) -> Dict[str, Any]:
        """Render sidebar and return filter selections"""
        with st.sidebar:
            st.markdown("### ⚙️ Configuration")
            
            # Data source
            with st.expander("📊 Data Source", expanded=False):
                sheet_url = st.text_input(
                    "Google Sheets URL",
                    value=self.config.DEFAULT_SHEET_URL,
                    help="Enter the public Google Sheets URL"
                )
                
                gid = st.text_input(
                    "Sheet GID",
                    value=self.config.DEFAULT_GID,
                    help="Sheet ID from the URL"
                )
            
            st.markdown("---")
            
            # Display mode
            st.markdown("### 🎯 Display Mode")
            display_mode = st.radio(
                "Select Mode",
                ["📈 Simple", "🚀 Advanced"],
                index=1,
                help="Simple: Core signals only | Advanced: Full analysis"
            )
            
            # Quick filters
            st.markdown("### 🔍 Smart Filters")
            
            # Signal filter
            signal_filter = st.multiselect(
                "Signal Types",
                options=[s.value for s in SignalType],
                default=[SignalType.STRONG_BUY.value, SignalType.BUY.value],
                help="Filter by trading signals"
            )
            
            # Score range
            score_range = st.slider(
                "Power Score Range",
                min_value=0,
                max_value=100,
                value=(50, 100),
                step=5,
                help="Filter by power score"
            )
            
            # Advanced filters (collapsible)
            with st.expander("🔧 Advanced Filters"):
                # Sector filter
                sector_filter = st.multiselect(
                    "Sectors",
                    options=["All"],
                    default=["All"],
                    help="Filter by sector (populated after data loads)"
                )
                
                # Lifecycle filter
                lifecycle_filter = st.multiselect(
                    "Lifecycle Stages",
                    options=[stage.value for stage in LifecycleStage],
                    default=["All"],
                    help="Filter by lifecycle stage"
                )
            
            st.markdown("---")
            
            # Action buttons
            col1, col2 = st.columns(2)
            
            with col1:
                refresh_clicked = st.button(
                    "🔄 Refresh",
                    use_container_width=True,
                    help="Reload data from source"
                )
            
            with col2:
                help_clicked = st.button(
                    "❓ Help",
                    use_container_width=True,
                    help="Show user guide"
                )
            
            if help_clicked:
                st.info("""
                **Quick Guide:**
                
                🚀 **Strong Buy**: Score > 85, High confidence
                ✅ **Buy**: Score > 70, Good setup
                ⏸️ **Hold**: Neutral, wait for clarity
                📉 **Sell**: Score < 30, Consider exit
                
                **Lifecycle Stages:**
                - Accumulation: Smart money entering
                - Early Markup: Beginning of trend
                - Distribution: Smart money exiting
                - Markdown: Downtrend phase
                """)
            
            return {
                'sheet_url': sheet_url,
                'gid': gid,
                'display_mode': display_mode,
                'signal_filter': signal_filter,
                'score_range': score_range,
                'sector_filter': sector_filter,
                'lifecycle_filter': lifecycle_filter,
                'refresh_clicked': refresh_clicked
            }
    
    def apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to dataframe"""
        filtered_df = df.copy()
        
        # Signal filter
        if filters['signal_filter']:
            filtered_df = filtered_df[filtered_df['signal'].isin(filters['signal_filter'])]
        
        # Score range
        filtered_df = filtered_df[
            (filtered_df['power_score'] >= filters['score_range'][0]) &
            (filtered_df['power_score'] <= filters['score_range'][1])
        ]
        
        # Sector filter
        if 'All' not in filters['sector_filter'] and filters['sector_filter']:
            filtered_df = filtered_df[filtered_df['sector'].isin(filters['sector_filter'])]
        
        # Lifecycle filter
        if 'All' not in filters['lifecycle_filter'] and filters['lifecycle_filter']:
            if 'lifecycle_stage' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['lifecycle_stage'].isin(filters['lifecycle_filter'])]
        
        return filtered_df
    
    def render_simple_mode(self, df: pd.DataFrame):
        """Render simple mode interface"""
        # Top signals
        st.markdown("### 🎯 Top Trading Signals")
        
        # Quick stats
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            display_count = st.selectbox(
                "Show top",
                [10, 20, 50, 100],
                index=1
            )
        
        with col2:
            sort_by = st.selectbox(
                "Sort by",
                ["Power Score", "Momentum", "Volume", "Confidence"],
                index=0
            )
        
        # Display table
        sort_column = {
            "Power Score": "power_score",
            "Momentum": "momentum_score",
            "Volume": "volume_score",
            "Confidence": "confidence"
        }[sort_by]
        
        top_stocks = df.nlargest(display_count, sort_column)
        
        # Simple table view
        simple_df = pd.DataFrame({
            'Rank': top_stocks['overall_rank'],
            'Ticker': top_stocks['ticker'],
            'Company': top_stocks['company_name'],
            'Signal': top_stocks['signal'],
            'Score': top_stocks['power_score'].round(1),
            'Price': top_stocks['price'].apply(lambda x: f"₹{x:,.2f}"),
            'Target': top_stocks['target'].apply(lambda x: f"₹{x:,.2f}"),
            'Stop Loss': top_stocks['stop_loss'].apply(lambda x: f"₹{x:,.2f}")
        })
        
        st.dataframe(simple_df, use_container_width=True, height=600)
        
        # Key insights
        st.markdown("### 💡 Quick Insights")
        
        col1, col2, col3 = st.columns(3)
        
        strong_buys = df[df['signal'] == SignalType.STRONG_BUY.value]
        high_confidence = df[df['confidence'] > 80]
        high_volume = df[df['rvol'] > 2]
        
        with col1:
            if len(strong_buys) > 0:
                st.success(f"🚀 {len(strong_buys)} Strong Buy signals found!")
                st.caption(f"Top pick: **{strong_buys.iloc[0]['ticker']}**")
        
        with col2:
            if len(high_confidence) > 0:
                st.info(f"🎯 {len(high_confidence)} high confidence trades")
                st.caption("Confidence > 80%")
        
        with col3:
            if len(high_volume) > 0:
                st.warning(f"🔊 {len(high_volume)} stocks with volume surge")
                st.caption("Volume > 2x average")
    
    def render_advanced_mode(self, df: pd.DataFrame):
        """Render advanced mode interface with full analysis"""
        # Create tabs
        tabs = st.tabs([
            "📊 Signals",
            "📈 Analysis",
            "🔍 Patterns",
            "⚖️ Risk Analysis",
            "📥 Export"
        ])
        
        # Tab 1: Signals
        with tabs[0]:
            st.markdown("### 🎯 Advanced Trading Signals")
            
            # Display options
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                display_count = st.selectbox(
                    "Display count",
                    [20, 50, 100],
                    key="adv_display_count"
                )
            
            with col2:
                show_lifecycle = st.checkbox(
                    "Show Lifecycle",
                    value=True,
                    help="Display lifecycle stage analysis"
                )
            
            with col3:
                show_patterns = st.checkbox(
                    "Show Patterns",
                    value=True,
                    help="Display volume and position patterns"
                )
            
            # Advanced table
            display_cols = [
                'overall_rank', 'ticker', 'company_name', 'signal',
                'power_score', 'confidence', 'price', 'ret_1d', 'rvol'
            ]
            
            if show_lifecycle and 'lifecycle_stage' in df.columns:
                display_cols.extend(['lifecycle_stage', 'future_potential_score'])
            
            if show_patterns:
                if 'volume_pattern' in df.columns:
                    display_cols.append('volume_pattern')
                if 'position_opportunity' in df.columns:
                    display_cols.append('position_opportunity')
            
            display_cols.extend(['stop_loss', 'target', 'sector'])
            
            # Get unique columns
            display_cols = [col for col in display_cols if col in df.columns]
            
            top_stocks = df.nlargest(display_count, 'power_score')[display_cols].copy()
            
            # Format numeric columns
            for col in ['power_score', 'confidence', 'future_potential_score']:
                if col in top_stocks.columns:
                    top_stocks[col] = top_stocks[col].round(1)
            
            for col in ['price', 'stop_loss', 'target']:
                if col in top_stocks.columns:
                    top_stocks[col] = top_stocks[col].apply(lambda x: f"₹{x:,.2f}")
            
            if 'ret_1d' in top_stocks.columns:
                top_stocks['ret_1d'] = top_stocks['ret_1d'].apply(lambda x: f"{x:+.2f}%")
            
            if 'rvol' in top_stocks.columns:
                top_stocks['rvol'] = top_stocks['rvol'].apply(lambda x: f"{x:.1f}x")
            
            st.dataframe(top_stocks, use_container_width=True, height=600)
        
        # Tab 2: Analysis
        with tabs[1]:
            st.markdown("### 📊 Market Analysis")
            
            # Signal distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig_signals = self.viz_engine.create_signal_distribution_chart(df)
                st.plotly_chart(fig_signals, use_container_width=True)
            
            with col2:
                fig_sectors = self.viz_engine.create_sector_performance_bubble(df)
                st.plotly_chart(fig_sectors, use_container_width=True)
            
            # Lifecycle heatmap
            if 'lifecycle_stage' in df.columns:
                st.markdown("### 🔄 Lifecycle Analysis")
                fig_lifecycle = self.viz_engine.create_lifecycle_heatmap(df)
                st.plotly_chart(fig_lifecycle, use_container_width=True)
        
        # Tab 3: Patterns
        with tabs[2]:
            st.markdown("### 🔍 Pattern Analysis")
            
            if 'volume_pattern' in df.columns and 'position_opportunity' in df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Volume patterns
                    vol_dist = df['volume_pattern'].value_counts()
                    fig_vol = px.bar(
                        x=vol_dist.index,
                        y=vol_dist.values,
                        title="Volume Pattern Distribution",
                        labels={'x': 'Pattern', 'y': 'Count'}
                    )
                    st.plotly_chart(fig_vol, use_container_width=True)
                
                with col2:
                    # Position opportunities
                    pos_dist = df['position_opportunity'].value_counts()
                    fig_pos = px.pie(
                        values=pos_dist.values,
                        names=pos_dist.index,
                        title="Position Opportunity Distribution"
                    )
                    st.plotly_chart(fig_pos, use_container_width=True)
            
            # Pattern combinations
            st.markdown("### 🎯 High Probability Setups")
            
            # Find best pattern combinations
            if all(col in df.columns for col in ['lifecycle_stage', 'volume_pattern', 'position_opportunity']):
                high_prob = df[
                    (df['power_score'] > 80) &
                    (df['confidence'] > 70)
                ]
                
                if not high_prob.empty:
                    pattern_summary = high_prob.groupby(
                        ['lifecycle_stage', 'volume_pattern', 'position_opportunity']
                    ).agg({
                        'ticker': 'count',
                        'power_score': 'mean',
                        'confidence': 'mean'
                    }).round(1)
                    
                    pattern_summary.columns = ['Count', 'Avg Score', 'Avg Confidence']
                    pattern_summary = pattern_summary.sort_values('Count', ascending=False).head(10)
                    
                    st.dataframe(pattern_summary, use_container_width=True)
        
        # Tab 4: Risk Analysis
        with tabs[3]:
            st.markdown("### ⚖️ Risk-Reward Analysis")
            
            # Risk-reward scatter
            fig_risk = self.viz_engine.create_risk_reward_scatter(df)
            st.plotly_chart(fig_risk, use_container_width=True)
            
            # Position sizing recommendations
            st.markdown("### 💼 Position Sizing Guide")
            
            position_guide = pd.DataFrame({
                'Volatility Level': ['Low (<2%)', 'Medium (2-3%)', 'High (3-5%)', 'Very High (>5%)'],
                'Recommended Position Size': ['5% of capital', '4% of capital', '3% of capital', '2% of capital'],
                'Stop Loss': ['5%', '5%', '5%', '5%'],
                'Risk per Trade': ['0.25%', '0.20%', '0.15%', '0.10%']
            })
            
            st.dataframe(position_guide, use_container_width=True)
        
        # Tab 5: Export
        with tabs[4]:
            st.markdown("### 📥 Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Excel Report")
                st.markdown("""
                Comprehensive Excel report includes:
                - Executive summary
                - Buy signals with targets
                - Sector analysis
                - Lifecycle analysis
                - All stocks ranked
                """)
                
                if st.button("Generate Excel Report", use_container_width=True):
                    with st.spinner("Generating report..."):
                        excel_file = self.export_engine.generate_excel_report(df)
                    
                    st.download_button(
                        label="📥 Download Excel Report",
                        data=excel_file,
                        file_name=f"wave_detection_ultimate_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
            
            with col2:
                st.markdown("#### 📄 Trading Signals CSV")
                st.markdown("""
                Quick CSV export includes:
                - Buy/Strong Buy signals only
                - Entry, stop loss, target prices
                - Position sizing
                - Ready for broker upload
                """)
                
                if st.button("Generate CSV", use_container_width=True):
                    csv_data = self.export_engine.generate_trading_signals_csv(df)
                    
                    st.download_button(
                        label="📥 Download Trading Signals",
                        data=csv_data,
                        file_name=f"trading_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
    
    def run(self):
        """Main application entry point"""
        try:
            # Inject CSS
            self.ui.render_custom_css()
            
            # Render header
            self.ui.render_header()
            
            # Get sidebar inputs
            filters = self.render_sidebar()
            
            # Handle refresh
            if filters['refresh_clicked']:
                st.cache_data.clear()
                st.rerun()
            
            # Load and process data
            try:
                with st.spinner("🔄 Loading and analyzing market data..."):
                    df = self.load_and_process_data(
                        filters['sheet_url'],
                        filters['gid']
                    )
                
                # Update sector filter options
                if 'sector' in df.columns:
                    unique_sectors = ['All'] + sorted(df['sector'].unique().tolist())
                    # Note: Can't dynamically update multiselect options in current Streamlit
                
            except Exception as e:
                st.error(f"❌ Failed to load data: {str(e)}")
                st.info("Please check your data source and try again.")
                st.stop()
            
            # Apply filters
            filtered_df = self.apply_filters(df, filters)
            
            if filtered_df.empty:
                st.warning("No stocks match the selected filters. Try adjusting your criteria.")
                st.stop()
            
            # Display metrics
            self.ui.render_metrics_row(filtered_df)
            
            # Render based on display mode
            if filters['display_mode'] == "📈 Simple":
                self.render_simple_mode(filtered_df)
            else:
                self.render_advanced_mode(filtered_df)
            
            # Footer
            st.markdown("---")
            st.markdown(
                f"""
                <div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
                    <p><strong>Wave Detection 8.0</strong> - Ultimate Trading Analytics Platform</p>
                    <p>Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
                    Data refreshes every {self.config.CACHE_TTL // 60} minutes</p>
                    <p style='font-size: 0.875rem; margin-top: 1rem;'>
                        Built with ❤️ for professional traders | 
                        <a href='#' style='color: #3498db;'>Documentation</a> | 
                        <a href='#' style='color: #3498db;'>Support</a>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        except Exception as e:
            logger.error(f"Application error: {str(e)}")
            st.error("An unexpected error occurred. Please refresh the page.")
            if st.checkbox("Show error details"):
                st.exception(e)

# ============================================
# ENTRY POINT
# ============================================

def main():
    """Application entry point"""
    app = WaveDetectionApp()
    app.run()

if __name__ == "__main__":
    main()
