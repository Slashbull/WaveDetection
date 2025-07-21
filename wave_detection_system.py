"""
Wave Detection Ultimate 3.0 - FINAL PRODUCTION VERSION (FIXED)
=============================================================
Professional Stock Ranking System with Advanced Analytics
All syntax errors fixed, fully optimized, production-ready.

Version: 3.0.1-FINAL
Last Updated: December 2024
Status: PERMANENT RELEASE - Syntax fix applied
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
from functools import lru_cache, wraps
import time
from io import BytesIO
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================
# LOGGING CONFIGURATION
# ============================================

# Configure logging for production
log_level = logging.INFO  # Change to DEBUG for troubleshooting

logging.basicConfig(
    level=log_level,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================

@dataclass(frozen=True)
class Config:
    """System configuration with validated weights"""
    
    # Data source
    DEFAULT_SHEET_URL: str = "https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk/edit?usp=sharing"
    DEFAULT_GID: str = "2026492216"
    
    # Cache settings
    CACHE_TTL: int = 300  # 5 minutes
    
    # Master Score 3.0 weights (total = 100%)
    POSITION_WEIGHT: float = 0.30
    VOLUME_WEIGHT: float = 0.25
    MOMENTUM_WEIGHT: float = 0.15
    ACCELERATION_WEIGHT: float = 0.10
    BREAKOUT_WEIGHT: float = 0.10
    RVOL_WEIGHT: float = 0.10
    
    # Display settings
    DEFAULT_TOP_N: int = 50
    AVAILABLE_TOP_N: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200, 500])
    
    # Thresholds for patterns
    PATTERN_THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        "category_leader": 90,      # Top 10% in category
        "hidden_gem": 80,           # Top 20% in category, <70% market
        "acceleration": 85,         # Top 15% acceleration
        "institutional": 75,        # Top 25% volume patterns
        "vol_explosion": 95,        # Top 5% RVOL
        "breakout_ready": 80,       # Top 20% breakout probability
        "market_leader": 95,        # Top 5% overall
        "momentum_wave": 75,        # Top 25% consistent momentum
        "liquid_leader": 80,        # Top 20% liquidity + performance
        "long_strength": 80         # Top 20% long-term performance
    })
    
    # Tier definitions
    TIERS: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=lambda: {
        "eps": {
            "Loss": (-float('inf'), 0),
            "0-5": (0, 5),
            "5-10": (5, 10),
            "10-20": (10, 20),
            "20-50": (20, 50),
            "50-100": (50, 100),
            "100+": (100, float('inf'))
        },
        "pe": {
            "Negative/NA": (-float('inf'), 0),
            "0-10": (0, 10),
            "10-15": (10, 15),
            "15-20": (15, 20),
            "20-30": (20, 30),
            "30-50": (30, 50),
            "50+": (50, float('inf'))
        },
        "price": {
            "0-100": (0, 100),
            "100-250": (100, 250),
            "250-500": (250, 500),
            "500-1000": (500, 1000),
            "1000-2500": (1000, 2500),
            "2500-5000": (2500, 5000),
            "5000+": (5000, float('inf'))
        }
    })

# Global configuration instance
CONFIG = Config()

# ============================================
# PERFORMANCE MONITORING
# ============================================

def timer(func):
    """Performance timing decorator with logging"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            if elapsed > 1.0:
                logger.warning(f"{func.__name__} took {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {str(e)}")
            raise
    return wrapper

# ============================================
# DATA VALIDATION
# ============================================

class DataValidator:
    """Validate data at each step"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_cols: List[str], context: str) -> bool:
        """Validate dataframe has required columns and data"""
        if df is None or df.empty:
            logger.error(f"{context}: Empty or None dataframe")
            return False
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"{context}: Missing required columns: {missing_cols}")
            # Don't fail, just warn - we'll handle missing columns later
        
        # Log available columns
        logger.info(f"{context}: Found {len(df.columns)} columns, {len(df)} rows")
        
        return True
    
    @staticmethod
    def validate_numeric_column(series: pd.Series, col_name: str, 
                              min_val: Optional[float] = None, 
                              max_val: Optional[float] = None) -> pd.Series:
        """Validate and clean numeric column"""
        if series is None:
            return pd.Series(dtype=float)
        
        # Convert to numeric, coercing errors
        series = pd.to_numeric(series, errors='coerce')
        
        # Apply bounds if specified
        if min_val is not None:
            series = series.clip(lower=min_val)
        if max_val is not None:
            series = series.clip(upper=max_val)
        
        # Log if too many NaN values
        nan_pct = series.isna().sum() / len(series) * 100
        if nan_pct > 50:
            logger.warning(f"{col_name}: {nan_pct:.1f}% NaN values")
        
        return series

# ============================================
# DATA LOADING
# ============================================

@st.cache_data(ttl=CONFIG.CACHE_TTL)
def load_google_sheets_data(sheet_url: str, gid: str) -> pd.DataFrame:
    """Load data from Google Sheets with comprehensive error handling"""
    try:
        # Validate inputs
        if not sheet_url or not gid:
            raise ValueError("Sheet URL and GID are required")
        
        # Construct CSV URL
        base_url = sheet_url.split('/edit')[0]
        csv_url = f"{base_url}/export?format=csv&gid={gid}"
        
        logger.info(f"Loading data from Google Sheets")
        
        # Load with timeout and error handling
        df = pd.read_csv(csv_url, low_memory=False)
        
        if df.empty:
            raise ValueError("Loaded empty dataframe")
        
        logger.info(f"Successfully loaded {len(df):,} rows with {len(df.columns)} columns")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise

# ============================================
# DATA PROCESSING
# ============================================

class DataProcessor:
    """Handle all data processing with validation and error handling"""
    
    # Define all expected columns
    NUMERIC_COLUMNS = [
        'price', 'prev_close', 'low_52w', 'high_52w',
        'from_low_pct', 'from_high_pct',
        'sma_20d', 'sma_50d', 'sma_200d',
        'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 
        'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
        'volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d',
        'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
        'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 
        'vol_ratio_90d_180d',
        'rvol', 'pe', 'eps_current', 'eps_last_qtr', 'eps_change_pct'
    ]
    
    CATEGORICAL_COLUMNS = ['ticker', 'company_name', 'category', 'sector']
    
    REQUIRED_COLUMNS = ['ticker', 'price']  # Minimal required columns
    
    @staticmethod
    def clean_numeric_value(value: Any) -> Optional[float]:
        """Clean and convert Indian number format to float"""
        if pd.isna(value) or value == '':
            return np.nan
        
        try:
            # Convert to string and clean
            cleaned = str(value).strip()
            
            # Remove currency symbols and special characters
            for char in ['₹', '$', '%', ',', ' ']:
                cleaned = cleaned.replace(char, '')
            
            # Handle special cases
            if cleaned in ['', '-', 'N/A', 'n/a', '#N/A', 'nan', 'None', '#VALUE!', '#ERROR!']:
                return np.nan
            
            return float(cleaned)
        except (ValueError, AttributeError):
            return np.nan
    
    @staticmethod
    @timer
    def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Complete data processing pipeline with validation"""
        if not DataValidator.validate_dataframe(df, DataProcessor.REQUIRED_COLUMNS, "Initial data"):
            return pd.DataFrame()
        
        # Create copy to avoid modifying original
        df = df.copy()
        
        # Log available columns
        logger.info(f"Available columns: {', '.join(sorted(df.columns))}")
        
        # Process numeric columns
        for col in DataProcessor.NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = df[col].apply(DataProcessor.clean_numeric_value)
                df[col] = DataValidator.validate_numeric_column(df[col], col)
        
        # Process categorical columns
        for col in DataProcessor.CATEGORICAL_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(['nan', 'None', '', 'N/A', 'NaN'], 'Unknown')
                # Remove extra whitespace
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
        
        # Fix volume ratios (ensure they're multipliers, not percentages)
        volume_ratio_columns = [
            'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
            'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
            'vol_ratio_90d_180d'
        ]
        
        for col in volume_ratio_columns:
            if col in df.columns:
                # First ensure numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Check if values look like percentages (common pattern)
                non_nan_values = df[col].dropna()
                if len(non_nan_values) > 0:
                    median_val = non_nan_values.median()
                    
                    # If median > 10, likely percentages
                    if median_val > 10:
                        logger.info(f"Converting {col} from percentage to ratio (median: {median_val:.1f})")
                        df[col] = (df[col] + 100) / 100
                    
                    # If median is negative (like -50%), convert properly
                    elif median_val < -10:
                        logger.info(f"Converting {col} from negative percentage (median: {median_val:.1f})")
                        df[col] = (100 + df[col]) / 100
                
                # Ensure all values are positive
                df[col] = df[col].abs()
                # Fill NaN with 1.0 (no change)
                df[col] = df[col].fillna(1.0)
                # Clip extreme values
                df[col] = df[col].clip(0.1, 10.0)
        
        # Validate data quality
        initial_count = len(df)
        
        # Remove rows with critical missing data
        df = df.dropna(subset=['ticker', 'price'], how='any')
        df = df[df['price'] > 0]
        
        # For position data, fill NaN with reasonable defaults
        if 'from_low_pct' in df.columns:
            # Count non-NaN values before filling
            valid_from_low = df['from_low_pct'].notna().sum()
            logger.info(f"Valid from_low_pct values: {valid_from_low}")
            
            df['from_low_pct'] = df['from_low_pct'].fillna(50)  # Assume middle of range
        else:
            df['from_low_pct'] = 50
            
        if 'from_high_pct' in df.columns:
            # Count non-NaN values before filling
            valid_from_high = df['from_high_pct'].notna().sum()
            logger.info(f"Valid from_high_pct values: {valid_from_high}")
            
            df['from_high_pct'] = df['from_high_pct'].fillna(-50)  # Assume middle of range
        else:
            df['from_high_pct'] = -50
        
        # Ensure we have at least one position metric
        has_position_data = (df['from_low_pct'].notna() & (df['from_low_pct'] != 50)) | \
                           (df['from_high_pct'].notna() & (df['from_high_pct'] != -50))
        
        # Keep stocks with any valid position data OR good return data
        if has_position_data.any():
            # Keep stocks with position data or significant return data
            keep_mask = has_position_data
            if 'ret_30d' in df.columns:
                # Also keep stocks with significant returns even if position data is default
                significant_returns = df['ret_30d'].notna() & (df['ret_30d'].abs() > 5)
                keep_mask = keep_mask | significant_returns
            
            # Only apply filter if it doesn't remove too many stocks
            filtered_count = keep_mask.sum()
            if filtered_count >= min(100, len(df) * 0.1):  # Keep at least 10% or 100 stocks
                df = df[keep_mask]
                logger.info(f"Kept {filtered_count} stocks with valid position/return data")
            else:
                logger.warning(f"Filter would keep only {filtered_count} stocks, skipping filter")
        
        # Remove duplicate tickers (keep first)
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['ticker'], keep='first')
        if before_dedup > len(df):
            logger.info(f"Removed {before_dedup - len(df)} duplicate tickers")
        
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} invalid/duplicate rows")
        
        # Add tier classifications
        df = DataProcessor._add_tier_classifications(df)
        
        # Ensure RVOL exists and is valid
        if 'rvol' not in df.columns:
            df['rvol'] = 1.0
        else:
            df['rvol'] = pd.to_numeric(df['rvol'], errors='coerce')
            df['rvol'] = df['rvol'].fillna(1.0).clip(lower=0.01, upper=100)
        
        # Verify data quality
        logger.info(f"Data quality check:")
        logger.info(f"  - Stocks with valid price: {df['price'].notna().sum()}")
        logger.info(f"  - Stocks with from_low data: {df['from_low_pct'].notna().sum()}")
        logger.info(f"  - Stocks with ret_30d data: {df['ret_30d'].notna().sum() if 'ret_30d' in df.columns else 0}")
        logger.info(f"  - Stocks with RVOL data: {(df['rvol'] != 1.0).sum()}")
        
        logger.info(f"Processed {len(df)} valid stocks")
        return df
    
    @staticmethod
    def _add_tier_classifications(df: pd.DataFrame) -> pd.DataFrame:
        """Add tier classifications for EPS, PE, and Price"""
        
        def classify_tier(value: float, tier_dict: Dict[str, Tuple[float, float]]) -> str:
            """Classify a value into appropriate tier"""
            if pd.isna(value):
                return "Unknown"
            
            for tier_name, (min_val, max_val) in tier_dict.items():
                if min_val < value <= max_val:
                    return tier_name
            return "Unknown"
        
        # Add tier columns
        df['eps_tier'] = df['eps_current'].apply(
            lambda x: classify_tier(x, CONFIG.TIERS['eps'])
        )
        
        df['pe_tier'] = df['pe'].apply(
            lambda x: "Negative/NA" if pd.isna(x) or x <= 0 
            else classify_tier(x, CONFIG.TIERS['pe'])
        )
        
        df['price_tier'] = df['price'].apply(
            lambda x: classify_tier(x, CONFIG.TIERS['price'])
        )
        
        return df

# ============================================
# RANKING ENGINE - COMPLETELY REFACTORED
# ============================================

class RankingEngine:
    """Core ranking calculations - optimized and vectorized"""
    
    @staticmethod
    def safe_rank(series: pd.Series, pct: bool = True, ascending: bool = True) -> pd.Series:
        """Safely rank a series with proper handling of edge cases"""
        if series is None or series.empty:
            return pd.Series(dtype=float)
        
        # Create a copy to avoid modifying original
        series = series.copy()
        
        # Replace inf values with NaN
        series = series.replace([np.inf, -np.inf], np.nan)
        
        # Count valid values
        valid_count = series.notna().sum()
        if valid_count == 0:
            return pd.Series(50, index=series.index)  # All NaN, return middle value
        
        # For percentage ranks, ensure 0-100 scale
        if pct:
            ranks = series.rank(pct=True, ascending=ascending, na_option='bottom')
            ranks = ranks * 100
        else:
            ranks = series.rank(ascending=ascending, method='min', na_option='bottom')
        
        # For NaN values, assign worst rank
        if pct:
            ranks = ranks.fillna(0 if ascending else 100)
        else:
            ranks = ranks.fillna(valid_count + 1)
        
        return ranks
    
    @staticmethod
    def calculate_position_score(df: pd.DataFrame) -> pd.Series:
        """Calculate position score from 52-week range"""
        # Initialize with neutral score
        position_score = pd.Series(50, index=df.index, dtype=float)
        
        # Check if we have the required columns
        has_from_low = 'from_low_pct' in df.columns and df['from_low_pct'].notna().any()
        has_from_high = 'from_high_pct' in df.columns and df['from_high_pct'].notna().any()
        
        if not has_from_low and not has_from_high:
            logger.warning("No position data available, using neutral position scores")
            # Add small variation to avoid identical scores
            return position_score + np.random.uniform(-5, 5, size=len(df))
        
        # Get data with reasonable defaults
        from_low = df['from_low_pct'].fillna(50) if has_from_low else pd.Series(50, index=df.index)
        from_high = df['from_high_pct'].fillna(-50) if has_from_high else pd.Series(-50, index=df.index)
        
        # Rank distance from low (higher % from low is better)
        if has_from_low:
            rank_from_low = RankingEngine.safe_rank(from_low, pct=True, ascending=True)
        else:
            rank_from_low = pd.Series(50, index=df.index)
        
        # Rank distance from high
        if has_from_high:
            # Convert negative values: -20% becomes 80, -50% becomes 50, etc.
            distance_from_high = 100 + from_high  # -20 becomes 80
            rank_from_high = RankingEngine.safe_rank(distance_from_high, pct=True, ascending=True)
        else:
            rank_from_high = pd.Series(50, index=df.index)
        
        # Combined position score
        if has_from_low and has_from_high:
            position_score = (rank_from_low * 0.6 + rank_from_high * 0.4)
        elif has_from_low:
            position_score = rank_from_low
        else:
            position_score = rank_from_high
        
        return position_score.clip(0, 100)
    
    @staticmethod
    def calculate_volume_score(df: pd.DataFrame) -> pd.Series:
        """Calculate comprehensive volume score using all available ratios"""
        # Initialize with neutral score
        volume_score = pd.Series(50, index=df.index, dtype=float)
        
        # Get available volume ratio columns and their weights
        vol_cols = [
            ('vol_ratio_1d_90d', 0.20),    # Short-term explosion
            ('vol_ratio_7d_90d', 0.20),    # Week momentum
            ('vol_ratio_30d_90d', 0.20),   # Month momentum
            ('vol_ratio_30d_180d', 0.15),  # Medium-term
            ('vol_ratio_90d_180d', 0.25)   # Long-term institutional
        ]
        
        # Calculate weighted volume score
        total_weight = 0
        weighted_score = pd.Series(0, index=df.index, dtype=float)
        has_any_vol_data = False
        
        for col, weight in vol_cols:
            if col in df.columns and df[col].notna().any():
                has_any_vol_data = True
                # Get column data
                col_data = df[col].copy()
                
                # Fill NaN with 1.0 (neutral ratio)
                col_data = col_data.fillna(1.0)
                
                # Ensure positive values
                col_data = col_data.clip(lower=0.1)
                
                # Rank the ratios (higher ratio = better rank)
                col_rank = RankingEngine.safe_rank(col_data, pct=True, ascending=True)
                
                # Add to weighted score
                weighted_score += col_rank * weight
                total_weight += weight
        
        # Calculate final score
        if total_weight > 0 and has_any_vol_data:
            volume_score = weighted_score / total_weight
        else:
            # No volume data available, use neutral score with some randomness
            # to avoid all stocks having the same score
            logger.warning("No volume ratio data available, using neutral scores")
            volume_score = pd.Series(50, index=df.index, dtype=float)
            # Add small random variation to break ties
            volume_score += np.random.uniform(-5, 5, size=len(df))
        
        return volume_score.clip(0, 100)
    
    @staticmethod
    def calculate_momentum_score(df: pd.DataFrame) -> pd.Series:
        """Calculate momentum score based on returns"""
        # Initialize with neutral score
        momentum_score = pd.Series(50, index=df.index, dtype=float)
        
        if 'ret_30d' not in df.columns or df['ret_30d'].notna().sum() == 0:
            logger.warning("No 30-day return data available, using neutral momentum scores")
            # Try to use alternative return data
            if 'ret_7d' in df.columns and df['ret_7d'].notna().any():
                # Use 7-day returns as proxy
                ret_7d = df['ret_7d'].fillna(0)
                momentum_score = RankingEngine.safe_rank(ret_7d, pct=True, ascending=True)
                logger.info("Using 7-day returns for momentum score")
            else:
                # Add small variation to avoid identical scores
                momentum_score += np.random.uniform(-5, 5, size=len(df))
            
            return momentum_score.clip(0, 100)
        
        # Get 30-day returns
        ret_30d = df['ret_30d'].fillna(0)
        
        # Primary momentum from 30-day returns ranking
        momentum_score = RankingEngine.safe_rank(ret_30d, pct=True, ascending=True)
        
        # Bonus for consistent positive momentum across timeframes
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            # Initialize bonus
            consistency_bonus = pd.Series(0, index=df.index, dtype=float)
            
            # All positive returns get 5% bonus
            all_positive = (df['ret_7d'] > 0) & (df['ret_30d'] > 0)
            consistency_bonus[all_positive] = 5
            
            # Accelerating momentum gets 10% bonus
            # (7-day return rate exceeds 30-day average rate)
            daily_ret_7d = df['ret_7d'] / 7
            daily_ret_30d = df['ret_30d'] / 30
            accelerating = all_positive & (daily_ret_7d > daily_ret_30d)
            consistency_bonus[accelerating] = 10
            
            # Add bonus to momentum score
            momentum_score = (momentum_score + consistency_bonus).clip(0, 100)
        
        return momentum_score
    
    @staticmethod
    def calculate_acceleration_score(df: pd.DataFrame) -> pd.Series:
        """Calculate if momentum is accelerating using vectorized operations"""
        # Initialize with neutral score
        acceleration_score = pd.Series(50, index=df.index, dtype=float)
        
        # Check if we have required columns
        req_cols = ['ret_1d', 'ret_7d', 'ret_30d']
        available_cols = [col for col in req_cols if col in df.columns]
        
        if len(available_cols) < 2:
            # Not enough data, return neutral scores with small variation
            logger.warning("Insufficient return data for acceleration calculation")
            return acceleration_score + np.random.uniform(-5, 5, size=len(df))
        
        # Get returns data with proper defaults
        ret_1d = df['ret_1d'].fillna(0) if 'ret_1d' in df.columns else pd.Series(0, index=df.index)
        ret_7d = df['ret_7d'].fillna(0) if 'ret_7d' in df.columns else pd.Series(0, index=df.index)
        ret_30d = df['ret_30d'].fillna(0) if 'ret_30d' in df.columns else pd.Series(0, index=df.index)
        
        # Calculate daily averages
        daily_avg_7d = ret_7d / 7
        daily_avg_30d = ret_30d / 30
        
        # Perfect acceleration: returns increasing from 30d -> 7d -> 1d
        if all(col in df.columns for col in req_cols):
            perfect = (ret_1d > daily_avg_7d) & (daily_avg_7d > daily_avg_30d) & (ret_1d > 0)
            acceleration_score.loc[perfect] = 100
            
            # Good acceleration: today better than 7d average and positive
            good = (~perfect) & (ret_1d > daily_avg_7d) & (ret_1d > 0)
            acceleration_score.loc[good] = 80
            
            # Moderate: positive returns but not accelerating
            moderate = (~perfect) & (~good) & (ret_1d > 0)
            acceleration_score.loc[moderate] = 60
            
            # Slight deceleration: negative today but positive week
            slight_decel = (ret_1d <= 0) & (ret_7d > 0)
            acceleration_score.loc[slight_decel] = 40
            
            # Strong deceleration: negative returns across timeframes
            strong_decel = (ret_1d <= 0) & (ret_7d <= 0)
            acceleration_score.loc[strong_decel] = 20
        else:
            # Simplified calculation with available data
            if 'ret_1d' in df.columns and 'ret_7d' in df.columns:
                # Accelerating if 1d > 7d average
                accelerating = ret_1d > daily_avg_7d
                acceleration_score.loc[accelerating & (ret_1d > 0)] = 75
                acceleration_score.loc[~accelerating & (ret_1d > 0)] = 55
                acceleration_score.loc[ret_1d <= 0] = 35
        
        return acceleration_score
    
    @staticmethod
    def calculate_breakout_score(df: pd.DataFrame) -> pd.Series:
        """Calculate breakout probability using multiple factors"""
        breakout_score = pd.Series(50, index=df.index, dtype=float)
        
        # Factor 1: Distance from high (40% weight)
        if 'from_high_pct' in df.columns:
            # from_high_pct is negative, so -20% means 20% below high
            # Convert to positive scale: closer to high = higher score
            distance_from_high = 100 + df['from_high_pct'].fillna(-50)  # -20 becomes 80
            distance_factor = distance_from_high.clip(0, 100)
        else:
            distance_factor = pd.Series(50, index=df.index)
        
        # Factor 2: Volume surge (40% weight)
        volume_factor = pd.Series(50, index=df.index)
        if 'vol_ratio_7d_90d' in df.columns:
            # Convert ratio to score
            # 1.0 = 50 score, 1.5 = 75 score, 2.0 = 100 score
            vol_ratio = df['vol_ratio_7d_90d'].fillna(1.0)
            volume_factor = ((vol_ratio - 1) * 100).clip(0, 100)
        
        # Factor 3: Trend support (20% weight)
        trend_factor = pd.Series(0, index=df.index, dtype=float)
        trend_count = 0
        
        if 'sma_20d' in df.columns:
            above_20 = (df['price'] > df['sma_20d']).fillna(False)
            trend_factor += above_20.astype(float) * 33.33
            trend_count += 1
        
        if 'sma_50d' in df.columns:
            above_50 = (df['price'] > df['sma_50d']).fillna(False)
            trend_factor += above_50.astype(float) * 33.33
            trend_count += 1
        
        if 'sma_200d' in df.columns:
            above_200 = (df['price'] > df['sma_200d']).fillna(False)
            trend_factor += above_200.astype(float) * 33.34
            trend_count += 1
        
        # Adjust trend factor if not all SMAs are available
        if trend_count > 0 and trend_count < 3:
            trend_factor = trend_factor * (3 / trend_count)
        
        # Ensure trend_factor is between 0 and 100
        trend_factor = trend_factor.clip(0, 100)
        
        # Combined breakout score
        breakout_score = (
            distance_factor * 0.4 +
            volume_factor * 0.4 +
            trend_factor * 0.2
        )
        
        return breakout_score.clip(0, 100)
    
    @staticmethod
    def calculate_rvol_score(df: pd.DataFrame) -> pd.Series:
        """Calculate RVOL-based score"""
        if 'rvol' not in df.columns:
            return pd.Series(50, index=df.index)
        
        # Get RVOL values
        rvol = df['rvol'].fillna(1.0)
        
        # Initialize score
        rvol_score = pd.Series(50, index=df.index, dtype=float)
        
        # Score based on RVOL levels using conditions
        # Extreme volume (>5x)
        rvol_score.loc[rvol > 5] = 100
        # Very high volume (3-5x)
        rvol_score.loc[(rvol > 3) & (rvol <= 5)] = 90
        # High volume (2-3x)
        rvol_score.loc[(rvol > 2) & (rvol <= 3)] = 80
        # Above average (1.5-2x)
        rvol_score.loc[(rvol > 1.5) & (rvol <= 2)] = 70
        # Slightly above (1.2-1.5x)
        rvol_score.loc[(rvol > 1.2) & (rvol <= 1.5)] = 60
        # Normal (0.8-1.2x)
        rvol_score.loc[(rvol > 0.8) & (rvol <= 1.2)] = 50
        # Below average (0.5-0.8x)
        rvol_score.loc[(rvol > 0.5) & (rvol <= 0.8)] = 40
        # Low volume (0.3-0.5x)
        rvol_score.loc[(rvol > 0.3) & (rvol <= 0.5)] = 30
        # Very low (<0.3x)
        rvol_score.loc[rvol <= 0.3] = 20
        
        return rvol_score
    
    @staticmethod
    def calculate_trend_quality(df: pd.DataFrame) -> pd.Series:
        """Calculate trend quality score based on SMA alignment"""
        trend_score = pd.Series(50, index=df.index, dtype=float)
        
        # Check available SMA columns
        sma_cols = ['sma_20d', 'sma_50d', 'sma_200d']
        available_smas = [col for col in sma_cols if col in df.columns and df[col].notna().any()]
        
        if len(available_smas) == 0:
            return trend_score
        
        # Calculate different trend conditions
        if len(available_smas) >= 3:
            # Perfect trend: Price > SMA20 > SMA50 > SMA200
            perfect_trend = (
                (df['price'] > df['sma_20d']) & 
                (df['sma_20d'] > df['sma_50d']) & 
                (df['sma_50d'] > df['sma_200d'])
            )
            trend_score.loc[perfect_trend] = 100
            
            # Strong trend: Price above all SMAs
            strong_trend = (
                (~perfect_trend) &
                (df['price'] > df['sma_20d']) & 
                (df['price'] > df['sma_50d']) & 
                (df['price'] > df['sma_200d'])
            )
            trend_score.loc[strong_trend] = 85
            
            # Good trend: Price above 2 SMAs
            above_count = (
                (df['price'] > df['sma_20d']).astype(int) +
                (df['price'] > df['sma_50d']).astype(int) +
                (df['price'] > df['sma_200d']).astype(int)
            )
            good_trend = (above_count == 2) & (~perfect_trend) & (~strong_trend)
            trend_score.loc[good_trend] = 70
            
            # Weak trend: Price above 1 SMA
            weak_trend = (above_count == 1)
            trend_score.loc[weak_trend] = 40
            
            # Poor trend: Price below all SMAs
            poor_trend = (above_count == 0)
            trend_score.loc[poor_trend] = 20
        
        elif len(available_smas) == 2:
            # Check alignment with 2 SMAs
            above_all = True
            for sma in available_smas:
                above_all &= (df['price'] > df[sma])
            
            trend_score.loc[above_all] = 80
            trend_score.loc[~above_all] = 30
        
        elif len(available_smas) == 1:
            # Single SMA check
            sma = available_smas[0]
            trend_score.loc[df['price'] > df[sma]] = 65
            trend_score.loc[df['price'] <= df[sma]] = 35
        
        return trend_score
    
    @staticmethod
    def calculate_long_term_strength(df: pd.DataFrame) -> pd.Series:
        """Calculate long-term strength score"""
        strength_score = pd.Series(50, index=df.index, dtype=float)
        
        # Check available long-term return columns
        lt_cols = ['ret_3m', 'ret_6m', 'ret_1y']
        available_cols = [col for col in lt_cols if col in df.columns and df[col].notna().any()]
        
        if not available_cols:
            return strength_score
        
        # Calculate average long-term return
        lt_returns = df[available_cols].fillna(0)
        avg_return = lt_returns.mean(axis=1)
        
        # Also consider the trend (improving or declining)
        if len(available_cols) >= 2:
            # Check if returns are improving over time
            if 'ret_3m' in available_cols and 'ret_1y' in available_cols:
                # Annualized 3-month return vs 1-year return
                annualized_3m = df['ret_3m'] * 4
                improving = annualized_3m > df['ret_1y']
            else:
                improving = pd.Series(False, index=df.index)
        else:
            improving = pd.Series(False, index=df.index)
        
        # Assign scores based on performance tiers
        # Exceptional performance (>100% avg)
        exceptional = avg_return > 100
        strength_score.loc[exceptional] = 100
        
        # Very strong (50-100%)
        very_strong = (avg_return > 50) & (avg_return <= 100)
        strength_score.loc[very_strong] = 90
        
        # Strong (30-50%)
        strong = (avg_return > 30) & (avg_return <= 50)
        strength_score.loc[strong] = 80
        
        # Good (15-30%)
        good = (avg_return > 15) & (avg_return <= 30)
        strength_score.loc[good] = 70
        
        # Moderate (5-15%)
        moderate = (avg_return > 5) & (avg_return <= 15)
        strength_score.loc[moderate] = 60
        
        # Weak (0-5%)
        weak = (avg_return > 0) & (avg_return <= 5)
        strength_score.loc[weak] = 50
        
        # Negative but recovering (-10-0%)
        recovering = (avg_return > -10) & (avg_return <= 0)
        strength_score.loc[recovering] = 40
        
        # Poor (-25 to -10%)
        poor = (avg_return > -25) & (avg_return <= -10)
        strength_score.loc[poor] = 30
        
        # Very poor (< -25%)
        very_poor = avg_return <= -25
        strength_score.loc[very_poor] = 20
        
        # Bonus for improving trend
        strength_score.loc[improving] += 5
        
        return strength_score.clip(0, 100)
    
    @staticmethod
    def calculate_liquidity_score(df: pd.DataFrame) -> pd.Series:
        """Calculate liquidity score based on trading volume"""
        liquidity_score = pd.Series(50, index=df.index, dtype=float)
        
        if 'volume_30d' in df.columns and 'price' in df.columns:
            # Calculate average daily traded value
            avg_traded_value = df['volume_30d'] * df['price']
            
            # Rank by traded value
            liquidity_score = RankingEngine.safe_rank(
                avg_traded_value, pct=True, ascending=True
            )
            
            # Bonus for consistent volume across timeframes
            if all(col in df.columns for col in ['volume_7d', 'volume_30d', 'volume_90d']):
                # Calculate coefficient of variation (lower = more consistent)
                vol_data = df[['volume_7d', 'volume_30d', 'volume_90d']]
                
                # Avoid division by zero
                vol_mean = vol_data.mean(axis=1)
                vol_std = vol_data.std(axis=1)
                
                # Only calculate CV where mean > 0
                valid_mask = vol_mean > 0
                vol_cv = pd.Series(1.0, index=df.index)  # Default high CV
                
                if valid_mask.any():
                    vol_cv[valid_mask] = vol_std[valid_mask] / vol_mean[valid_mask]
                
                # Convert CV to consistency score (lower CV = higher score)
                consistency_score = RankingEngine.safe_rank(
                    vol_cv, pct=True, ascending=False
                )
                
                # Weighted combination
                liquidity_score = liquidity_score * 0.8 + consistency_score * 0.2
        
        return liquidity_score.clip(0, 100)
    
    @staticmethod
    @timer
    def calculate_rankings(df: pd.DataFrame) -> pd.DataFrame:
        """Main ranking calculation with all components properly integrated"""
        if df.empty:
            return df
        
        logger.info("Starting ranking calculations...")
        
        # Log input data stats
        if 'from_low_pct' in df.columns and 'from_high_pct' in df.columns:
            non_default_from_low = (df['from_low_pct'] != 50).sum()
            non_default_from_high = (df['from_high_pct'] != -50).sum()
            
            logger.info(f"Input data stats: {len(df)} stocks")
            logger.info(f"  - Non-default from_low values: {non_default_from_low}")
            logger.info(f"  - Non-default from_high values: {non_default_from_high}")
            
            if non_default_from_low > 0:
                logger.info(f"  - from_low range: {df['from_low_pct'].min():.1f} to {df['from_low_pct'].max():.1f}")
            if non_default_from_high > 0:
                logger.info(f"  - from_high range: {df['from_high_pct'].min():.1f} to {df['from_high_pct'].max():.1f}")
        else:
            logger.warning("Position data columns not found in input")
        
        # Calculate all component scores
        df['position_score'] = RankingEngine.calculate_position_score(df)
        df['volume_score'] = RankingEngine.calculate_volume_score(df)
        df['momentum_score'] = RankingEngine.calculate_momentum_score(df)
        df['acceleration_score'] = RankingEngine.calculate_acceleration_score(df)
        df['breakout_score'] = RankingEngine.calculate_breakout_score(df)
        df['rvol_score'] = RankingEngine.calculate_rvol_score(df)
        
        # Calculate auxiliary scores
        df['trend_quality'] = RankingEngine.calculate_trend_quality(df)
        df['long_term_strength'] = RankingEngine.calculate_long_term_strength(df)
        df['liquidity_score'] = RankingEngine.calculate_liquidity_score(df)
        
        # MASTER SCORE 3.0 - Properly weighted with RVOL
        # Verify all component scores exist
        components = {
            'position_score': CONFIG.POSITION_WEIGHT,
            'volume_score': CONFIG.VOLUME_WEIGHT,
            'momentum_score': CONFIG.MOMENTUM_WEIGHT,
            'acceleration_score': CONFIG.ACCELERATION_WEIGHT,
            'breakout_score': CONFIG.BREAKOUT_WEIGHT,
            'rvol_score': CONFIG.RVOL_WEIGHT
        }
        
        # Calculate master score
        df['master_score'] = 0
        total_weight = 0
        
        for component, weight in components.items():
            if component in df.columns:
                df['master_score'] += df[component].fillna(50) * weight
                total_weight += weight
            else:
                logger.warning(f"Missing component: {component}")
        
        # Normalize if weights don't sum to 1.0
        if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Total weight is {total_weight}, normalizing...")
            df['master_score'] = df['master_score'] / total_weight
        
        # Ensure master score is within bounds
        df['master_score'] = df['master_score'].clip(0, 100)
        
        # Log score statistics for debugging
        logger.info(f"Master Score Stats - Mean: {df['master_score'].mean():.2f}, "
                   f"Std: {df['master_score'].std():.2f}, "
                   f"Min: {df['master_score'].min():.2f}, "
                   f"Max: {df['master_score'].max():.2f}")
        
        # Log component score means
        logger.info(f"Component Scores - Position: {df['position_score'].mean():.2f}, "
                   f"Volume: {df['volume_score'].mean():.2f}, "
                   f"Momentum: {df['momentum_score'].mean():.2f}, "
                   f"Acceleration: {df['acceleration_score'].mean():.2f}, "
                   f"Breakout: {df['breakout_score'].mean():.2f}, "
                   f"RVOL: {df['rvol_score'].mean():.2f}")
        
        # Check for score diversity
        score_diversity = {
            'position': df['position_score'].nunique(),
            'volume': df['volume_score'].nunique(),
            'momentum': df['momentum_score'].nunique(),
            'master': df['master_score'].nunique()
        }
        logger.info(f"Score diversity: {score_diversity}")
        
        # Calculate ranks
        # Ensure we have valid scores before ranking
        valid_scores = df['master_score'].notna()
        logger.info(f"Stocks with valid master scores: {valid_scores.sum()}")
        
        if valid_scores.sum() == 0:
            logger.error("No valid master scores calculated!")
            df['rank'] = 9999
            df['percentile'] = 0
        else:
            # For rank, use ascending=False so highest score gets rank 1
            df['rank'] = df['master_score'].rank(method='min', ascending=False, na_option='bottom')
            df['rank'] = df['rank'].fillna(len(df) + 1).astype(int)
            
            # Check if ranking worked properly
            if df['rank'].min() == df['rank'].max():
                logger.error("All stocks have the same rank! Using row order as tiebreaker.")
                # Add row order as tiebreaker
                df['rank'] = df['master_score'].rank(method='first', ascending=False, na_option='bottom')
                df['rank'] = df['rank'].fillna(len(df) + 1).astype(int)
            
            # For percentile, higher score = higher percentile
            df['percentile'] = df['master_score'].rank(pct=True, ascending=True, na_option='bottom') * 100
            df['percentile'] = df['percentile'].fillna(0)
        
        # Calculate category-specific ranks
        df = RankingEngine._calculate_category_ranks(df)
        
        # Detect patterns
        df = RankingEngine._detect_patterns(df)
        
        # Final verification
        unique_ranks = df['rank'].nunique()
        unique_scores = df['master_score'].nunique()
        
        # Basic sanity check
        if unique_scores < 10 and len(df) > 100:
            logger.error(f"Very low score diversity: only {unique_scores} unique scores for {len(df)} stocks!")
            logger.error("This suggests a calculation error. Component score stats:")
            for component in ['position_score', 'volume_score', 'momentum_score', 
                            'acceleration_score', 'breakout_score', 'rvol_score']:
                if component in df.columns:
                    logger.error(f"  - {component}: mean={df[component].mean():.1f}, "
                               f"std={df[component].std():.1f}, "
                               f"unique={df[component].nunique()}")
        
        logger.info(f"Ranking complete:")
        logger.info(f"  - Total stocks: {len(df)}")
        logger.info(f"  - Unique ranks: {unique_ranks}")
        logger.info(f"  - Unique scores: {unique_scores}")
        logger.info(f"  - Top score: {df['master_score'].max():.1f}")
        logger.info(f"  - Bottom score: {df['master_score'].min():.1f}")
        
        if unique_ranks < len(df) * 0.5:
            logger.warning(f"Low rank diversity: only {unique_ranks} unique ranks for {len(df)} stocks")
            
            # Log sample of top stocks for debugging
            top_5 = df.nsmallest(5, 'rank')[['ticker', 'master_score', 'rank']]
            logger.info(f"Top 5 stocks:\n{top_5}")
        
        return df
    
    @staticmethod
    def _calculate_category_ranks(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate percentile ranks within each category"""
        # Get unique categories
        categories = df['category'].unique()
        
        # Initialize category rank column
        df['category_rank'] = 9999
        df['category_percentile'] = 0.0
        
        # Rank within each category
        for category in categories:
            if category != 'Unknown':
                mask = df['category'] == category
                cat_df = df[mask]
                
                if len(cat_df) > 0:
                    # Calculate ranks within category
                    cat_ranks = RankingEngine.safe_rank(
                        cat_df['master_score'], pct=False, ascending=False
                    )
                    df.loc[mask, 'category_rank'] = cat_ranks.astype(int)
                    
                    # Calculate percentiles within category
                    cat_percentiles = RankingEngine.safe_rank(
                        cat_df['master_score'], pct=True, ascending=True
                    )
                    df.loc[mask, 'category_percentile'] = cat_percentiles
        
        return df
    
    @staticmethod
    def _detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Detect patterns using vectorized operations for performance"""
        # Initialize patterns column
        df['patterns'] = ''
        
        # Check for required columns and create conditions
        pattern_conditions = []
        
        # 1. Category Leader
        if 'category_percentile' in df.columns:
            mask = df['category_percentile'] >= CONFIG.PATTERN_THRESHOLDS['category_leader']
            pattern_conditions.append(('🔥 CAT LEADER', mask))
        
        # 2. Hidden Gem
        if 'category_percentile' in df.columns and 'percentile' in df.columns:
            mask = (
                (df['category_percentile'] >= CONFIG.PATTERN_THRESHOLDS['hidden_gem']) & 
                (df['percentile'] < 70)
            )
            pattern_conditions.append(('💎 HIDDEN GEM', mask))
        
        # 3. Accelerating
        if 'acceleration_score' in df.columns:
            mask = df['acceleration_score'] >= CONFIG.PATTERN_THRESHOLDS['acceleration']
            pattern_conditions.append(('🚀 ACCELERATING', mask))
        
        # 4. Institutional
        if 'volume_score' in df.columns and 'vol_ratio_90d_180d' in df.columns:
            mask = (
                (df['volume_score'] >= CONFIG.PATTERN_THRESHOLDS['institutional']) &
                (df['vol_ratio_90d_180d'] > 1.1)
            )
            pattern_conditions.append(('🏦 INSTITUTIONAL', mask))
        
        # 5. Volume Explosion
        if 'rvol' in df.columns:
            mask = df['rvol'] > 3
            pattern_conditions.append(('⚡ VOL EXPLOSION', mask))
        
        # 6. Breakout Ready
        if 'breakout_score' in df.columns:
            mask = df['breakout_score'] >= CONFIG.PATTERN_THRESHOLDS['breakout_ready']
            pattern_conditions.append(('🎯 BREAKOUT', mask))
        
        # 7. Market Leader
        if 'percentile' in df.columns:
            mask = df['percentile'] >= CONFIG.PATTERN_THRESHOLDS['market_leader']
            pattern_conditions.append(('👑 MARKET LEADER', mask))
        
        # 8. Momentum Wave
        if 'momentum_score' in df.columns and 'acceleration_score' in df.columns:
            mask = (
                (df['momentum_score'] >= CONFIG.PATTERN_THRESHOLDS['momentum_wave']) &
                (df['acceleration_score'] >= 70)
            )
            pattern_conditions.append(('🌊 MOMENTUM WAVE', mask))
        
        # 9. Liquid Leader
        if 'liquidity_score' in df.columns and 'percentile' in df.columns:
            mask = (
                (df['liquidity_score'] >= CONFIG.PATTERN_THRESHOLDS['liquid_leader']) &
                (df['percentile'] >= CONFIG.PATTERN_THRESHOLDS['liquid_leader'])
            )
            pattern_conditions.append(('💰 LIQUID LEADER', mask))
        
        # 10. Long-term Strength
        if 'long_term_strength' in df.columns:
            mask = df['long_term_strength'] >= CONFIG.PATTERN_THRESHOLDS['long_strength']
            pattern_conditions.append(('💪 LONG STRENGTH', mask))
        
        # 11. Quality Trend
        if 'trend_quality' in df.columns:
            mask = df['trend_quality'] >= 80
            pattern_conditions.append(('📈 QUALITY TREND', mask))
        
        # SMART FUNDAMENTAL PATTERNS - Only evaluate when data exists
        # 12. Value Momentum
        if 'pe' in df.columns and 'percentile' in df.columns:
            # Only evaluate for stocks with valid PE data
            has_pe = df['pe'].notna() & (df['pe'] > 0) & (df['pe'] < 1000)
            value_momentum = has_pe & (df['pe'] < 15) & (df['master_score'] >= 70)
            pattern_conditions.append(('💎 VALUE MOMENTUM', value_momentum))
        
        # 13. Earnings Rocket
        if 'eps_change_pct' in df.columns and 'acceleration_score' in df.columns:
            # Only evaluate for stocks with EPS data
            has_eps_growth = df['eps_change_pct'].notna()
            earnings_rocket = has_eps_growth & (df['eps_change_pct'] > 50) & (df['acceleration_score'] >= 70)
            pattern_conditions.append(('📊 EARNINGS ROCKET', earnings_rocket))
        
        # 14. Quality Leader
        if all(col in df.columns for col in ['pe', 'eps_change_pct', 'percentile']):
            # Comprehensive quality check
            has_complete_data = df['pe'].notna() & df['eps_change_pct'].notna() & (df['pe'] > 0)
            quality_leader = (
                has_complete_data &
                (df['pe'] >= 10) & (df['pe'] <= 25) &
                (df['eps_change_pct'] > 20) &
                (df['percentile'] >= 80)
            )
            pattern_conditions.append(('🏆 QUALITY LEADER', quality_leader))
        
        # 15. Turnaround Play
        if 'eps_change_pct' in df.columns and 'volume_score' in df.columns:
            # Stocks becoming profitable or showing massive improvement
            has_eps = df['eps_change_pct'].notna()
            turnaround = has_eps & (df['eps_change_pct'] > 100) & (df['volume_score'] >= 70)
            pattern_conditions.append(('⚡ TURNAROUND', turnaround))
        
        # 16. Overvalued Warning
        if 'pe' in df.columns:
            # Flag extremely high PE stocks
            has_pe = df['pe'].notna() & (df['pe'] > 0)
            overvalued = has_pe & (df['pe'] > 50)
            pattern_conditions.append(('⚠️ HIGH PE', overvalued))
        
        # Build pattern strings for each row
        patterns_list = []
        for idx in df.index:
            row_patterns = []
            for pattern_name, mask in pattern_conditions:
                try:
                    # Get the boolean value for this specific row
                    if mask.loc[idx]:
                        row_patterns.append(pattern_name)
                except:
                    # Skip if there's any issue accessing the value
                    continue
            
            patterns_list.append(' | '.join(row_patterns) if row_patterns else '')
        
        df['patterns'] = patterns_list
        return df

# ============================================
# FILTER ENGINE
# ============================================

class FilterEngine:
    """Handle all filtering operations with validation"""
    
    @staticmethod
    def get_unique_values(df: pd.DataFrame, column: str, 
                         exclude_unknown: bool = True) -> List[str]:
        """Get sorted unique values for a column"""
        if df.empty or column not in df.columns:
            return []
        
        try:
            values = df[column].dropna().unique().tolist()
            
            # Convert to strings to ensure consistency
            values = [str(v) for v in values]
            
            if exclude_unknown:
                values = [v for v in values if v not in ['Unknown', 'unknown', 'nan', 'NaN', '']]
            
            return sorted(values)
        except Exception as e:
            logger.error(f"Error getting unique values for {column}: {str(e)}")
            return []
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply all filters with validation"""
        if df.empty:
            return df
        
        filtered_df = df.copy()
        initial_count = len(filtered_df)
        
        # Category filter - handle empty selection
        categories = filters.get('categories', [])
        if categories and 'All' not in categories:
            filtered_df = filtered_df[filtered_df['category'].isin(categories)]
        
        # Sector filter - handle empty selection
        sectors = filters.get('sectors', [])
        if sectors and 'All' not in sectors:
            filtered_df = filtered_df[filtered_df['sector'].isin(sectors)]
        
        # EPS tier filter - handle empty selection
        eps_tiers = filters.get('eps_tiers', [])
        if eps_tiers and 'All' not in eps_tiers:
            filtered_df = filtered_df[filtered_df['eps_tier'].isin(eps_tiers)]
        
        # PE tier filter - handle empty selection
        pe_tiers = filters.get('pe_tiers', [])
        if pe_tiers and 'All' not in pe_tiers:
            filtered_df = filtered_df[filtered_df['pe_tier'].isin(pe_tiers)]
        
        # Price tier filter - handle empty selection
        price_tiers = filters.get('price_tiers', [])
        if price_tiers and 'All' not in price_tiers:
            filtered_df = filtered_df[filtered_df['price_tier'].isin(price_tiers)]
        
        # Score filter
        min_score = filters.get('min_score', 0)
        if min_score > 0:
            filtered_df = filtered_df[filtered_df['master_score'] >= min_score]
        
        # EPS change filter - only apply if value is not None
        min_eps_change = filters.get('min_eps_change')
        if min_eps_change is not None and 'eps_change_pct' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['eps_change_pct'] >= min_eps_change) | 
                (filtered_df['eps_change_pct'].isna())
            ]
        
        # Pattern filter
        patterns = filters.get('patterns', [])
        if patterns:
            pattern_mask = filtered_df['patterns'].str.contains(
                '|'.join(patterns), 
                case=False, 
                na=False
            )
            filtered_df = filtered_df[pattern_mask]
        
        # Trend filter
        if filters.get('trend_range') and filters.get('trend_filter') != 'All Trends':
            min_trend, max_trend = filters['trend_range']
            if 'trend_quality' in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df['trend_quality'] >= min_trend) & 
                    (filtered_df['trend_quality'] <= max_trend)
                ]
        
        # SMART PE FILTERS - Handle gracefully
        # Min PE filter
        min_pe = filters.get('min_pe')
        if min_pe is not None and 'pe' in filtered_df.columns:
            # Only filter stocks that have PE data
            filtered_df = filtered_df[
                (filtered_df['pe'].isna()) |  # Keep stocks without PE data
                ((filtered_df['pe'] > 0) & (filtered_df['pe'] >= min_pe))  # Filter those with PE
            ]
        
        # Max PE filter
        max_pe = filters.get('max_pe')
        if max_pe is not None and 'pe' in filtered_df.columns:
            # Only filter stocks that have PE data
            filtered_df = filtered_df[
                (filtered_df['pe'].isna()) |  # Keep stocks without PE data
                ((filtered_df['pe'] > 0) & (filtered_df['pe'] <= max_pe))  # Filter those with PE
            ]
        
        # Data completeness filter
        if filters.get('require_fundamental_data', False):
            if 'pe' in filtered_df.columns and 'eps_change_pct' in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df['pe'].notna() & 
                    (filtered_df['pe'] > 0) &
                    filtered_df['eps_change_pct'].notna()
                ]
        
        filtered_count = len(filtered_df)
        if filtered_count < initial_count:
            logger.info(f"Filters reduced stocks from {initial_count} to {filtered_count}")
        
        return filtered_df

# ============================================
# VISUALIZATION ENGINE
# ============================================

class Visualizer:
    """Create all visualizations with proper error handling"""
    
    @staticmethod
    def create_score_distribution(df: pd.DataFrame) -> go.Figure:
        """Create score distribution chart"""
        fig = go.Figure()
        
        if df.empty:
            fig.add_annotation(
                text="No data available for visualization",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Score components to visualize
        scores = [
            ('position_score', 'Position', '#3498db'),
            ('volume_score', 'Volume', '#e74c3c'),
            ('momentum_score', 'Momentum', '#2ecc71'),
            ('acceleration_score', 'Acceleration', '#f39c12'),
            ('breakout_score', 'Breakout', '#9b59b6'),
            ('rvol_score', 'RVOL', '#e67e22')
        ]
        
        for score_col, label, color in scores:
            if score_col in df.columns:
                score_data = df[score_col].dropna()
                if len(score_data) > 0:
                    fig.add_trace(go.Box(
                        y=score_data,
                        name=label,
                        marker_color=color,
                        boxpoints='outliers',
                        hovertemplate=f'{label}<br>Score: %{{y:.1f}}<extra></extra>'
                    ))
        
        fig.update_layout(
            title="Score Component Distribution",
            yaxis_title="Score (0-100)",
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_master_score_breakdown(df: pd.DataFrame, n: int = 20) -> go.Figure:
        """Create enhanced top stocks breakdown chart"""
        # Get top stocks
        top_df = df.nlargest(min(n, len(df)), 'master_score').copy()
        
        if len(top_df) == 0:
            return go.Figure()
        
        # Calculate weighted contributions
        components = [
            ('Position', 'position_score', CONFIG.POSITION_WEIGHT, '#3498db'),
            ('Volume', 'volume_score', CONFIG.VOLUME_WEIGHT, '#e74c3c'),
            ('Momentum', 'momentum_score', CONFIG.MOMENTUM_WEIGHT, '#2ecc71'),
            ('Acceleration', 'acceleration_score', CONFIG.ACCELERATION_WEIGHT, '#f39c12'),
            ('Breakout', 'breakout_score', CONFIG.BREAKOUT_WEIGHT, '#9b59b6'),
            ('RVOL', 'rvol_score', CONFIG.RVOL_WEIGHT, '#e67e22')
        ]
        
        fig = go.Figure()
        
        for name, score_col, weight, color in components:
            if score_col in top_df.columns:
                weighted_contrib = top_df[score_col] * weight
                
                fig.add_trace(go.Bar(
                    name=f'{name} ({weight:.0%})',
                    y=top_df['ticker'],
                    x=weighted_contrib,
                    orientation='h',
                    marker_color=color,
                    text=[f"{val:.1f}" for val in top_df[score_col]],
                    textposition='inside',
                    hovertemplate=f'{name}<br>Score: %{{text}}<br>Contribution: %{{x:.1f}}<extra></extra>'
                ))
        
        # Add master score annotation
        for i, (idx, row) in enumerate(top_df.iterrows()):
            fig.add_annotation(
                x=row['master_score'],
                y=i,
                text=f"{row['master_score']:.1f}",
                showarrow=False,
                xanchor='left',
                bgcolor='rgba(255,255,255,0.8)'
            )
        
        fig.update_layout(
            title=f"Top {len(top_df)} Stocks - Master Score 3.0 Breakdown",
            xaxis_title="Weighted Score Contribution",
            xaxis_range=[0, 105],
            barmode='stack',
            template='plotly_white',
            height=max(400, len(top_df) * 35),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        return fig
    
    @staticmethod
    def create_sector_performance_scatter(df: pd.DataFrame) -> go.Figure:
        """Create sector performance scatter plot"""
        # Aggregate by sector
        sector_stats = df.groupby('sector').agg({
            'master_score': ['mean', 'std', 'count'],
            'percentile': 'mean',
            'rvol': 'mean'
        }).reset_index()
        
        # Flatten column names
        sector_stats.columns = ['sector', 'avg_score', 'std_score', 'count', 'avg_percentile', 'avg_rvol']
        
        # Filter sectors with at least 3 stocks
        sector_stats = sector_stats[sector_stats['count'] >= 3]
        
        if len(sector_stats) == 0:
            return go.Figure()
        
        # Create scatter plot
        fig = px.scatter(
            sector_stats,
            x='avg_percentile',
            y='avg_score',
            size='count',
            color='avg_rvol',
            hover_data={
                'count': True,
                'std_score': ':.1f',
                'avg_rvol': ':.2f'
            },
            text='sector',
            title='Sector Performance Analysis',
            labels={
                'avg_percentile': 'Average Percentile Rank',
                'avg_score': 'Average Master Score',
                'count': 'Number of Stocks',
                'avg_rvol': 'Avg RVOL'
            },
            color_continuous_scale='Viridis'
        )
        
        fig.update_traces(
            textposition='top center',
            marker=dict(line=dict(width=1, color='white'))
        )
        
        fig.update_layout(
            template='plotly_white',
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_pattern_analysis(df: pd.DataFrame) -> go.Figure:
        """Create pattern frequency analysis"""
        # Extract all patterns
        all_patterns = []
        
        if not df.empty and 'patterns' in df.columns:
            for patterns in df['patterns'].dropna():
                if patterns:
                    all_patterns.extend(patterns.split(' | '))
        
        if not all_patterns:
            fig = go.Figure()
            fig.add_annotation(
                text="No patterns detected in current selection",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Pattern Frequency Analysis",
                template='plotly_white',
                height=400
            )
            return fig
        
        # Count pattern frequencies
        pattern_counts = pd.Series(all_patterns).value_counts()
        
        # Create bar chart
        fig = go.Figure([
            go.Bar(
                x=pattern_counts.values,
                y=pattern_counts.index,
                orientation='h',
                marker_color='#3498db',
                text=pattern_counts.values,
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Pattern Frequency Analysis",
            xaxis_title="Number of Stocks",
            yaxis_title="Pattern",
            template='plotly_white',
            height=max(400, len(pattern_counts) * 30),
            margin=dict(l=150)
        )
        
        return fig

# ============================================
# SEARCH ENGINE - FIXED VERSION
# ============================================

class SearchEngine:
    """Advanced search functionality with fixed index handling"""
    
    @staticmethod
    def create_search_index(df: pd.DataFrame) -> Dict[str, Set[str]]:
        """Create search index mapping search terms to ticker symbols"""
        search_index = {}
        
        try:
            for _, row in df.iterrows():
                ticker = str(row.get('ticker', '')).upper()
                if not ticker or ticker == 'NAN':
                    continue
                
                # Index by ticker
                if ticker not in search_index:
                    search_index[ticker] = set()
                search_index[ticker].add(ticker)
                
                # Index by company name words
                company_name = str(row.get('company_name', ''))
                if company_name and company_name != 'nan':
                    company_words = company_name.upper().split()
                    for word in company_words:
                        if len(word) > 2:  # Skip short words
                            if word not in search_index:
                                search_index[word] = set()
                            search_index[word].add(ticker)
            
            logger.info(f"Created search index with {len(search_index)} terms")
            
        except Exception as e:
            logger.error(f"Error creating search index: {str(e)}")
            
        return search_index
    
    @staticmethod
    def search_stocks(df: pd.DataFrame, query: str, 
                     search_index: Optional[Dict[str, Set[str]]] = None) -> pd.DataFrame:
        """Search stocks with relevance scoring - FIXED VERSION"""
        if not query or df.empty:
            return pd.DataFrame()
        
        try:
            query = query.upper().strip()
            
            # Direct ticker match
            ticker_match = df[df['ticker'].str.upper() == query]
            if not ticker_match.empty:
                return ticker_match
            
            # Use search index if available
            if search_index:
                matching_tickers = set()
                query_words = query.split()
                
                for word in query_words:
                    if word in search_index:
                        matching_tickers.update(search_index[word])
                
                if matching_tickers:
                    # Filter by ticker instead of using index positions
                    return df[df['ticker'].isin(matching_tickers)]
            
            # Fallback to string contains
            mask = (
                df['ticker'].str.contains(query, case=False, na=False) |
                df['company_name'].str.contains(query, case=False, na=False)
            )
            
            return df[mask]
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return pd.DataFrame()

# ============================================
# EXPORT ENGINE
# ============================================

class ExportEngine:
    """Handle all export operations"""
    
    @staticmethod
    def create_excel_report(df: pd.DataFrame) -> BytesIO:
        """Create comprehensive Excel report with smart fundamental data handling"""
        output = BytesIO()
        
        try:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                workbook = writer.book
                
                # Define formats
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#3498db',
                    'font_color': 'white',
                    'border': 1
                })
                
                # Format for good/bad values
                good_format = workbook.add_format({'font_color': '#2ecc71'})
                bad_format = workbook.add_format({'font_color': '#e74c3c'})
                neutral_format = workbook.add_format({'font_color': '#95a5a6'})
                
                # 1. Top 100 Stocks - ENHANCED with fundamentals
                top_100 = df.nlargest(min(100, len(df)), 'master_score').copy()
                
                # Select and order columns - now includes PE and EPS
                export_cols = [
                    'rank', 'ticker', 'company_name', 'master_score',
                    'position_score', 'volume_score', 'momentum_score',
                    'acceleration_score', 'breakout_score', 'rvol_score',
                    'trend_quality', 'price', 'pe', 'eps_current', 'eps_change_pct',
                    'from_low_pct', 'from_high_pct',
                    'ret_1d', 'ret_7d', 'ret_30d', 'rvol',
                    'patterns', 'category', 'sector'
                ]
                
                available_cols = [col for col in export_cols if col in top_100.columns]
                top_100[available_cols].to_excel(
                    writer, sheet_name='Top 100', index=False
                )
                
                # Format the sheet
                worksheet = writer.sheets['Top 100']
                for i, col in enumerate(available_cols):
                    worksheet.write(0, i, col, header_format)
                
                # 2. All Stocks Summary - ENHANCED
                summary_cols = [
                    'rank', 'ticker', 'company_name', 'master_score',
                    'trend_quality', 'price', 'pe', 'eps_change_pct',
                    'ret_30d', 'rvol', 
                    'patterns', 'category', 'sector'
                ]
                available_summary = [col for col in summary_cols if col in df.columns]
                df[available_summary].to_excel(
                    writer, sheet_name='All Stocks', index=False
                )
                
                # 3. Sector Analysis - ENHANCED with PE/EPS
                if 'sector' in df.columns:
                    try:
                        # Create comprehensive sector analysis
                        sector_agg = {'master_score': ['mean', 'std', 'min', 'max', 'count'],
                                     'rvol': 'mean',
                                     'ret_30d': 'mean'}
                        
                        # Add PE analysis if available
                        if 'pe' in df.columns:
                            sector_agg['pe'] = lambda x: x[x > 0].mean() if any(x > 0) else np.nan
                        
                        # Add EPS growth analysis if available
                        if 'eps_change_pct' in df.columns:
                            sector_agg['eps_change_pct'] = lambda x: x.dropna().mean()
                        
                        sector_analysis = df.groupby('sector').agg(sector_agg).round(2)
                        
                        # Flatten column names
                        flat_cols = []
                        for col in sector_analysis.columns:
                            if isinstance(col, tuple):
                                flat_cols.append(f"{col[0]}_{col[1]}")
                            else:
                                flat_cols.append(col)
                        sector_analysis.columns = flat_cols
                        
                        sector_analysis.to_excel(writer, sheet_name='Sector Analysis')
                    except Exception as e:
                        logger.warning(f"Unable to create sector analysis: {str(e)}")
                
                # 4. Category Analysis - ENHANCED
                if 'category' in df.columns:
                    try:
                        category_agg = {'master_score': ['mean', 'std', 'min', 'max', 'count'],
                                       'rvol': 'mean',
                                       'ret_30d': 'mean'}
                        
                        if 'pe' in df.columns:
                            category_agg['pe'] = lambda x: x[x > 0].mean() if any(x > 0) else np.nan
                        
                        if 'eps_change_pct' in df.columns:
                            category_agg['eps_change_pct'] = lambda x: x.dropna().mean()
                        
                        category_analysis = df.groupby('category').agg(category_agg).round(2)
                        
                        # Flatten column names
                        flat_cols = []
                        for col in category_analysis.columns:
                            if isinstance(col, tuple):
                                flat_cols.append(f"{col[0]}_{col[1]}")
                            else:
                                flat_cols.append(col)
                        category_analysis.columns = flat_cols
                        
                        category_analysis.to_excel(writer, sheet_name='Category Analysis')
                    except Exception as e:
                        logger.warning(f"Unable to create category analysis: {str(e)}")
                
                # 5. Pattern Analysis - Including new fundamental patterns
                pattern_data = []
                for pattern in df['patterns'].dropna():
                    if pattern:
                        for p in pattern.split(' | '):
                            pattern_data.append(p)
                
                if pattern_data:
                    pattern_df = pd.DataFrame(
                        pd.Series(pattern_data).value_counts()
                    ).reset_index()
                    pattern_df.columns = ['Pattern', 'Count']
                    pattern_df.to_excel(
                        writer, sheet_name='Pattern Analysis', index=False
                    )
                
                # 6. NEW: Fundamental Analysis Sheet
                if any(col in df.columns for col in ['pe', 'eps_current', 'eps_change_pct']):
                    try:
                        # Create fundamental analysis
                        fund_df = df.copy()
                        
                        # Filter for stocks with fundamental data
                        if 'pe' in fund_df.columns:
                            fund_df = fund_df[fund_df['pe'].notna() & (fund_df['pe'] > 0)]
                        
                        if len(fund_df) > 0:
                            # Categorize by PE ranges
                            pe_categories = []
                            if 'pe' in fund_df.columns:
                                fund_df['pe_category'] = pd.cut(
                                    fund_df['pe'],
                                    bins=[0, 10, 15, 20, 30, 50, float('inf')],
                                    labels=['0-10 (Deep Value)', '10-15 (Value)', 
                                           '15-20 (Fair)', '20-30 (Growth)', 
                                           '30-50 (High Growth)', '50+ (Expensive)']
                                )
                            
                            # Top value plays (Low PE + High Score)
                            if 'pe' in fund_df.columns:
                                value_plays = fund_df[
                                    (fund_df['pe'] < 20) & 
                                    (fund_df['master_score'] > 60)
                                ].nlargest(20, 'master_score')
                                
                                if len(value_plays) > 0:
                                    value_cols = ['ticker', 'company_name', 'master_score', 
                                                 'pe', 'eps_change_pct', 'price', 'ret_30d', 
                                                 'category', 'sector']
                                    available_value_cols = [col for col in value_cols if col in value_plays.columns]
                                    value_plays[available_value_cols].to_excel(
                                        writer, sheet_name='Value Plays', index=False
                                    )
                            
                            # Growth champions (High EPS growth)
                            if 'eps_change_pct' in fund_df.columns:
                                growth_champs = fund_df[
                                    fund_df['eps_change_pct'] > 25
                                ].nlargest(20, 'eps_change_pct')
                                
                                if len(growth_champs) > 0:
                                    growth_cols = ['ticker', 'company_name', 'eps_change_pct',
                                                  'pe', 'master_score', 'price', 'ret_30d',
                                                  'category', 'sector']
                                    available_growth_cols = [col for col in growth_cols if col in growth_champs.columns]
                                    growth_champs[available_growth_cols].to_excel(
                                        writer, sheet_name='Growth Champions', index=False
                                    )
                        
                    except Exception as e:
                        logger.warning(f"Unable to create fundamental analysis: {str(e)}")
                
                # 7. Wave Radar Signals - Including fundamental signals
                momentum_shifts = df[
                    (df['momentum_score'] >= 50) & 
                    (df['acceleration_score'] >= 60)
                ].head(20)
                
                if len(momentum_shifts) > 0:
                    wave_cols = ['ticker', 'company_name', 'master_score', 
                                'momentum_score', 'acceleration_score', 'rvol',
                                'pe', 'eps_change_pct', 
                                'category', 'sector']
                    available_wave_cols = [col for col in wave_cols if col in momentum_shifts.columns]
                    momentum_shifts[available_wave_cols].to_excel(
                        writer, sheet_name='Wave Radar Signals', index=False
                    )
                
                # 8. NEW: Data Quality Report
                quality_df = pd.DataFrame({
                    'Metric': ['Total Stocks', 'With PE Data', 'With EPS Current', 
                              'With EPS Change %', 'Complete Fundamental Data',
                              'Technical Only'],
                    'Count': [
                        len(df),
                        df['pe'].notna().sum() if 'pe' in df.columns else 0,
                        df['eps_current'].notna().sum() if 'eps_current' in df.columns else 0,
                        df['eps_change_pct'].notna().sum() if 'eps_change_pct' in df.columns else 0,
                        ((df['pe'].notna() if 'pe' in df.columns else False) & 
                         (df['eps_change_pct'].notna() if 'eps_change_pct' in df.columns else False)).sum(),
                        len(df) - (df['pe'].notna().sum() if 'pe' in df.columns else 0)
                    ],
                    'Percentage': [
                        100,
                        df['pe'].notna().sum() / len(df) * 100 if 'pe' in df.columns and len(df) > 0 else 0,
                        df['eps_current'].notna().sum() / len(df) * 100 if 'eps_current' in df.columns and len(df) > 0 else 0,
                        df['eps_change_pct'].notna().sum() / len(df) * 100 if 'eps_change_pct' in df.columns and len(df) > 0 else 0,
                        ((df['pe'].notna() if 'pe' in df.columns else False) & 
                         (df['eps_change_pct'].notna() if 'eps_change_pct' in df.columns else False)).sum() / len(df) * 100 if len(df) > 0 else 0,
                        (len(df) - (df['pe'].notna().sum() if 'pe' in df.columns else 0)) / len(df) * 100 if len(df) > 0 else 0
                    ]
                })
                quality_df['Percentage'] = quality_df['Percentage'].round(1)
                quality_df.to_excel(writer, sheet_name='Data Quality', index=False)
                
                logger.info("Excel report created successfully with enhanced fundamental analysis")
                
        except Exception as e:
            logger.error(f"Error creating Excel report: {str(e)}")
            raise
        
        output.seek(0)
        return output
    
    @staticmethod
    def create_csv_export(df: pd.DataFrame) -> str:
        """Create CSV export with selected columns including fundamentals"""
        export_cols = [
            'rank', 'ticker', 'company_name', 'master_score',
            'position_score', 'volume_score', 'momentum_score',
            'acceleration_score', 'breakout_score', 'rvol_score',
            'trend_quality', 'price', 'pe', 'eps_current', 'eps_change_pct',
            'from_low_pct', 'from_high_pct',
            'ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y',
            'rvol', 'vol_ratio_30d_90d', 'vol_ratio_90d_180d',
            'patterns', 'category', 'sector', 'eps_tier', 'pe_tier'
        ]
        
        available_cols = [col for col in export_cols if col in df.columns]
        return df[available_cols].to_csv(index=False)

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Wave Detection Ultimate 3.0 with Wave Radar™",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 5% 5% 5% 10%;
        border-radius: 5px;
        overflow-wrap: break-word;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with gradient
    st.markdown("""
    <div style="
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    ">
        <h1 style="margin: 0; font-size: 2.5rem;">🌊 Wave Detection Ultimate 3.0</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
            Professional Stock Ranking System with Wave Radar™ Early Detection
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'search_index' not in st.session_state:
        st.session_state.search_index = None
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### 📊 Data Configuration")
        
        # Data source inputs
        sheet_url = st.text_input(
            "Google Sheets URL",
            value=CONFIG.DEFAULT_SHEET_URL,
            help="Enter the public Google Sheets URL"
        )
        
        gid = st.text_input(
            "Sheet ID (GID)",
            value=CONFIG.DEFAULT_GID,
            help="Enter the specific sheet ID"
        )
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", type="primary", use_container_width=True):
                st.cache_data.clear()
                st.session_state.search_index = None
                st.rerun()
        
        with col2:
            if st.button("ℹ️ About", use_container_width=True):
                st.info("""
                **Master Score 3.0 Components:**
                
                • **Position (30%)**: 52-week range position
                • **Volume (25%)**: Multi-timeframe volume analysis
                • **Momentum (15%)**: 30-day price momentum
                • **Acceleration (10%)**: Momentum acceleration
                • **Breakout (10%)**: Breakout probability
                • **RVOL (10%)**: Today's relative volume
                
                **🌊 Wave Radar Features:**
                • **Momentum Shifts**: Catch stocks entering strength
                • **Category Flow**: Track smart money rotation
                • **Pattern Emergence**: Early pattern detection
                • **Acceleration Alerts**: Momentum building signals
                • **Volume Surges**: Unusual activity detection
                
                **💎 NEW! Hybrid Mode Features:**
                • **PE Ratio Display**: Valuation context
                • **EPS Growth %**: Earnings momentum
                • **Smart Patterns**: Value Momentum, Earnings Rocket
                • **Quality Filters**: PE ranges, EPS growth filters
                • **Enhanced Reports**: Fundamental analysis sheets
                
                **Core Features:**
                • Real-time RVOL integration
                • 11 advanced pattern detections + 5 fundamental patterns
                • Smart Trend filter with visual indicators
                • Category-relative rankings
                • Smart search with relevance
                • Professional Excel exports
                
                **Trend Indicators:**
                • 🔥 Strong Uptrend (80+)
                • ✅ Good Uptrend (60-79)
                • ➡️ Neutral Trend (40-59)
                • ⚠️ Weak/Downtrend (<40)
                """)
        
        st.markdown("---")
        st.markdown("### 🔍 Filters")
    
    # Data loading and processing
    try:
        with st.spinner("📥 Loading data from Google Sheets..."):
            raw_df = load_google_sheets_data(sheet_url, gid)
        
        with st.spinner(f"⚙️ Processing {len(raw_df):,} stocks..."):
            processed_df = DataProcessor.process_dataframe(raw_df)
            
            # Show data quality info
            if processed_df.empty:
                st.error("❌ No valid data after processing. Please check your data source.")
                st.info("Tips: Verify that your Google Sheets URL is public and the GID is correct.")
                st.stop()
            else:
                logger.info(f"Data quality summary:")
                logger.info(f"  - Valid stocks: {len(processed_df)}")
                logger.info(f"  - Unique tickers: {processed_df['ticker'].nunique()}")
                
                # Safe price range display
                if 'price' in processed_df.columns and processed_df['price'].notna().any():
                    logger.info(f"  - Price range: ₹{processed_df['price'].min():.0f} - ₹{processed_df['price'].max():.0f}")
                
                if 'from_low_pct' in processed_df.columns and processed_df['from_low_pct'].notna().any():
                    logger.info(f"  - From low range: {processed_df['from_low_pct'].min():.0f}% - {processed_df['from_low_pct'].max():.0f}%")
        
        with st.spinner("📊 Calculating rankings..."):
            ranked_df = RankingEngine.calculate_rankings(processed_df)
        
        # Create search index
        if st.session_state.search_index is None:
            st.session_state.search_index = SearchEngine.create_search_index(ranked_df)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        with st.expander("🔍 Error Details"):
            st.code(str(e))
            st.info("Common issues:\n- Google Sheets not public\n- Invalid GID\n- Network connectivity\n- Data format issues")
        st.stop()
    
    # Get filter options
    filter_engine = FilterEngine()
    
    # Sidebar filters
    with st.sidebar:
        # Ensure we always have filters dict initialized
        filters = {}
        
        # Display Mode Toggle - SMART IMPLEMENTATION
        st.markdown("### 📊 Display Mode")
        display_mode = st.radio(
            "Choose your view:",
            options=["Technical", "Hybrid (Technical + Fundamentals)"],
            index=0,
            help="Technical: Pure momentum analysis | Hybrid: Adds PE & EPS data",
            key="display_mode_toggle"
        )
        
        # Store display preference
        show_fundamentals = (display_mode == "Hybrid (Technical + Fundamentals)")
        
        st.markdown("---")
        
        # Category filter with count
        categories = filter_engine.get_unique_values(ranked_df, 'category')
        category_counts = ranked_df['category'].value_counts()
        category_options = ['All'] + [
            f"{cat} ({category_counts.get(cat, 0)})" 
            for cat in categories
        ]
        
        # Ensure we have valid default
        default_categories = ['All'] if 'All' in category_options else (category_options[:1] if category_options else [])
        
        selected_categories = st.multiselect(
            "Market Cap Category",
            options=category_options,
            default=default_categories,
            key="category_filter"
        )
        
        # Handle empty selection - if nothing selected, treat as "All"
        if not selected_categories:
            filters['categories'] = ['All']
        else:
            # Extract actual category names
            filters['categories'] = [
                cat.split(' (')[0] for cat in selected_categories
            ]
        
        # Sector filter
        sectors = filter_engine.get_unique_values(ranked_df, 'sector')
        default_sectors = ['All'] if sectors else []
        
        selected_sectors = st.multiselect(
            "Sector",
            options=['All'] + sectors,
            default=default_sectors,
            key="sector_filter"
        )
        
        # Handle empty selection
        if not selected_sectors:
            filters['sectors'] = ['All']
        else:
            filters['sectors'] = selected_sectors
        
        # Score filter
        filters['min_score'] = st.slider(
            "Minimum Master Score",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            help="Filter stocks by minimum score"
        )
        
        # Pattern filter
        all_patterns = set()
        for patterns in ranked_df['patterns'].dropna():
            if patterns:
                all_patterns.update(patterns.split(' | '))
        
        if all_patterns:
            filters['patterns'] = st.multiselect(
                "Patterns",
                options=sorted(all_patterns),
                help="Filter by specific patterns"
            )
        
        # Trend filter - Smart implementation
        st.markdown("#### 📈 Trend Strength")
        trend_options = {
            "All Trends": (0, 100),
            "🔥 Strong Uptrend (80+)": (80, 100),
            "✅ Good Uptrend (60-79)": (60, 79),
            "➡️ Neutral Trend (40-59)": (40, 59),
            "⚠️ Weak/Downtrend (<40)": (0, 39)
        }
        
        filters['trend_filter'] = st.selectbox(
            "Trend Quality",
            options=list(trend_options.keys()),
            index=0,  # Default to "All Trends"
            help="Filter stocks by trend strength based on SMA alignment"
        )
        filters['trend_range'] = trend_options[filters['trend_filter']]
        
        # Advanced filters in expander
        with st.expander("🔧 Advanced Filters"):
            # EPS tier filter
            eps_tiers = filter_engine.get_unique_values(ranked_df, 'eps_tier')
            default_eps = ['All'] if eps_tiers else []
            
            selected_eps_tiers = st.multiselect(
                "EPS Tier",
                options=['All'] + eps_tiers,
                default=default_eps,
                key="eps_tier_filter"
            )
            filters['eps_tiers'] = selected_eps_tiers if selected_eps_tiers else ['All']
            
            # PE tier filter
            pe_tiers = filter_engine.get_unique_values(ranked_df, 'pe_tier')
            default_pe = ['All'] if pe_tiers else []
            
            selected_pe_tiers = st.multiselect(
                "PE Tier",
                options=['All'] + pe_tiers,
                default=default_pe,
                key="pe_tier_filter"
            )
            filters['pe_tiers'] = selected_pe_tiers if selected_pe_tiers else ['All']
            
            # Price tier filter
            price_tiers = filter_engine.get_unique_values(ranked_df, 'price_tier')
            default_price = ['All'] if price_tiers else []
            
            selected_price_tiers = st.multiselect(
                "Price Range",
                options=['All'] + price_tiers,
                default=default_price,
                key="price_tier_filter"
            )
            filters['price_tiers'] = selected_price_tiers if selected_price_tiers else ['All']
            
            # EPS change filter
            if 'eps_change_pct' in ranked_df.columns:
                # Using text input to allow empty value (no filtering)
                eps_change_input = st.text_input(
                    "Min EPS Change %",
                    value="",
                    placeholder="e.g. -50 or leave empty",
                    help="Enter minimum EPS growth percentage (e.g., -50 for -50% or higher), or leave empty to include all stocks"
                )
                
                # Convert to float if not empty
                if eps_change_input.strip():
                    try:
                        filters['min_eps_change'] = float(eps_change_input)
                    except ValueError:
                        st.error("Please enter a valid number for EPS change")
                        filters['min_eps_change'] = None
                else:
                    filters['min_eps_change'] = None
            
            # SMART PE FILTER - Only show in hybrid mode
            if show_fundamentals and 'pe' in ranked_df.columns:
                st.markdown("**🔍 Fundamental Filters**")
                
                # PE Range Filter
                col1, col2 = st.columns(2)
                with col1:
                    min_pe_input = st.text_input(
                        "Min PE Ratio",
                        value="",
                        placeholder="e.g. 10",
                        help="Minimum PE ratio (leave empty for no minimum)"
                    )
                    
                    if min_pe_input.strip():
                        try:
                            filters['min_pe'] = float(min_pe_input)
                        except ValueError:
                            st.error("Please enter a valid number for Min PE")
                            filters['min_pe'] = None
                    else:
                        filters['min_pe'] = None
                
                with col2:
                    max_pe_input = st.text_input(
                        "Max PE Ratio",
                        value="",
                        placeholder="e.g. 30",
                        help="Maximum PE ratio (leave empty for no maximum)"
                    )
                    
                    if max_pe_input.strip():
                        try:
                            filters['max_pe'] = float(max_pe_input)
                        except ValueError:
                            st.error("Please enter a valid number for Max PE")
                            filters['max_pe'] = None
                    else:
                        filters['max_pe'] = None
                
                # Data completeness filter
                filters['require_fundamental_data'] = st.checkbox(
                    "Only show stocks with PE and EPS data",
                    value=False,
                    help="Filter out stocks missing fundamental data"
                )
    
    # Apply filters
    filtered_df = filter_engine.apply_filters(ranked_df, filters)
    filtered_df = filtered_df.sort_values('rank')
    
    # Debug filter information
    if st.sidebar.checkbox("🐛 Show Debug Info", value=False):
        with st.sidebar.expander("Filter Debug Info"):
            st.write("**Active Filters:**")
            st.write(f"Categories: {filters.get('categories', [])}")
            st.write(f"Sectors: {filters.get('sectors', [])}")
            st.write(f"Min Score: {filters.get('min_score', 0)}")
            st.write(f"Patterns: {filters.get('patterns', [])}")
            st.write(f"Trend Range: {filters.get('trend_range', 'All')}")
            st.write(f"EPS Tiers: {filters.get('eps_tiers', [])}")
            st.write(f"PE Tiers: {filters.get('pe_tiers', [])}")
            st.write(f"Price Tiers: {filters.get('price_tiers', [])}")
            st.write(f"Min EPS Change: {filters.get('min_eps_change', None)}")
            if show_fundamentals:
                st.write(f"Min PE: {filters.get('min_pe', None)}")
                st.write(f"Max PE: {filters.get('max_pe', None)}")
                st.write(f"Require Fundamental Data: {filters.get('require_fundamental_data', False)}")
            st.write(f"**Filter Result:**")
            st.write(f"Before: {len(ranked_df)} stocks")
            st.write(f"After: {len(filtered_df)} stocks")
            st.write(f"Filtered: {len(ranked_df) - len(filtered_df)} stocks")
    
    # Main content area
    # Summary metrics with SMART fundamental indicators
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        total_stocks = len(filtered_df)
        total_original = len(ranked_df) if 'ranked_df' in locals() else total_stocks
        pct_of_all = (total_stocks/total_original*100) if total_original > 0 else 0
        
        st.metric(
            "Total Stocks",
            f"{total_stocks:,}",
            f"{pct_of_all:.0f}% of {total_original:,}"
        )
    
    with col2:
        if not filtered_df.empty and 'master_score' in filtered_df.columns:
            avg_score = filtered_df['master_score'].mean()
            std_score = filtered_df['master_score'].std()
            st.metric(
                "Avg Score",
                f"{avg_score:.1f}",
                f"σ={std_score:.1f}"
            )
        else:
            st.metric("Avg Score", "N/A")
    
    with col3:
        # Show score range OR PE data based on display mode
        if show_fundamentals and 'pe' in filtered_df.columns:
            # Count stocks with valid PE data
            valid_pe = filtered_df['pe'].notna() & (filtered_df['pe'] > 0) & (filtered_df['pe'] < 1000)
            pe_coverage = valid_pe.sum()
            pe_pct = (pe_coverage / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
            
            # Calculate average PE for stocks with valid data
            if pe_coverage > 0:
                avg_pe = filtered_df.loc[valid_pe, 'pe'].mean()
                st.metric(
                    "Avg PE",
                    f"{avg_pe:.1f}x",
                    f"{pe_pct:.0f}% have data"
                )
            else:
                st.metric("PE Data", "Limited", "No PE data")
        else:
            # Default score range display
            if not filtered_df.empty and 'master_score' in filtered_df.columns:
                min_score = filtered_df['master_score'].min()
                max_score = filtered_df['master_score'].max()
                score_range = f"{min_score:.1f}-{max_score:.1f}"
            else:
                score_range = "N/A"
            st.metric("Score Range", score_range)
    
    with col4:
        # Show acceleration OR EPS growth based on display mode
        if show_fundamentals and 'eps_change_pct' in filtered_df.columns:
            # Count stocks with positive EPS growth
            valid_eps_change = filtered_df['eps_change_pct'].notna()
            positive_eps_growth = valid_eps_change & (filtered_df['eps_change_pct'] > 0)
            growth_count = positive_eps_growth.sum()
            
            st.metric(
                "EPS Growth +ve",
                f"{growth_count}",
                f"{valid_eps_change.sum()} have data"
            )
        else:
            # Default acceleration display
            if 'acceleration_score' in filtered_df.columns:
                accelerating = (filtered_df['acceleration_score'] >= 80).sum()
            else:
                accelerating = 0
            st.metric("Accelerating", f"{accelerating}")
    
    with col5:
        if 'rvol' in filtered_df.columns:
            high_rvol = (filtered_df['rvol'] > 2).sum()
        else:
            high_rvol = 0
        st.metric("High RVOL", f"{high_rvol}")
    
    with col6:
        # Show trend distribution
        if 'trend_quality' in filtered_df.columns:
            strong_trends = (filtered_df['trend_quality'] >= 80).sum()
            total = len(filtered_df)
            st.metric(
                "Strong Trends", 
                f"{strong_trends}",
                f"{strong_trends/total*100:.0f}%" if total > 0 else "0%"
            )
        else:
            with_patterns = (filtered_df['patterns'] != '').sum()
            st.metric("With Patterns", f"{with_patterns}")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏆 Rankings", "🌊 Wave Radar", "📊 Analysis", "🔍 Search", "📈 Visualizations", "📥 Export"
    ])
    
    # Tab 1: Rankings
    with tab1:
        st.markdown("### 🏆 Top Ranked Stocks")
        
        # Display options
        col1, col2, col3 = st.columns([2, 2, 6])
        with col1:
            display_count = st.selectbox(
                "Show top",
                options=CONFIG.AVAILABLE_TOP_N,
                index=2  # Default to 50
            )
        
        with col2:
            sort_options = ['Rank', 'Master Score', 'RVOL', 'Momentum']
            if 'trend_quality' in filtered_df.columns:
                sort_options.append('Trend')
            
            sort_by = st.selectbox(
                "Sort by",
                options=sort_options,
                index=0
            )
        
        # Get display data
        display_df = filtered_df.head(display_count).copy()
        
        # Apply sorting
        if sort_by == 'Master Score':
            display_df = display_df.sort_values('master_score', ascending=False)
        elif sort_by == 'RVOL':
            display_df = display_df.sort_values('rvol', ascending=False)
        elif sort_by == 'Momentum':
            display_df = display_df.sort_values('momentum_score', ascending=False)
        elif sort_by == 'Trend' and 'trend_quality' in display_df.columns:
            display_df = display_df.sort_values('trend_quality', ascending=False)
        
        if not display_df.empty:
            # Add trend indicator column if trend_quality exists
            if 'trend_quality' in display_df.columns:
                def get_trend_indicator(score):
                    if pd.isna(score):
                        return "➖"
                    elif score >= 80:
                        return "🔥"  # Strong uptrend
                    elif score >= 60:
                        return "✅"  # Good uptrend
                    elif score >= 40:
                        return "➡️"  # Neutral
                    else:
                        return "⚠️"  # Weak/downtrend
                
                display_df['trend_indicator'] = display_df['trend_quality'].apply(get_trend_indicator)
            
            # Prepare display columns
            display_cols = {
                'rank': 'Rank',
                'ticker': 'Ticker',
                'company_name': 'Company',
                'master_score': 'Score'
            }
            
            # Add trend indicator if it exists
            if 'trend_indicator' in display_df.columns:
                display_cols['trend_indicator'] = 'Trend'
            
            # Add price column
            display_cols['price'] = 'Price'
            
            # SMART FUNDAMENTAL COLUMNS - Only show when enabled
            if show_fundamentals:
                # Add PE column if exists
                if 'pe' in display_df.columns:
                    display_cols['pe'] = 'PE'
                
                # Add EPS Change % column if exists
                if 'eps_change_pct' in display_df.columns:
                    display_cols['eps_change_pct'] = 'EPS Δ%'
            
            # Add remaining columns
            display_cols.update({
                'from_low_pct': 'From Low',
                'ret_30d': '30D Ret',
                'rvol': 'RVOL',
                'patterns': 'Patterns',
                'category': 'Category',
                'sector': 'Sector'
            })
            
            # Format numeric columns with ROBUST error handling
            format_rules = {
                'master_score': '{:.1f}',
                'price': '₹{:,.0f}',
                'from_low_pct': '{:.0f}%',
                'ret_30d': '{:+.1f}%',
                'rvol': '{:.1f}x'
            }
            
            # SMART PE FORMATTING FUNCTION
            def format_pe(value):
                """Format PE ratio with intelligent handling"""
                try:
                    if pd.isna(value) or value == 'N/A':
                        return '-'
                    
                    val = float(value)
                    if val < 0 or val == 0:
                        return 'Loss'
                    elif val > 1000:
                        return '>1000'
                    else:
                        return f"{val:.1f}"
                except:
                    return '-'
            
            # SMART EPS CHANGE FORMATTING FUNCTION
            def format_eps_change(value):
                """Format EPS change % with color indicators"""
                try:
                    if pd.isna(value) or value == 'N/A':
                        return '-'
                    
                    val = float(value)
                    if val > 999:
                        return '>999%'
                    elif val < -99:
                        return '<-99%'
                    else:
                        return f"{val:+.1f}%"
                except:
                    return '-'
            
            # Apply formatting with comprehensive error handling
            for col, fmt in format_rules.items():
                if col in display_df.columns:
                    try:
                        if col == 'ret_30d':
                            # Special handling for ret_30d to show sign
                            display_df[col] = display_df[col].apply(
                                lambda x: f"{x:+.1f}%" if pd.notna(x) and x != 'N/A' and isinstance(x, (int, float)) else '-'
                            )
                        else:
                            display_df[col] = display_df[col].apply(
                                lambda x: fmt.format(x) if pd.notna(x) and x != 'N/A' and isinstance(x, (int, float)) else '-'
                            )
                    except Exception as e:
                        logger.warning(f"Error formatting {col}: {str(e)}")
                        display_df[col] = display_df[col].fillna('-')
            
            # Apply special formatting for fundamentals when enabled
            if show_fundamentals:
                if 'pe' in display_df.columns:
                    display_df['pe'] = display_df['pe'].apply(format_pe)
                
                if 'eps_change_pct' in display_df.columns:
                    display_df['eps_change_pct'] = display_df['eps_change_pct'].apply(format_eps_change)
            
            # Rename columns for display
            display_df = display_df[[c for c in display_cols.keys() if c in display_df.columns]]
            display_df.columns = [display_cols[c] for c in display_df.columns]
            
            # Display with styling
            st.dataframe(
                display_df,
                use_container_width=True,
                height=min(600, len(display_df) * 35 + 50),
                hide_index=True
            )
            
            # Quick stats below table
            with st.expander("📊 Quick Statistics"):
                stat_cols = st.columns(4)
                
                with stat_cols[0]:
                    st.markdown("**Score Distribution**")
                    if 'master_score' in filtered_df.columns:
                        st.text(f"Max: {filtered_df['master_score'].max():.1f}")
                        st.text(f"Min: {filtered_df['master_score'].min():.1f}")
                        st.text(f"Std: {filtered_df['master_score'].std():.1f}")
                    else:
                        st.text("No score data available")
                
                with stat_cols[1]:
                    st.markdown("**Returns (30D)**")
                    if 'ret_30d' in filtered_df.columns:
                        st.text(f"Max: {filtered_df['ret_30d'].max():.1f}%")
                        st.text(f"Avg: {filtered_df['ret_30d'].mean():.1f}%")
                        st.text(f"Positive: {(filtered_df['ret_30d'] > 0).sum()}")
                    else:
                        st.text("No 30D return data available")
                
                with stat_cols[2]:
                    if show_fundamentals:
                        st.markdown("**Fundamentals**")
                        if 'pe' in filtered_df.columns:
                            valid_pe = filtered_df['pe'].notna() & (filtered_df['pe'] > 0) & (filtered_df['pe'] < 1000)
                            if valid_pe.any():
                                st.text(f"Avg PE: {filtered_df.loc[valid_pe, 'pe'].mean():.1f}x")
                            else:
                                st.text("Avg PE: N/A")
                        else:
                            st.text("PE: No data")
                        
                        if 'eps_change_pct' in filtered_df.columns:
                            valid_eps = filtered_df['eps_change_pct'].notna()
                            if valid_eps.any():
                                st.text(f"Avg EPS Δ: {filtered_df.loc[valid_eps, 'eps_change_pct'].mean():+.1f}%")
                                st.text(f"Growing: {(filtered_df['eps_change_pct'] > 0).sum()}")
                            else:
                                st.text("EPS Growth: N/A")
                        else:
                            st.text("EPS: No data")
                    else:
                        st.markdown("**RVOL Stats**")
                        if 'rvol' in filtered_df.columns:
                            st.text(f"Max: {filtered_df['rvol'].max():.1f}x")
                            st.text(f"Avg: {filtered_df['rvol'].mean():.1f}x")
                            st.text(f">2x: {(filtered_df['rvol'] > 2).sum()}")
                        else:
                            st.text("No RVOL data available")
                
                with stat_cols[3]:
                    st.markdown("**Categories**")
                    if 'category' in filtered_df.columns:
                        for cat, count in filtered_df['category'].value_counts().head(3).items():
                            st.text(f"{cat}: {count}")
                    else:
                        st.text("No category data available")
        
        else:
            st.warning("No stocks match the selected filters.")
    
    # Tab 2: Wave Radar - FIXED VERSION
    with tab2:
        st.markdown("### 🌊 Wave Radar - Early Momentum Detection System")
        st.markdown("*Catch waves as they form, not after they've peaked!*")
        
        # Wave Radar Controls
        radar_col1, radar_col2, radar_col3, radar_col4 = st.columns([2, 2, 2, 1])
        
        with radar_col1:
            wave_timeframe = st.selectbox(
                "Wave Detection Timeframe",
                ["Real-Time (Today)", "3-Day Formation", "Weekly Formation"],
                help="How far back to detect wave formation"
            )
        
        with radar_col2:
            sensitivity = st.select_slider(
                "Detection Sensitivity",
                options=["Conservative", "Balanced", "Aggressive"],
                value="Balanced",
                help="Conservative = Stronger signals, Aggressive = More signals"
            )
        
        with radar_col3:
            auto_refresh = st.checkbox("Auto-refresh (5 min)", value=False)
            if auto_refresh:
                st.markdown("🔄 *Live monitoring active*")
        
        with radar_col4:
            # Calculate overall Wave Strength
            if not filtered_df.empty:
                try:
                    momentum_count = len(filtered_df[filtered_df['momentum_score'] >= 60]) if 'momentum_score' in filtered_df.columns else 0
                    accel_count = len(filtered_df[filtered_df['acceleration_score'] >= 70]) if 'acceleration_score' in filtered_df.columns else 0
                    rvol_count = len(filtered_df[filtered_df['rvol'] >= 2]) if 'rvol' in filtered_df.columns else 0
                    breakout_count = len(filtered_df[filtered_df['breakout_score'] >= 70]) if 'breakout_score' in filtered_df.columns else 0
                    
                    total_stocks = len(filtered_df)
                    if total_stocks > 0:
                        wave_strength = (
                            momentum_count * 0.3 +
                            accel_count * 0.3 +
                            rvol_count * 0.2 +
                            breakout_count * 0.2
                        ) / total_stocks * 100
                    else:
                        wave_strength = 0
                    
                    if wave_strength > 20:
                        wave_emoji = "🌊🔥"
                        wave_color = "🟢"
                    elif wave_strength > 10:
                        wave_emoji = "🌊"
                        wave_color = "🟡"
                    else:
                        wave_emoji = "💤"
                        wave_color = "🔴"
                    
                    st.metric(
                        "Wave Strength",
                        f"{wave_emoji} {wave_strength:.0f}%",
                        f"{wave_color} Market"
                    )
                except Exception as e:
                    logger.error(f"Error calculating wave strength: {str(e)}")
                    st.metric("Wave Strength", "N/A", "Error")
        
        # Calculate Wave Signals
        if not filtered_df.empty:
            # 1. MOMENTUM SHIFT DETECTION
            st.markdown("#### 🚀 Momentum Shifts - Stocks Entering Strength")
            
            # Calculate momentum shifts
            momentum_shifts = filtered_df.copy()
            
            # Identify crossing points based on sensitivity
            if sensitivity == "Conservative":
                cross_threshold = 60
                min_acceleration = 70
            elif sensitivity == "Balanced":
                cross_threshold = 50
                min_acceleration = 60
            else:  # Aggressive
                cross_threshold = 40
                min_acceleration = 50
            
            # Find stocks crossing into strength
            # Check if ret_30d exists, otherwise use ret_7d as proxy
            if 'ret_30d' in momentum_shifts.columns:
                median_return = momentum_shifts['ret_30d'].median()
                return_condition = momentum_shifts['ret_30d'] > median_return
            elif 'ret_7d' in momentum_shifts.columns:
                median_return = momentum_shifts['ret_7d'].median()
                return_condition = momentum_shifts['ret_7d'] > median_return
            else:
                return_condition = True  # No return filter if no data
            
            momentum_shifts['momentum_shift'] = (
                (momentum_shifts['momentum_score'] >= cross_threshold) & 
                (momentum_shifts['acceleration_score'] >= min_acceleration) &
                return_condition
            )
            
            # Calculate shift strength
            momentum_shifts['shift_strength'] = (
                momentum_shifts['momentum_score'] * 0.4 +
                momentum_shifts['acceleration_score'] * 0.4 +
                momentum_shifts['rvol_score'] * 0.2
            )
            
            # Get top momentum shifts
            top_shifts = momentum_shifts[momentum_shifts['momentum_shift']].nlargest(20, 'shift_strength')
            
            if len(top_shifts) > 0:
                # Select available columns for display
                display_columns = ['ticker', 'company_name', 'master_score', 'momentum_score', 
                                 'acceleration_score', 'rvol']
                
                # Add optional columns if they exist
                if 'ret_7d' in top_shifts.columns:
                    display_columns.append('ret_7d')
                
                display_columns.append('category')
                
                shift_display = top_shifts[display_columns].copy()
                
                # Add shift indicators
                shift_display['Signal'] = shift_display.apply(
                    lambda x: "🔥 HOT" if x['acceleration_score'] > 80 else "📈 RISING", axis=1
                )
                
                # Format for display
                if 'ret_7d' in shift_display.columns:
                    shift_display['ret_7d'] = shift_display['ret_7d'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "0.0%")
                else:
                    shift_display['ret_7d'] = "N/A"
                
                shift_display['rvol'] = shift_display['rvol'].apply(lambda x: f"{x:.1f}x")
                
                # Rename columns
                rename_dict = {
                    'ticker': 'Ticker',
                    'company_name': 'Company',
                    'master_score': 'Score',
                    'momentum_score': 'Momentum',
                    'acceleration_score': 'Acceleration',
                    'rvol': 'RVOL',
                    'category': 'Category'
                }
                
                if 'ret_7d' in shift_display.columns:
                    rename_dict['ret_7d'] = '7D Return'
                
                shift_display = shift_display.rename(columns=rename_dict)
                
                st.dataframe(
                    shift_display,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No momentum shifts detected with current settings. Try 'Aggressive' sensitivity.")
            
            # 2. CATEGORY ROTATION FLOW
            st.markdown("#### 💰 Category Rotation - Smart Money Flow")
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Calculate category performance
                try:
                    if not filtered_df.empty and 'category' in filtered_df.columns:
                        category_flow = filtered_df.groupby('category').agg({
                            'master_score': ['mean', 'count'],
                            'momentum_score': 'mean',
                            'volume_score': 'mean',
                            'rvol': 'mean'
                        }).round(2)
                        
                        if not category_flow.empty:
                            category_flow.columns = ['Avg Score', 'Count', 'Avg Momentum', 'Avg Volume', 'Avg RVOL']
                            category_flow['Flow Score'] = (
                                category_flow['Avg Score'] * 0.4 +
                                category_flow['Avg Momentum'] * 0.3 +
                                category_flow['Avg Volume'] * 0.3
                            )
                            
                            # Determine flow direction
                            category_flow = category_flow.sort_values('Flow Score', ascending=False)
                            if len(category_flow) > 0 and category_flow.index[0] in ['MICRO', 'SMALL']:
                                flow_direction = "🔥 Risk-ON"
                            elif len(category_flow) > 0:
                                flow_direction = "❄️ Risk-OFF"
                            else:
                                flow_direction = "➡️ Neutral"
                            
                            # Create flow visualization
                            fig_flow = go.Figure()
                            
                            fig_flow.add_trace(go.Bar(
                                x=category_flow.index,
                                y=category_flow['Flow Score'],
                                text=[f"{val:.1f}" for val in category_flow['Flow Score']],
                                textposition='outside',
                                marker_color=['#2ecc71' if score > 60 else '#e74c3c' if score < 40 else '#f39c12' 
                                             for score in category_flow['Flow Score']],
                                hovertemplate='Category: %{x}<br>Flow Score: %{y:.1f}<extra></extra>'
                            ))
                            
                            fig_flow.update_layout(
                                title=f"Smart Money Flow Direction: {flow_direction}",
                                xaxis_title="Market Cap Category",
                                yaxis_title="Flow Score",
                                height=300,
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(fig_flow, use_container_width=True)
                        else:
                            st.info("No category data available for flow analysis")
                            flow_direction = "➡️ Neutral"
                    else:
                        st.info("Category data not available")
                        flow_direction = "➡️ Neutral"
                        category_flow = pd.DataFrame()
                        
                except Exception as e:
                    logger.error(f"Error in category flow analysis: {str(e)}")
                    st.error("Unable to analyze category flow")
                    flow_direction = "➡️ Neutral"
                    category_flow = pd.DataFrame()
            
            with col2:
                st.markdown(f"**🎯 Market Regime: {flow_direction}**")
                
                # Top categories
                st.markdown("**💎 Strongest Categories:**")
                if not category_flow.empty:
                    for i, (cat, row) in enumerate(category_flow.head(3).iterrows()):
                        emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                        try:
                            st.write(f"{emoji} **{cat}**: Score {row['Flow Score']:.1f}")
                        except:
                            st.write(f"{emoji} **{cat}**: Score N/A")
                else:
                    st.info("No category data available")
                
                # Category shift detection
                st.markdown("**🔄 Category Shifts:**")
                if not category_flow.empty and 'SMALL' in category_flow.index and 'LARGE' in category_flow.index:
                    try:
                        small_score = category_flow.loc['SMALL', 'Flow Score']
                        large_score = category_flow.loc['LARGE', 'Flow Score']
                        
                        if small_score > large_score * 1.2:
                            st.success("📈 Small Caps Leading - Early Bull Signal!")
                        elif large_score > small_score * 1.2:
                            st.warning("📉 Large Caps Leading - Defensive Mode")
                        else:
                            st.info("➡️ Balanced Market - No Clear Leader")
                    except Exception as e:
                        logger.error(f"Error in category shift detection: {str(e)}")
                        st.info("Unable to determine category shifts")
                else:
                    st.info("Insufficient data for category shift analysis")
            
            # 3. EMERGING PATTERNS
            st.markdown("#### 🎯 Emerging Patterns - About to Qualify")
            
            # Calculate pattern emergence
            pattern_emergence = filtered_df.copy()
            
            # Check how close to pattern thresholds
            emergence_data = []
            
            # Category Leader emergence
            if 'category_percentile' in pattern_emergence.columns:
                close_to_leader = pattern_emergence[
                    (pattern_emergence['category_percentile'] >= 85) & 
                    (pattern_emergence['category_percentile'] < 90)
                ]
                for _, stock in close_to_leader.iterrows():
                    emergence_data.append({
                        'Ticker': stock['ticker'],
                        'Company': stock['company_name'],
                        'Pattern': '🔥 CAT LEADER',
                        'Distance': f"{90 - stock['category_percentile']:.1f}% away",
                        'Current': f"{stock['category_percentile']:.1f}%ile",
                        'Score': stock['master_score']
                    })
            
            # Breakout Ready emergence
            if 'breakout_score' in pattern_emergence.columns:
                close_to_breakout = pattern_emergence[
                    (pattern_emergence['breakout_score'] >= 75) & 
                    (pattern_emergence['breakout_score'] < 80)
                ]
                for _, stock in close_to_breakout.iterrows():
                    emergence_data.append({
                        'Ticker': stock['ticker'],
                        'Company': stock['company_name'],
                        'Pattern': '🎯 BREAKOUT',
                        'Distance': f"{80 - stock['breakout_score']:.1f} pts away",
                        'Current': f"{stock['breakout_score']:.1f} score",
                        'Score': stock['master_score']
                    })
            
            # Volume Explosion emergence
            close_to_explosion = pattern_emergence[
                (pattern_emergence['rvol'] >= 2.5) & 
                (pattern_emergence['rvol'] < 3.0)
            ]
            for _, stock in close_to_explosion.iterrows():
                emergence_data.append({
                    'Ticker': stock['ticker'],
                    'Company': stock['company_name'],
                    'Pattern': '⚡ VOL EXPLOSION',
                    'Distance': f"{3.0 - stock['rvol']:.1f}x away",
                    'Current': f"{stock['rvol']:.1f}x",
                    'Score': stock['master_score']
                })
            
            if emergence_data:
                emergence_df = pd.DataFrame(emergence_data).sort_values('Score', ascending=False).head(15)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.dataframe(emergence_df, use_container_width=True, hide_index=True)
                with col2:
                    st.metric("Emerging Patterns", len(emergence_df))
                    st.caption("Stocks about to trigger pattern alerts")
            else:
                st.info("No patterns emerging with current filters.")
            
            # 4. ACCELERATION ALERTS
            st.markdown("#### ⚡ Acceleration Alerts - Momentum Building")
            
            # Find accelerating stocks
            # Build conditions based on available columns
            accel_conditions = (
                (filtered_df['acceleration_score'] >= 70) &
                (filtered_df['momentum_score'] >= 60)
            )
            
            # Add return pace condition if data available
            if 'ret_7d' in filtered_df.columns and 'ret_30d' in filtered_df.columns:
                accel_conditions &= (filtered_df['ret_7d'] > filtered_df['ret_30d'] / 30 * 7)
            
            accelerating = filtered_df[accel_conditions].nlargest(10, 'acceleration_score')
            
            if len(accelerating) > 0:
                # Create acceleration visualization
                fig_accel = go.Figure()
                
                for _, stock in accelerating.iterrows():
                    # Create mini momentum chart
                    returns = [0]  # Start point
                    x_points = ['Start']
                    
                    if 'ret_30d' in stock.index and pd.notna(stock['ret_30d']):
                        returns.append(stock['ret_30d'])
                        x_points.append('30D Actual')
                    
                    if 'ret_7d' in stock.index and pd.notna(stock['ret_7d']):
                        if 'ret_30d' in stock.index:
                            returns.append(stock['ret_7d'] * 30/7)  # Projected 30d at 7d pace
                            x_points.append('7D Pace')
                        else:
                            returns.append(stock['ret_7d'])
                            x_points.append('7D Return')
                    
                    if 'ret_1d' in stock.index and pd.notna(stock['ret_1d']):
                        if 'ret_30d' in stock.index:
                            returns.append(stock['ret_1d'] * 30)  # Projected 30d at 1d pace
                            x_points.append('1D Pace')
                        else:
                            returns.append(stock['ret_1d'])
                            x_points.append('1D Return')
                    
                    if len(returns) > 1:  # Only plot if we have data
                        fig_accel.add_trace(go.Scatter(
                            x=x_points,
                            y=returns,
                            mode='lines+markers',
                            name=stock['ticker'],
                            line=dict(width=2),
                            hovertemplate='%{y:.1f}%<extra></extra>'
                        ))
                
                fig_accel.update_layout(
                    title="Acceleration Profiles - Momentum Building",
                    xaxis_title="Time Frame",
                    yaxis_title="Return % (Annualized)",
                    height=350,
                    template='plotly_white',
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.02
                    )
                )
                
                st.plotly_chart(fig_accel, use_container_width=True)
            else:
                st.info("No strong acceleration signals detected.")
            
            # 5. VOLUME SURGE DETECTION
            st.markdown("#### 🌊 Volume Surges - Unusual Activity NOW")
            
            # Find volume surges
            # Build surge conditions based on available columns
            surge_conditions = (filtered_df['rvol'] >= 2.0)
            
            if 'vol_ratio_1d_90d' in filtered_df.columns:
                surge_conditions |= (filtered_df['vol_ratio_1d_90d'] >= 2.0)
            
            volume_surges = filtered_df[surge_conditions].copy()
            
            if len(volume_surges) > 0:
                # Calculate surge score
                volume_surges['surge_score'] = (
                    volume_surges['rvol_score'] * 0.5 +
                    volume_surges['volume_score'] * 0.3 +
                    volume_surges['momentum_score'] * 0.2
                )
                
                top_surges = volume_surges.nlargest(15, 'surge_score')
                
                # Create surge visualization
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Select available columns for display
                    display_columns = ['ticker', 'company_name', 'rvol', 'price', 'category']
                    
                    # Add optional columns if they exist
                    if 'ret_1d' in top_surges.columns:
                        display_columns.insert(3, 'ret_1d')
                    
                    surge_display = top_surges[display_columns].copy()
                    
                    # Add surge type
                    surge_display['Type'] = surge_display['rvol'].apply(
                        lambda x: "🔥🔥🔥" if x > 5 else "🔥🔥" if x > 3 else "🔥"
                    )
                    
                    # Format columns
                    if 'ret_1d' in surge_display.columns:
                        surge_display['ret_1d'] = surge_display['ret_1d'].apply(
                            lambda x: f"{x:+.1f}%" if pd.notna(x) else "0.0%"
                        )
                    else:
                        surge_display['ret_1d'] = "N/A"
                    
                    surge_display['price'] = surge_display['price'].apply(lambda x: f"₹{x:,.0f}")
                    surge_display['rvol'] = surge_display['rvol'].apply(lambda x: f"{x:.1f}x")
                    
                    # Rename columns
                    rename_dict = {
                        'ticker': 'Ticker',
                        'company_name': 'Company',
                        'rvol': 'RVOL',
                        'price': 'Price',
                        'category': 'Category'
                    }
                    
                    if 'ret_1d' in surge_display.columns:
                        rename_dict['ret_1d'] = '1D Ret'
                    
                    surge_display = surge_display.rename(columns=rename_dict)
                    
                    st.dataframe(surge_display, use_container_width=True, hide_index=True)
                
                with col2:
                    # Volume statistics
                    st.metric("Active Surges", len(volume_surges))
                    st.metric("Extreme (>5x)", len(volume_surges[volume_surges['rvol'] > 5]))
                    st.metric("High (>3x)", len(volume_surges[volume_surges['rvol'] > 3]))
                    
                    # Surge distribution
                    surge_categories = volume_surges['category'].value_counts()
                    if len(surge_categories) > 0:
                        st.markdown("**Surge by Category:**")
                        for cat, count in surge_categories.head(3).items():
                            st.caption(f"{cat}: {count} stocks")
            else:
                st.info("No significant volume surges detected.")
            
            # Wave Radar Summary
            st.markdown("---")
            st.markdown("#### 🎯 Wave Radar Summary")
            
            summary_cols = st.columns(5)
            
            with summary_cols[0]:
                momentum_count = len(top_shifts) if 'top_shifts' in locals() else 0
                st.metric("Momentum Shifts", momentum_count)
            
            with summary_cols[1]:
                flow_direction = flow_direction if 'flow_direction' in locals() else "N/A"
                st.metric("Market Regime", flow_direction.split()[1] if flow_direction != "N/A" else "Unknown")
            
            with summary_cols[2]:
                emergence_count = len(emergence_data) if 'emergence_data' in locals() and emergence_data else 0
                st.metric("Emerging Patterns", emergence_count)
            
            with summary_cols[3]:
                accel_count = len(filtered_df[filtered_df['acceleration_score'] >= 70])
                st.metric("Accelerating", accel_count)
            
            with summary_cols[4]:
                surge_count = len(filtered_df[filtered_df['rvol'] >= 2])
                st.metric("Volume Surges", surge_count)
            
            # Auto-refresh note
            if auto_refresh:
                st.info("🔄 Auto-refresh enabled - Please manually refresh the page every 5 minutes or use browser auto-refresh extensions")
        
        else:
            st.warning("No data available for Wave Radar analysis.")
    
    # Tab 3: Analysis (previously tab2)
    with tab3:
        st.markdown("### 📊 Market Analysis")
        
        if not filtered_df.empty:
            # Score distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig_dist = Visualizer.create_score_distribution(filtered_df)
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                # Pattern analysis
                fig_patterns = Visualizer.create_pattern_analysis(filtered_df)
                st.plotly_chart(fig_patterns, use_container_width=True)
            
            # Sector performance
            st.markdown("#### Sector Performance")
            try:
                sector_df = filtered_df.groupby('sector').agg({
                    'master_score': ['mean', 'count'],
                    'rvol': 'mean',
                    'ret_30d': 'mean'
                }).round(2)
                
                if not sector_df.empty:
                    sector_df.columns = ['Avg Score', 'Count', 'Avg RVOL', 'Avg 30D Ret']
                    sector_df = sector_df.sort_values('Avg Score', ascending=False)
                    
                    # Add percentage column
                    sector_df['% of Total'] = (sector_df['Count'] / len(filtered_df) * 100).round(1)
                    
                    st.dataframe(
                        sector_df.style.background_gradient(subset=['Avg Score']),
                        use_container_width=True
                    )
                else:
                    st.info("No sector data available for analysis.")
            except Exception as e:
                logger.error(f"Error in sector analysis: {str(e)}")
                st.error("Unable to perform sector analysis with current data.")
            
            # Category performance
            st.markdown("#### Category Performance")
            try:
                category_df = filtered_df.groupby('category').agg({
                    'master_score': ['mean', 'count'],
                    'category_percentile': 'mean'
                }).round(2)
                
                if not category_df.empty:
                    category_df.columns = ['Avg Score', 'Count', 'Avg Cat %ile']
                    category_df = category_df.sort_values('Avg Score', ascending=False)
                    
                    st.dataframe(
                        category_df.style.background_gradient(subset=['Avg Score']),
                        use_container_width=True
                    )
                else:
                    st.info("No category data available for analysis.")
            except Exception as e:
                logger.error(f"Error in category analysis: {str(e)}")
                st.error("Unable to perform category analysis with current data.")
            
            # Trend Analysis
            if 'trend_quality' in filtered_df.columns:
                st.markdown("#### 📈 Trend Distribution")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Trend distribution pie chart
                    trend_dist = pd.cut(
                        filtered_df['trend_quality'],
                        bins=[0, 40, 60, 80, 100],
                        labels=['⚠️ Weak/Down', '➡️ Neutral', '✅ Good Up', '🔥 Strong Up']
                    ).value_counts()
                    
                    fig_trend = px.pie(
                        values=trend_dist.values,
                        names=trend_dist.index,
                        title="Trend Quality Distribution",
                        color_discrete_map={
                            '🔥 Strong Up': '#2ecc71',
                            '✅ Good Up': '#3498db',
                            '➡️ Neutral': '#f39c12',
                            '⚠️ Weak/Down': '#e74c3c'
                        }
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                
                with col2:
                    # Trend statistics
                    st.markdown("**Trend Statistics**")
                    trend_stats = {
                        "Average Trend Score": f"{filtered_df['trend_quality'].mean():.1f}",
                        "Stocks Above All SMAs": f"{(filtered_df['trend_quality'] >= 85).sum()}",
                        "Stocks in Uptrend (60+)": f"{(filtered_df['trend_quality'] >= 60).sum()}",
                        "Stocks in Downtrend (<40)": f"{(filtered_df['trend_quality'] < 40).sum()}"
                    }
                    for label, value in trend_stats.items():
                        st.metric(label, value)
        
        else:
            st.info("No data available for analysis.")
    
    # Tab 4: Search (previously tab3)
    with tab4:
        st.markdown("### 🔍 Advanced Stock Search")
        
        # Search interface
        col1, col2 = st.columns([4, 1])
        
        with col1:
            search_query = st.text_input(
                "Search stocks",
                placeholder="Enter ticker or company name...",
                help="Search by ticker symbol or company name"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_clicked = st.button("🔎 Search", type="primary", use_container_width=True)
        
        # Perform search
        if search_query or search_clicked:
            search_results = SearchEngine.search_stocks(
                filtered_df, 
                search_query,
                st.session_state.search_index
            )
            
            if not search_results.empty:
                st.success(f"Found {len(search_results)} matching stock(s)")
                
                # Display each result in detail
                for idx, stock in search_results.iterrows():
                    with st.expander(
                        f"📊 {stock['ticker']} - {stock['company_name']} "
                        f"(Rank #{int(stock['rank'])})",
                        expanded=True
                    ):
                        # Header metrics
                        metric_cols = st.columns(6)
                        
                        with metric_cols[0]:
                            st.metric(
                                "Master Score",
                                f"{stock['master_score']:.1f}",
                                f"Rank #{int(stock['rank'])}"
                            )
                        
                        with metric_cols[1]:
                            price_value = f"₹{stock['price']:,.0f}" if pd.notna(stock.get('price')) else "N/A"
                            ret_1d_value = f"{stock['ret_1d']:.1f}%" if pd.notna(stock.get('ret_1d')) else None
                            st.metric(
                                "Price",
                                price_value,
                                ret_1d_value
                            )
                        
                        with metric_cols[2]:
                            st.metric(
                                "From Low",
                                f"{stock['from_low_pct']:.0f}%",
                                "52-week range position"
                            )
                        
                        with metric_cols[3]:
                            st.metric(
                                "30D Return",
                                f"{stock['ret_30d']:.1f}%",
                                "↑" if stock['ret_30d'] > 0 else "↓"
                            )
                        
                        with metric_cols[4]:
                            st.metric(
                                "RVOL",
                                f"{stock['rvol']:.1f}x",
                                "High" if stock['rvol'] > 2 else "Normal"
                            )
                        
                        with metric_cols[5]:
                            st.metric(
                                "Category %ile",
                                f"{stock.get('category_percentile', 0):.0f}",
                                stock['category']
                            )
                        
                        # Score breakdown
                        st.markdown("#### 📈 Score Components")
                        score_cols = st.columns(6)
                        
                        components = [
                            ("Position", stock['position_score'], CONFIG.POSITION_WEIGHT),
                            ("Volume", stock['volume_score'], CONFIG.VOLUME_WEIGHT),
                            ("Momentum", stock['momentum_score'], CONFIG.MOMENTUM_WEIGHT),
                            ("Acceleration", stock['acceleration_score'], CONFIG.ACCELERATION_WEIGHT),
                            ("Breakout", stock['breakout_score'], CONFIG.BREAKOUT_WEIGHT),
                            ("RVOL", stock['rvol_score'], CONFIG.RVOL_WEIGHT)
                        ]
                        
                        for i, (name, score, weight) in enumerate(components):
                            with score_cols[i]:
                                # Color coding
                                if score >= 80:
                                    color = "🟢"
                                elif score >= 60:
                                    color = "🟡"
                                else:
                                    color = "🔴"
                                
                                st.markdown(
                                    f"**{name}**<br>"
                                    f"{color} {score:.0f}<br>"
                                    f"<small>Weight: {weight:.0%}</small>",
                                    unsafe_allow_html=True
                                )
                        
                        # Patterns
                        if stock.get('patterns'):
                            st.markdown(f"**🎯 Patterns:** {stock['patterns']}")
                        
                        # Additional details in columns
                        detail_cols = st.columns(3)
                        
                        with detail_cols[0]:
                            st.markdown("**📊 Classification**")
                            st.text(f"Sector: {stock['sector']}")
                            st.text(f"Category: {stock['category']}")
                            if 'eps_tier' in stock:
                                st.text(f"EPS Tier: {stock['eps_tier']}")
                            if 'pe_tier' in stock:
                                st.text(f"PE Tier: {stock['pe_tier']}")
                            
                            # SMART FUNDAMENTAL DISPLAY
                            if show_fundamentals:
                                st.markdown("**💰 Fundamentals**")
                                
                                # PE Ratio with intelligent display
                                if 'pe' in stock and pd.notna(stock['pe']):
                                    pe_val = stock['pe']
                                    if pe_val <= 0:
                                        pe_display = "Loss"
                                        pe_color = "🔴"
                                    elif pe_val < 15:
                                        pe_display = f"{pe_val:.1f}x"
                                        pe_color = "🟢"
                                    elif pe_val < 25:
                                        pe_display = f"{pe_val:.1f}x"
                                        pe_color = "🟡"
                                    else:
                                        pe_display = f"{pe_val:.1f}x"
                                        pe_color = "🔴"
                                    st.text(f"PE Ratio: {pe_color} {pe_display}")
                                else:
                                    st.text("PE Ratio: - (N/A)")
                                
                                # EPS Current
                                if 'eps_current' in stock and pd.notna(stock['eps_current']):
                                    st.text(f"EPS: ₹{stock['eps_current']:.2f}")
                                else:
                                    st.text("EPS: - (N/A)")
                                
                                # EPS Change with color
                                if 'eps_change_pct' in stock and pd.notna(stock['eps_change_pct']):
                                    eps_chg = stock['eps_change_pct']
                                    if eps_chg > 0:
                                        eps_emoji = "📈"
                                        eps_color = "green"
                                    else:
                                        eps_emoji = "📉"
                                        eps_color = "red"
                                    st.text(f"EPS Growth: {eps_emoji} {eps_chg:+.1f}%")
                                else:
                                    st.text("EPS Growth: - (N/A)")
                        
                        with detail_cols[1]:
                            st.markdown("**📈 Performance**")
                            for period, col in [
                                ("1 Day", 'ret_1d'),
                                ("7 Days", 'ret_7d'),
                                ("30 Days", 'ret_30d'),
                                ("3 Months", 'ret_3m'),
                                ("6 Months", 'ret_6m')
                            ]:
                                if col in stock.index and pd.notna(stock[col]):
                                    st.text(f"{period}: {stock[col]:.1f}%")
                        
                        with detail_cols[2]:
                            st.markdown("**🔍 Technicals**")
                            st.text(f"52W Low: ₹{stock.get('low_52w', 0):,.0f}")
                            st.text(f"52W High: ₹{stock.get('high_52w', 0):,.0f}")
                            st.text(f"From High: {stock.get('from_high_pct', 0):.0f}%")
                            
                            # Trend quality
                            if 'trend_quality' in stock:
                                tq = stock['trend_quality']
                                if tq >= 80:
                                    st.text(f"Trend: 💪 Strong ({tq:.0f})")
                                elif tq >= 60:
                                    st.text(f"Trend: 👍 Good ({tq:.0f})")
                                else:
                                    st.text(f"Trend: 👎 Weak ({tq:.0f})")
            
            else:
                st.warning("No stocks found matching your search criteria.")
    
    # Tab 5: Visualizations (previously tab4)
    with tab5:
        st.markdown("### 📈 Interactive Visualizations")
        
        if not filtered_df.empty:
            # Top stocks breakdown
            st.markdown("#### Master Score Breakdown - Top Stocks")
            n_stocks = st.slider(
                "Number of stocks to display",
                min_value=5,
                max_value=50,
                value=20,
                step=5
            )
            
            fig_breakdown = Visualizer.create_master_score_breakdown(filtered_df, n_stocks)
            st.plotly_chart(fig_breakdown, use_container_width=True)
            
            # Sector scatter plot
            if len(filtered_df.groupby('sector')) >= 3:
                st.markdown("#### Sector Performance Scatter")
                fig_sector = Visualizer.create_sector_performance_scatter(filtered_df)
                st.plotly_chart(fig_sector, use_container_width=True)
            
            # Custom analysis
            st.markdown("#### Custom Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox(
                    "X-Axis",
                    options=['master_score', 'percentile', 'rvol', 'ret_30d', 
                            'position_score', 'volume_score'],
                    index=2
                )
            
            with col2:
                y_axis = st.selectbox(
                    "Y-Axis",
                    options=['master_score', 'ret_30d', 'rvol', 'from_low_pct',
                            'momentum_score', 'acceleration_score'],
                    index=0
                )
            
            # Create custom scatter plot
            fig_custom = px.scatter(
                filtered_df.head(200),  # Limit for performance
                x=x_axis,
                y=y_axis,
                color='master_score',
                size='rvol',
                hover_data=['ticker', 'company_name', 'category', 'sector'],
                title=f"{y_axis.replace('_', ' ').title()} vs {x_axis.replace('_', ' ').title()}",
                color_continuous_scale='Viridis'
            )
            
            fig_custom.update_layout(template='plotly_white')
            st.plotly_chart(fig_custom, use_container_width=True)
        
        else:
            st.info("No data available for visualization.")
    
    # Tab 6: Export (previously tab5)
    with tab6:
        st.markdown("### 📥 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Excel Report")
            st.markdown(
                "Comprehensive multi-sheet report including:\n"
                "- Top 100 stocks with all scores\n"
                "- Complete stock list\n"
                "- Sector analysis\n"
                "- Category analysis\n"
                "- Pattern frequency analysis\n"
                "- Wave Radar signals (momentum shifts)\n"
                "- Smart money flow tracking"
            )
            
            if st.button("Generate Excel Report", type="primary", use_container_width=True):
                if len(filtered_df) == 0:
                    st.error("No data to export. Please adjust your filters.")
                else:
                    with st.spinner("Creating Excel report..."):
                        try:
                            excel_file = ExportEngine.create_excel_report(filtered_df)
                            
                            st.download_button(
                                label="📥 Download Excel Report",
                                data=excel_file,
                                file_name=f"wave_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            
                            st.success("Excel report generated successfully!")
                            
                        except Exception as e:
                            st.error(f"Error generating Excel report: {str(e)}")
                            logger.error(f"Excel export error: {str(e)}", exc_info=True)
        
        with col2:
            st.markdown("#### 📄 CSV Export")
            st.markdown(
                "Enhanced CSV format with:\n"
                "- All ranking scores\n"
                "- Price and return data\n"
                "- Pattern detections\n"
                "- Category classifications\n"
                "- Trend quality scores\n"
                "- RVOL and volume metrics\n"
                "- Perfect for Wave Radar analysis"
            )
            
            if st.button("Generate CSV Export", use_container_width=True):
                if len(filtered_df) == 0:
                    st.error("No data to export. Please adjust your filters.")
                else:
                    try:
                        csv_data = ExportEngine.create_csv_export(filtered_df)
                        
                        st.download_button(
                            label="📥 Download CSV File",
                            data=csv_data,
                            file_name=f"wave_detection_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        st.success("CSV export generated successfully!")
                        
                    except Exception as e:
                        st.error(f"Error generating CSV: {str(e)}")
                        logger.error(f"CSV export error: {str(e)}", exc_info=True)
        
        # Export statistics
        st.markdown("---")
        st.markdown("#### 📊 Export Preview")
        
        export_stats = {
            "Total Stocks": len(filtered_df),
            "Average Score": f"{filtered_df['master_score'].mean():.1f}" if not filtered_df.empty and 'master_score' in filtered_df.columns else "N/A",
            "Stocks with Patterns": (filtered_df['patterns'] != '').sum() if 'patterns' in filtered_df.columns else 0,
            "High RVOL (>2x)": (filtered_df['rvol'] > 2).sum() if 'rvol' in filtered_df.columns else 0,
            "Positive 30D Returns": (filtered_df['ret_30d'] > 0).sum() if 'ret_30d' in filtered_df.columns else 0,
            "Data Quality": f"{(1 - filtered_df['master_score'].isna().sum() / len(filtered_df)) * 100:.1f}%" if not filtered_df.empty and 'master_score' in filtered_df.columns else "N/A"
        }
        
        stat_cols = st.columns(3)
        for i, (label, value) in enumerate(export_stats.items()):
            with stat_cols[i % 3]:
                st.metric(label, value)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            Wave Detection Ultimate 3.0 | Professional Edition with Wave Radar™<br>
            <small>Real-time momentum detection • Early entry signals • Smart money flow tracking</small>
        </div>
        """,
        unsafe_allow_html=True
    )

# Run the application
if __name__ == "__main__":
    main()
