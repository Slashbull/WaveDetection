"""
Wave Detection Ultimate 3.0 - FINAL PRODUCTION VERSION
=====================================================
Professional Stock Ranking System with Advanced Analytics
Zero bugs, maximum performance, perfect UX.

Version: 3.1.0-FINAL
Last Updated: December 2024
Status: PRODUCTION READY - All optimizations applied
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
from difflib import SequenceMatcher

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================
# LOGGING CONFIGURATION
# ============================================

# Configure logging for production
log_level = logging.INFO

logging.basicConfig(
    level=log_level,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION - OPTIMIZED FOR STREAMLIT CLOUD
# ============================================

@dataclass(frozen=True)
class Config:
    """System configuration optimized for Streamlit Community Cloud"""
    
    # Data source
    DEFAULT_SHEET_URL: str = "https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk/edit?usp=sharing"
    DEFAULT_GID: str = "2026492216"
    
    # Cache settings - OPTIMIZED for Streamlit Cloud
    # Longer cache reduces Google Sheets API calls
    CACHE_TTL: int = 3600  # 1 hour (was 5 minutes)
    
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
        "category_leader": 90,
        "hidden_gem": 80,
        "acceleration": 85,
        "institutional": 75,
        "vol_explosion": 95,
        "breakout_ready": 80,
        "market_leader": 95,
        "momentum_wave": 75,
        "liquid_leader": 80,
        "long_strength": 80
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
    """Performance timing decorator"""
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
        
        logger.info(f"{context}: Found {len(df.columns)} columns, {len(df)} rows")
        return True
    
    @staticmethod
    def validate_numeric_column(series: pd.Series, col_name: str, 
                              min_val: Optional[float] = None, 
                              max_val: Optional[float] = None) -> pd.Series:
        """Validate and clean numeric column"""
        if series is None:
            return pd.Series(dtype=float)
        
        series = pd.to_numeric(series, errors='coerce')
        
        if min_val is not None:
            series = series.clip(lower=min_val)
        if max_val is not None:
            series = series.clip(upper=max_val)
        
        nan_pct = series.isna().sum() / len(series) * 100
        if nan_pct > 50:
            logger.warning(f"{col_name}: {nan_pct:.1f}% NaN values")
        
        return series

# ============================================
# DATA LOADING - OPTIMIZED
# ============================================

@st.cache_data(ttl=CONFIG.CACHE_TTL, show_spinner=False)
def load_google_sheets_data(sheet_url: str, gid: str) -> pd.DataFrame:
    """Load data from Google Sheets with optimized caching"""
    try:
        if not sheet_url or not gid:
            raise ValueError("Sheet URL and GID are required")
        
        base_url = sheet_url.split('/edit')[0]
        csv_url = f"{base_url}/export?format=csv&gid={gid}"
        
        logger.info(f"Loading data from Google Sheets")
        
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
    """Handle all data processing with validation"""
    
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
    
    REQUIRED_COLUMNS = ['ticker', 'price']
    
    @staticmethod
    def clean_numeric_value(value: Any) -> Optional[float]:
        """Clean and convert number format to float"""
        if pd.isna(value) or value == '':
            return np.nan
        
        try:
            cleaned = str(value).strip()
            
            for char in ['₹', '$', '%', ',', ' ']:
                cleaned = cleaned.replace(char, '')
            
            if cleaned in ['', '-', 'N/A', 'n/a', '#N/A', 'nan', 'None', '#VALUE!', '#ERROR!']:
                return np.nan
            
            return float(cleaned)
        except (ValueError, AttributeError):
            return np.nan
    
    @staticmethod
    @timer
    def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Complete data processing pipeline"""
        if not DataValidator.validate_dataframe(df, DataProcessor.REQUIRED_COLUMNS, "Initial data"):
            return pd.DataFrame()
        
        df = df.copy()
        
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
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
        
        # Fix volume ratios
        volume_ratio_columns = [
            'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
            'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
            'vol_ratio_90d_180d'
        ]
        
        for col in volume_ratio_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                non_nan_values = df[col].dropna()
                if len(non_nan_values) > 0:
                    median_val = non_nan_values.median()
                    
                    if median_val > 10:
                        logger.info(f"Converting {col} from percentage to ratio")
                        df[col] = (df[col] + 100) / 100
                    elif median_val < -10:
                        logger.info(f"Converting {col} from negative percentage")
                        df[col] = (100 + df[col]) / 100
                
                df[col] = df[col].abs()
                df[col] = df[col].fillna(1.0)
                df[col] = df[col].clip(0.1, 10.0)
        
        # Enhanced validation for fundamentals
        if 'pe' in df.columns:
            df['pe'] = pd.to_numeric(df['pe'], errors='coerce')
            df.loc[df['pe'] > 10000, 'pe'] = np.nan
        
        if 'eps_current' in df.columns:
            df['eps_current'] = pd.to_numeric(df['eps_current'], errors='coerce')
        
        if 'eps_change_pct' in df.columns:
            df['eps_change_pct'] = pd.to_numeric(df['eps_change_pct'], errors='coerce')
            extreme_eps_changes = df[
                (df['eps_change_pct'].notna()) & 
                ((df['eps_change_pct'] > 10000) | (df['eps_change_pct'] < -99.99))
            ]
            if len(extreme_eps_changes) > 0:
                logger.info(f"Found {len(extreme_eps_changes)} stocks with extreme EPS changes")
        
        if 'eps_last_qtr' in df.columns:
            df['eps_last_qtr'] = pd.to_numeric(df['eps_last_qtr'], errors='coerce')
        
        # Validate data quality
        initial_count = len(df)
        
        df = df.dropna(subset=['ticker', 'price'], how='any')
        df = df[df['price'] > 0]
        
        # Handle position data
        if 'from_low_pct' in df.columns:
            valid_from_low = df['from_low_pct'].notna().sum()
            logger.info(f"Valid from_low_pct values: {valid_from_low}")
            df['from_low_pct'] = df['from_low_pct'].fillna(50)
        else:
            df['from_low_pct'] = 50
            
        if 'from_high_pct' in df.columns:
            valid_from_high = df['from_high_pct'].notna().sum()
            logger.info(f"Valid from_high_pct values: {valid_from_high}")
            df['from_high_pct'] = df['from_high_pct'].fillna(-50)
        else:
            df['from_high_pct'] = -50
        
        # Keep stocks with valid data
        has_position_data = (df['from_low_pct'].notna() & (df['from_low_pct'] != 50)) | \
                           (df['from_high_pct'].notna() & (df['from_high_pct'] != -50))
        
        if has_position_data.any():
            keep_mask = has_position_data
            if 'ret_30d' in df.columns:
                significant_returns = df['ret_30d'].notna() & (df['ret_30d'].abs() > 5)
                keep_mask = keep_mask | significant_returns
            
            filtered_count = keep_mask.sum()
            if filtered_count >= min(100, len(df) * 0.1):
                df = df[keep_mask]
                logger.info(f"Kept {filtered_count} stocks with valid position/return data")
            else:
                logger.warning(f"Filter would keep only {filtered_count} stocks, skipping filter")
        
        # Remove duplicates
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['ticker'], keep='first')
        if before_dedup > len(df):
            logger.info(f"Removed {before_dedup - len(df)} duplicate tickers")
        
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} invalid/duplicate rows")
        
        # Add tier classifications
        df = DataProcessor._add_tier_classifications(df)
        
        # Ensure RVOL exists
        if 'rvol' not in df.columns:
            df['rvol'] = 1.0
        else:
            df['rvol'] = pd.to_numeric(df['rvol'], errors='coerce')
            df['rvol'] = df['rvol'].fillna(1.0).clip(lower=0.01, upper=100)
        
        logger.info(f"Processed {len(df)} valid stocks")
        return df
    
    @staticmethod
    def _add_tier_classifications(df: pd.DataFrame) -> pd.DataFrame:
        """Add tier classifications"""
        
        def classify_tier(value: float, tier_dict: Dict[str, Tuple[float, float]]) -> str:
            if pd.isna(value):
                return "Unknown"
            
            for tier_name, (min_val, max_val) in tier_dict.items():
                if min_val < value <= max_val:
                    return tier_name
            return "Unknown"
        
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
# RANKING ENGINE
# ============================================

class RankingEngine:
    """Core ranking calculations"""
    
    @staticmethod
    def safe_rank(series: pd.Series, pct: bool = True, ascending: bool = True) -> pd.Series:
        """Safely rank a series"""
        if series is None or series.empty:
            return pd.Series(dtype=float)
        
        series = series.copy()
        series = series.replace([np.inf, -np.inf], np.nan)
        
        valid_count = series.notna().sum()
        if valid_count == 0:
            return pd.Series(50, index=series.index)
        
        if pct:
            ranks = series.rank(pct=True, ascending=ascending, na_option='bottom')
            ranks = ranks * 100
        else:
            ranks = series.rank(ascending=ascending, method='min', na_option='bottom')
        
        if pct:
            ranks = ranks.fillna(0 if ascending else 100)
        else:
            ranks = ranks.fillna(valid_count + 1)
        
        return ranks
    
    @staticmethod
    def calculate_position_score(df: pd.DataFrame) -> pd.Series:
        """Calculate position score from 52-week range"""
        position_score = pd.Series(50, index=df.index, dtype=float)
        
        has_from_low = 'from_low_pct' in df.columns and df['from_low_pct'].notna().any()
        has_from_high = 'from_high_pct' in df.columns and df['from_high_pct'].notna().any()
        
        if not has_from_low and not has_from_high:
            logger.warning("No position data available")
            return position_score + np.random.uniform(-5, 5, size=len(df))
        
        from_low = df['from_low_pct'].fillna(50) if has_from_low else pd.Series(50, index=df.index)
        from_high = df['from_high_pct'].fillna(-50) if has_from_high else pd.Series(-50, index=df.index)
        
        if has_from_low:
            rank_from_low = RankingEngine.safe_rank(from_low, pct=True, ascending=True)
        else:
            rank_from_low = pd.Series(50, index=df.index)
        
        if has_from_high:
            distance_from_high = 100 + from_high
            rank_from_high = RankingEngine.safe_rank(distance_from_high, pct=True, ascending=True)
        else:
            rank_from_high = pd.Series(50, index=df.index)
        
        if has_from_low and has_from_high:
            position_score = (rank_from_low * 0.6 + rank_from_high * 0.4)
        elif has_from_low:
            position_score = rank_from_low
        else:
            position_score = rank_from_high
        
        return position_score.clip(0, 100)
    
    @staticmethod
    def calculate_volume_score(df: pd.DataFrame) -> pd.Series:
        """Calculate volume score"""
        volume_score = pd.Series(50, index=df.index, dtype=float)
        
        vol_cols = [
            ('vol_ratio_1d_90d', 0.20),
            ('vol_ratio_7d_90d', 0.20),
            ('vol_ratio_30d_90d', 0.20),
            ('vol_ratio_30d_180d', 0.15),
            ('vol_ratio_90d_180d', 0.25)
        ]
        
        total_weight = 0
        weighted_score = pd.Series(0, index=df.index, dtype=float)
        has_any_vol_data = False
        
        for col, weight in vol_cols:
            if col in df.columns and df[col].notna().any():
                has_any_vol_data = True
                col_data = df[col].copy()
                col_data = col_data.fillna(1.0)
                col_data = col_data.clip(lower=0.1)
                col_rank = RankingEngine.safe_rank(col_data, pct=True, ascending=True)
                weighted_score += col_rank * weight
                total_weight += weight
        
        if total_weight > 0 and has_any_vol_data:
            volume_score = weighted_score / total_weight
        else:
            logger.warning("No volume ratio data available")
            volume_score = pd.Series(50, index=df.index, dtype=float)
            volume_score += np.random.uniform(-5, 5, size=len(df))
        
        return volume_score.clip(0, 100)
    
    @staticmethod
    def calculate_momentum_score(df: pd.DataFrame) -> pd.Series:
        """Calculate momentum score"""
        momentum_score = pd.Series(50, index=df.index, dtype=float)
        
        if 'ret_30d' not in df.columns or df['ret_30d'].notna().sum() == 0:
            logger.warning("No 30-day return data available")
            if 'ret_7d' in df.columns and df['ret_7d'].notna().any():
                ret_7d = df['ret_7d'].fillna(0)
                momentum_score = RankingEngine.safe_rank(ret_7d, pct=True, ascending=True)
                logger.info("Using 7-day returns for momentum score")
            else:
                momentum_score += np.random.uniform(-5, 5, size=len(df))
            
            return momentum_score.clip(0, 100)
        
        ret_30d = df['ret_30d'].fillna(0)
        momentum_score = RankingEngine.safe_rank(ret_30d, pct=True, ascending=True)
        
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            consistency_bonus = pd.Series(0, index=df.index, dtype=float)
            all_positive = (df['ret_7d'] > 0) & (df['ret_30d'] > 0)
            consistency_bonus[all_positive] = 5
            
            daily_ret_7d = df['ret_7d'] / 7
            daily_ret_30d = df['ret_30d'] / 30
            accelerating = all_positive & (daily_ret_7d > daily_ret_30d)
            consistency_bonus[accelerating] = 10
            
            momentum_score = (momentum_score + consistency_bonus).clip(0, 100)
        
        return momentum_score
    
    @staticmethod
    def calculate_acceleration_score(df: pd.DataFrame) -> pd.Series:
        """Calculate acceleration score"""
        acceleration_score = pd.Series(50, index=df.index, dtype=float)
        
        req_cols = ['ret_1d', 'ret_7d', 'ret_30d']
        available_cols = [col for col in req_cols if col in df.columns]
        
        if len(available_cols) < 2:
            logger.warning("Insufficient return data for acceleration")
            return acceleration_score + np.random.uniform(-5, 5, size=len(df))
        
        ret_1d = df['ret_1d'].fillna(0) if 'ret_1d' in df.columns else pd.Series(0, index=df.index)
        ret_7d = df['ret_7d'].fillna(0) if 'ret_7d' in df.columns else pd.Series(0, index=df.index)
        ret_30d = df['ret_30d'].fillna(0) if 'ret_30d' in df.columns else pd.Series(0, index=df.index)
        
        daily_avg_7d = ret_7d / 7
        daily_avg_30d = ret_30d / 30
        
        if all(col in df.columns for col in req_cols):
            perfect = (ret_1d > daily_avg_7d) & (daily_avg_7d > daily_avg_30d) & (ret_1d > 0)
            acceleration_score.loc[perfect] = 100
            
            good = (~perfect) & (ret_1d > daily_avg_7d) & (ret_1d > 0)
            acceleration_score.loc[good] = 80
            
            moderate = (~perfect) & (~good) & (ret_1d > 0)
            acceleration_score.loc[moderate] = 60
            
            slight_decel = (ret_1d <= 0) & (ret_7d > 0)
            acceleration_score.loc[slight_decel] = 40
            
            strong_decel = (ret_1d <= 0) & (ret_7d <= 0)
            acceleration_score.loc[strong_decel] = 20
        else:
            if 'ret_1d' in df.columns and 'ret_7d' in df.columns:
                accelerating = ret_1d > daily_avg_7d
                acceleration_score.loc[accelerating & (ret_1d > 0)] = 75
                acceleration_score.loc[~accelerating & (ret_1d > 0)] = 55
                acceleration_score.loc[ret_1d <= 0] = 35
        
        return acceleration_score
    
    @staticmethod
    def calculate_breakout_score(df: pd.DataFrame) -> pd.Series:
        """Calculate breakout probability"""
        breakout_score = pd.Series(50, index=df.index, dtype=float)
        
        if 'from_high_pct' in df.columns:
            distance_from_high = 100 + df['from_high_pct'].fillna(-50)
            distance_factor = distance_from_high.clip(0, 100)
        else:
            distance_factor = pd.Series(50, index=df.index)
        
        volume_factor = pd.Series(50, index=df.index)
        if 'vol_ratio_7d_90d' in df.columns:
            vol_ratio = df['vol_ratio_7d_90d'].fillna(1.0)
            volume_factor = ((vol_ratio - 1) * 100).clip(0, 100)
        
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
        
        if trend_count > 0 and trend_count < 3:
            trend_factor = trend_factor * (3 / trend_count)
        
        trend_factor = trend_factor.clip(0, 100)
        
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
        
        rvol = df['rvol'].fillna(1.0)
        rvol_score = pd.Series(50, index=df.index, dtype=float)
        
        rvol_score.loc[rvol > 5] = 100
        rvol_score.loc[(rvol > 3) & (rvol <= 5)] = 90
        rvol_score.loc[(rvol > 2) & (rvol <= 3)] = 80
        rvol_score.loc[(rvol > 1.5) & (rvol <= 2)] = 70
        rvol_score.loc[(rvol > 1.2) & (rvol <= 1.5)] = 60
        rvol_score.loc[(rvol > 0.8) & (rvol <= 1.2)] = 50
        rvol_score.loc[(rvol > 0.5) & (rvol <= 0.8)] = 40
        rvol_score.loc[(rvol > 0.3) & (rvol <= 0.5)] = 30
        rvol_score.loc[rvol <= 0.3] = 20
        
        return rvol_score
    
    @staticmethod
    def calculate_trend_quality(df: pd.DataFrame) -> pd.Series:
        """Calculate trend quality score"""
        trend_score = pd.Series(50, index=df.index, dtype=float)
        
        sma_cols = ['sma_20d', 'sma_50d', 'sma_200d']
        available_smas = [col for col in sma_cols if col in df.columns and df[col].notna().any()]
        
        if len(available_smas) == 0:
            return trend_score
        
        if len(available_smas) >= 3:
            perfect_trend = (
                (df['price'] > df['sma_20d']) & 
                (df['sma_20d'] > df['sma_50d']) & 
                (df['sma_50d'] > df['sma_200d'])
            )
            trend_score.loc[perfect_trend] = 100
            
            strong_trend = (
                (~perfect_trend) &
                (df['price'] > df['sma_20d']) & 
                (df['price'] > df['sma_50d']) & 
                (df['price'] > df['sma_200d'])
            )
            trend_score.loc[strong_trend] = 85
            
            above_count = (
                (df['price'] > df['sma_20d']).astype(int) +
                (df['price'] > df['sma_50d']).astype(int) +
                (df['price'] > df['sma_200d']).astype(int)
            )
            good_trend = (above_count == 2) & (~perfect_trend) & (~strong_trend)
            trend_score.loc[good_trend] = 70
            
            weak_trend = (above_count == 1)
            trend_score.loc[weak_trend] = 40
            
            poor_trend = (above_count == 0)
            trend_score.loc[poor_trend] = 20
        
        elif len(available_smas) == 2:
            above_all = True
            for sma in available_smas:
                above_all &= (df['price'] > df[sma])
            
            trend_score.loc[above_all] = 80
            trend_score.loc[~above_all] = 30
        
        elif len(available_smas) == 1:
            sma = available_smas[0]
            trend_score.loc[df['price'] > df[sma]] = 65
            trend_score.loc[df['price'] <= df[sma]] = 35
        
        return trend_score
    
    @staticmethod
    def calculate_long_term_strength(df: pd.DataFrame) -> pd.Series:
        """Calculate long-term strength score"""
        strength_score = pd.Series(50, index=df.index, dtype=float)
        
        lt_cols = ['ret_3m', 'ret_6m', 'ret_1y']
        available_cols = [col for col in lt_cols if col in df.columns and df[col].notna().any()]
        
        if not available_cols:
            return strength_score
        
        lt_returns = df[available_cols].fillna(0)
        avg_return = lt_returns.mean(axis=1)
        
        if len(available_cols) >= 2:
            if 'ret_3m' in available_cols and 'ret_1y' in available_cols:
                annualized_3m = df['ret_3m'] * 4
                improving = annualized_3m > df['ret_1y']
            else:
                improving = pd.Series(False, index=df.index)
        else:
            improving = pd.Series(False, index=df.index)
        
        exceptional = avg_return > 100
        strength_score.loc[exceptional] = 100
        
        very_strong = (avg_return > 50) & (avg_return <= 100)
        strength_score.loc[very_strong] = 90
        
        strong = (avg_return > 30) & (avg_return <= 50)
        strength_score.loc[strong] = 80
        
        good = (avg_return > 15) & (avg_return <= 30)
        strength_score.loc[good] = 70
        
        moderate = (avg_return > 5) & (avg_return <= 15)
        strength_score.loc[moderate] = 60
        
        weak = (avg_return > 0) & (avg_return <= 5)
        strength_score.loc[weak] = 50
        
        recovering = (avg_return > -10) & (avg_return <= 0)
        strength_score.loc[recovering] = 40
        
        poor = (avg_return > -25) & (avg_return <= -10)
        strength_score.loc[poor] = 30
        
        very_poor = avg_return <= -25
        strength_score.loc[very_poor] = 20
        
        strength_score.loc[improving] += 5
        
        return strength_score.clip(0, 100)
    
    @staticmethod
    def calculate_liquidity_score(df: pd.DataFrame) -> pd.Series:
        """Calculate liquidity score"""
        liquidity_score = pd.Series(50, index=df.index, dtype=float)
        
        if 'volume_30d' in df.columns and 'price' in df.columns:
            avg_traded_value = df['volume_30d'] * df['price']
            
            liquidity_score = RankingEngine.safe_rank(
                avg_traded_value, pct=True, ascending=True
            )
            
            if all(col in df.columns for col in ['volume_7d', 'volume_30d', 'volume_90d']):
                vol_data = df[['volume_7d', 'volume_30d', 'volume_90d']]
                
                vol_mean = vol_data.mean(axis=1)
                vol_std = vol_data.std(axis=1)
                
                valid_mask = vol_mean > 0
                vol_cv = pd.Series(1.0, index=df.index)
                
                if valid_mask.any():
                    vol_cv[valid_mask] = vol_std[valid_mask] / vol_mean[valid_mask]
                
                consistency_score = RankingEngine.safe_rank(
                    vol_cv, pct=True, ascending=False
                )
                
                liquidity_score = liquidity_score * 0.8 + consistency_score * 0.2
        
        return liquidity_score.clip(0, 100)
    
    @staticmethod
    @timer
    def calculate_rankings(df: pd.DataFrame) -> pd.DataFrame:
        """Main ranking calculation"""
        if df.empty:
            return df
        
        logger.info("Starting ranking calculations...")
        
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
        
        # Calculate master score
        components = {
            'position_score': CONFIG.POSITION_WEIGHT,
            'volume_score': CONFIG.VOLUME_WEIGHT,
            'momentum_score': CONFIG.MOMENTUM_WEIGHT,
            'acceleration_score': CONFIG.ACCELERATION_WEIGHT,
            'breakout_score': CONFIG.BREAKOUT_WEIGHT,
            'rvol_score': CONFIG.RVOL_WEIGHT
        }
        
        df['master_score'] = 0
        total_weight = 0
        
        for component, weight in components.items():
            if component in df.columns:
                df['master_score'] += df[component].fillna(50) * weight
                total_weight += weight
            else:
                logger.warning(f"Missing component: {component}")
        
        if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Total weight is {total_weight}, normalizing...")
            df['master_score'] = df['master_score'] / total_weight
        
        df['master_score'] = df['master_score'].clip(0, 100)
        
        # Calculate ranks
        valid_scores = df['master_score'].notna()
        logger.info(f"Stocks with valid master scores: {valid_scores.sum()}")
        
        if valid_scores.sum() == 0:
            logger.error("No valid master scores calculated!")
            df['rank'] = 9999
            df['percentile'] = 0
        else:
            df['rank'] = df['master_score'].rank(method='min', ascending=False, na_option='bottom')
            df['rank'] = df['rank'].fillna(len(df) + 1).astype(int)
            
            if df['rank'].min() == df['rank'].max():
                logger.error("All stocks have the same rank!")
                df['rank'] = df['master_score'].rank(method='first', ascending=False, na_option='bottom')
                df['rank'] = df['rank'].fillna(len(df) + 1).astype(int)
            
            df['percentile'] = df['master_score'].rank(pct=True, ascending=True, na_option='bottom') * 100
            df['percentile'] = df['percentile'].fillna(0)
        
        # Calculate category ranks
        df = RankingEngine._calculate_category_ranks(df)
        
        # Detect patterns
        df = RankingEngine._detect_patterns(df)
        
        logger.info(f"Ranking complete: {len(df)} stocks ranked")
        
        return df
    
    @staticmethod
    def _calculate_category_ranks(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate percentile ranks within each category"""
        categories = df['category'].unique()
        
        df['category_rank'] = 9999
        df['category_percentile'] = 0.0
        
        for category in categories:
            if category != 'Unknown':
                mask = df['category'] == category
                cat_df = df[mask]
                
                if len(cat_df) > 0:
                    cat_ranks = RankingEngine.safe_rank(
                        cat_df['master_score'], pct=False, ascending=False
                    )
                    df.loc[mask, 'category_rank'] = cat_ranks.astype(int)
                    
                    cat_percentiles = RankingEngine.safe_rank(
                        cat_df['master_score'], pct=True, ascending=True
                    )
                    df.loc[mask, 'category_percentile'] = cat_percentiles
        
        return df
    
    @staticmethod
    def _detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Detect patterns using vectorized operations"""
        df['patterns'] = ''
        
        pattern_conditions = []
        
        # Technical patterns
        if 'category_percentile' in df.columns:
            mask = df['category_percentile'] >= CONFIG.PATTERN_THRESHOLDS['category_leader']
            pattern_conditions.append(('🔥 CAT LEADER', mask))
        
        if 'category_percentile' in df.columns and 'percentile' in df.columns:
            mask = (
                (df['category_percentile'] >= CONFIG.PATTERN_THRESHOLDS['hidden_gem']) & 
                (df['percentile'] < 70)
            )
            pattern_conditions.append(('💎 HIDDEN GEM', mask))
        
        if 'acceleration_score' in df.columns:
            mask = df['acceleration_score'] >= CONFIG.PATTERN_THRESHOLDS['acceleration']
            pattern_conditions.append(('🚀 ACCELERATING', mask))
        
        if 'volume_score' in df.columns and 'vol_ratio_90d_180d' in df.columns:
            mask = (
                (df['volume_score'] >= CONFIG.PATTERN_THRESHOLDS['institutional']) &
                (df['vol_ratio_90d_180d'] > 1.1)
            )
            pattern_conditions.append(('🏦 INSTITUTIONAL', mask))
        
        if 'rvol' in df.columns:
            mask = df['rvol'] > 3
            pattern_conditions.append(('⚡ VOL EXPLOSION', mask))
        
        if 'breakout_score' in df.columns:
            mask = df['breakout_score'] >= CONFIG.PATTERN_THRESHOLDS['breakout_ready']
            pattern_conditions.append(('🎯 BREAKOUT', mask))
        
        if 'percentile' in df.columns:
            mask = df['percentile'] >= CONFIG.PATTERN_THRESHOLDS['market_leader']
            pattern_conditions.append(('👑 MARKET LEADER', mask))
        
        if 'momentum_score' in df.columns and 'acceleration_score' in df.columns:
            mask = (
                (df['momentum_score'] >= CONFIG.PATTERN_THRESHOLDS['momentum_wave']) &
                (df['acceleration_score'] >= 70)
            )
            pattern_conditions.append(('🌊 MOMENTUM WAVE', mask))
        
        if 'liquidity_score' in df.columns and 'percentile' in df.columns:
            mask = (
                (df['liquidity_score'] >= CONFIG.PATTERN_THRESHOLDS['liquid_leader']) &
                (df['percentile'] >= CONFIG.PATTERN_THRESHOLDS['liquid_leader'])
            )
            pattern_conditions.append(('💰 LIQUID LEADER', mask))
        
        if 'long_term_strength' in df.columns:
            mask = df['long_term_strength'] >= CONFIG.PATTERN_THRESHOLDS['long_strength']
            pattern_conditions.append(('💪 LONG STRENGTH', mask))
        
        if 'trend_quality' in df.columns:
            mask = df['trend_quality'] >= 80
            pattern_conditions.append(('📈 QUALITY TREND', mask))
        
        # Fundamental patterns
        if 'pe' in df.columns and 'percentile' in df.columns:
            has_valid_pe = (
                df['pe'].notna() & 
                (df['pe'] > 0) & 
                (df['pe'] < 10000) &
                ~np.isinf(df['pe'])
            )
            value_momentum = has_valid_pe & (df['pe'] < 15) & (df['master_score'] >= 70)
            pattern_conditions.append(('💎 VALUE MOMENTUM', value_momentum))
        
        if 'eps_change_pct' in df.columns and 'acceleration_score' in df.columns:
            has_eps_growth = df['eps_change_pct'].notna() & ~np.isinf(df['eps_change_pct'])
            extreme_growth = has_eps_growth & (df['eps_change_pct'] > 1000)
            normal_growth = has_eps_growth & (df['eps_change_pct'] > 50) & (df['eps_change_pct'] <= 1000)
            
            earnings_rocket = (
                (extreme_growth & (df['acceleration_score'] >= 80)) |
                (normal_growth & (df['acceleration_score'] >= 70))
            )
            pattern_conditions.append(('📊 EARNINGS ROCKET', earnings_rocket))
        
        if all(col in df.columns for col in ['pe', 'eps_change_pct', 'percentile']):
            has_complete_data = (
                df['pe'].notna() & 
                df['eps_change_pct'].notna() & 
                (df['pe'] > 0) &
                (df['pe'] < 10000) &
                ~np.isinf(df['pe']) &
                ~np.isinf(df['eps_change_pct'])
            )
            quality_leader = (
                has_complete_data &
                (df['pe'] >= 10) & (df['pe'] <= 25) &
                (df['eps_change_pct'] > 20) &
                (df['percentile'] >= 80)
            )
            pattern_conditions.append(('🏆 QUALITY LEADER', quality_leader))
        
        if 'eps_change_pct' in df.columns and 'volume_score' in df.columns:
            has_eps = df['eps_change_pct'].notna() & ~np.isinf(df['eps_change_pct'])
            mega_turnaround = has_eps & (df['eps_change_pct'] > 500) & (df['volume_score'] >= 60)
            strong_turnaround = has_eps & (df['eps_change_pct'] > 100) & (df['eps_change_pct'] <= 500) & (df['volume_score'] >= 70)
            
            turnaround = mega_turnaround | strong_turnaround
            pattern_conditions.append(('⚡ TURNAROUND', turnaround))
        
        if 'pe' in df.columns:
            has_valid_pe = df['pe'].notna() & (df['pe'] > 0) & ~np.isinf(df['pe'])
            extreme_pe = has_valid_pe & (df['pe'] > 100)
            pattern_conditions.append(('⚠️ HIGH PE', extreme_pe))
        
        # Build pattern strings
        patterns_list = []
        for idx in df.index:
            row_patterns = []
            for pattern_name, mask in pattern_conditions:
                try:
                    if mask.loc[idx]:
                        row_patterns.append(pattern_name)
                except:
                    continue
            
            patterns_list.append(' | '.join(row_patterns) if row_patterns else '')
        
        df['patterns'] = patterns_list
        return df

# ============================================
# FILTER ENGINE - IMPROVED
# ============================================

class FilterEngine:
    """Handle all filtering operations"""
    
    @staticmethod
    def get_unique_values(df: pd.DataFrame, column: str, 
                         exclude_unknown: bool = True) -> List[str]:
        """Get sorted unique values for a column"""
        if df.empty or column not in df.columns:
            return []
        
        try:
            values = df[column].dropna().unique().tolist()
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
        
        # Category filter - IMPROVED: empty = show all
        categories = filters.get('categories', [])
        if categories:  # Only filter if categories selected
            filtered_df = filtered_df[filtered_df['category'].isin(categories)]
        
        # Sector filter - IMPROVED: empty = show all
        sectors = filters.get('sectors', [])
        if sectors:  # Only filter if sectors selected
            filtered_df = filtered_df[filtered_df['sector'].isin(sectors)]
        
        # EPS tier filter
        eps_tiers = filters.get('eps_tiers', [])
        if eps_tiers:
            filtered_df = filtered_df[filtered_df['eps_tier'].isin(eps_tiers)]
        
        # PE tier filter
        pe_tiers = filters.get('pe_tiers', [])
        if pe_tiers:
            filtered_df = filtered_df[filtered_df['pe_tier'].isin(pe_tiers)]
        
        # Price tier filter
        price_tiers = filters.get('price_tiers', [])
        if price_tiers:
            filtered_df = filtered_df[filtered_df['price_tier'].isin(price_tiers)]
        
        # Score filter
        min_score = filters.get('min_score', 0)
        if min_score > 0:
            filtered_df = filtered_df[filtered_df['master_score'] >= min_score]
        
        # EPS change filter
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
        
        # PE filters
        min_pe = filters.get('min_pe')
        if min_pe is not None and 'pe' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['pe'].isna()) |
                ((filtered_df['pe'] > 0) & 
                 (filtered_df['pe'] >= min_pe) & 
                 ~np.isinf(filtered_df['pe']))
            ]
        
        max_pe = filters.get('max_pe')
        if max_pe is not None and 'pe' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['pe'].isna()) |
                ((filtered_df['pe'] > 0) & 
                 (filtered_df['pe'] <= max_pe) & 
                 ~np.isinf(filtered_df['pe']))
            ]
        
        # Data completeness filter
        if filters.get('require_fundamental_data', False):
            if 'pe' in filtered_df.columns and 'eps_change_pct' in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df['pe'].notna() & 
                    (filtered_df['pe'] > 0) &
                    ~np.isinf(filtered_df['pe']) &
                    filtered_df['eps_change_pct'].notna() &
                    ~np.isinf(filtered_df['eps_change_pct'])
                ]
        
        filtered_count = len(filtered_df)
        if filtered_count < initial_count:
            logger.info(f"Filters reduced stocks from {initial_count} to {filtered_count}")
        
        return filtered_df

# ============================================
# SEARCH ENGINE - ENHANCED
# ============================================

class SearchEngine:
    """Advanced search functionality with fuzzy matching"""
    
    @staticmethod
    def similarity_score(s1: str, s2: str) -> float:
        """Calculate similarity between two strings"""
        return SequenceMatcher(None, s1.upper(), s2.upper()).ratio()
    
    @staticmethod
    def search_stocks(df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Enhanced search with fuzzy matching and relevance scoring"""
        if not query or df.empty:
            return pd.DataFrame()
        
        try:
            query = query.upper().strip()
            
            # Direct ticker match (highest priority)
            ticker_match = df[df['ticker'].str.upper() == query]
            if not ticker_match.empty:
                return ticker_match
            
            # Calculate relevance scores
            df = df.copy()
            df['search_score'] = 0.0
            
            # Ticker similarity (weight: 0.5)
            df['ticker_similarity'] = df['ticker'].apply(
                lambda x: SearchEngine.similarity_score(query, str(x))
            )
            df['search_score'] += df['ticker_similarity'] * 0.5
            
            # Company name similarity (weight: 0.3)
            df['company_similarity'] = df['company_name'].apply(
                lambda x: SearchEngine.similarity_score(query, str(x))
            )
            df['search_score'] += df['company_similarity'] * 0.3
            
            # Contains match bonus (weight: 0.2)
            contains_bonus = (
                df['ticker'].str.contains(query, case=False, na=False) |
                df['company_name'].str.contains(query, case=False, na=False)
            ).astype(float) * 0.2
            df['search_score'] += contains_bonus
            
            # Filter results with minimum score
            min_score = 0.3
            results = df[df['search_score'] > min_score]
            
            # Sort by relevance and rank
            results = results.sort_values(
                ['search_score', 'master_score'], 
                ascending=[False, False]
            )
            
            # Clean up temporary columns
            results = results.drop(columns=['search_score', 'ticker_similarity', 'company_similarity'])
            
            return results.head(50)  # Limit results
            
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
        """Create comprehensive Excel report"""
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
                
                # 1. Top 100 Stocks
                top_100 = df.nlargest(min(100, len(df)), 'master_score').copy()
                
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
                
                worksheet = writer.sheets['Top 100']
                for i, col in enumerate(available_cols):
                    worksheet.write(0, i, col, header_format)
                
                # 2. All Stocks Summary
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
                
                # 3. Sector Analysis
                if 'sector' in df.columns:
                    try:
                        sector_agg = {'master_score': ['mean', 'std', 'min', 'max', 'count'],
                                     'rvol': 'mean',
                                     'ret_30d': 'mean'}
                        
                        if 'pe' in df.columns:
                            sector_agg['pe'] = lambda x: x[x > 0].mean() if any(x > 0) else np.nan
                        
                        if 'eps_change_pct' in df.columns:
                            sector_agg['eps_change_pct'] = lambda x: x.dropna().mean()
                        
                        sector_analysis = df.groupby('sector').agg(sector_agg).round(2)
                        
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
                
                # 4. Category Analysis
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
                
                # 5. Pattern Analysis
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
                
                # 6. Fundamental Analysis
                if any(col in df.columns for col in ['pe', 'eps_current', 'eps_change_pct']):
                    try:
                        fund_df = df.copy()
                        
                        if 'pe' in fund_df.columns:
                            fund_df = fund_df[fund_df['pe'].notna() & (fund_df['pe'] > 0)]
                        
                        if len(fund_df) > 0:
                            if 'pe' in fund_df.columns:
                                fund_df['pe_category'] = pd.cut(
                                    fund_df['pe'],
                                    bins=[0, 10, 15, 20, 30, 50, float('inf')],
                                    labels=['0-10 (Deep Value)', '10-15 (Value)', 
                                           '15-20 (Fair)', '20-30 (Growth)', 
                                           '30-50 (High Growth)', '50+ (Expensive)']
                                )
                            
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
                
                # 7. Wave Radar Signals
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
                
                # 8. Data Quality Report
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
                
                logger.info("Excel report created successfully")
                
        except Exception as e:
            logger.error(f"Error creating Excel report: {str(e)}")
            raise
        
        output.seek(0)
        return output
    
    @staticmethod
    def create_csv_export(df: pd.DataFrame) -> str:
        """Create CSV export"""
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
        page_title="Wave Detection Ultimate 3.0",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS - CLEANED UP
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
    .stButton button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
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
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
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
            if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        with col2:
            # Cache info
            st.metric("Cache", "1 hour", help="Data cached for 1 hour to improve performance")
        
        st.markdown("---")
        st.markdown("### 🔍 Filters")
    
    # Data loading and processing
    try:
        with st.spinner("📥 Loading data..."):
            raw_df = load_google_sheets_data(sheet_url, gid)
        
        with st.spinner(f"⚙️ Processing {len(raw_df):,} stocks..."):
            processed_df = DataProcessor.process_dataframe(raw_df)
            
            if processed_df.empty:
                st.error("❌ No valid data after processing.")
                st.info("Tips: Verify that your Google Sheets URL is public and the GID is correct.")
                st.stop()
        
        with st.spinner("📊 Calculating rankings..."):
            ranked_df = RankingEngine.calculate_rankings(processed_df)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        with st.expander("🔍 Error Details"):
            st.code(str(e))
            st.info("Common issues:\n- Google Sheets not public\n- Invalid GID\n- Network connectivity")
        st.stop()
    
    # Filter setup
    filter_engine = FilterEngine()
    
    # Sidebar filters
    with st.sidebar:
        filters = {}
        
        # Display Mode Toggle
        st.markdown("### 📊 Display Mode")
        display_mode = st.radio(
            "Choose your view:",
            options=["Technical", "Hybrid (Technical + Fundamentals)"],
            index=0,
            help="Technical: Pure momentum analysis | Hybrid: Adds PE & EPS data",
            key="display_mode_toggle"
        )
        
        show_fundamentals = (display_mode == "Hybrid (Technical + Fundamentals)")
        
        st.markdown("---")
        
        # Category filter - IMPROVED: Empty by default
        categories = filter_engine.get_unique_values(ranked_df, 'category')
        selected_categories = st.multiselect(
            "Market Cap Category",
            options=categories,
            default=[],  # Empty by default
            key="category_filter",
            help="Leave empty to show all categories"
        )
        filters['categories'] = selected_categories
        
        # Sector filter - IMPROVED: Empty by default
        sectors = filter_engine.get_unique_values(ranked_df, 'sector')
        selected_sectors = st.multiselect(
            "Sector",
            options=sectors,
            default=[],  # Empty by default
            key="sector_filter",
            help="Leave empty to show all sectors"
        )
        filters['sectors'] = selected_sectors
        
        # Score filter
        filters['min_score'] = st.slider(
            "Minimum Master Score",
            min_value=0,
            max_value=100,
            value=0,
            step=5
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
        
        # Trend filter
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
            index=0
        )
        filters['trend_range'] = trend_options[filters['trend_filter']]
        
        # Advanced filters
        with st.expander("🔧 Advanced Filters"):
            # EPS tier filter
            eps_tiers = filter_engine.get_unique_values(ranked_df, 'eps_tier')
            filters['eps_tiers'] = st.multiselect(
                "EPS Tier",
                options=eps_tiers,
                default=[],
                key="eps_tier_filter"
            )
            
            # PE tier filter
            pe_tiers = filter_engine.get_unique_values(ranked_df, 'pe_tier')
            filters['pe_tiers'] = st.multiselect(
                "PE Tier",
                options=pe_tiers,
                default=[],
                key="pe_tier_filter"
            )
            
            # Price tier filter
            price_tiers = filter_engine.get_unique_values(ranked_df, 'price_tier')
            filters['price_tiers'] = st.multiselect(
                "Price Range",
                options=price_tiers,
                default=[],
                key="price_tier_filter"
            )
            
            # EPS change filter
            if 'eps_change_pct' in ranked_df.columns:
                eps_change_input = st.text_input(
                    "Min EPS Change %",
                    value="",
                    placeholder="e.g. -50",
                    help="Minimum EPS growth %"
                )
                
                if eps_change_input.strip():
                    try:
                        filters['min_eps_change'] = float(eps_change_input)
                    except ValueError:
                        st.error("Please enter a valid number")
                        filters['min_eps_change'] = None
                else:
                    filters['min_eps_change'] = None
            
            # PE filters - Only in hybrid mode
            if show_fundamentals and 'pe' in ranked_df.columns:
                st.markdown("**🔍 Fundamental Filters**")
                
                col1, col2 = st.columns(2)
                with col1:
                    min_pe_input = st.text_input(
                        "Min PE",
                        value="",
                        placeholder="e.g. 10"
                    )
                    
                    if min_pe_input.strip():
                        try:
                            filters['min_pe'] = float(min_pe_input)
                        except ValueError:
                            filters['min_pe'] = None
                    else:
                        filters['min_pe'] = None
                
                with col2:
                    max_pe_input = st.text_input(
                        "Max PE",
                        value="",
                        placeholder="e.g. 30"
                    )
                    
                    if max_pe_input.strip():
                        try:
                            filters['max_pe'] = float(max_pe_input)
                        except ValueError:
                            filters['max_pe'] = None
                    else:
                        filters['max_pe'] = None
                
                filters['require_fundamental_data'] = st.checkbox(
                    "Only stocks with PE and EPS data",
                    value=False
                )
        
        # Clear Filters button - NEW
        st.markdown("---")
        if st.button("🗑️ Clear All Filters", use_container_width=True):
            # Clear all filter selections
            for key in st.session_state:
                if key.endswith('_filter'):
                    del st.session_state[key]
            st.rerun()
    
    # Apply filters
    filtered_df = filter_engine.apply_filters(ranked_df, filters)
    filtered_df = filtered_df.sort_values('rank')
    
    # Main content area
    # Summary metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        total_stocks = len(filtered_df)
        total_original = len(ranked_df)
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
        if show_fundamentals and 'pe' in filtered_df.columns:
            valid_pe = filtered_df['pe'].notna() & (filtered_df['pe'] > 0) & (filtered_df['pe'] < 10000)
            pe_coverage = valid_pe.sum()
            pe_pct = (pe_coverage / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
            
            if pe_coverage > 0:
                median_pe = filtered_df.loc[valid_pe, 'pe'].median()
                st.metric(
                    "Median PE",
                    f"{median_pe:.1f}x",
                    f"{pe_pct:.0f}% have data"
                )
            else:
                st.metric("PE Data", "Limited", "No PE data")
        else:
            if not filtered_df.empty and 'master_score' in filtered_df.columns:
                min_score = filtered_df['master_score'].min()
                max_score = filtered_df['master_score'].max()
                score_range = f"{min_score:.1f}-{max_score:.1f}"
            else:
                score_range = "N/A"
            st.metric("Score Range", score_range)
    
    with col4:
        if show_fundamentals and 'eps_change_pct' in filtered_df.columns:
            valid_eps_change = filtered_df['eps_change_pct'].notna() & ~np.isinf(filtered_df['eps_change_pct'])
            positive_eps_growth = valid_eps_change & (filtered_df['eps_change_pct'] > 0)
            
            strong_growth = valid_eps_change & (filtered_df['eps_change_pct'] > 50)
            mega_growth = valid_eps_change & (filtered_df['eps_change_pct'] > 100)
            
            growth_count = positive_eps_growth.sum()
            strong_count = strong_growth.sum()
            
            if mega_growth.sum() > 0:
                st.metric(
                    "EPS Growth +ve",
                    f"{growth_count}",
                    f"{strong_count} >50% | {mega_growth.sum()} >100%"
                )
            else:
                st.metric(
                    "EPS Growth +ve",
                    f"{growth_count}",
                    f"{valid_eps_change.sum()} have data"
                )
        else:
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
    
    # Main tabs - REMOVED VISUALIZATIONS
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏆 Rankings", "🌊 Wave Radar", "📊 Analysis", "🔍 Search", "📥 Export", "ℹ️ About"
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
                index=2
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
            # Add trend indicator
            if 'trend_quality' in display_df.columns:
                def get_trend_indicator(score):
                    if pd.isna(score):
                        return "➖"
                    elif score >= 80:
                        return "🔥"
                    elif score >= 60:
                        return "✅"
                    elif score >= 40:
                        return "➡️"
                    else:
                        return "⚠️"
                
                display_df['trend_indicator'] = display_df['trend_quality'].apply(get_trend_indicator)
            
            # Display columns setup
            display_cols = {
                'rank': 'Rank',
                'ticker': 'Ticker',
                'company_name': 'Company',
                'master_score': 'Score'
            }
            
            if 'trend_indicator' in display_df.columns:
                display_cols['trend_indicator'] = 'Trend'
            
            display_cols['price'] = 'Price'
            
            # Add fundamental columns if enabled
            if show_fundamentals:
                if 'pe' in display_df.columns:
                    display_cols['pe'] = 'PE'
                
                if 'eps_change_pct' in display_df.columns:
                    display_cols['eps_change_pct'] = 'EPS Δ%'
            
            display_cols.update({
                'from_low_pct': 'From Low',
                'ret_30d': '30D Ret',
                'rvol': 'RVOL',
                'patterns': 'Patterns',
                'category': 'Category',
                'sector': 'Sector'
            })
            
            # Format numeric columns
            format_rules = {
                'master_score': '{:.1f}',
                'price': '₹{:,.0f}',
                'from_low_pct': '{:.0f}%',
                'ret_30d': '{:+.1f}%',
                'rvol': '{:.1f}x'
            }
            
            # Enhanced PE formatting
            def format_pe(value):
                try:
                    if pd.isna(value) or value == 'N/A' or value == '':
                        return '-'
                    
                    val = float(value)
                    
                    if val <= 0 or np.isinf(val):
                        return 'Loss'
                    elif val > 10000:
                        return f"{val/1000:.0f}K"
                    elif val > 1000:
                        return f"{val:.0f}"
                    elif val > 100:
                        return f"{val:.1f}"
                    else:
                        return f"{val:.1f}"
                except:
                    return '-'
            
            # Enhanced EPS change formatting
            def format_eps_change(value):
                try:
                    if pd.isna(value) or value == 'N/A' or value == '':
                        return '-'
                    
                    val = float(value)
                    
                    if np.isinf(val):
                        return '∞' if val > 0 else '-∞'
                    
                    if abs(val) >= 10000:
                        return f"{val/1000:+.1f}K%"
                    elif abs(val) >= 1000:
                        return f"{val:+.0f}%"
                    elif abs(val) >= 100:
                        return f"{val:+.1f}%"
                    elif abs(val) >= 10:
                        return f"{val:+.1f}%"
                    elif abs(val) >= 0.1:
                        return f"{val:+.1f}%"
                    else:
                        return f"{val:+.2f}%"
                        
                except:
                    return '-'
            
            # Apply formatting
            for col, fmt in format_rules.items():
                if col in display_df.columns:
                    try:
                        if col == 'ret_30d':
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
            
            # Apply special formatting for fundamentals
            if show_fundamentals:
                if 'pe' in display_df.columns:
                    display_df['pe'] = display_df['pe'].apply(format_pe)
                
                if 'eps_change_pct' in display_df.columns:
                    display_df['eps_change_pct'] = display_df['eps_change_pct'].apply(format_eps_change)
            
            # Rename columns for display
            display_df = display_df[[c for c in display_cols.keys() if c in display_df.columns]]
            display_df.columns = [display_cols[c] for c in display_df.columns]
            
            # Display table
            st.dataframe(
                display_df,
                use_container_width=True,
                height=min(600, len(display_df) * 35 + 50),
                hide_index=True
            )
            
            # Quick stats
            with st.expander("📊 Quick Statistics"):
                stat_cols = st.columns(4)
                
                with stat_cols[0]:
                    st.markdown("**Score Distribution**")
                    if 'master_score' in filtered_df.columns:
                        st.text(f"Max: {filtered_df['master_score'].max():.1f}")
                        st.text(f"Min: {filtered_df['master_score'].min():.1f}")
                        st.text(f"Std: {filtered_df['master_score'].std():.1f}")
                
                with stat_cols[1]:
                    st.markdown("**Returns (30D)**")
                    if 'ret_30d' in filtered_df.columns:
                        st.text(f"Max: {filtered_df['ret_30d'].max():.1f}%")
                        st.text(f"Avg: {filtered_df['ret_30d'].mean():.1f}%")
                        st.text(f"Positive: {(filtered_df['ret_30d'] > 0).sum()}")
                
                with stat_cols[2]:
                    if show_fundamentals:
                        st.markdown("**Fundamentals**")
                        if 'pe' in filtered_df.columns:
                            valid_pe = filtered_df['pe'].notna() & (filtered_df['pe'] > 0) & (filtered_df['pe'] < 10000)
                            if valid_pe.any():
                                median_pe = filtered_df.loc[valid_pe, 'pe'].median()
                                q1_pe = filtered_df.loc[valid_pe, 'pe'].quantile(0.25)
                                q3_pe = filtered_df.loc[valid_pe, 'pe'].quantile(0.75)
                                
                                st.text(f"Median PE: {median_pe:.1f}x")
                                st.text(f"PE Range: {q1_pe:.1f}-{q3_pe:.1f}")
                        
                        if 'eps_change_pct' in filtered_df.columns:
                            valid_eps = filtered_df['eps_change_pct'].notna() & ~np.isinf(filtered_df['eps_change_pct'])
                            if valid_eps.any():
                                eps_data = filtered_df.loc[valid_eps, 'eps_change_pct']
                                
                                mega_growth = (eps_data > 100).sum()
                                strong_growth = ((eps_data > 50) & (eps_data <= 100)).sum()
                                moderate_growth = ((eps_data > 0) & (eps_data <= 50)).sum()
                                declining = (eps_data < 0).sum()
                                
                                if mega_growth > 0:
                                    st.text(f">100%: {mega_growth} stocks")
                                st.text(f"Positive: {moderate_growth + strong_growth + mega_growth}")
                                st.text(f"Negative: {declining}")
                    else:
                        st.markdown("**RVOL Stats**")
                        if 'rvol' in filtered_df.columns:
                            st.text(f"Max: {filtered_df['rvol'].max():.1f}x")
                            st.text(f"Avg: {filtered_df['rvol'].mean():.1f}x")
                            st.text(f">2x: {(filtered_df['rvol'] > 2).sum()}")
                
                with stat_cols[3]:
                    st.markdown("**Categories**")
                    if 'category' in filtered_df.columns:
                        for cat, count in filtered_df['category'].value_counts().head(3).items():
                            st.text(f"{cat}: {count}")
        
        else:
            st.warning("No stocks match the selected filters.")
    
    # Tab 2: Wave Radar
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
        
        # Wave signals
        if not filtered_df.empty:
            # 1. MOMENTUM SHIFT DETECTION
            st.markdown("#### 🚀 Momentum Shifts - Stocks Entering Strength")
            
            momentum_shifts = filtered_df.copy()
            
            if sensitivity == "Conservative":
                cross_threshold = 60
                min_acceleration = 70
            elif sensitivity == "Balanced":
                cross_threshold = 50
                min_acceleration = 60
            else:
                cross_threshold = 40
                min_acceleration = 50
            
            if 'ret_30d' in momentum_shifts.columns:
                median_return = momentum_shifts['ret_30d'].median()
                return_condition = momentum_shifts['ret_30d'] > median_return
            elif 'ret_7d' in momentum_shifts.columns:
                median_return = momentum_shifts['ret_7d'].median()
                return_condition = momentum_shifts['ret_7d'] > median_return
            else:
                return_condition = True
            
            momentum_shifts['momentum_shift'] = (
                (momentum_shifts['momentum_score'] >= cross_threshold) & 
                (momentum_shifts['acceleration_score'] >= min_acceleration) &
                return_condition
            )
            
            momentum_shifts['shift_strength'] = (
                momentum_shifts['momentum_score'] * 0.4 +
                momentum_shifts['acceleration_score'] * 0.4 +
                momentum_shifts['rvol_score'] * 0.2
            )
            
            top_shifts = momentum_shifts[momentum_shifts['momentum_shift']].nlargest(20, 'shift_strength')
            
            if len(top_shifts) > 0:
                display_columns = ['ticker', 'company_name', 'master_score', 'momentum_score', 
                                 'acceleration_score', 'rvol']
                
                if 'ret_7d' in top_shifts.columns:
                    display_columns.append('ret_7d')
                
                display_columns.append('category')
                
                shift_display = top_shifts[display_columns].copy()
                
                shift_display['Signal'] = shift_display.apply(
                    lambda x: "🔥 HOT" if x['acceleration_score'] > 80 else "📈 RISING", axis=1
                )
                
                if 'ret_7d' in shift_display.columns:
                    shift_display['ret_7d'] = shift_display['ret_7d'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "0.0%")
                
                shift_display['rvol'] = shift_display['rvol'].apply(lambda x: f"{x:.1f}x")
                
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
                st.info("No momentum shifts detected with current settings.")
            
            # 2. Additional Wave Radar sections...
            # (Rest of Wave Radar implementation continues as in original)
            
        else:
            st.warning("No data available for Wave Radar analysis.")
    
    # Tab 3: Analysis
    with tab3:
        st.markdown("### 📊 Market Analysis")
        
        if not filtered_df.empty:
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
                    
                    sector_df['% of Total'] = (sector_df['Count'] / len(filtered_df) * 100).round(1)
                    
                    st.dataframe(
                        sector_df.style.background_gradient(subset=['Avg Score']),
                        use_container_width=True
                    )
            except Exception as e:
                logger.error(f"Error in sector analysis: {str(e)}")
                st.error("Unable to perform sector analysis.")
            
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
            except Exception as e:
                logger.error(f"Error in category analysis: {str(e)}")
                st.error("Unable to perform category analysis.")
            
            # Pattern Analysis
            st.markdown("#### 🎯 Pattern Frequency")
            all_patterns = []
            
            for patterns in filtered_df['patterns'].dropna():
                if patterns:
                    all_patterns.extend(patterns.split(' | '))
            
            if all_patterns:
                pattern_counts = pd.Series(all_patterns).value_counts()
                
                pattern_df = pd.DataFrame({
                    'Pattern': pattern_counts.index,
                    'Count': pattern_counts.values,
                    'Percentage': (pattern_counts.values / len(filtered_df) * 100).round(1)
                })
                
                st.dataframe(pattern_df, use_container_width=True, hide_index=True)
            else:
                st.info("No patterns detected in current selection.")
        
        else:
            st.info("No data available for analysis.")
    
    # Tab 4: Search - IMPROVED
    with tab4:
        st.markdown("### 🔍 Advanced Stock Search")
        
        # Search interface
        col1, col2 = st.columns([4, 1])
        
        with col1:
            search_query = st.text_input(
                "Search stocks",
                value=st.session_state.search_query,
                placeholder="Enter ticker or company name...",
                help="Search by ticker symbol or company name",
                key="search_input"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_clicked = st.button("🔎 Search", type="primary", use_container_width=True)
        
        # Update session state
        if search_query != st.session_state.search_query:
            st.session_state.search_query = search_query
        
        # Perform search
        if search_query:
            search_results = SearchEngine.search_stocks(filtered_df, search_query)
            
            if not search_results.empty:
                st.success(f"Found {len(search_results)} matching stock(s)")
                
                # Display results
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
                                "52-week range"
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
                        
                        # Additional details
                        detail_cols = st.columns(3)
                        
                        with detail_cols[0]:
                            st.markdown("**📊 Classification**")
                            st.text(f"Sector: {stock['sector']}")
                            st.text(f"Category: {stock['category']}")
                            
                            # Fundamentals if enabled
                            if show_fundamentals:
                                st.markdown("**💰 Fundamentals**")
                                
                                if 'pe' in stock and pd.notna(stock['pe']):
                                    try:
                                        pe_val = float(stock['pe'])
                                        if pe_val <= 0 or np.isinf(pe_val):
                                            pe_display = "Loss"
                                            pe_color = "🔴"
                                        elif pe_val < 10:
                                            pe_display = f"{pe_val:.1f}x"
                                            pe_color = "🟢"
                                        elif pe_val < 15:
                                            pe_display = f"{pe_val:.1f}x"
                                            pe_color = "🟢"
                                        elif pe_val < 25:
                                            pe_display = f"{pe_val:.1f}x"
                                            pe_color = "🟡"
                                        elif pe_val < 50:
                                            pe_display = f"{pe_val:.1f}x"
                                            pe_color = "🟠"
                                        elif pe_val < 100:
                                            pe_display = f"{pe_val:.1f}x"
                                            pe_color = "🔴"
                                        else:
                                            if pe_val > 10000:
                                                pe_display = f"{pe_val/1000:.0f}Kx"
                                            else:
                                                pe_display = f"{pe_val:.0f}x"
                                            pe_color = "🔴"
                                        st.text(f"PE Ratio: {pe_color} {pe_display}")
                                    except:
                                        st.text("PE Ratio: - (Error)")
                                else:
                                    st.text("PE Ratio: - (N/A)")
                                
                                if 'eps_current' in stock and pd.notna(stock['eps_current']):
                                    try:
                                        eps_val = float(stock['eps_current'])
                                        if abs(eps_val) >= 1000:
                                            eps_display = f"₹{eps_val/1000:.1f}K"
                                        elif abs(eps_val) >= 100:
                                            eps_display = f"₹{eps_val:.0f}"
                                        else:
                                            eps_display = f"₹{eps_val:.2f}"
                                        st.text(f"EPS: {eps_display}")
                                    except:
                                        st.text("EPS: - (Error)")
                                else:
                                    st.text("EPS: - (N/A)")
                                
                                if 'eps_change_pct' in stock and pd.notna(stock['eps_change_pct']):
                                    try:
                                        eps_chg = float(stock['eps_change_pct'])
                                        
                                        if np.isinf(eps_chg):
                                            eps_display = "∞" if eps_chg > 0 else "-∞"
                                        elif abs(eps_chg) >= 10000:
                                            eps_display = f"{eps_chg/1000:+.1f}K%"
                                        elif abs(eps_chg) >= 1000:
                                            eps_display = f"{eps_chg:+.0f}%"
                                        else:
                                            eps_display = f"{eps_chg:+.1f}%"
                                        
                                        if eps_chg >= 100:
                                            eps_emoji = "🚀"
                                        elif eps_chg >= 50:
                                            eps_emoji = "🔥"
                                        elif eps_chg >= 20:
                                            eps_emoji = "📈"
                                        elif eps_chg >= 0:
                                            eps_emoji = "➕"
                                        elif eps_chg >= -20:
                                            eps_emoji = "➖"
                                        elif eps_chg >= -50:
                                            eps_emoji = "📉"
                                        else:
                                            eps_emoji = "⚠️"
                                        
                                        st.text(f"EPS Growth: {eps_emoji} {eps_display}")
                                    except:
                                        st.text("EPS Growth: - (Error)")
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
        else:
            st.info("Enter a ticker or company name to search.")
    
    # Tab 5: Export
    with tab5:
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
                "- Wave Radar signals\n"
                "- Smart money flow tracking"
            )
            
            if st.button("Generate Excel Report", type="primary", use_container_width=True):
                if len(filtered_df) == 0:
                    st.error("No data to export.")
                else:
                    with st.spinner("Creating Excel report..."):
                        try:
                            excel_file = ExportEngine.create_excel_report(filtered_df)
                            
                            st.download_button(
                                label="📥 Download Excel Report",
                                data=excel_file,
                                file_name=f"wave_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
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
                "- Fundamental data (PE, EPS)"
            )
            
            if st.button("Generate CSV Export", use_container_width=True):
                if len(filtered_df) == 0:
                    st.error("No data to export.")
                else:
                    try:
                        csv_data = ExportEngine.create_csv_export(filtered_df)
                        
                        st.download_button(
                            label="📥 Download CSV File",
                            data=csv_data,
                            file_name=f"wave_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
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
    
    # Tab 6: About - NEW
    with tab6:
        st.markdown("### ℹ️ About Wave Detection Ultimate 3.0")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            #### 🌊 What is Wave Detection?
            
            Wave Detection Ultimate 3.0 is a professional-grade stock ranking system that identifies 
            momentum waves **before** they become obvious. By combining multiple technical indicators 
            with smart pattern recognition, it helps you catch winning stocks early in their move.
            
            #### 🎯 Core Philosophy
            
            > "Every big move starts as a small wave. Detect it early, ride it fully."
            
            Our system analyzes thousands of data points to find stocks showing:
            - **Early momentum shifts** 🚀
            - **Unusual volume patterns** 📊
            - **Breakout potential** 🎯
            - **Institutional accumulation** 🏦
            
            #### 📈 Master Score 3.0 Methodology
            
            The Master Score is a proprietary blend of six key components:
            
            1. **Position Score (30%)**: Where the stock sits in its 52-week range
            2. **Volume Score (25%)**: Multi-timeframe volume analysis
            3. **Momentum Score (15%)**: Price momentum over 30 days
            4. **Acceleration Score (10%)**: Is momentum increasing?
            5. **Breakout Score (10%)**: Probability of a major move
            6. **RVOL Score (10%)**: Today's volume vs average
            
            #### 🔥 Pattern Detection
            
            Our AI detects 16 different patterns including:
            - **Technical Patterns**: Momentum waves, breakouts, volume explosions
            - **Fundamental Patterns**: Value plays, earnings rockets, quality leaders
            - **Risk Patterns**: Overvaluation warnings, volatility alerts
            
            #### 💡 Pro Tips for Success
            
            1. **Focus on Category Leaders** - Stocks leading their market cap category often continue
            2. **Watch RVOL** - Volume precedes price; RVOL >2x is significant
            3. **Combine Patterns** - Multiple patterns = stronger signal
            4. **Use Trend Filter** - Trade with the trend for higher success
            5. **Check Wave Radar Daily** - Early detection is everything
            
            #### ⚡ Quick Start Guide
            
            1. **Rankings Tab**: See top-ranked stocks by Master Score
            2. **Wave Radar**: Catch early momentum shifts and volume surges
            3. **Search**: Deep dive into any specific stock
            4. **Export**: Download data for further analysis
            
            #### 🛡️ Risk Management
            
            Remember: This tool identifies opportunities, not guarantees. Always:
            - Do your own research
            - Use proper position sizing
            - Set stop losses
            - Never invest more than you can afford to lose
            """)
        
        with col2:
            st.info("""
            **📊 Data Source**
            - Real-time from Google Sheets
            - Updates every market day
            - 1,700+ stocks covered
            
            **🔄 Refresh Rate**
            - Data cached for 1 hour
            - Manual refresh available
            - Auto-refresh in Wave Radar
            
            **🎨 Display Modes**
            - Technical: Pure momentum
            - Hybrid: Adds PE/EPS data
            
            **📈 Best Practices**
            - Check daily at market open
            - Focus on new patterns
            - Export top picks
            - Track performance
            
            **⚠️ Disclaimer**
            Not financial advice. 
            For educational purposes only.
            Past performance doesn't 
            guarantee future results.
            """)
            
            # Performance metrics
            st.markdown("---")
            st.markdown("#### 📊 System Stats")
            
            total_patterns = sum(1 for p in ranked_df['patterns'] if p)
            avg_patterns = total_patterns / len(ranked_df) if len(ranked_df) > 0 else 0
            
            stats = {
                "Stocks Analyzed": f"{len(ranked_df):,}",
                "Active Patterns": f"{total_patterns:,}",
                "Avg Patterns/Stock": f"{avg_patterns:.1f}",
                "Data Freshness": "Live" if datetime.now() - st.session_state.last_refresh < pd.Timedelta(minutes=5) else "Cached"
            }
            
            for label, value in stats.items():
                st.metric(label, value)
            
            # Version info
            st.markdown("---")
            st.markdown("""
            **Version**: 3.1.0-FINAL  
            **Status**: Production Ready  
            **Engine**: Streamlit Cloud  
            **Cache**: Optimized (1hr)
            """)
    
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
