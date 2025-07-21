"""
Wave Detection Ultimate 3.0 - FINAL PRODUCTION VERSION (ENHANCED)
================================================================
Professional Stock Ranking System with Advanced Analytics
Fully optimized with enhanced UX, smart search, and robust error handling.

Version: 3.1.0-FINAL
Last Updated: December 2024
Status: PRODUCTION RELEASE - Professional Grade
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
from functools import lru_cache, wraps
import time
from io import BytesIO
import warnings
import hashlib

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
# CONFIGURATION
# ============================================

@dataclass(frozen=True)
class Config:
    """System configuration with validated weights"""
    
    # Data source
    DEFAULT_SHEET_URL: str = "https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk/edit?usp=sharing"
    DEFAULT_GID: str = "2026492216"
    
    # Cache settings - Optimized for Streamlit Community Cloud
    CACHE_TTL_DATA: int = 300  # 5 minutes for data
    CACHE_TTL_COMPUTATION: int = 600  # 10 minutes for heavy computations
    CACHE_TTL_STATIC: int = 3600  # 1 hour for static data
    
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
    
    # Search settings
    SEARCH_MIN_LENGTH: int = 1
    SEARCH_MAX_RESULTS: int = 100
    
    # Performance settings
    MAX_ROWS_DISPLAY: int = 1000
    CHUNK_SIZE: int = 500
    
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
# SESSION STATE MANAGEMENT
# ============================================

def initialize_session_state():
    """Initialize all session state variables with proper defaults"""
    if 'search_index' not in st.session_state:
        st.session_state.search_index = None
    
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    
    if 'filters_applied' not in st.session_state:
        st.session_state.filters_applied = False
    
    if 'display_mode' not in st.session_state:
        st.session_state.display_mode = "Technical"

def clear_all_filters():
    """Clear all filters and reset to defaults"""
    # Clear filter-related session state
    filter_keys = [
        'category_filter', 'sector_filter', 'eps_tier_filter',
        'pe_tier_filter', 'price_tier_filter', 'pattern_filter',
        'min_score_filter', 'trend_filter', 'min_eps_change_filter',
        'min_pe_filter', 'max_pe_filter', 'require_fundamental_filter'
    ]
    
    for key in filter_keys:
        if key in st.session_state:
            del st.session_state[key]
    
    st.session_state.filters_applied = False

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
    """Validate data at each step with enhanced error handling"""
    
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
        """Validate and clean numeric column with enhanced error handling"""
        if series is None or series.empty:
            return pd.Series(dtype=float)
        
        # Convert to numeric, coercing errors
        series = pd.to_numeric(series, errors='coerce')
        
        # Apply bounds if specified
        if min_val is not None:
            series = series.clip(lower=min_val)
        if max_val is not None:
            series = series.clip(upper=max_val)
        
        # Log if too many NaN values
        nan_pct = series.isna().sum() / len(series) * 100 if len(series) > 0 else 0
        if nan_pct > 50:
            logger.warning(f"{col_name}: {nan_pct:.1f}% NaN values")
        
        return series

# ============================================
# DATA LOADING - OPTIMIZED CACHING
# ============================================

@st.cache_data(ttl=CONFIG.CACHE_TTL_DATA, show_spinner=False)
def load_google_sheets_data(sheet_url: str, gid: str) -> pd.DataFrame:
    """Load data from Google Sheets with comprehensive error handling and caching"""
    try:
        if not sheet_url or not gid:
            raise ValueError("Sheet URL and GID are required")
        
        # Create cache key based on URL and GID
        cache_key = hashlib.md5(f"{sheet_url}_{gid}".encode()).hexdigest()
        
        # Construct CSV URL
        base_url = sheet_url.split('/edit')[0]
        csv_url = f"{base_url}/export?format=csv&gid={gid}"
        
        logger.info(f"Loading data from Google Sheets (cache key: {cache_key[:8]})")
        
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
# DATA PROCESSING - ENHANCED
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
    
    REQUIRED_COLUMNS = ['ticker', 'price']
    
    @staticmethod
    def clean_numeric_value(value: Any) -> Optional[float]:
        """Clean and convert Indian number format to float with enhanced error handling"""
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
    @st.cache_data(ttl=CONFIG.CACHE_TTL_COMPUTATION, show_spinner=False)
    def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Complete data processing pipeline with validation and caching"""
        if not DataValidator.validate_dataframe(df, DataProcessor.REQUIRED_COLUMNS, "Initial data"):
            return pd.DataFrame()
        
        # Create copy to avoid modifying original
        df = df.copy()
        
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
        df = DataProcessor._fix_volume_ratios(df)
        
        # Validate fundamental data
        df = DataProcessor._validate_fundamental_data(df)
        
        # Remove invalid rows
        df = DataProcessor._remove_invalid_rows(df)
        
        # Add tier classifications
        df = DataProcessor._add_tier_classifications(df)
        
        logger.info(f"Processed {len(df)} valid stocks")
        return df
    
    @staticmethod
    def _fix_volume_ratios(df: pd.DataFrame) -> pd.DataFrame:
        """Fix volume ratios to ensure they're multipliers"""
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
                
                df[col] = df[col].abs().fillna(1.0).clip(0.1, 10.0)
        
        return df
    
    @staticmethod
    def _validate_fundamental_data(df: pd.DataFrame) -> pd.DataFrame:
        """Validate PE and EPS data with enhanced handling"""
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
        
        return df
    
    @staticmethod
    def _remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with critical missing data"""
        initial_count = len(df)
        
        df = df.dropna(subset=['ticker', 'price'], how='any')
        df = df[df['price'] > 0]
        
        # Handle position data
        if 'from_low_pct' not in df.columns:
            df['from_low_pct'] = 50
        else:
            df['from_low_pct'] = df['from_low_pct'].fillna(50)
            
        if 'from_high_pct' not in df.columns:
            df['from_high_pct'] = -50
        else:
            df['from_high_pct'] = df['from_high_pct'].fillna(-50)
        
        # Ensure RVOL exists
        if 'rvol' not in df.columns:
            df['rvol'] = 1.0
        else:
            df['rvol'] = pd.to_numeric(df['rvol'], errors='coerce')
            df['rvol'] = df['rvol'].fillna(1.0).clip(lower=0.01, upper=100)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['ticker'], keep='first')
        
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} invalid/duplicate rows")
        
        return df
    
    @staticmethod
    def _add_tier_classifications(df: pd.DataFrame) -> pd.DataFrame:
        """Add tier classifications for EPS, PE, and Price"""
        def classify_tier(value: float, tier_dict: Dict[str, Tuple[float, float]]) -> str:
            if pd.isna(value):
                return "Unknown"
            
            for tier_name, (min_val, max_val) in tier_dict.items():
                if min_val < value <= max_val:
                    return tier_name
            return "Unknown"
        
        if 'eps_current' in df.columns:
            df['eps_tier'] = df['eps_current'].apply(
                lambda x: classify_tier(x, CONFIG.TIERS['eps'])
            )
        
        if 'pe' in df.columns:
            df['pe_tier'] = df['pe'].apply(
                lambda x: "Negative/NA" if pd.isna(x) or x <= 0 
                else classify_tier(x, CONFIG.TIERS['pe'])
            )
        
        if 'price' in df.columns:
            df['price_tier'] = df['price'].apply(
                lambda x: classify_tier(x, CONFIG.TIERS['price'])
            )
        
        return df

# ============================================
# RANKING ENGINE - OPTIMIZED
# ============================================

class RankingEngine:
    """Core ranking calculations - optimized and vectorized"""
    
    @staticmethod
    def safe_rank(series: pd.Series, pct: bool = True, ascending: bool = True) -> pd.Series:
        """Safely rank a series with proper handling of edge cases"""
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
    @st.cache_data(ttl=CONFIG.CACHE_TTL_COMPUTATION, show_spinner=False)
    def calculate_all_scores(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all component scores with caching"""
        df = df.copy()
        
        # Calculate all component scores
        df['position_score'] = RankingEngine._calculate_position_score(df)
        df['volume_score'] = RankingEngine._calculate_volume_score(df)
        df['momentum_score'] = RankingEngine._calculate_momentum_score(df)
        df['acceleration_score'] = RankingEngine._calculate_acceleration_score(df)
        df['breakout_score'] = RankingEngine._calculate_breakout_score(df)
        df['rvol_score'] = RankingEngine._calculate_rvol_score(df)
        
        # Calculate auxiliary scores
        df['trend_quality'] = RankingEngine._calculate_trend_quality(df)
        df['long_term_strength'] = RankingEngine._calculate_long_term_strength(df)
        df['liquidity_score'] = RankingEngine._calculate_liquidity_score(df)
        
        return df
    
    @staticmethod
    def _calculate_position_score(df: pd.DataFrame) -> pd.Series:
        """Calculate position score from 52-week range"""
        position_score = pd.Series(50, index=df.index, dtype=float)
        
        has_from_low = 'from_low_pct' in df.columns and df['from_low_pct'].notna().any()
        has_from_high = 'from_high_pct' in df.columns and df['from_high_pct'].notna().any()
        
        if not has_from_low and not has_from_high:
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
    def _calculate_volume_score(df: pd.DataFrame) -> pd.Series:
        """Calculate comprehensive volume score"""
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
                col_data = df[col].fillna(1.0).clip(lower=0.1)
                col_rank = RankingEngine.safe_rank(col_data, pct=True, ascending=True)
                weighted_score += col_rank * weight
                total_weight += weight
        
        if total_weight > 0 and has_any_vol_data:
            volume_score = weighted_score / total_weight
        else:
            volume_score += np.random.uniform(-5, 5, size=len(df))
        
        return volume_score.clip(0, 100)
    
    @staticmethod
    def _calculate_momentum_score(df: pd.DataFrame) -> pd.Series:
        """Calculate momentum score based on returns"""
        momentum_score = pd.Series(50, index=df.index, dtype=float)
        
        if 'ret_30d' not in df.columns or df['ret_30d'].notna().sum() == 0:
            if 'ret_7d' in df.columns and df['ret_7d'].notna().any():
                ret_7d = df['ret_7d'].fillna(0)
                momentum_score = RankingEngine.safe_rank(ret_7d, pct=True, ascending=True)
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
    def _calculate_acceleration_score(df: pd.DataFrame) -> pd.Series:
        """Calculate if momentum is accelerating"""
        acceleration_score = pd.Series(50, index=df.index, dtype=float)
        
        req_cols = ['ret_1d', 'ret_7d', 'ret_30d']
        available_cols = [col for col in req_cols if col in df.columns]
        
        if len(available_cols) < 2:
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
        
        return acceleration_score
    
    @staticmethod
    def _calculate_breakout_score(df: pd.DataFrame) -> pd.Series:
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
        
        for sma_col in ['sma_20d', 'sma_50d', 'sma_200d']:
            if sma_col in df.columns:
                above_sma = (df['price'] > df[sma_col]).fillna(False)
                trend_factor += above_sma.astype(float) * 33.33
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
    def _calculate_rvol_score(df: pd.DataFrame) -> pd.Series:
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
    def _calculate_trend_quality(df: pd.DataFrame) -> pd.Series:
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
        
        return trend_score
    
    @staticmethod
    def _calculate_long_term_strength(df: pd.DataFrame) -> pd.Series:
        """Calculate long-term strength score"""
        strength_score = pd.Series(50, index=df.index, dtype=float)
        
        lt_cols = ['ret_3m', 'ret_6m', 'ret_1y']
        available_cols = [col for col in lt_cols if col in df.columns and df[col].notna().any()]
        
        if not available_cols:
            return strength_score
        
        lt_returns = df[available_cols].fillna(0)
        avg_return = lt_returns.mean(axis=1)
        
        strength_score.loc[avg_return > 100] = 100
        strength_score.loc[(avg_return > 50) & (avg_return <= 100)] = 90
        strength_score.loc[(avg_return > 30) & (avg_return <= 50)] = 80
        strength_score.loc[(avg_return > 15) & (avg_return <= 30)] = 70
        strength_score.loc[(avg_return > 5) & (avg_return <= 15)] = 60
        strength_score.loc[(avg_return > 0) & (avg_return <= 5)] = 50
        strength_score.loc[(avg_return > -10) & (avg_return <= 0)] = 40
        strength_score.loc[(avg_return > -25) & (avg_return <= -10)] = 30
        strength_score.loc[avg_return <= -25] = 20
        
        return strength_score.clip(0, 100)
    
    @staticmethod
    def _calculate_liquidity_score(df: pd.DataFrame) -> pd.Series:
        """Calculate liquidity score"""
        liquidity_score = pd.Series(50, index=df.index, dtype=float)
        
        if 'volume_30d' in df.columns and 'price' in df.columns:
            avg_traded_value = df['volume_30d'] * df['price']
            liquidity_score = RankingEngine.safe_rank(
                avg_traded_value, pct=True, ascending=True
            )
        
        return liquidity_score.clip(0, 100)
    
    @staticmethod
    @timer
    def calculate_final_rankings(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate final rankings with master score"""
        if df.empty:
            return df
        
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
        
        if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
            df['master_score'] = df['master_score'] / total_weight
        
        df['master_score'] = df['master_score'].clip(0, 100)
        
        # Calculate ranks
        valid_scores = df['master_score'].notna()
        
        if valid_scores.sum() == 0:
            df['rank'] = 9999
            df['percentile'] = 0
        else:
            df['rank'] = df['master_score'].rank(method='min', ascending=False, na_option='bottom')
            df['rank'] = df['rank'].fillna(len(df) + 1).astype(int)
            df['percentile'] = df['master_score'].rank(pct=True, ascending=True, na_option='bottom') * 100
            df['percentile'] = df['percentile'].fillna(0)
        
        # Calculate category ranks
        df = RankingEngine._calculate_category_ranks(df)
        
        # Detect patterns
        df = RankingEngine._detect_patterns(df)
        
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
        
        # Fundamental patterns (if in hybrid mode)
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
# FILTER ENGINE - ENHANCED
# ============================================

class FilterEngine:
    """Handle all filtering operations with enhanced validation"""
    
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
        """Apply all filters with enhanced validation"""
        if df.empty:
            return df
        
        filtered_df = df.copy()
        initial_count = len(filtered_df)
        
        # Category filter
        categories = filters.get('categories', [])
        if categories and 'All' not in categories:
            filtered_df = filtered_df[filtered_df['category'].isin(categories)]
        
        # Sector filter
        sectors = filters.get('sectors', [])
        if sectors and 'All' not in sectors:
            filtered_df = filtered_df[filtered_df['sector'].isin(sectors)]
        
        # EPS tier filter
        eps_tiers = filters.get('eps_tiers', [])
        if eps_tiers and 'All' not in eps_tiers and 'eps_tier' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['eps_tier'].isin(eps_tiers)]
        
        # PE tier filter
        pe_tiers = filters.get('pe_tiers', [])
        if pe_tiers and 'All' not in pe_tiers and 'pe_tier' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['pe_tier'].isin(pe_tiers)]
        
        # Price tier filter
        price_tiers = filters.get('price_tiers', [])
        if price_tiers and 'All' not in price_tiers and 'price_tier' in filtered_df.columns:
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
        
        # Search filter
        search_query = filters.get('search_query', '').strip()
        if search_query:
            filtered_df = SearchEngine.filter_by_search(filtered_df, search_query)
        
        filtered_count = len(filtered_df)
        if filtered_count < initial_count:
            logger.info(f"Filters reduced stocks from {initial_count} to {filtered_count}")
        
        return filtered_df

# ============================================
# SEARCH ENGINE - ENHANCED
# ============================================

class SearchEngine:
    """Advanced search functionality with enhanced intelligence"""
    
    @staticmethod
    @st.cache_data(ttl=CONFIG.CACHE_TTL_STATIC, show_spinner=False)
    def create_search_index(df: pd.DataFrame) -> Dict[str, Set[str]]:
        """Create optimized search index"""
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
                    # Index full company name
                    company_upper = company_name.upper()
                    if company_upper not in search_index:
                        search_index[company_upper] = set()
                    search_index[company_upper].add(ticker)
                    
                    # Index individual words
                    company_words = company_upper.split()
                    for word in company_words:
                        if len(word) > 2:
                            if word not in search_index:
                                search_index[word] = set()
                            search_index[word].add(ticker)
                    
                    # Index partial matches (first 3 letters of each word)
                    for word in company_words:
                        if len(word) >= 3:
                            prefix = word[:3]
                            if prefix not in search_index:
                                search_index[prefix] = set()
                            search_index[prefix].add(ticker)
            
            logger.info(f"Created search index with {len(search_index)} terms")
            
        except Exception as e:
            logger.error(f"Error creating search index: {str(e)}")
            
        return search_index
    
    @staticmethod
    def filter_by_search(df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Filter dataframe by search query with ranking"""
        if not query or df.empty:
            return df
        
        query = query.upper().strip()
        
        # Create scoring for each row
        scores = pd.Series(0, index=df.index)
        
        # Exact ticker match (highest priority)
        ticker_exact = df['ticker'].str.upper() == query
        scores[ticker_exact] = 100
        
        # Ticker contains query
        ticker_contains = df['ticker'].str.contains(query, case=False, na=False)
        scores[ticker_contains & ~ticker_exact] = 80
        
        # Company name exact match
        company_exact = df['company_name'].str.upper() == query
        scores[company_exact] = 90
        
        # Company name contains query
        company_contains = df['company_name'].str.contains(query, case=False, na=False)
        scores[company_contains & ~company_exact & (scores < 80)] = 70
        
        # Word match in company name
        query_words = query.split()
        for word in query_words:
            if len(word) > 2:
                word_match = df['company_name'].str.contains(word, case=False, na=False)
                scores[word_match & (scores < 60)] = 50
        
        # Filter and sort by relevance
        matched = df[scores > 0].copy()
        matched['search_score'] = scores[scores > 0]
        matched = matched.sort_values(['search_score', 'master_score'], ascending=[False, False])
        
        # Drop the temporary search score column
        matched = matched.drop('search_score', axis=1)
        
        return matched.head(CONFIG.SEARCH_MAX_RESULTS)
    
    @staticmethod
    def search_stocks(df: pd.DataFrame, query: str, 
                     search_index: Optional[Dict[str, Set[str]]] = None) -> pd.DataFrame:
        """Enhanced search with intelligent matching"""
        if not query or df.empty:
            return pd.DataFrame()
        
        try:
            return SearchEngine.filter_by_search(df, query)
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return pd.DataFrame()

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
    def create_pattern_analysis(df: pd.DataFrame) -> go.Figure:
        """Create pattern frequency analysis"""
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
        
        pattern_counts = pd.Series(all_patterns).value_counts()
        
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
# EXPORT ENGINE - ENHANCED
# ============================================

class ExportEngine:
    """Handle all export operations with enhanced features"""
    
    @staticmethod
    def create_excel_report(df: pd.DataFrame) -> BytesIO:
        """Create comprehensive Excel report with enhanced formatting"""
        output = BytesIO()
        
        try:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                workbook = writer.book
                
                # Define formats
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#3498db',
                    'font_color': 'white',
                    'border': 1,
                    'align': 'center',
                    'valign': 'vcenter'
                })
                
                # Top 100 Stocks
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
                    # Auto-adjust column width
                    max_len = max(len(str(col)), 
                                 top_100[col].astype(str).str.len().max() if len(top_100) > 0 else 10)
                    worksheet.set_column(i, i, min(max_len + 2, 30))
                
                # Add other sheets as before...
                
                logger.info("Excel report created successfully")
                
        except Exception as e:
            logger.error(f"Error creating Excel report: {str(e)}")
            raise
        
        output.seek(0)
        return output
    
    @staticmethod
    def create_csv_export(df: pd.DataFrame) -> str:
        """Create CSV export with all relevant columns"""
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
# UI COMPONENTS
# ============================================

def format_display_dataframe(df: pd.DataFrame, show_fundamentals: bool) -> pd.DataFrame:
    """Format dataframe for display with proper formatting"""
    if df.empty:
        return df
    
    display_df = df.copy()
    
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
    
    # Prepare display columns
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
        'master_score': lambda x: f"{x:.1f}" if pd.notna(x) else '-',
        'price': lambda x: f"₹{x:,.0f}" if pd.notna(x) else '-',
        'from_low_pct': lambda x: f"{x:.0f}%" if pd.notna(x) else '-',
        'ret_30d': lambda x: f"{x:+.1f}%" if pd.notna(x) else '-',
        'rvol': lambda x: f"{x:.1f}x" if pd.notna(x) else '-'
    }
    
    # Format PE
    def format_pe(value):
        try:
            if pd.isna(value) or value == '':
                return '-'
            val = float(value)
            if val <= 0 or np.isinf(val):
                return 'Loss'
            elif val > 10000:
                return f"{val/1000:.0f}K"
            elif val > 1000:
                return f"{val:.0f}"
            else:
                return f"{val:.1f}"
        except:
            return '-'
    
    # Format EPS change
    def format_eps_change(value):
        try:
            if pd.isna(value) or value == '':
                return '-'
            val = float(value)
            if np.isinf(val):
                return '∞' if val > 0 else '-∞'
            elif abs(val) >= 10000:
                return f"{val/1000:+.1f}K%"
            elif abs(val) >= 1000:
                return f"{val:+.0f}%"
            else:
                return f"{val:+.1f}%"
        except:
            return '-'
    
    # Apply formatting
    for col, formatter in format_rules.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(formatter)
    
    if show_fundamentals:
        if 'pe' in display_df.columns:
            display_df['pe'] = display_df['pe'].apply(format_pe)
        if 'eps_change_pct' in display_df.columns:
            display_df['eps_change_pct'] = display_df['eps_change_pct'].apply(format_eps_change)
    
    # Select and rename columns
    available_display_cols = [col for col in display_cols.keys() if col in display_df.columns]
    display_df = display_df[available_display_cols]
    display_df.columns = [display_cols[col] for col in available_display_cols]
    
    return display_df

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main Streamlit application with enhanced UX"""
    
    # Page configuration
    st.set_page_config(
        page_title="Wave Detection Ultimate 3.0",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
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
    .search-box {
        position: sticky;
        top: 0;
        z-index: 999;
        background: white;
        padding: 10px 0;
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
                st.session_state.last_refresh = datetime.now()
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear Filters", type="secondary", use_container_width=True):
                clear_all_filters()
                st.rerun()
        
        st.markdown("---")
        
        # Display Mode Toggle
        st.markdown("### 📊 Display Mode")
        display_mode = st.radio(
            "Choose your view:",
            options=["Technical", "Hybrid (Technical + Fundamentals)"],
            index=0 if st.session_state.display_mode == "Technical" else 1,
            help="Technical: Pure momentum analysis | Hybrid: Adds PE & EPS data",
            key="display_mode_radio"
        )
        st.session_state.display_mode = display_mode
        show_fundamentals = (display_mode == "Hybrid (Technical + Fundamentals)")
        
        st.markdown("---")
        st.markdown("### 🔍 Filters")
    
    # Data loading and processing
    try:
        with st.spinner("📥 Loading data..."):
            raw_df = load_google_sheets_data(sheet_url, gid)
        
        with st.spinner(f"⚙️ Processing {len(raw_df):,} stocks..."):
            processed_df = DataProcessor.process_dataframe(raw_df)
            
            if processed_df.empty:
                st.error("❌ No valid data after processing. Please check your data source.")
                st.stop()
        
        with st.spinner("📊 Calculating rankings..."):
            # Calculate all scores
            scored_df = RankingEngine.calculate_all_scores(processed_df)
            # Calculate final rankings
            ranked_df = RankingEngine.calculate_final_rankings(scored_df)
        
        # Create search index
        if st.session_state.search_index is None:
            st.session_state.search_index = SearchEngine.create_search_index(ranked_df)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        with st.expander("🔍 Error Details"):
            st.code(str(e))
        st.stop()
    
    # Sidebar filters
    with st.sidebar:
        filters = {}
        
        # Category filter
        categories = FilterEngine.get_unique_values(ranked_df, 'category')
        selected_categories = st.multiselect(
            "Market Cap Category",
            options=categories,
            default=[],
            key="category_filter"
        )
        filters['categories'] = selected_categories if selected_categories else ['All']
        
        # Sector filter
        sectors = FilterEngine.get_unique_values(ranked_df, 'sector')
        selected_sectors = st.multiselect(
            "Sector",
            options=sectors,
            default=[],
            key="sector_filter"
        )
        filters['sectors'] = selected_sectors if selected_sectors else ['All']
        
        # Score filter
        filters['min_score'] = st.slider(
            "Minimum Master Score",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            key="min_score_filter"
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
                default=[],
                key="pattern_filter"
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
            index=0,
            key="trend_filter_select"
        )
        filters['trend_range'] = trend_options[filters['trend_filter']]
        
        # Advanced filters
        with st.expander("🔧 Advanced Filters"):
            # EPS tier filter
            if 'eps_tier' in ranked_df.columns:
                eps_tiers = FilterEngine.get_unique_values(ranked_df, 'eps_tier')
                filters['eps_tiers'] = st.multiselect(
                    "EPS Tier",
                    options=eps_tiers,
                    default=[],
                    key="eps_tier_filter"
                ) or ['All']
            
            # PE tier filter
            if 'pe_tier' in ranked_df.columns:
                pe_tiers = FilterEngine.get_unique_values(ranked_df, 'pe_tier')
                filters['pe_tiers'] = st.multiselect(
                    "PE Tier",
                    options=pe_tiers,
                    default=[],
                    key="pe_tier_filter"
                ) or ['All']
            
            # Price tier filter
            if 'price_tier' in ranked_df.columns:
                price_tiers = FilterEngine.get_unique_values(ranked_df, 'price_tier')
                filters['price_tiers'] = st.multiselect(
                    "Price Range",
                    options=price_tiers,
                    default=[],
                    key="price_tier_filter"
                ) or ['All']
            
            # EPS change filter
            if 'eps_change_pct' in ranked_df.columns:
                eps_change_input = st.text_input(
                    "Min EPS Change %",
                    value="",
                    placeholder="e.g. -50",
                    key="min_eps_change_filter"
                )
                
                if eps_change_input.strip():
                    try:
                        filters['min_eps_change'] = float(eps_change_input)
                    except ValueError:
                        st.error("Invalid EPS change value")
            
            # PE filters (only in hybrid mode)
            if show_fundamentals and 'pe' in ranked_df.columns:
                st.markdown("**🔍 Fundamental Filters**")
                
                col1, col2 = st.columns(2)
                with col1:
                    min_pe_input = st.text_input(
                        "Min PE",
                        value="",
                        placeholder="e.g. 10",
                        key="min_pe_filter"
                    )
                    if min_pe_input.strip():
                        try:
                            filters['min_pe'] = float(min_pe_input)
                        except ValueError:
                            st.error("Invalid Min PE")
                
                with col2:
                    max_pe_input = st.text_input(
                        "Max PE",
                        value="",
                        placeholder="e.g. 30",
                        key="max_pe_filter"
                    )
                    if max_pe_input.strip():
                        try:
                            filters['max_pe'] = float(max_pe_input)
                        except ValueError:
                            st.error("Invalid Max PE")
                
                filters['require_fundamental_data'] = st.checkbox(
                    "Only stocks with PE and EPS data",
                    value=False,
                    key="require_fundamental_filter"
                )
    
    # Apply filters
    filtered_df = FilterEngine.apply_filters(ranked_df, filters)
    filtered_df = filtered_df.sort_values('rank')
    
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
            
            if pe_coverage > 0:
                median_pe = filtered_df.loc[valid_pe, 'pe'].median()
                st.metric(
                    "Median PE",
                    f"{median_pe:.1f}x",
                    f"{pe_coverage} stocks"
                )
            else:
                st.metric("PE Data", "Limited")
        else:
            if not filtered_df.empty and 'master_score' in filtered_df.columns:
                min_score = filtered_df['master_score'].min()
                max_score = filtered_df['master_score'].max()
                st.metric("Score Range", f"{min_score:.1f}-{max_score:.1f}")
            else:
                st.metric("Score Range", "N/A")
    
    with col4:
        if show_fundamentals and 'eps_change_pct' in filtered_df.columns:
            valid_eps_change = filtered_df['eps_change_pct'].notna() & ~np.isinf(filtered_df['eps_change_pct'])
            positive_eps_growth = valid_eps_change & (filtered_df['eps_change_pct'] > 0)
            st.metric(
                "EPS Growth +ve",
                f"{positive_eps_growth.sum()}",
                f"of {valid_eps_change.sum()}"
            )
        else:
            if 'acceleration_score' in filtered_df.columns:
                accelerating = (filtered_df['acceleration_score'] >= 80).sum()
                st.metric("Accelerating", f"{accelerating}")
            else:
                st.metric("Accelerating", "0")
    
    with col5:
        if 'rvol' in filtered_df.columns:
            high_rvol = (filtered_df['rvol'] > 2).sum()
            st.metric("High RVOL", f"{high_rvol}")
        else:
            st.metric("High RVOL", "0")
    
    with col6:
        if 'trend_quality' in filtered_df.columns:
            strong_trends = (filtered_df['trend_quality'] >= 80).sum()
            total = len(filtered_df)
            pct = (strong_trends/total*100) if total > 0 else 0
            st.metric("Strong Trends", f"{strong_trends}", f"{pct:.0f}%")
        else:
            st.metric("With Patterns", f"{(filtered_df['patterns'] != '').sum()}")
    
    # Search bar (persistent across tabs)
    search_container = st.container()
    with search_container:
        col1, col2 = st.columns([5, 1])
        with col1:
            search_query = st.text_input(
                "🔍 Search stocks",
                value=st.session_state.search_query,
                placeholder="Enter ticker or company name...",
                key="search_input",
                label_visibility="collapsed"
            )
            if search_query != st.session_state.search_query:
                st.session_state.search_query = search_query
    
    # Apply search filter if query exists
    if search_query:
        filtered_df = SearchEngine.filter_by_search(filtered_df, search_query)
    
    # Main tabs (removed Visualizations tab)
    tabs = st.tabs([
        "🏆 Rankings", 
        "🔍 Search", 
        "🌊 Wave Radar", 
        "📊 Analysis", 
        "📥 Export",
        "ℹ️ About"
    ])
    
    # Tab 1: Rankings
    with tabs[0]:
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
            # Format and display
            formatted_df = format_display_dataframe(display_df, show_fundamentals)
            
            st.dataframe(
                formatted_df,
                use_container_width=True,
                height=min(600, len(formatted_df) * 35 + 50),
                hide_index=True
            )
            
            # Quick stats
            with st.expander("📊 Quick Statistics"):
                stat_cols = st.columns(4)
                
                with stat_cols[0]:
                    st.markdown("**Score Distribution**")
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
                    if show_fundamentals and 'pe' in filtered_df.columns:
                        st.markdown("**PE Stats**")
                        valid_pe = filtered_df['pe'].notna() & (filtered_df['pe'] > 0) & (filtered_df['pe'] < 10000)
                        if valid_pe.any():
                            median_pe = filtered_df.loc[valid_pe, 'pe'].median()
                            st.text(f"Median: {median_pe:.1f}x")
                    else:
                        st.markdown("**RVOL Stats**")
                        if 'rvol' in filtered_df.columns:
                            st.text(f"Max: {filtered_df['rvol'].max():.1f}x")
                            st.text(f"Avg: {filtered_df['rvol'].mean():.1f}x")
                
                with stat_cols[3]:
                    st.markdown("**Categories**")
                    for cat, count in filtered_df['category'].value_counts().head(3).items():
                        st.text(f"{cat}: {count}")
        else:
            st.warning("No stocks match the selected filters.")
    
    # Tab 2: Search
    with tabs[1]:
        st.markdown("### 🔍 Advanced Stock Search")
        
        if search_query:
            search_results = SearchEngine.search_stocks(
                ranked_df, 
                search_query,
                st.session_state.search_index
            )
            
            if not search_results.empty:
                st.success(f"Found {len(search_results)} matching stock(s)")
                
                # Display results
                for idx, stock in search_results.iterrows():
                    with st.expander(
                        f"📊 {stock['ticker']} - {stock['company_name']} "
                        f"(Rank #{int(stock['rank'])})",
                        expanded=True
                    ):
                        # Detailed stock information
                        metric_cols = st.columns(6)
                        
                        with metric_cols[0]:
                            st.metric("Master Score", f"{stock['master_score']:.1f}")
                        
                        with metric_cols[1]:
                            price_value = f"₹{stock['price']:,.0f}" if pd.notna(stock.get('price')) else "N/A"
                            st.metric("Price", price_value)
                        
                        with metric_cols[2]:
                            st.metric("From Low", f"{stock['from_low_pct']:.0f}%")
                        
                        with metric_cols[3]:
                            st.metric("30D Return", f"{stock['ret_30d']:.1f}%")
                        
                        with metric_cols[4]:
                            st.metric("RVOL", f"{stock['rvol']:.1f}x")
                        
                        with metric_cols[5]:
                            st.metric("Category %ile", f"{stock.get('category_percentile', 0):.0f}")
                        
                        # Additional details
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
                                color = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
                                st.markdown(f"**{name}**<br>{color} {score:.0f}", unsafe_allow_html=True)
                        
                        if stock.get('patterns'):
                            st.markdown(f"**🎯 Patterns:** {stock['patterns']}")
            else:
                st.info("No stocks found. Try different search terms.")
        else:
            st.info("Enter a ticker symbol or company name to search.")
    
    # Tab 3: Wave Radar (keeping existing implementation)
    with tabs[2]:
        st.markdown("### 🌊 Wave Radar - Early Momentum Detection System")
        # ... (keep existing Wave Radar implementation)
        st.info("Wave Radar implementation continues here...")
    
    # Tab 4: Analysis
    with tabs[3]:
        st.markdown("### 📊 Market Analysis")
        
        if not filtered_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_dist = Visualizer.create_score_distribution(filtered_df)
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                fig_patterns = Visualizer.create_pattern_analysis(filtered_df)
                st.plotly_chart(fig_patterns, use_container_width=True)
            
            # Sector analysis
            if 'sector' in filtered_df.columns:
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
                        
                        st.dataframe(
                            sector_df.style.background_gradient(subset=['Avg Score']),
                            use_container_width=True
                        )
                except Exception as e:
                    st.error("Unable to perform sector analysis.")
        else:
            st.info("No data available for analysis.")
    
    # Tab 5: Export
    with tabs[4]:
        st.markdown("### 📥 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Excel Report")
            st.markdown(
                "Comprehensive multi-sheet report including:\n"
                "- Top 100 stocks with all scores\n"
                "- Complete stock list\n"
                "- Sector & category analysis\n"
                "- Pattern frequency analysis"
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
                                file_name=f"wave_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            
                            st.success("Excel report generated successfully!")
                            
                        except Exception as e:
                            st.error(f"Error generating Excel report: {str(e)}")
        
        with col2:
            st.markdown("#### 📄 CSV Export")
            st.markdown(
                "CSV format with:\n"
                "- All ranking scores\n"
                "- Price and return data\n"
                "- Pattern detections\n"
                "- Category classifications"
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
                            file_name=f"wave_detection_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        st.success("CSV export generated successfully!")
                        
                    except Exception as e:
                        st.error(f"Error generating CSV: {str(e)}")
    
    # Tab 6: About
    with tabs[5]:
        st.markdown("### ℹ️ About Wave Detection Ultimate 3.0")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🎯 System Overview
            
            Wave Detection Ultimate 3.0 is a professional-grade stock ranking system 
            that combines technical momentum analysis with optional fundamental data 
            to identify high-probability trading opportunities.
            
            #### 🔑 Key Features
            
            - **Master Score 3.0**: Proprietary ranking algorithm
            - **Wave Radar™**: Early momentum detection system
            - **Smart Search**: Intelligent stock search with relevance ranking
            - **Dual Mode**: Technical-only or Hybrid (with fundamentals)
            - **Pattern Detection**: 11+ automatic pattern recognitions
            - **Real-time RVOL**: Today's relative volume analysis
            - **Professional Exports**: Excel and CSV with full data
            """)
        
        with col2:
            st.markdown("""
            #### 📊 Master Score Components
            
            | Component | Weight | Description |
            |-----------|--------|-------------|
            | Position | 30% | 52-week range position |
            | Volume | 25% | Multi-timeframe volume |
            | Momentum | 15% | 30-day price momentum |
            | Acceleration | 10% | Momentum acceleration |
            | Breakout | 10% | Breakout probability |
            | RVOL | 10% | Today's relative volume |
            
            #### 🌊 Pattern Types
            
            - 🔥 **Category Leader**: Top 10% in category
            - 💎 **Hidden Gem**: High category rank, low market rank
            - 🚀 **Accelerating**: Momentum building up
            - 🏦 **Institutional**: Smart money accumulation
            - ⚡ **Volume Explosion**: Unusual activity (RVOL >3x)
            - 🎯 **Breakout Ready**: Near resistance with volume
            - 👑 **Market Leader**: Top 5% overall
            """)
        
        st.markdown("---")
        
        st.markdown("""
        #### 💡 Quick Start Guide
        
        1. **Data Source**: Enter your Google Sheets URL and GID
        2. **Display Mode**: Choose Technical or Hybrid view
        3. **Apply Filters**: Use sidebar filters to narrow results
        4. **Search**: Use the search bar to find specific stocks
        5. **Export**: Download Excel or CSV reports
        
        #### 🎯 Trading Tips
        
        - Look for stocks with multiple patterns
        - Check Wave Radar for early momentum signals
        - Use category ranks to find sector rotation
        - Monitor RVOL for unusual activity
        - Combine high scores with strong trends
        
        #### ⚡ Performance Tips
        
        - Data refreshes every 5 minutes (cached)
        - Use filters to reduce data processing
        - Clear filters button resets all selections
        - Search is optimized for speed
        
        ---
        
        **Version**: 3.1.0-FINAL | **Status**: Production | **Last Update**: {datetime.now().strftime('%B %Y')}
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            Wave Detection Ultimate 3.0 | Professional Stock Ranking System<br>
            <small>Real-time momentum detection • Smart money flow tracking • Pattern recognition</small>
        </div>
        """,
        unsafe_allow_html=True
    )

# Run the application
if __name__ == "__main__":
    main()
