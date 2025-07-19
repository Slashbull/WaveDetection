"""
Wave Detection System 6.0 - Professional Trading Analytics Platform
=================================================================
A production-ready stock analysis system with comprehensive scoring,
filtering, and ranking capabilities.

Version 6.0.1 Features:
- Classic 4-pillar scoring system (Momentum, Position, Volume, Quality)
- Enhanced scoring with pattern detection and lifecycle analysis
- Future potential scoring based on lifecycle positioning
- Momentum acceleration detection
- Volume pattern recognition (accumulation/distribution)
- Position opportunity identification (breakout/reversal/momentum)
- EPS momentum tracking
- Multi-mode display (Classic/Enhanced/Future Potential)
- Comprehensive filtering by category, sector, and tiers
- Professional visualizations and reporting

Author: Professional Implementation
Version: 6.0.1 (Enhanced)
Status: Production Ready
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
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings
import time
from functools import wraps, lru_cache
import re

# ============================================
# CONFIGURATION & CONSTANTS
# ============================================

# Suppress warnings in production
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Page configuration
try:
    st.set_page_config(
        page_title="Wave Detection 6.0 | Professional Analytics",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except:
    pass

# ============================================
# CONSTANTS
# ============================================

class Config:
    """Application configuration constants"""
    
    # Data source
    DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk/edit?usp=sharing"
    DEFAULT_GID = "2026492216"
    
    # Cache settings
    CACHE_TTL = 300  # 5 minutes
    
    # Scoring weights
    MOMENTUM_WEIGHT = 0.30
    POSITION_WEIGHT = 0.25
    VOLUME_WEIGHT = 0.25
    QUALITY_WEIGHT = 0.20
    
    # Display settings
    TOP_STOCKS_DISPLAY = 50
    DECIMAL_PLACES = 2
    
    # Tier definitions
    EPS_TIERS = {
        "Below 5": (-np.inf, 5),
        "5-15": (5, 15),
        "15-35": (15, 35),
        "35-55": (35, 55),
        "55-75": (55, 75),
        "75-95": (75, 95),
        "Above 95": (95, np.inf)
    }
    
    PRICE_TIERS = {
        "Below 100": (-np.inf, 100),
        "100-200": (100, 200),
        "200-500": (200, 500),
        "500-1000": (500, 1000),
        "1000-2000": (1000, 2000),
        "2000-5000": (2000, 5000),
        "Above 5000": (5000, np.inf)
    }
    
    PE_TIERS = {
        "Below 10": (-np.inf, 10),
        "10-20": (10, 20),
        "20-30": (20, 30),
        "30-50": (30, 50),
        "50-75": (50, 75),
        "Above 75": (75, np.inf),
        "Negative/NA": (None, None)  # Special handling
    }

# ============================================
# DATA MODELS
# ============================================

@dataclass
class StockScore:
    """Complete stock scoring information"""
    ticker: str
    company_name: str
    category: str
    sector: str
    price: float
    
    # Tier classifications
    eps_tier: str
    price_tier: str
    pe_tier: str
    
    # Individual scores
    momentum_score: float
    position_score: float
    volume_score: float
    quality_score: float
    master_score: float
    
    # Rank
    rank: int
    percentile: float
    
    # Additional metrics
    metrics: Dict[str, Any] = field(default_factory=dict)

# ============================================
# UTILITY FUNCTIONS
# ============================================

def timer_decorator(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        if execution_time > 1.0:
            logger.warning(f"{func.__name__} took {execution_time:.2f} seconds")
        
        return result
    return wrapper

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers"""
    try:
        if denominator == 0 or pd.isna(denominator):
            return default
        return numerator / denominator
    except:
        return default

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert to float"""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except:
        return default

# ============================================
# DATA PROCESSING
# ============================================

class DataProcessor:
    """Handles all data loading and cleaning operations"""
    
    @staticmethod
    def clean_indian_number_format(value: str) -> Optional[float]:
        """Clean Indian formatted numbers (₹, commas, %)"""
        if pd.isna(value) or value == '':
            return np.nan
            
        try:
            # Convert to string and clean
            cleaned = str(value)
            
            # Remove currency symbol
            cleaned = cleaned.replace('₹', '')
            
            # Remove percentage sign
            cleaned = cleaned.replace('%', '')
            
            # Remove commas
            cleaned = cleaned.replace(',', '')
            
            # Remove any extra spaces
            cleaned = cleaned.strip()
            
            # Handle special cases
            if cleaned in ['', '-', 'N/A', 'n/a', '#N/A', 'nan', 'None']:
                return np.nan
                
            # Convert to float
            return float(cleaned)
            
        except Exception as e:
            logger.debug(f"Failed to convert '{value}': {e}")
            return np.nan
    
    @staticmethod
    @timer_decorator
    def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Complete dataframe processing with error handling"""
        if df.empty:
            logger.warning("Empty dataframe provided for processing")
            return df
            
        try:
            # Create a copy to avoid modifying original
            processed_df = df.copy()
            
            # Define column groups for processing
            numeric_columns = {
                'price_cols': ['price', 'prev_close', 'low_52w', 'high_52w', 
                              'sma_20d', 'sma_50d', 'sma_200d'],
                'return_cols': ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 
                               'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y'],
                'volume_cols': ['volume_1d', 'volume_7d', 'volume_30d', 
                               'volume_90d', 'volume_180d'],
                'ratio_cols': ['from_low_pct', 'from_high_pct', 'rvol',
                              'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
                              'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 
                              'vol_ratio_90d_180d'],
                'fundamental_cols': ['pe', 'eps_current', 'eps_last_qtr', 'eps_change_pct']
            }
            
            # Process all numeric columns
            all_numeric_cols = []
            for col_group in numeric_columns.values():
                all_numeric_cols.extend(col_group)
            
            # Clean numeric columns
            for col in all_numeric_cols:
                if col in processed_df.columns:
                    processed_df[col] = processed_df[col].apply(DataProcessor.clean_indian_number_format)
            
            # Special handling for market cap
            if 'market_cap' in processed_df.columns:
                processed_df['market_cap_clean'] = processed_df['market_cap'].apply(
                    lambda x: DataProcessor.clean_market_cap(x)
                )
            
            # Clean categorical columns
            categorical_cols = ['ticker', 'company_name', 'category', 'sector']
            for col in categorical_cols:
                if col in processed_df.columns:
                    processed_df[col] = processed_df[col].astype(str).str.strip()
                    processed_df[col] = processed_df[col].replace(['nan', 'None', '', 'N/A'], 'Unknown')
            
            # Fix volume ratios - they are percentage changes, not ratios
            # Convert from percentage change to multiplier
            for col in numeric_columns['ratio_cols']:
                if col in processed_df.columns and 'vol_ratio' in col:
                    # -56.61% becomes 0.4339 (100-56.61)/100
                    processed_df[col] = (100 + processed_df[col]) / 100
            
            # Remove invalid rows (only price must be valid and positive)
            initial_count = len(processed_df)
            processed_df = processed_df[
                processed_df['price'].notna() & 
                (processed_df['price'] > 0)
            ]
            final_count = len(processed_df)
            
            if initial_count != final_count:
                logger.info(f"Removed {initial_count - final_count} invalid rows")
            
            # Add tier classifications
            processed_df = DataProcessor.add_tiers(processed_df)
            
            return processed_df
            
        except Exception as e:
            logger.error(f"Error processing dataframe: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def clean_market_cap(value: str) -> float:
        """Clean market cap values"""
        try:
            if pd.isna(value):
                return np.nan
                
            cleaned = str(value)
            cleaned = cleaned.replace('₹', '')
            cleaned = cleaned.replace(',', '')
            
            # Handle Cr (Crores) and Lakh
            if 'Cr' in cleaned:
                cleaned = cleaned.replace('Cr', '').strip()
                return float(cleaned)
            elif 'Lakh' in cleaned:
                cleaned = cleaned.replace('Lakh', '').strip()
                return float(cleaned) / 100  # Convert to Crores
            else:
                return float(cleaned)
                
        except:
            return np.nan
    
    @staticmethod
    def add_tiers(df: pd.DataFrame) -> pd.DataFrame:
        """Add tier classifications"""
        try:
            # EPS Tier
            df['eps_tier'] = df['eps_current'].apply(DataProcessor.get_eps_tier)
            
            # Price Tier
            df['price_tier'] = df['price'].apply(DataProcessor.get_price_tier)
            
            # PE Tier
            df['pe_tier'] = df['pe'].apply(DataProcessor.get_pe_tier)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding tiers: {e}")
            return df
    
    @staticmethod
    def get_eps_tier(eps: float) -> str:
        """Determine EPS tier"""
        if pd.isna(eps):
            return "Unknown"
            
        for tier_name, (min_val, max_val) in Config.EPS_TIERS.items():
            if min_val < eps <= max_val:
                return tier_name
        return "Unknown"
    
    @staticmethod
    def get_price_tier(price: float) -> str:
        """Determine price tier"""
        if pd.isna(price):
            return "Unknown"
            
        for tier_name, (min_val, max_val) in Config.PRICE_TIERS.items():
            if min_val < price <= max_val:
                return tier_name
        return "Unknown"
    
    @staticmethod
    def get_pe_tier(pe: float) -> str:
        """Determine PE tier"""
        if pd.isna(pe) or pe <= 0:
            return "Negative/NA"
            
        for tier_name, (min_val, max_val) in Config.PE_TIERS.items():
            if tier_name == "Negative/NA":
                continue
            if min_val < pe <= max_val:
                return tier_name
        return "Unknown"

# ============================================
# SCORING ENGINE
# ============================================

class ScoringEngine:
    """Calculates all scores for stocks"""
    
    @staticmethod
    def calculate_momentum_score(row: pd.Series) -> float:
        """Calculate momentum score using multiple timeframes"""
        try:
            # Get returns with fallback to 0
            ret_1d = safe_float(row.get('ret_1d', 0))
            ret_7d = safe_float(row.get('ret_7d', 0))
            ret_30d = safe_float(row.get('ret_30d', 0))
            ret_3m = safe_float(row.get('ret_3m', 0))
            ret_1y = safe_float(row.get('ret_1y', 0))
            
            # Calculate daily average returns for consistency
            daily_avg_7d = safe_divide(ret_7d, 7)
            daily_avg_30d = safe_divide(ret_30d, 30)
            daily_avg_3m = safe_divide(ret_3m, 90)
            daily_avg_1y = safe_divide(ret_1y, 365)
            
            # Weighted momentum score
            momentum = (
                ret_1d * 0.35 +          # Today's return (35%)
                daily_avg_7d * 0.25 +    # Weekly average (25%)
                daily_avg_30d * 0.20 +   # Monthly average (20%)
                daily_avg_3m * 0.10 +    # 3-month average (10%)
                daily_avg_1y * 0.10      # Yearly average (10%)
            )
            
            # Scale to 0-100 range
            # Assuming daily returns typically range from -10% to +10%
            momentum_scaled = np.clip((momentum + 10) * 5, 0, 100)
            
            return momentum_scaled
            
        except Exception as e:
            logger.debug(f"Error calculating momentum score: {e}")
            return 0.0
    
    @staticmethod
    def calculate_momentum_acceleration(row: pd.Series) -> float:
        """Calculate if momentum is accelerating or decelerating"""
        try:
            # Get returns
            ret_1d = safe_float(row.get('ret_1d', 0))
            ret_7d = safe_float(row.get('ret_7d', 0))
            ret_30d = safe_float(row.get('ret_30d', 0))
            ret_3m = safe_float(row.get('ret_3m', 0))
            
            # Calculate daily averages
            daily_1d = ret_1d
            daily_7d = safe_divide(ret_7d, 7)
            daily_30d = safe_divide(ret_30d, 30)
            daily_3m = safe_divide(ret_3m, 90)
            
            # Check acceleration patterns
            accel_score = 0
            
            # Today vs Week
            if daily_1d > daily_7d:
                accel_score += 33.33
            
            # Week vs Month
            if daily_7d > daily_30d:
                accel_score += 33.33
            
            # Month vs Quarter
            if daily_30d > daily_3m:
                accel_score += 33.34
            
            # Bonus for strong acceleration
            if daily_1d > daily_7d > daily_30d > daily_3m:
                accel_score = 100  # Perfect acceleration
            
            return accel_score
            
        except Exception as e:
            logger.debug(f"Error calculating momentum acceleration: {e}")
            return 0.0
    
    @staticmethod
    def calculate_momentum_divergence(row: pd.Series) -> Dict[str, float]:
        """Calculate momentum divergence between short and long term"""
        try:
            # Short-term momentum (1d, 3d, 7d)
            short_term = (
                safe_float(row.get('ret_1d', 0)) +
                safe_float(row.get('ret_3d', 0)) +
                safe_float(row.get('ret_7d', 0))
            ) / 3
            
            # Long-term momentum (30d, 3m, 6m)
            long_term = (
                safe_float(row.get('ret_30d', 0)) +
                safe_float(row.get('ret_3m', 0)) +
                safe_float(row.get('ret_6m', 0))
            ) / 3
            
            divergence = short_term - long_term
            
            return {
                'short_term_momentum': short_term,
                'long_term_momentum': long_term,
                'momentum_divergence': divergence
            }
            
        except Exception as e:
            logger.debug(f"Error calculating momentum divergence: {e}")
            return {
                'short_term_momentum': 0,
                'long_term_momentum': 0,
                'momentum_divergence': 0
            }
    
    @staticmethod
    def calculate_position_score(row: pd.Series) -> float:
        """Calculate position score based on price levels"""
        try:
            price = safe_float(row.get('price', 0))
            low_52w = safe_float(row.get('low_52w', 0))
            high_52w = safe_float(row.get('high_52w', 0))
            sma_200d = safe_float(row.get('sma_200d', price))
            
            from_low_pct = safe_float(row.get('from_low_pct', 0))
            from_high_pct = safe_float(row.get('from_high_pct', 0))
            
            # Position in 52-week range (0-100)
            range_position = 0
            if high_52w > low_52w:
                range_position = ((price - low_52w) / (high_52w - low_52w)) * 100
            
            # Distance from high (less negative is better)
            high_distance = 100 + from_high_pct  # Convert negative % to positive scale
            
            # Position relative to 200 SMA
            sma_position = 0
            if sma_200d > 0:
                sma_position = ((price / sma_200d) - 1) * 100  # Percentage above/below
            
            # Combined position score
            position = (
                from_low_pct * 0.4 +          # Distance from low (40%)
                high_distance * 0.3 +          # Room to grow (30%)
                np.clip(sma_position, -50, 50) * 0.3  # SMA position (30%)
            )
            
            # Normalize to 0-100
            position_scaled = np.clip(position, 0, 100)
            
            return position_scaled
            
        except Exception as e:
            logger.debug(f"Error calculating position score: {e}")
            return 0.0
    
    @staticmethod
    def calculate_position_opportunity(row: pd.Series) -> Dict[str, Any]:
        """Calculate position-based opportunity score"""
        try:
            from_high_pct = safe_float(row.get('from_high_pct', 0))
            from_low_pct = safe_float(row.get('from_low_pct', 0))
            ret_30d = safe_float(row.get('ret_30d', 0))
            ret_1d = safe_float(row.get('ret_1d', 0))
            ret_7d = safe_float(row.get('ret_7d', 0))
            price = safe_float(row.get('price', 0))
            sma_20d = safe_float(row.get('sma_20d', price))
            sma_50d = safe_float(row.get('sma_50d', price))
            sma_200d = safe_float(row.get('sma_200d', price))
            eps_change_pct = safe_float(row.get('eps_change_pct', 0))
            
            opportunity_type = "NEUTRAL"
            opportunity_score = 50
            
            # Breakout opportunity
            if (from_high_pct > -10 and ret_30d > 0 and 
                price > sma_20d and price > sma_50d and price > sma_200d):
                opportunity_type = "BREAKOUT"
                opportunity_score = 85 + min(ret_30d, 15)
            
            # Reversal opportunity
            elif (from_high_pct < -50 and from_low_pct < 30 and
                  ret_1d > 0 and ret_7d > 0 and eps_change_pct > 0):
                opportunity_type = "REVERSAL"
                opportunity_score = 90 + min(eps_change_pct / 10, 10)
            
            # Momentum opportunity
            elif (from_low_pct > 50 and from_high_pct > -30 and
                  ret_7d > 0 and ret_30d > 0):
                opportunity_type = "MOMENTUM"
                opportunity_score = 75 + min(ret_7d, 25)
            
            # Avoid zone
            elif (from_high_pct < -20 and from_high_pct > -40 and
                  ret_30d < 0):
                opportunity_type = "AVOID"
                opportunity_score = 20
            
            return {
                'position_opportunity': opportunity_type,
                'position_opportunity_score': np.clip(opportunity_score, 0, 100)
            }
            
        except Exception as e:
            logger.debug(f"Error calculating position opportunity: {e}")
            return {
                'position_opportunity': 'UNKNOWN',
                'position_opportunity_score': 50
            }
    
    @staticmethod
    def calculate_trend_quality(row: pd.Series) -> Dict[str, float]:
        """Calculate trend quality metrics"""
        try:
            price = safe_float(row.get('price', 0))
            sma_20d = safe_float(row.get('sma_20d', price))
            sma_50d = safe_float(row.get('sma_50d', price))
            sma_200d = safe_float(row.get('sma_200d', price))
            
            # MA alignment score
            ma_alignment = 0
            if price > sma_20d:
                ma_alignment += 25
            if sma_20d > sma_50d:
                ma_alignment += 25
            if sma_50d > sma_200d:
                ma_alignment += 25
            if price > sma_200d:
                ma_alignment += 25
            
            # MA spread (trend strength)
            spread_20_50 = abs((sma_20d - sma_50d) / sma_50d * 100) if sma_50d > 0 else 0
            spread_50_200 = abs((sma_50d - sma_200d) / sma_200d * 100) if sma_200d > 0 else 0
            
            # Trend quality score
            trend_quality = ma_alignment * 0.6 + min(spread_20_50 + spread_50_200, 40) * 0.4
            
            return {
                'ma_alignment_score': ma_alignment,
                'trend_spread': spread_20_50 + spread_50_200,
                'trend_quality_score': trend_quality
            }
            
        except Exception as e:
            logger.debug(f"Error calculating trend quality: {e}")
            return {
                'ma_alignment_score': 0,
                'trend_spread': 0,
                'trend_quality_score': 0
            }
    
    @staticmethod
    def calculate_volume_score(row: pd.Series) -> float:
        """Calculate volume score with corrected ratios"""
        try:
            # Get volume metrics
            rvol = safe_float(row.get('rvol', 1.0))
            
            # Volume ratios are now multipliers (after conversion)
            vol_1d_90d = safe_float(row.get('vol_ratio_1d_90d', 1.0))
            vol_30d_90d = safe_float(row.get('vol_ratio_30d_90d', 1.0))
            
            # Calculate volume score components
            # RVOL component (0-100 scale, capped at 5x)
            rvol_score = min(rvol * 20, 100)
            
            # Short-term volume trend (0-100 scale)
            short_trend_score = min(vol_1d_90d * 50, 100)
            
            # Medium-term volume trend (0-100 scale)
            medium_trend_score = min(vol_30d_90d * 50, 100)
            
            # Combined volume score
            volume = (
                rvol_score * 0.40 +           # Current volume (40%)
                short_trend_score * 0.30 +    # Short-term trend (30%)
                medium_trend_score * 0.30     # Medium-term trend (30%)
            )
            
            return np.clip(volume, 0, 100)
            
        except Exception as e:
            logger.debug(f"Error calculating volume score: {e}")
            return 0.0
    
    @staticmethod
    def calculate_volume_acceleration(row: pd.Series) -> float:
        """Calculate volume acceleration - institutional footprint"""
        try:
            # Get volume ratios
            vol_30d_90d = safe_float(row.get('vol_ratio_30d_90d', 1.0))
            vol_30d_180d = safe_float(row.get('vol_ratio_30d_180d', 1.0))
            
            # Calculate acceleration
            volume_acceleration = vol_30d_90d - vol_30d_180d
            
            # Convert to score (0-100)
            # Strong accumulation: > 0.15
            # Accumulation: > 0.05
            # Distribution: < -0.05
            # Strong distribution: < -0.15
            
            if volume_acceleration > 0.15:
                return 100
            elif volume_acceleration > 0.10:
                return 85
            elif volume_acceleration > 0.05:
                return 70
            elif volume_acceleration > 0:
                return 55
            elif volume_acceleration > -0.05:
                return 45
            elif volume_acceleration > -0.10:
                return 30
            elif volume_acceleration > -0.15:
                return 15
            else:
                return 0
                
        except Exception as e:
            logger.debug(f"Error calculating volume acceleration: {e}")
            return 50.0
    
    @staticmethod
    def calculate_volume_pattern(row: pd.Series) -> Dict[str, Any]:
        """Detect volume patterns - accumulation/distribution"""
        try:
            # Get all volume ratios
            vol_1d_90d = safe_float(row.get('vol_ratio_1d_90d', 1.0))
            vol_7d_90d = safe_float(row.get('vol_ratio_7d_90d', 1.0))
            vol_30d_90d = safe_float(row.get('vol_ratio_30d_90d', 1.0))
            rvol = safe_float(row.get('rvol', 1.0))
            
            # Pattern detection
            pattern = "NEUTRAL"
            pattern_score = 50
            
            # Expansion pattern (accumulation)
            if vol_1d_90d > vol_7d_90d > vol_30d_90d:
                pattern = "EXPANDING"
                pattern_score = 80 + (vol_1d_90d - vol_30d_90d) * 20
            
            # Contraction pattern (distribution)
            elif vol_1d_90d < vol_7d_90d < vol_30d_90d:
                pattern = "CONTRACTING"
                pattern_score = 20 - (vol_30d_90d - vol_1d_90d) * 20
            
            # Spike pattern (climax)
            elif rvol > 3 and vol_1d_90d > 2:
                pattern = "SPIKE"
                pattern_score = 70
            
            # Dormant pattern
            elif rvol < 0.5 and vol_30d_90d < 0.7:
                pattern = "DORMANT"
                pattern_score = 30
            
            return {
                'volume_pattern': pattern,
                'volume_pattern_score': np.clip(pattern_score, 0, 100)
            }
            
        except Exception as e:
            logger.debug(f"Error detecting volume pattern: {e}")
            return {
                'volume_pattern': 'UNKNOWN',
                'volume_pattern_score': 50
            }
    
    @staticmethod
    def calculate_quality_score(row: pd.Series) -> float:
        """Calculate fundamental quality score"""
        try:
            pe = safe_float(row.get('pe', 50))
            eps_change_pct = safe_float(row.get('eps_change_pct', 0))
            eps_current = safe_float(row.get('eps_current', 0))
            
            # EPS growth component (0-100, capped at 100% growth)
            eps_growth_score = min(max(eps_change_pct, -50) + 50, 100)
            
            # PE value component (inverse scoring - lower PE is better)
            pe_score = 0
            if pe > 0:
                if pe < 10:
                    pe_score = 90
                elif pe < 20:
                    pe_score = 70
                elif pe < 30:
                    pe_score = 50
                elif pe < 50:
                    pe_score = 30
                else:
                    pe_score = 10
            
            # EPS level component
            eps_level_score = 0
            if eps_current > 0:
                if eps_current >= 100:
                    eps_level_score = 100
                elif eps_current >= 50:
                    eps_level_score = 80
                elif eps_current >= 20:
                    eps_level_score = 60
                elif eps_current >= 10:
                    eps_level_score = 40
                else:
                    eps_level_score = 20
            
            # Combined quality score
            quality = (
                eps_growth_score * 0.40 +    # EPS growth (40%)
                pe_score * 0.30 +            # PE valuation (30%)
                eps_level_score * 0.30       # EPS level (30%)
            )
            
            return np.clip(quality, 0, 100)
            
        except Exception as e:
            logger.debug(f"Error calculating quality score: {e}")
            return 0.0
    
    @staticmethod
    def calculate_eps_momentum(row: pd.Series) -> Dict[str, float]:
        """Calculate EPS momentum - fundamental acceleration"""
        try:
            eps_current = safe_float(row.get('eps_current', 0))
            eps_last_qtr = safe_float(row.get('eps_last_qtr', 0))
            eps_change_pct = safe_float(row.get('eps_change_pct', 0))
            
            # EPS momentum score
            eps_momentum_score = 50  # Base score
            
            # Strong growth
            if eps_change_pct > 50:
                eps_momentum_score = 90
            elif eps_change_pct > 30:
                eps_momentum_score = 80
            elif eps_change_pct > 20:
                eps_momentum_score = 70
            elif eps_change_pct > 10:
                eps_momentum_score = 60
            elif eps_change_pct > 0:
                eps_momentum_score = 55
            elif eps_change_pct > -10:
                eps_momentum_score = 45
            elif eps_change_pct > -20:
                eps_momentum_score = 30
            else:
                eps_momentum_score = 20
            
            # Consistency bonus
            if eps_current > eps_last_qtr and eps_last_qtr > 0:
                eps_momentum_score = min(eps_momentum_score + 10, 100)
            
            return {
                'eps_momentum_score': eps_momentum_score,
                'eps_growth_rate': eps_change_pct
            }
            
        except Exception as e:
            logger.debug(f"Error calculating EPS momentum: {e}")
            return {
                'eps_momentum_score': 50,
                'eps_growth_rate': 0
            }
    
    @staticmethod
    def detect_lifecycle_stage(row: pd.Series) -> Dict[str, Any]:
        """Detect which lifecycle stage the stock is in"""
        try:
            # Get all necessary metrics
            from_high_pct = safe_float(row.get('from_high_pct', 0))
            from_low_pct = safe_float(row.get('from_low_pct', 0))
            
            # Volume metrics
            vol_30d_90d = safe_float(row.get('vol_ratio_30d_90d', 1.0))
            vol_30d_180d = safe_float(row.get('vol_ratio_30d_180d', 1.0))
            volume_acceleration = vol_30d_90d - vol_30d_180d
            
            # Returns
            ret_1d = safe_float(row.get('ret_1d', 0))
            ret_7d = safe_float(row.get('ret_7d', 0))
            ret_30d = safe_float(row.get('ret_30d', 0))
            
            # Moving averages
            price = safe_float(row.get('price', 0))
            sma_50d = safe_float(row.get('sma_50d', price))
            sma_200d = safe_float(row.get('sma_200d', price))
            
            # Fundamentals
            eps_change_pct = safe_float(row.get('eps_change_pct', 0))
            pe = safe_float(row.get('pe', 50))
            
            # Momentum divergence
            divergence_data = ScoringEngine.calculate_momentum_divergence(row)
            momentum_divergence = divergence_data['momentum_divergence']
            
            # Default stage
            stage = "UNKNOWN"
            stage_score = 50
            future_potential = 50
            
            # STAGE 1: ACCUMULATION (Smart money entering)
            if (from_high_pct < -50 and volume_acceleration > 0 and
                ret_1d > -2 and ret_7d > -5 and eps_change_pct > 0):
                stage = "ACCUMULATION"
                stage_score = 85
                future_potential = 90 + min(volume_acceleration * 50, 10)
            
            # STAGE 2: EARLY MARKUP (Starting the big move)
            elif (price > sma_50d and momentum_divergence > 0 and
                  volume_acceleration > 0 and from_low_pct > 20):
                stage = "EARLY_MARKUP"
                stage_score = 90
                future_potential = 85 + min(momentum_divergence, 15)
            
            # STAGE 2: LATE MARKUP (Move maturing)
            elif (price > sma_200d and from_high_pct > -20 and
                  momentum_divergence < 0 and from_low_pct > 100):
                stage = "LATE_MARKUP"
                stage_score = 60
                future_potential = 40
            
            # STAGE 3: DISTRIBUTION (Smart money exiting)
            elif (from_high_pct > -10 and volume_acceleration < 0 and
                  momentum_divergence < 0 and pe > 50):
                stage = "DISTRIBUTION"
                stage_score = 30
                future_potential = 20
            
            # STAGE 4: MARKDOWN (Downtrend)
            elif (price < sma_50d and price < sma_200d and
                  ret_30d < -10 and volume_acceleration < -0.1):
                stage = "MARKDOWN"
                stage_score = 10
                future_potential = 10
            
            # RECOVERY: Potential turnaround
            elif (from_high_pct < -30 and ret_7d > 0 and ret_1d > 0 and
                  volume_acceleration > 0.05):
                stage = "RECOVERY"
                stage_score = 70
                future_potential = 75
            
            return {
                'lifecycle_stage': stage,
                'lifecycle_score': stage_score,
                'future_potential_score': future_potential
            }
            
        except Exception as e:
            logger.debug(f"Error detecting lifecycle stage: {e}")
            return {
                'lifecycle_stage': 'UNKNOWN',
                'lifecycle_score': 50,
                'future_potential_score': 50
            }
    
    @staticmethod
    @timer_decorator
    def calculate_all_scores(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all scores for the dataframe"""
        if df.empty:
            return df
            
        try:
            # Calculate original scores
            logger.info("Calculating momentum scores...")
            df['momentum_score'] = df.apply(ScoringEngine.calculate_momentum_score, axis=1)
            
            logger.info("Calculating position scores...")
            df['position_score'] = df.apply(ScoringEngine.calculate_position_score, axis=1)
            
            logger.info("Calculating volume scores...")
            df['volume_score'] = df.apply(ScoringEngine.calculate_volume_score, axis=1)
            
            logger.info("Calculating quality scores...")
            df['quality_score'] = df.apply(ScoringEngine.calculate_quality_score, axis=1)
            
            # Calculate master score (original)
            df['master_score'] = (
                df['momentum_score'] * Config.MOMENTUM_WEIGHT +
                df['position_score'] * Config.POSITION_WEIGHT +
                df['volume_score'] * Config.VOLUME_WEIGHT +
                df['quality_score'] * Config.QUALITY_WEIGHT
            )
            
            # Calculate enhanced scores
            logger.info("Calculating enhanced scores...")
            
            # Momentum acceleration
            df['momentum_acceleration_score'] = df.apply(ScoringEngine.calculate_momentum_acceleration, axis=1)
            
            # Momentum divergence
            divergence_results = df.apply(ScoringEngine.calculate_momentum_divergence, axis=1, result_type='expand')
            df['short_term_momentum'] = divergence_results['short_term_momentum']
            df['long_term_momentum'] = divergence_results['long_term_momentum']
            df['momentum_divergence'] = divergence_results['momentum_divergence']
            
            # Volume acceleration
            df['volume_acceleration_score'] = df.apply(ScoringEngine.calculate_volume_acceleration, axis=1)
            
            # Volume pattern
            volume_pattern_results = df.apply(ScoringEngine.calculate_volume_pattern, axis=1, result_type='expand')
            df['volume_pattern'] = volume_pattern_results['volume_pattern']
            df['volume_pattern_score'] = volume_pattern_results['volume_pattern_score']
            
            # Position opportunity
            position_opp_results = df.apply(ScoringEngine.calculate_position_opportunity, axis=1, result_type='expand')
            df['position_opportunity'] = position_opp_results['position_opportunity']
            df['position_opportunity_score'] = position_opp_results['position_opportunity_score']
            
            # Trend quality
            trend_results = df.apply(ScoringEngine.calculate_trend_quality, axis=1, result_type='expand')
            df['ma_alignment_score'] = trend_results['ma_alignment_score']
            df['trend_spread'] = trend_results['trend_spread']
            df['trend_quality_score'] = trend_results['trend_quality_score']
            
            # EPS momentum
            eps_results = df.apply(ScoringEngine.calculate_eps_momentum, axis=1, result_type='expand')
            df['eps_momentum_score'] = eps_results['eps_momentum_score']
            df['eps_growth_rate'] = eps_results['eps_growth_rate']
            
            # Lifecycle detection
            logger.info("Detecting lifecycle stages...")
            lifecycle_results = df.apply(ScoringEngine.detect_lifecycle_stage, axis=1, result_type='expand')
            df['lifecycle_stage'] = lifecycle_results['lifecycle_stage']
            df['lifecycle_score'] = lifecycle_results['lifecycle_score']
            df['future_potential_score'] = lifecycle_results['future_potential_score']
            
            # Calculate enhanced master score
            df['enhanced_master_score'] = (
                df['momentum_acceleration_score'] * 0.20 +
                df['volume_pattern_score'] * 0.20 +
                df['position_opportunity_score'] * 0.20 +
                df['trend_quality_score'] * 0.15 +
                df['eps_momentum_score'] * 0.15 +
                df['lifecycle_score'] * 0.10
            )
            
            # Future-weighted score (prioritizes future potential)
            df['future_weighted_score'] = (
                df['future_potential_score'] * 0.40 +
                df['momentum_acceleration_score'] * 0.20 +
                df['volume_acceleration_score'] * 0.20 +
                df['eps_momentum_score'] * 0.20
            )
            
            # Add rankings
            df['rank'] = df['master_score'].rank(ascending=False, method='min').astype(int)
            df['percentile'] = df['master_score'].rank(pct=True, method='average') * 100
            df['enhanced_rank'] = df['enhanced_master_score'].rank(ascending=False, method='min').astype(int)
            df['future_rank'] = df['future_weighted_score'].rank(ascending=False, method='min').astype(int)
            
            logger.info(f"Scoring completed for {len(df)} stocks")
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating scores: {e}")
            return df

# ============================================
# DATA LOADING
# ============================================

@st.cache_data(ttl=Config.CACHE_TTL)
def load_market_data(sheet_url: str, gid: str) -> pd.DataFrame:
    """Load data from Google Sheets with caching"""
    try:
        # Construct CSV URL
        base_url = sheet_url.split('/edit')[0]
        csv_url = f"{base_url}/export?format=csv&gid={gid}"
        
        logger.info(f"Loading data from: {csv_url}")
        
        # Load data
        df = pd.read_csv(csv_url)
        
        if df.empty:
            logger.warning("Loaded empty dataframe")
            return pd.DataFrame()
            
        logger.info(f"Successfully loaded {len(df)} rows")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        st.error(f"❌ Failed to load data: {str(e)}")
        return pd.DataFrame()

# ============================================
# VISUALIZATION
# ============================================

class Visualizer:
    """Handle all visualizations"""
    
    @staticmethod
    def create_score_distribution(df: pd.DataFrame) -> go.Figure:
        """Create score distribution chart"""
        fig = go.Figure()
        
        scores = ['momentum_score', 'position_score', 'volume_score', 'quality_score']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for score, color in zip(scores, colors):
            fig.add_trace(go.Box(
                y=df[score],
                name=score.replace('_', ' ').title(),
                marker_color=color,
                boxpoints='outliers'
            ))
        
        fig.update_layout(
            title="Score Distribution Analysis",
            yaxis_title="Score (0-100)",
            showlegend=False,
            height=400
        )
        
    @staticmethod
    def create_lifecycle_distribution(df: pd.DataFrame) -> go.Figure:
        """Create lifecycle stage distribution chart"""
        if 'lifecycle_stage' not in df.columns:
            return go.Figure()
        
        # Count stocks in each stage
        stage_counts = df['lifecycle_stage'].value_counts()
        
        # Define colors for each stage
        stage_colors = {
            'ACCUMULATION': '#27AE60',      # Green
            'EARLY_MARKUP': '#3498DB',      # Blue
            'LATE_MARKUP': '#F39C12',       # Orange
            'DISTRIBUTION': '#E74C3C',      # Red
            'MARKDOWN': '#95A5A6',          # Gray
            'RECOVERY': '#9B59B6',          # Purple
            'UNKNOWN': '#BDC3C7'            # Light gray
        }
        
        colors = [stage_colors.get(stage, '#BDC3C7') for stage in stage_counts.index]
        
        fig = go.Figure(data=[
            go.Bar(
                x=stage_counts.index,
                y=stage_counts.values,
                marker_color=colors,
                text=stage_counts.values,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Lifecycle Stage Distribution",
            xaxis_title="Lifecycle Stage",
            yaxis_title="Number of Stocks",
            height=400,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_future_potential_scatter(df: pd.DataFrame) -> go.Figure:
        """Create scatter plot of current score vs future potential"""
        if 'future_potential_score' not in df.columns:
            return go.Figure()
        
        top_100 = df.nlargest(100, 'master_score')
        
        fig = px.scatter(
            top_100,
            x='master_score',
            y='future_potential_score',
            size='volume_score',
            color='lifecycle_stage',
            hover_data=['ticker', 'company_name', 'sector'],
            title='Current Score vs Future Potential',
            labels={
                'master_score': 'Current Score',
                'future_potential_score': 'Future Potential Score',
                'lifecycle_stage': 'Lifecycle Stage'
            },
            color_discrete_map={
                'ACCUMULATION': '#27AE60',
                'EARLY_MARKUP': '#3498DB',
                'LATE_MARKUP': '#F39C12',
                'DISTRIBUTION': '#E74C3C',
                'MARKDOWN': '#95A5A6',
                'RECOVERY': '#9B59B6',
                'UNKNOWN': '#BDC3C7'
            }
        )
        
        # Add diagonal line
        fig.add_trace(
            go.Scatter(
                x=[0, 100],
                y=[0, 100],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                showlegend=False
            )
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    @staticmethod
    def create_top_stocks_chart(df: pd.DataFrame, n: int = 20) -> go.Figure:
        """Create horizontal bar chart of top stocks"""
        top_df = df.nlargest(n, 'master_score')
        
        fig = go.Figure()
        
        # Add individual score components
        fig.add_trace(go.Bar(
            name='Momentum',
            y=top_df['ticker'],
            x=top_df['momentum_score'] * Config.MOMENTUM_WEIGHT,
            orientation='h',
            marker_color='#FF6B6B'
        ))
        
        fig.add_trace(go.Bar(
            name='Position',
            y=top_df['ticker'],
            x=top_df['position_score'] * Config.POSITION_WEIGHT,
            orientation='h',
            marker_color='#4ECDC4'
        ))
        
        fig.add_trace(go.Bar(
            name='Volume',
            y=top_df['ticker'],
            x=top_df['volume_score'] * Config.VOLUME_WEIGHT,
            orientation='h',
            marker_color='#45B7D1'
        ))
        
        fig.add_trace(go.Bar(
            name='Quality',
            y=top_df['ticker'],
            x=top_df['quality_score'] * Config.QUALITY_WEIGHT,
            orientation='h',
            marker_color='#96CEB4'
        ))
        
        fig.update_layout(
            title=f"Top {n} Stocks - Score Breakdown",
            xaxis_title="Weighted Score",
            barmode='stack',
            height=600,
            showlegend=True,
            legend=dict(x=0.7, y=1)
        )
        
        return fig
    
    @staticmethod
    def create_sector_performance(df: pd.DataFrame) -> go.Figure:
        """Create sector performance bubble chart"""
        sector_stats = df.groupby('sector').agg({
            'master_score': 'mean',
            'momentum_score': 'mean',
            'ticker': 'count'
        }).reset_index()
        
        sector_stats = sector_stats[sector_stats['ticker'] >= 3]  # Min 3 stocks
        
        fig = px.scatter(
            sector_stats,
            x='momentum_score',
            y='master_score',
            size='ticker',
            color='master_score',
            hover_data=['ticker'],
            text='sector',
            title='Sector Performance Analysis',
            labels={
                'momentum_score': 'Average Momentum Score',
                'master_score': 'Average Master Score',
                'ticker': 'Number of Stocks'
            },
            color_continuous_scale='Viridis'
        )
        
        fig.update_traces(textposition='top center')
        fig.update_layout(height=500)
        
        return fig

# ============================================
# REPORT GENERATION
# ============================================

def generate_excel_report(df: pd.DataFrame) -> BytesIO:
    """Generate comprehensive Excel report"""
    output = BytesIO()
    
    try:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#2C3E50',
                'font_color': 'white',
                'border': 1,
                'align': 'center'
            })
            
            number_format = workbook.add_format({'num_format': '#,##0.00'})
            percent_format = workbook.add_format({'num_format': '0.00%'})
            
            # Sheet 1: Top 100 Stocks (Classic)
            top_100 = df.nlargest(100, 'master_score')[[
                'rank', 'ticker', 'company_name', 'category', 'sector',
                'price', 'eps_tier', 'price_tier', 'pe_tier',
                'master_score', 'momentum_score', 'position_score', 
                'volume_score', 'quality_score',
                'ret_1d', 'ret_7d', 'ret_30d', 'rvol'
            ]].copy()
            
            top_100.to_excel(writer, sheet_name='Top 100 Classic', index=False)
            
            # Sheet 2: Top 100 Enhanced (if available)
            if 'enhanced_master_score' in df.columns:
                top_100_enhanced = df.nlargest(100, 'enhanced_master_score')[[
                    'enhanced_rank', 'ticker', 'company_name', 'lifecycle_stage',
                    'enhanced_master_score', 'future_potential_score',
                    'momentum_acceleration_score', 'volume_acceleration_score',
                    'volume_pattern', 'position_opportunity',
                    'eps_momentum_score', 'trend_quality_score',
                    'price', 'ret_1d', 'ret_7d', 'rvol'
                ]].copy()
                
                top_100_enhanced.to_excel(writer, sheet_name='Top 100 Enhanced', index=False)
            
            # Sheet 3: Lifecycle Analysis (if available)
            if 'lifecycle_stage' in df.columns:
                lifecycle_summary = df.groupby('lifecycle_stage').agg({
                    'ticker': 'count',
                    'master_score': 'mean',
                    'enhanced_master_score': 'mean',
                    'future_potential_score': 'mean',
                    'momentum_acceleration_score': 'mean',
                    'volume_acceleration_score': 'mean',
                    'ret_1d': 'mean',
                    'ret_7d': 'mean',
                    'ret_30d': 'mean'
                }).round(2)
                
                lifecycle_summary.to_excel(writer, sheet_name='Lifecycle Analysis')
            
            # Sheet 4: All Stocks Ranked
            all_ranked = df.sort_values('rank')[[
                'rank', 'ticker', 'company_name', 'master_score',
                'category', 'sector', 'price'
            ]].copy()
            
            if 'lifecycle_stage' in df.columns:
                all_ranked['lifecycle_stage'] = df.sort_values('rank')['lifecycle_stage']
                all_ranked['future_potential_score'] = df.sort_values('rank')['future_potential_score']
            
            all_ranked.to_excel(writer, sheet_name='All Stocks Ranked', index=False)
            
            # Sheet 5: Sector Analysis
            sector_analysis = df.groupby('sector').agg({
                'master_score': ['mean', 'std', 'count'],
                'momentum_score': 'mean',
                'volume_score': 'mean',
                'quality_score': 'mean'
            }).round(2)
            
            if 'future_potential_score' in df.columns:
                sector_future = df.groupby('sector')['future_potential_score'].mean().round(2)
                sector_analysis['future_potential'] = sector_future
            
            sector_analysis.to_excel(writer, sheet_name='Sector Analysis')
            
            # Sheet 6: Category Analysis
            category_analysis = df.groupby('category').agg({
                'master_score': ['mean', 'std', 'count'],
                'momentum_score': 'mean',
                'volume_score': 'mean',
                'quality_score': 'mean'
            }).round(2)
            
            if 'future_potential_score' in df.columns:
                category_future = df.groupby('category')['future_potential_score'].mean().round(2)
                category_analysis['future_potential'] = category_future
            
            category_analysis.to_excel(writer, sheet_name='Category Analysis')
            
            # Format all sheets
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                worksheet.freeze_panes(1, 0)
            
        output.seek(0)
        return output
        
    except Exception as e:
        logger.error(f"Error generating Excel report: {e}")
        return output

# ============================================
# CUSTOM CSS
# ============================================

def load_custom_css():
    """Load custom CSS for better UI"""
    st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 0;
    }
    
    /* Headers */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        height: 100%;
    }
    
    .metric-card h3 {
        margin: 0 0 0.5rem 0;
        color: #2C3E50;
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #3498DB;
    }
    
    /* Score badges */
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .score-high {
        background: #27AE60;
        color: white;
    }
    
    .score-medium {
        background: #F39C12;
        color: white;
    }
    
    .score-low {
        background: #E74C3C;
        color: white;
    }
    
    /* Tables */
    .dataframe {
        font-size: 0.9rem;
    }
    
    .dataframe th {
        background: #2C3E50 !important;
        color: white !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 0.05em;
    }
    
    /* Filters */
    .stSelectbox label {
        font-weight: 600;
        color: #2C3E50;
        margin-bottom: 0.5rem;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border-radius: 5px;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.2);
    }
    
    /* Info boxes */
    .stInfo {
        background: #E8F4FD;
        border-left: 4px solid #3498DB;
    }
    
    .stSuccess {
        background: #D5E8D4;
        border-left: 4px solid #27AE60;
    }
    
    .stWarning {
        background: #FFF3CD;
        border-left: 4px solid #F39C12;
    }
    
    .stError {
        background: #F8D7DA;
        border-left: 4px solid #E74C3C;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main application entry point"""
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
    
    # Load custom CSS
    load_custom_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Wave Detection 6.0</h1>
        <p>Professional Stock Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Data source
        st.markdown("### 📁 Data Source")
        sheet_url = st.text_input(
            "Google Sheets URL",
            value=Config.DEFAULT_SHEET_URL,
            help="Enter the Google Sheets URL containing stock data"
        )
        
        gid = st.text_input(
            "Sheet GID",
            value=Config.DEFAULT_GID,
            help="Enter the GID of the specific sheet"
        )
        
        # Refresh button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Data", help="Clear cache and reload data"):
                st.cache_data.clear()
                st.rerun()
        
        with col2:
            if st.button("ℹ️ Help", help="Show help information"):
                st.info("This platform analyzes stocks using a 4-pillar scoring system: "
                       "Momentum, Position, Volume, and Quality.")
        
        # Filters
        st.markdown("---")
        st.markdown("### 🔍 Smart Filters")
    
    # Load and process data
    with st.spinner("Loading market data..."):
        raw_df = load_market_data(sheet_url, gid)
    
    if raw_df.empty:
        st.error("❌ No data loaded. Please check the URL and GID.")
        st.stop()
    
    # Process data
    with st.spinner(f"Processing {len(raw_df):,} stocks..."):
        processed_df = DataProcessor.process_dataframe(raw_df)
    
    if processed_df.empty:
        st.error("❌ No valid data after processing.")
        st.stop()
    
    # Calculate scores
    with st.spinner("Calculating scores..."):
        scored_df = ScoringEngine.calculate_all_scores(processed_df)
    
    # Get unique values for filters (excluding 'Unknown')
    categories = ['All'] + sorted([c for c in scored_df['category'].unique() if c != 'Unknown'])
    sectors = ['All'] + sorted([s for s in scored_df['sector'].unique() if s != 'Unknown'])
    eps_tiers = ['All'] + sorted([e for e in scored_df['eps_tier'].unique() if e != 'Unknown'])
    price_tiers = ['All'] + sorted([p for p in scored_df['price_tier'].unique() if p != 'Unknown'])
    pe_tiers = ['All'] + sorted([p for p in scored_df['pe_tier'].unique() if p != 'Unknown'])
    
    # Sidebar filters
    with st.sidebar:
        # Category filter
        selected_categories = st.multiselect(
            "Market Cap Category",
            options=categories,
            default=['All'],
            help="Filter by market capitalization category"
        )
        
        # Sector filter
        selected_sectors = st.multiselect(
            "Sector",
            options=sectors,
            default=['All'],
            help="Filter by business sector"
        )
        
        # EPS tier filter
        selected_eps_tiers = st.multiselect(
            "EPS Tier",
            options=eps_tiers,
            default=['All'],
            help="Filter by earnings per share tier"
        )
        
        # Price tier filter
        selected_price_tiers = st.multiselect(
            "Price Tier",
            options=price_tiers,
            default=['All'],
            help="Filter by price range tier"
        )
        
        # PE tier filter
        selected_pe_tiers = st.multiselect(
            "PE Tier",
            options=pe_tiers,
            default=['All'],
            help="Filter by price-to-earnings ratio tier"
        )
        
        # Score range filter
        st.markdown("### 📊 Score Range")
        min_score = st.slider(
            "Minimum Master Score",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            help="Filter stocks by minimum master score"
        )
    
    # Apply filters
    filtered_df = scored_df.copy()
    
    if 'All' not in selected_categories:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
    
    if 'All' not in selected_sectors:
        filtered_df = filtered_df[filtered_df['sector'].isin(selected_sectors)]
    
    if 'All' not in selected_eps_tiers:
        filtered_df = filtered_df[filtered_df['eps_tier'].isin(selected_eps_tiers)]
    
    if 'All' not in selected_price_tiers:
        filtered_df = filtered_df[filtered_df['price_tier'].isin(selected_price_tiers)]
    
    if 'All' not in selected_pe_tiers:
        filtered_df = filtered_df[filtered_df['pe_tier'].isin(selected_pe_tiers)]
    
    filtered_df = filtered_df[filtered_df['master_score'] >= min_score]
    
    # Main content area
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_stocks = len(filtered_df)
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Stocks</h3>
            <div class="value">{total_stocks:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_score = filtered_df['master_score'].mean() if not filtered_df.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg Score</h3>
            <div class="value">{avg_score:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Show lifecycle distribution
        if not filtered_df.empty and 'lifecycle_stage' in filtered_df.columns:
            accumulation_count = len(filtered_df[filtered_df['lifecycle_stage'].isin(['ACCUMULATION', 'EARLY_MARKUP'])])
            pct = (accumulation_count / total_stocks * 100) if total_stocks > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>Early Stage</h3>
                <div class="value">{pct:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            top_performer = filtered_df.nlargest(1, 'master_score')['ticker'].iloc[0] if not filtered_df.empty else "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Top Stock</h3>
                <div class="value">{top_performer}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        # Show momentum acceleration
        if not filtered_df.empty and 'momentum_acceleration_score' in filtered_df.columns:
            accelerating_pct = (filtered_df['momentum_acceleration_score'] > 70).mean() * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3>Accelerating</h3>
                <div class="value">{accelerating_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            bullish_pct = (filtered_df['momentum_score'] > 50).mean() * 100 if not filtered_df.empty else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>Bullish %</h3>
                <div class="value">{bullish_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col5:
        # Show volume patterns
        if not filtered_df.empty and 'volume_pattern' in filtered_df.columns:
            expanding_vol_pct = (filtered_df['volume_pattern'] == 'EXPANDING').mean() * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3>Vol Expanding</h3>
                <div class="value">{expanding_vol_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            high_volume_pct = (filtered_df['volume_score'] > 50).mean() * 100 if not filtered_df.empty else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>High Volume %</h3>
                <div class="value">{high_volume_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Top Stocks",
        "📊 Analysis",
        "📈 Visualizations",
        "🔍 Search",
        "📥 Export"
    ])
    
    with tab1:
        st.markdown("### 🏆 Top Performing Stocks")
        
        if not filtered_df.empty:
            # Display controls
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                display_count = st.selectbox(
                    "Number of stocks to display",
                    options=[10, 20, 50, 100],
                    index=1
                )
            
            with col2:
                scoring_mode = st.selectbox(
                    "Scoring Mode",
                    options=["Classic", "Enhanced", "Future Potential"],
                    index=1,
                    help="Classic: Original scoring | Enhanced: Advanced pattern detection | Future: Lifecycle-based potential"
                )
            
            # Get top stocks based on selected scoring mode
            if scoring_mode == "Classic":
                sort_column = 'master_score'
                rank_column = 'rank'
            elif scoring_mode == "Enhanced":
                sort_column = 'enhanced_master_score'
                rank_column = 'enhanced_rank'
            else:  # Future Potential
                sort_column = 'future_weighted_score'
                rank_column = 'future_rank'
            
            top_stocks = filtered_df.nlargest(display_count, sort_column)
            
            # Create display dataframe based on mode
            if scoring_mode == "Classic":
                display_columns = [
                    rank_column, 'ticker', 'company_name', 'master_score',
                    'momentum_score', 'position_score', 'volume_score', 'quality_score',
                    'price', 'ret_1d', 'ret_7d', 'rvol',
                    'category', 'sector', 'eps_tier', 'price_tier', 'pe_tier'
                ]
            else:
                display_columns = [
                    rank_column, 'ticker', 'company_name', sort_column,
                    'lifecycle_stage', 'future_potential_score',
                    'momentum_acceleration_score', 'volume_pattern',
                    'position_opportunity', 'eps_momentum_score',
                    'price', 'ret_1d', 'ret_7d', 'rvol',
                    'category', 'sector'
                ]
            
            display_df = top_stocks[display_columns].copy()
            
            # Format numeric columns for display
            numeric_format_cols = [
                'master_score', 'enhanced_master_score', 'future_weighted_score',
                'momentum_score', 'position_score', 'volume_score', 'quality_score',
                'momentum_acceleration_score', 'volume_pattern_score',
                'position_opportunity_score', 'eps_momentum_score',
                'future_potential_score', 'lifecycle_score'
            ]
            
            for col in numeric_format_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].round(1)
            
            if 'price' in display_df.columns:
                display_df['price'] = display_df['price'].apply(lambda x: f'₹{x:,.2f}' if pd.notna(x) else '')
            if 'ret_1d' in display_df.columns:
                display_df['ret_1d'] = display_df['ret_1d'].apply(lambda x: f'{x:.2f}%' if pd.notna(x) else '')
            if 'ret_7d' in display_df.columns:
                display_df['ret_7d'] = display_df['ret_7d'].apply(lambda x: f'{x:.2f}%' if pd.notna(x) else '')
            if 'rvol' in display_df.columns:
                display_df['rvol'] = display_df['rvol'].apply(lambda x: f'{x:.2f}x' if pd.notna(x) else '')
            
            # Rename columns for better display
            display_df = display_df.rename(columns={
                'rank': 'Rank',
                'enhanced_rank': 'Rank',
                'future_rank': 'Rank',
                'ticker': 'Ticker',
                'company_name': 'Company',
                'master_score': 'Score',
                'enhanced_master_score': 'Score',
                'future_weighted_score': 'Score',
                'lifecycle_stage': 'Stage',
                'future_potential_score': 'Potential',
                'momentum_acceleration_score': 'Mom Accel',
                'volume_pattern': 'Vol Pattern',
                'position_opportunity': 'Opportunity',
                'eps_momentum_score': 'EPS Mom'
            })
            
            # Display the table with custom styling
            st.dataframe(
                display_df,
                use_container_width=True,
                height=600
            )
            
            # Enhanced insights section
            st.markdown("### 💡 Enhanced Insights")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Lifecycle distribution
                lifecycle_dist = top_stocks['lifecycle_stage'].value_counts()
                st.metric(
                    "Top Stage",
                    lifecycle_dist.index[0] if len(lifecycle_dist) > 0 else "N/A",
                    f"{lifecycle_dist.iloc[0] if len(lifecycle_dist) > 0 else 0} stocks"
                )
            
            with col2:
                # Volume patterns
                vol_patterns = top_stocks['volume_pattern'].value_counts()
                accumulating = vol_patterns.get('EXPANDING', 0) + vol_patterns.get('ACCUMULATION', 0)
                st.metric(
                    "Accumulation",
                    f"{accumulating} stocks",
                    "Volume expanding" if accumulating > 0 else ""
                )
            
            with col3:
                # Position opportunities
                opportunities = top_stocks['position_opportunity'].value_counts()
                breakouts = opportunities.get('BREAKOUT', 0)
                reversals = opportunities.get('REVERSAL', 0)
                st.metric(
                    "Key Setups",
                    f"{breakouts + reversals}",
                    f"{breakouts} breakouts, {reversals} reversals"
                )
            
            with col4:
                # Average future potential
                avg_potential = top_stocks['future_potential_score'].mean()
                high_potential = (top_stocks['future_potential_score'] > 80).sum()
                st.metric(
                    "Avg Potential",
                    f"{avg_potential:.0f}/100",
                    f"{high_potential} high potential"
                )
            
            # Show comparison if in enhanced mode
            if scoring_mode != "Classic":
                st.markdown("### 📊 Classic vs Enhanced Comparison")
                
                comparison_df = top_stocks[['ticker', 'company_name', 'master_score', 
                                          'enhanced_master_score', 'future_weighted_score',
                                          'rank', 'enhanced_rank', 'future_rank']].head(20)
                
                comparison_df['Classic Rank'] = comparison_df['rank']
                comparison_df['Enhanced Rank'] = comparison_df['enhanced_rank']
                comparison_df['Rank Change'] = comparison_df['rank'] - comparison_df['enhanced_rank']
                
                st.dataframe(
                    comparison_df[['ticker', 'company_name', 'Classic Rank', 
                                 'Enhanced Rank', 'Rank Change']],
                    use_container_width=True
                )
        
        else:
            st.warning("No stocks match the selected filters.")
    
    with tab2:
        st.markdown("### 📊 Market Analysis")
        
        if not filtered_df.empty:
            # Score distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig_dist = Visualizer.create_score_distribution(filtered_df)
                st.plotly_chart(fig_dist, use_container_width=True, key="score_dist")
            
            with col2:
                # Category breakdown
                cat_stats = filtered_df.groupby('category').agg({
                    'master_score': ['mean', 'count']
                }).round(2)
                
                st.markdown("#### Category Performance")
                st.dataframe(cat_stats, use_container_width=True)
            
            # Enhanced Analysis Section
            if 'lifecycle_stage' in filtered_df.columns:
                st.markdown("#### 🔄 Lifecycle Stage Analysis")
                
                # Create lifecycle summary
                lifecycle_summary = filtered_df.groupby('lifecycle_stage').agg({
                    'ticker': 'count',
                    'master_score': 'mean',
                    'future_potential_score': 'mean',
                    'momentum_acceleration_score': 'mean',
                    'volume_acceleration_score': 'mean'
                }).round(2)
                
                lifecycle_summary.columns = ['Count', 'Avg Score', 'Avg Potential', 'Mom Accel', 'Vol Accel']
                lifecycle_summary = lifecycle_summary.sort_values('Avg Potential', ascending=False)
                
                st.dataframe(lifecycle_summary, use_container_width=True)
                
                # Pattern Analysis
                st.markdown("#### 📊 Pattern Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Volume patterns
                    vol_pattern_dist = filtered_df['volume_pattern'].value_counts()
                    st.markdown("**Volume Patterns**")
                    for pattern, count in vol_pattern_dist.items():
                        pct = count / len(filtered_df) * 100
                        st.write(f"{pattern}: {count} ({pct:.1f}%)")
                
                with col2:
                    # Position opportunities
                    pos_opp_dist = filtered_df['position_opportunity'].value_counts()
                    st.markdown("**Position Opportunities**")
                    for opp, count in pos_opp_dist.items():
                        pct = count / len(filtered_df) * 100
                        st.write(f"{opp}: {count} ({pct:.1f}%)")
                
                with col3:
                    # Momentum acceleration
                    high_accel = (filtered_df['momentum_acceleration_score'] > 80).sum()
                    low_accel = (filtered_df['momentum_acceleration_score'] < 20).sum()
                    st.markdown("**Momentum Status**")
                    st.write(f"Accelerating: {high_accel}")
                    st.write(f"Decelerating: {low_accel}")
                    st.write(f"Neutral: {len(filtered_df) - high_accel - low_accel}")
            
            # Sector performance
            st.markdown("#### Sector Performance")
            fig_sector = Visualizer.create_sector_performance(filtered_df)
            st.plotly_chart(fig_sector, use_container_width=True, key="sector_perf")
            
            # Correlation analysis
            st.markdown("#### Score Correlations")
            
            # Include enhanced scores if available
            if 'momentum_acceleration_score' in filtered_df.columns:
                score_cols = [
                    'momentum_score', 'position_score', 'volume_score', 'quality_score',
                    'momentum_acceleration_score', 'volume_acceleration_score',
                    'eps_momentum_score', 'future_potential_score'
                ]
            else:
                score_cols = ['momentum_score', 'position_score', 'volume_score', 'quality_score']
            
            # Filter to only existing columns
            score_cols = [col for col in score_cols if col in filtered_df.columns]
            
            correlation_matrix = filtered_df[score_cols].corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            
            fig_corr.update_layout(
                title="Score Correlation Matrix",
                height=500
            )
            
            st.plotly_chart(fig_corr, use_container_width=True, key="corr_matrix")
        
        else:
            st.warning("No data available for analysis.")
    
    with tab3:
        st.markdown("### 📈 Visualizations")
        
        if not filtered_df.empty:
            # Top stocks breakdown
            fig_top = Visualizer.create_top_stocks_chart(filtered_df, 20)
            st.plotly_chart(fig_top, use_container_width=True, key="top_stocks_chart")
            
            # Scatter plot - Master Score vs Momentum
            fig_scatter = px.scatter(
                filtered_df.nlargest(100, 'master_score'),
                x='momentum_score',
                y='master_score',
                size='volume_score',
                color='quality_score',
                hover_data=['ticker', 'company_name', 'price'],
                title='Master Score vs Momentum Score (Top 100)',
                labels={
                    'momentum_score': 'Momentum Score',
                    'master_score': 'Master Score',
                    'volume_score': 'Volume Score',
                    'quality_score': 'Quality Score'
                },
                color_continuous_scale='Viridis'
            )
            
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_plot")
            
            # Lifecycle analysis (if available)
            if 'lifecycle_stage' in filtered_df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Lifecycle distribution
                    fig_lifecycle = Visualizer.create_lifecycle_distribution(filtered_df)
                    st.plotly_chart(fig_lifecycle, use_container_width=True, key="lifecycle_dist")
                
                with col2:
                    # Future potential scatter
                    fig_potential = Visualizer.create_future_potential_scatter(filtered_df)
                    st.plotly_chart(fig_potential, use_container_width=True, key="future_potential")
            
            # Tier distribution
            col1, col2 = st.columns(2)
            
            with col1:
                eps_dist = filtered_df['eps_tier'].value_counts()
                fig_eps = px.pie(
                    values=eps_dist.values,
                    names=eps_dist.index,
                    title='EPS Tier Distribution',
                    hole=0.4
                )
                st.plotly_chart(fig_eps, use_container_width=True, key="eps_dist")
            
            with col2:
                pe_dist = filtered_df['pe_tier'].value_counts()
                fig_pe = px.pie(
                    values=pe_dist.values,
                    names=pe_dist.index,
                    title='PE Tier Distribution',
                    hole=0.4
                )
                st.plotly_chart(fig_pe, use_container_width=True, key="pe_dist")
        
        else:
            st.warning("No data available for visualization.")
    
    with tab4:
        st.markdown("### 🔍 Stock Search")
        
        # Search box
        search_term = st.text_input(
            "Search by ticker or company name",
            placeholder="Enter ticker symbol or company name..."
        )
        
        if search_term:
            # Search in both ticker and company name
            search_results = filtered_df[
                (filtered_df['ticker'].str.contains(search_term, case=False)) |
                (filtered_df['company_name'].str.contains(search_term, case=False))
            ]
            
            if not search_results.empty:
                st.success(f"Found {len(search_results)} matching stocks")
                
                # Display search results
                for _, stock in search_results.iterrows():
                    with st.expander(f"{stock['ticker']} - {stock['company_name']}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Master Score", f"{stock['master_score']:.1f}")
                            st.metric("Price", f"₹{stock['price']:,.2f}")
                            st.metric("Category", stock['category'])
                        
                        with col2:
                            st.metric("Momentum", f"{stock['momentum_score']:.1f}")
                            st.metric("Position", f"{stock['position_score']:.1f}")
                            st.metric("Sector", stock['sector'])
                        
                        with col3:
                            st.metric("Volume", f"{stock['volume_score']:.1f}")
                            st.metric("Quality", f"{stock['quality_score']:.1f}")
                            st.metric("Rank", f"#{stock['rank']}")
                        
                        # Additional details
                        st.markdown("#### Recent Performance")
                        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                        
                        with perf_col1:
                            st.metric("1 Day", f"{stock['ret_1d']:.2f}%")
                        with perf_col2:
                            st.metric("7 Days", f"{stock['ret_7d']:.2f}%")
                        with perf_col3:
                            st.metric("30 Days", f"{stock['ret_30d']:.2f}%")
                        with perf_col4:
                            st.metric("RVOL", f"{stock['rvol']:.2f}x")
            else:
                st.warning(f"No stocks found matching '{search_term}'")
    
    with tab5:
        st.markdown("### 📥 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Excel Report")
            st.markdown("Download comprehensive Excel report with multiple sheets:")
            st.markdown("- Top 100 stocks with full metrics")
            st.markdown("- All stocks ranked")
            st.markdown("- Sector analysis")
            st.markdown("- Category analysis")
            
            if st.button("📊 Generate Excel Report", key="excel_btn"):
                with st.spinner("Generating Excel report..."):
                    excel_file = generate_excel_report(filtered_df)
                    
                    st.download_button(
                        label="📥 Download Excel Report",
                        data=excel_file,
                        file_name=f"wave_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        with col2:
            st.markdown("#### CSV Export")
            st.markdown("Download filtered data as CSV for further analysis")
            
            if st.button("📄 Generate CSV", key="csv_btn"):
                csv_data = filtered_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"wave_detection_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # Summary statistics
        st.markdown("---")
        st.markdown("#### 📊 Summary Statistics")
        
        summary_stats = pd.DataFrame({
            'Metric': ['Total Stocks', 'Average Master Score', 'Stocks Above 70', 
                      'Stocks Above 50', 'Highest Score', 'Lowest Score'],
            'Value': [
                len(filtered_df),
                f"{filtered_df['master_score'].mean():.2f}",
                f"{(filtered_df['master_score'] > 70).sum()} ({(filtered_df['master_score'] > 70).mean() * 100:.1f}%)",
                f"{(filtered_df['master_score'] > 50).sum()} ({(filtered_df['master_score'] > 50).mean() * 100:.1f}%)",
                f"{filtered_df['master_score'].max():.2f}",
                f"{filtered_df['master_score'].min():.2f}"
            ]
        })
        
        st.table(summary_stats)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7F8C8D; padding: 2rem;">
        <p>Wave Detection 6.0 - Professional Stock Analytics Platform</p>
        <p>Data refreshes every 5 minutes | Last update: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"An error occurred: {str(e)}")
        st.error("Please refresh the page or contact support if the issue persists.")
