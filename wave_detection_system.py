"""
Wave Detection Ultimate - Professional Stock Ranking System
=========================================================
Complete implementation with all filters and robust error handling.

Version: 3.0.0 - Production Ready
Author: Professional Implementation
License: MIT
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
import time
from io import BytesIO
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================
# LOGGING CONFIGURATION
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================

@dataclass(frozen=True)
class Config:
    """System configuration"""
    
    # Data source
    DEFAULT_SHEET_URL: str = "https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk/edit?usp=sharing"
    DEFAULT_GID: str = "2026492216"
    
    # Cache settings
    CACHE_TTL: int = 300  # 5 minutes
    
    # ENHANCED Ranking weights - Master Score 2.0
    POSITION_WEIGHT: float = 0.35  # Reduced from 0.45
    VOLUME_WEIGHT: float = 0.30    # Reduced from 0.35  
    MOMENTUM_WEIGHT: float = 0.15  # Reduced from 0.20
    ACCELERATION_WEIGHT: float = 0.10  # NEW!
    BREAKOUT_WEIGHT: float = 0.10      # NEW!
    
    # Display settings
    DEFAULT_TOP_N: int = 50
    AVAILABLE_TOP_N: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200, 500])
    
    # Tier definitions
    EPS_TIERS: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "Loss": (-float('inf'), 0),
        "0-5": (0, 5),
        "5-10": (5, 10),
        "10-20": (10, 20),
        "20-50": (20, 50),
        "50-100": (50, 100),
        "100+": (100, float('inf'))
    })
    
    PE_TIERS: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "Negative/NA": (-float('inf'), 0),
        "0-10": (0, 10),
        "10-15": (10, 15),
        "15-20": (15, 20),
        "20-30": (20, 30),
        "30-50": (30, 50),
        "50+": (50, float('inf'))
    })
    
    PRICE_TIERS: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "0-100": (0, 100),
        "100-250": (100, 250),
        "250-500": (250, 500),
        "500-1000": (500, 1000),
        "1000-2500": (1000, 2500),
        "2500-5000": (2500, 5000),
        "5000+": (5000, float('inf'))
    })

# Global configuration
CONFIG = Config()

# ============================================
# PERFORMANCE MONITORING
# ============================================

def timer(func):
    """Performance timing decorator"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        if elapsed > 1.0:
            logger.warning(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper

# ============================================
# DATA LOADING
# ============================================

@st.cache_data(ttl=CONFIG.CACHE_TTL)
def load_google_sheets_data(sheet_url: str, gid: str) -> pd.DataFrame:
    """
    Load data from Google Sheets with proper caching
    
    Args:
        sheet_url: Google Sheets URL
        gid: Sheet ID
        
    Returns:
        Raw dataframe
    """
    try:
        # Construct CSV URL
        base_url = sheet_url.split('/edit')[0]
        csv_url = f"{base_url}/export?format=csv&gid={gid}"
        
        logger.info(f"Loading data from Google Sheets")
        
        # Load data
        df = pd.read_csv(csv_url)
        
        if df.empty:
            raise ValueError("Loaded empty dataframe")
        
        logger.info(f"Successfully loaded {len(df):,} rows")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise

# ============================================
# DATA PROCESSING
# ============================================

class DataProcessor:
    """Handle all data processing operations"""
    
    @staticmethod
    def clean_indian_number(value: Any) -> Optional[float]:
        """Clean Indian number format"""
        if pd.isna(value) or value == '':
            return np.nan
        
        try:
            cleaned = str(value)
            # Remove currency and special characters
            for char in ['₹', '$', '%', ',']:
                cleaned = cleaned.replace(char, '')
            cleaned = cleaned.strip()
            
            # Handle special cases
            if cleaned in ['', '-', 'N/A', 'n/a', '#N/A', 'nan', 'None']:
                return np.nan
            
            return float(cleaned)
        except:
            return np.nan
    
    @staticmethod
    @timer
    def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Complete data processing pipeline"""
        if df.empty:
            return pd.DataFrame()
        
        # Create copy
        df = df.copy()
        
        # Clean numeric columns
        numeric_columns = [
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
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(DataProcessor.clean_indian_number)
        
        # Clean categorical columns
        categorical_columns = ['ticker', 'company_name', 'category', 'sector']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(['nan', 'None', '', 'N/A'], 'Unknown')
        
        # Fix volume ratios (convert from % to multiplier)
        volume_ratio_columns = [
            'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
            'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
            'vol_ratio_90d_180d'
        ]
        
        for col in volume_ratio_columns:
            if col in df.columns:
                # Convert percentage to multiplier, handling NaN values
                df[col] = df[col].apply(lambda x: (100 + x) / 100 if pd.notna(x) else np.nan)
        
        # Remove invalid rows
        initial_count = len(df)
        df = df[df['price'].notna() & (df['price'] > 0)]
        df = df[df['from_low_pct'].notna()]
        df = df[df['from_high_pct'].notna()]
        
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} invalid rows")
        
        # Add tier classifications
        df = DataProcessor.add_tiers(df)
        
        return df
    
    @staticmethod
    def add_tiers(df: pd.DataFrame) -> pd.DataFrame:
        """Add tier classifications"""
        
        # EPS Tier
        def get_eps_tier(eps):
            if pd.isna(eps):
                return "Unknown"
            for tier_name, (min_val, max_val) in CONFIG.EPS_TIERS.items():
                if min_val < eps <= max_val:
                    return tier_name
            return "Unknown"
        
        # PE Tier
        def get_pe_tier(pe):
            if pd.isna(pe) or pe <= 0:
                return "Negative/NA"
            for tier_name, (min_val, max_val) in CONFIG.PE_TIERS.items():
                if tier_name != "Negative/NA" and min_val < pe <= max_val:
                    return tier_name
            return "Unknown"
        
        # Price Tier
        def get_price_tier(price):
            if pd.isna(price):
                return "Unknown"
            for tier_name, (min_val, max_val) in CONFIG.PRICE_TIERS.items():
                if min_val < price <= max_val:
                    return tier_name
            return "Unknown"
        
        df['eps_tier'] = df['eps_current'].apply(get_eps_tier)
        df['pe_tier'] = df['pe'].apply(get_pe_tier)
        df['price_tier'] = df['price'].apply(get_price_tier)
        
        return df

# ============================================
# RANKING ENGINE
# ============================================

class RankingEngine:
    """Core ranking calculations"""
    
    @staticmethod
    def safe_rank(series: pd.Series, pct: bool = False, ascending: bool = True) -> pd.Series:
        """Safely rank a series handling NaN values"""
        # Replace inf values with NaN
        series = series.replace([np.inf, -np.inf], np.nan)
        
        # Perform ranking
        if pct:
            return series.rank(pct=True, ascending=ascending, na_option='bottom')
        else:
            return series.rank(ascending=ascending, method='min', na_option='bottom')
    
    @staticmethod
    def calculate_advanced_volume_score(df: pd.DataFrame) -> pd.Series:
        """Calculate advanced volume score using ALL 7 volume ratios"""
        # Fill NaN values with 1.0 (no change) for all volume ratios
        volume_cols = ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
                      'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
                      'vol_ratio_90d_180d']
        
        for col in volume_cols:
            if col in df.columns:
                df[col] = df[col].fillna(1.0)
        
        # Rank all volume ratios
        df['rank_vol_1d_90d'] = RankingEngine.safe_rank(df['vol_ratio_1d_90d'], pct=True) * 100
        df['rank_vol_7d_90d'] = RankingEngine.safe_rank(df['vol_ratio_7d_90d'], pct=True) * 100
        df['rank_vol_30d_90d'] = RankingEngine.safe_rank(df['vol_ratio_30d_90d'], pct=True) * 100
        df['rank_vol_30d_180d'] = RankingEngine.safe_rank(df['vol_ratio_30d_180d'], pct=True) * 100
        df['rank_vol_90d_180d'] = RankingEngine.safe_rank(df['vol_ratio_90d_180d'], pct=True) * 100
        
        # Short-term explosion (20%)
        short_term = df['rank_vol_1d_90d'] * 0.2
        
        # Medium-term accumulation (40%)
        medium_term = (
            df['rank_vol_7d_90d'] * 0.2 +
            df['rank_vol_30d_90d'] * 0.2
        )
        
        # Long-term institutional (40%) - USING THE MISSING RATIOS!
        long_term = (
            df['rank_vol_30d_180d'] * 0.15 +
            df['rank_vol_90d_180d'] * 0.25  # Most important for institutional!
        )
        
        return short_term + medium_term + long_term
    
    @staticmethod
    def calculate_momentum_acceleration(df: pd.DataFrame) -> pd.Series:
        """Calculate if momentum is accelerating or decelerating"""
        # Ensure we have the data
        df['ret_1d'] = df['ret_1d'].fillna(0.0)
        df['ret_7d'] = df['ret_7d'].fillna(0.0)
        df['ret_30d'] = df['ret_30d'].fillna(0.0)
        
        # Calculate daily averages
        daily_avg_7d = df['ret_7d'] / 7
        daily_avg_30d = df['ret_30d'] / 30
        
        # Acceleration metrics
        daily_vs_weekly = df['ret_1d'] - daily_avg_7d
        weekly_vs_monthly = daily_avg_7d - daily_avg_30d
        
        # Create acceleration score
        acceleration_score = pd.Series(index=df.index, dtype=float)
        
        # Perfect acceleration: today > week > month
        perfect_accel = (df['ret_1d'] > daily_avg_7d) & (daily_avg_7d > daily_avg_30d)
        acceleration_score[perfect_accel] = 100
        
        # Good acceleration: today > week
        good_accel = (~perfect_accel) & (df['ret_1d'] > daily_avg_7d)
        acceleration_score[good_accel] = 70
        
        # Neutral
        neutral = (~perfect_accel) & (~good_accel) & (df['ret_1d'] > 0)
        acceleration_score[neutral] = 50
        
        # Deceleration
        acceleration_score[acceleration_score.isna()] = 30
        
        return acceleration_score
    
    @staticmethod
    def calculate_breakout_probability(df: pd.DataFrame) -> pd.Series:
        """Calculate probability of breakout based on multiple factors"""
        # Distance from high (closer = higher probability)
        distance_score = (100 + df['from_high_pct']) # -20% becomes 80
        distance_score = distance_score.clip(0, 100)
        
        # Volume pressure (average of 7d and 30d ratios)
        volume_pressure = (df['vol_ratio_7d_90d'] + df['vol_ratio_30d_90d']) / 2
        volume_pressure = (volume_pressure - 1) * 100  # Convert to percentage above normal
        volume_pressure = volume_pressure.clip(0, 100)
        
        # Trend support (how many SMAs is price above)
        trend_support = pd.Series(0, index=df.index)
        if 'sma_20d' in df.columns:
            trend_support += (df['price'] > df['sma_20d']).astype(int) * 33.33
        if 'sma_50d' in df.columns:
            trend_support += (df['price'] > df['sma_50d']).astype(int) * 33.33
        if 'sma_200d' in df.columns:
            trend_support += (df['price'] > df['sma_200d']).astype(int) * 33.34
        
        # Combined breakout probability
        breakout_prob = (
            distance_score * 0.4 +
            volume_pressure * 0.4 +
            trend_support * 0.2
        )
        
        return breakout_prob.clip(0, 100)
    
    @staticmethod
    def calculate_trend_quality(df: pd.DataFrame) -> pd.Series:
        """Calculate trend quality based on SMA alignment"""
        trend_score = pd.Series(0, index=df.index)
        
        # Check if we have SMA columns
        if all(col in df.columns for col in ['sma_20d', 'sma_50d', 'sma_200d']):
            # Price above each SMA
            trend_score += (df['price'] > df['sma_20d']).astype(int) * 25
            trend_score += (df['price'] > df['sma_50d']).astype(int) * 25
            trend_score += (df['price'] > df['sma_200d']).astype(int) * 25
            
            # Perfect alignment bonus
            perfect_alignment = (
                (df['price'] > df['sma_20d']) & 
                (df['sma_20d'] > df['sma_50d']) & 
                (df['sma_50d'] > df['sma_200d'])
            )
            trend_score[perfect_alignment] = 100
        else:
            # If no SMA data, use a neutral score
            trend_score = pd.Series(50, index=df.index)
        
        return trend_score
    
    @staticmethod
    @timer
    def calculate_rankings(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rankings using ENHANCED scoring system"""
        if df.empty:
            return df
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Calculate basic percentile ranks for position metrics
        df['rank_from_low'] = RankingEngine.safe_rank(df['from_low_pct'], pct=True) * 100
        df['rank_from_high'] = RankingEngine.safe_rank(100 + df['from_high_pct'], pct=True) * 100
        
        # Fill NaN values for returns
        df['ret_1d'] = df['ret_1d'].fillna(0.0)
        df['ret_7d'] = df['ret_7d'].fillna(0.0)
        df['ret_30d'] = df['ret_30d'].fillna(0.0)
        
        # Basic momentum ranks
        df['rank_ret_30d'] = RankingEngine.safe_rank(df['ret_30d'], pct=True) * 100
        
        # ENHANCED SCORING COMPONENTS
        
        # 1. Position Score (35%) - slightly reduced from 45%
        df['position_score'] = (
            df['rank_from_low'].fillna(50) * 0.6 +
            df['rank_from_high'].fillna(50) * 0.4
        )
        
        # 2. Advanced Volume Score (30%) - using ALL ratios
        df['volume_score'] = RankingEngine.calculate_advanced_volume_score(df)
        
        # 3. Momentum Score (15%) - reduced from 20%
        df['momentum_score'] = df['rank_ret_30d'].fillna(50)
        
        # 4. Momentum Acceleration (10%) - NEW!
        df['acceleration_score'] = RankingEngine.calculate_momentum_acceleration(df)
        
        # 5. Breakout Probability (10%) - NEW!
        df['breakout_score'] = RankingEngine.calculate_breakout_probability(df)
        
        # MASTER SCORE 2.0 - Enhanced formula
        df['master_score'] = (
            df['position_score'] * 0.35 +
            df['volume_score'] * 0.30 +
            df['momentum_score'] * 0.15 +
            df['acceleration_score'] * 0.10 +
            df['breakout_score'] * 0.10
        )
        
        # Add trend quality as a bonus (not part of main score)
        df['trend_quality'] = RankingEngine.calculate_trend_quality(df)
        
        # Final ranking
        df['rank'] = RankingEngine.safe_rank(df['master_score'], ascending=False)
        df['rank'] = df['rank'].fillna(9999).astype(int)
        
        df['percentile'] = RankingEngine.safe_rank(df['master_score'], pct=True) * 100
        df['percentile'] = df['percentile'].fillna(0)
        
        # Enhanced pattern detection
        df = RankingEngine.detect_smart_patterns(df)
        
        return df
    
    @staticmethod
    def detect_smart_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Detect advanced trading patterns using all available data"""
        patterns = []
        
        for idx, row in df.iterrows():
            stock_patterns = []
            
            # Get values with safe defaults
            from_high_pct = row.get('from_high_pct', -100)
            from_low_pct = row.get('from_low_pct', 0)
            vol_ratio_1d_90d = row.get('vol_ratio_1d_90d', 1.0)
            vol_ratio_7d_90d = row.get('vol_ratio_7d_90d', 1.0)
            vol_ratio_30d_90d = row.get('vol_ratio_30d_90d', 1.0)
            vol_ratio_90d_180d = row.get('vol_ratio_90d_180d', 1.0)
            ret_1d = row.get('ret_1d', 0)
            ret_7d = row.get('ret_7d', 0)
            ret_30d = row.get('ret_30d', 0)
            price = row.get('price', 0)
            sma_50d = row.get('sma_50d', price)
            percentile = row.get('percentile', 0)
            acceleration_score = row.get('acceleration_score', 50)
            breakout_score = row.get('breakout_score', 50)
            
            # INSTITUTIONAL ACCUMULATION
            if (vol_ratio_90d_180d > 1.1 and 
                from_low_pct < 60 and
                ret_30d > -5):
                stock_patterns.append("🏦 INSTITUTIONAL")
            
            # MOMENTUM EXPLOSION
            if acceleration_score > 90:
                stock_patterns.append("🚀 EXPLOSION")
            
            # VOLUME CLIMAX
            if (vol_ratio_1d_90d > 3 and
                abs(ret_1d) > 5):
                stock_patterns.append("⚡ CLIMAX")
            
            # BREAKOUT SETUP
            if breakout_score > 80:
                stock_patterns.append("🎯 BREAKOUT")
            
            # ACCUMULATION PATTERN
            if (vol_ratio_30d_90d > 1.2 and
                vol_ratio_90d_180d > 1.05 and
                from_low_pct < 50):
                stock_patterns.append("📈 ACCUMULATING")
            
            # LEADER
            if percentile > 95:
                stock_patterns.append("👑 LEADER")
            
            # STEALTH STRENGTH
            if (vol_ratio_90d_180d > 1.15 and
                ret_30d > 10 and
                vol_ratio_1d_90d < 1.5):
                stock_patterns.append("💎 STEALTH")
            
            patterns.append(" | ".join(stock_patterns) if stock_patterns else "")
        
        df['patterns'] = patterns
        return df

# ============================================
# FILTER ENGINE
# ============================================

class FilterEngine:
    """Handle all filtering operations"""
    
    @staticmethod
    def get_unique_values(df: pd.DataFrame, column: str) -> List[str]:
        """Get unique values for a column, excluding 'Unknown'"""
        if column not in df.columns:
            return []
        
        values = df[column].unique().tolist()
        # Remove 'Unknown' and sort
        values = [v for v in values if v != 'Unknown' and pd.notna(v)]
        return sorted(values)
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply all filters to dataframe"""
        filtered_df = df.copy()
        
        # Category filter
        if filters.get('categories') and 'All' not in filters['categories']:
            filtered_df = filtered_df[filtered_df['category'].isin(filters['categories'])]
        
        # Sector filter
        if filters.get('sectors') and 'All' not in filters['sectors']:
            filtered_df = filtered_df[filtered_df['sector'].isin(filters['sectors'])]
        
        # EPS tier filter
        if filters.get('eps_tiers') and 'All' not in filters['eps_tiers']:
            filtered_df = filtered_df[filtered_df['eps_tier'].isin(filters['eps_tiers'])]
        
        # PE tier filter
        if filters.get('pe_tiers') and 'All' not in filters['pe_tiers']:
            filtered_df = filtered_df[filtered_df['pe_tier'].isin(filters['pe_tiers'])]
        
        # Price tier filter
        if filters.get('price_tiers') and 'All' not in filters['price_tiers']:
            filtered_df = filtered_df[filtered_df['price_tier'].isin(filters['price_tiers'])]
        
        # Score filter
        if filters.get('min_score', 0) > 0:
            filtered_df = filtered_df[filtered_df['master_score'] >= filters['min_score']]
        
        # EPS change filter
        if filters.get('min_eps_change') is not None:
            filtered_df = filtered_df[
                (filtered_df['eps_change_pct'] >= filters['min_eps_change']) | 
                (filtered_df['eps_change_pct'].isna())
            ]
        
        return filtered_df

# ============================================
# VISUALIZATION ENGINE
# ============================================

class Visualizer:
    """Create all visualizations"""
    
    @staticmethod
    def create_score_distribution(df: pd.DataFrame) -> go.Figure:
        """Score distribution chart"""
        fig = go.Figure()
        
        scores = ['position_score', 'volume_score', 'momentum_score']
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for score, color in zip(scores, colors):
            if score in df.columns:
                # Filter out NaN values for box plot
                score_data = df[score].dropna()
                if len(score_data) > 0:
                    fig.add_trace(go.Box(
                        y=score_data,
                        name=score.replace('_', ' ').title(),
                        marker_color=color,
                        boxpoints='outliers'
                    ))
        
        fig.update_layout(
            title="Score Component Distribution",
            yaxis_title="Score (0-100)",
            template='plotly_white',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_top_stocks_chart(df: pd.DataFrame, n: int = 20) -> go.Figure:
        """Top stocks breakdown with enhanced scores"""
        # Get top stocks with valid scores
        valid_df = df[df['master_score'].notna()]
        top_df = valid_df.nlargest(min(n, len(valid_df)), 'master_score')
        
        if len(top_df) == 0:
            return go.Figure()
        
        fig = go.Figure()
        
        # Calculate weighted contributions for display
        fig.add_trace(go.Bar(
            name='Position',
            y=top_df['ticker'],
            x=top_df['position_score'] * 0.35,  # 35% weight
            orientation='h',
            marker_color='#3498db',
            text=[f"{x:.1f}" for x in top_df['position_score']],
            textposition='inside'
        ))
        
        fig.add_trace(go.Bar(
            name='Volume',
            y=top_df['ticker'],
            x=top_df['volume_score'] * 0.30,  # 30% weight
            orientation='h',
            marker_color='#e74c3c',
            text=[f"{x:.1f}" for x in top_df['volume_score']],
            textposition='inside'
        ))
        
        fig.add_trace(go.Bar(
            name='Momentum',
            y=top_df['ticker'],
            x=top_df['momentum_score'] * 0.15,  # 15% weight
            orientation='h',
            marker_color='#2ecc71',
            text=[f"{x:.1f}" for x in top_df['momentum_score']],
            textposition='inside'
        ))
        
        # Add new components if they exist
        if 'acceleration_score' in top_df.columns:
            fig.add_trace(go.Bar(
                name='Acceleration',
                y=top_df['ticker'],
                x=top_df['acceleration_score'] * 0.10,  # 10% weight
                orientation='h',
                marker_color='#f39c12',
                text=[f"{x:.1f}" for x in top_df['acceleration_score']],
                textposition='inside'
            ))
        
        if 'breakout_score' in top_df.columns:
            fig.add_trace(go.Bar(
                name='Breakout',
                y=top_df['ticker'],
                x=top_df['breakout_score'] * 0.10,  # 10% weight
                orientation='h',
                marker_color='#9b59b6',
                text=[f"{x:.1f}" for x in top_df['breakout_score']],
                textposition='inside'
            ))
        
        fig.update_layout(
            title=f"Top {len(top_df)} Stocks - Enhanced Score Breakdown",
            xaxis_title="Weighted Score Contribution",
            barmode='stack',
            template='plotly_white',
            height=max(400, len(top_df) * 30),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    @staticmethod
    def create_sector_performance(df: pd.DataFrame) -> go.Figure:
        """Sector performance analysis"""
        # Group by sector with valid scores
        valid_df = df[df['master_score'].notna()]
        
        sector_stats = valid_df.groupby('sector').agg({
            'master_score': 'mean',
            'position_score': 'mean',
            'ticker': 'count'
        }).reset_index()
        
        # Filter sectors with at least 3 stocks
        sector_stats = sector_stats[sector_stats['ticker'] >= 3]
        
        if len(sector_stats) == 0:
            return go.Figure()
        
        fig = px.scatter(
            sector_stats,
            x='position_score',
            y='master_score',
            size='ticker',
            color='master_score',
            hover_data=['ticker'],
            text='sector',
            title='Sector Performance Analysis',
            labels={
                'position_score': 'Average Position Score',
                'master_score': 'Average Master Score',
                'ticker': 'Number of Stocks'
            },
            color_continuous_scale='Viridis',
            template='plotly_white'
        )
        
        fig.update_traces(textposition='top center')
        fig.update_layout(height=500)
        
        return fig

# ============================================
# EXPORT ENGINE
# ============================================

class ExportEngine:
    """Handle data exports"""
    
    @staticmethod
    def create_excel_report(df: pd.DataFrame) -> BytesIO:
        """Create Excel report"""
        output = BytesIO()
        
        try:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Top 100 stocks
                valid_df = df[df['master_score'].notna()]
                top_100 = valid_df.nlargest(min(100, len(valid_df)), 'master_score')
                
                export_cols = [
                    'rank', 'ticker', 'company_name', 'master_score',
                    'position_score', 'volume_score', 'momentum_score',
                    'acceleration_score', 'breakout_score', 'trend_quality',
                    'price', 'from_low_pct', 'from_high_pct',
                    'ret_30d', 'vol_ratio_30d_90d', 'vol_ratio_90d_180d',
                    'patterns', 'category', 'sector', 'eps_tier', 'pe_tier', 'price_tier'
                ]
                export_cols = [col for col in export_cols if col in top_100.columns]
                top_100[export_cols].to_excel(writer, sheet_name='Top 100', index=False)
                
                # All stocks summary
                summary_cols = ['rank', 'ticker', 'company_name', 'master_score', 'category', 'sector']
                summary_cols = [col for col in summary_cols if col in df.columns]
                df[summary_cols].to_excel(writer, sheet_name='All Stocks', index=False)
                
                # Sector analysis
                sector_stats = valid_df.groupby('sector').agg({
                    'master_score': ['mean', 'std', 'count'],
                    'position_score': 'mean',
                    'volume_score': 'mean',
                    'momentum_score': 'mean'
                }).round(2)
                sector_stats.to_excel(writer, sheet_name='Sector Analysis')
                
                # Category analysis
                category_stats = valid_df.groupby('category').agg({
                    'master_score': ['mean', 'std', 'count'],
                    'position_score': 'mean',
                    'volume_score': 'mean',
                    'momentum_score': 'mean'
                }).round(2)
                category_stats.to_excel(writer, sheet_name='Category Analysis')
        
        except Exception as e:
            logger.error(f"Error creating Excel report: {e}")
            
        output.seek(0)
        return output

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main application"""
    
    # Page config
    st.set_page_config(
        page_title="Wave Detection Ultimate",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {height: 50px;}
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 5% 5% 5% 10%;
        border-radius: 5px;
        overflow-wrap: break-word;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; background: linear-gradient(90deg, #3498db, #2ecc71); color: white; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="margin: 0;">📊 Wave Detection Ultimate</h1>
        <p style="margin: 0;">Professional Stock Ranking System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Data Source")
        
        sheet_url = st.text_input(
            "Google Sheets URL",
            value=CONFIG.DEFAULT_SHEET_URL,
            help="Enter the Google Sheets URL"
        )
        
        gid = st.text_input(
            "Sheet ID (GID)",
            value=CONFIG.DEFAULT_GID,
            help="Enter the sheet ID"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", help="Clear cache and reload"):
                st.cache_data.clear()
                st.rerun()
        
        with col2:
            if st.button("ℹ️ Help"):
                st.info("""
                **Enhanced Master Score 2.0:**
                - Position (35%): Distance from 52w low/high
                - Volume (30%): ALL 7 volume ratios analyzed
                - Momentum (15%): 30-day returns
                - Acceleration (10%): Momentum building/fading
                - Breakout (10%): Probability of breakout
                """)
        
        st.markdown("---")
        st.markdown("### 🔍 Filters")
    
    # Load and process data
    try:
        with st.spinner("Loading data..."):
            raw_df = load_google_sheets_data(sheet_url, gid)
        
        with st.spinner(f"Processing {len(raw_df):,} stocks..."):
            processed_df = DataProcessor.process_dataframe(raw_df)
        
        with st.spinner("Calculating rankings..."):
            ranked_df = RankingEngine.calculate_rankings(processed_df)
        
        if ranked_df.empty:
            st.error("No valid data after processing.")
            st.stop()
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()
    
    # Get filter options
    filter_engine = FilterEngine()
    
    # Sidebar filters
    with st.sidebar:
        filters = {}
        
        # Category filter
        categories = ['All'] + filter_engine.get_unique_values(ranked_df, 'category')
        filters['categories'] = st.multiselect(
            "Category",
            options=categories,
            default=['All'],
            help="Filter by market cap category"
        )
        
        # Sector filter
        sectors = ['All'] + filter_engine.get_unique_values(ranked_df, 'sector')
        filters['sectors'] = st.multiselect(
            "Sector",
            options=sectors,
            default=['All'],
            help="Filter by sector"
        )
        
        # EPS tier filter
        eps_tiers = ['All'] + filter_engine.get_unique_values(ranked_df, 'eps_tier')
        filters['eps_tiers'] = st.multiselect(
            "EPS Tier",
            options=eps_tiers,
            default=['All'],
            help="Filter by EPS tier"
        )
        
        # PE tier filter
        pe_tiers = ['All'] + filter_engine.get_unique_values(ranked_df, 'pe_tier')
        filters['pe_tiers'] = st.multiselect(
            "PE Tier",
            options=pe_tiers,
            default=['All'],
            help="Filter by PE ratio tier"
        )
        
        # Price tier filter
        price_tiers = ['All'] + filter_engine.get_unique_values(ranked_df, 'price_tier')
        filters['price_tiers'] = st.multiselect(
            "Price Tier",
            options=price_tiers,
            default=['All'],
            help="Filter by price range"
        )
        
        # Score filter
        filters['min_score'] = st.slider(
            "Minimum Score",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            help="Filter by minimum master score"
        )
        
        # EPS change filter
        if 'eps_change_pct' in ranked_df.columns:
            filters['min_eps_change'] = st.number_input(
                "Min EPS Change %",
                min_value=-100.0,
                max_value=1000.0,
                value=0.0,
                step=10.0,
                help="Filter by minimum EPS change percentage"
            )
    
    # Apply filters
    filtered_df = filter_engine.apply_filters(ranked_df, filters)
    
    # Sort by rank
    filtered_df = filtered_df.sort_values('rank')
    
    # Main content
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Stocks", f"{len(filtered_df):,}")
    
    with col2:
        valid_scores = filtered_df['master_score'].dropna()
        avg_score = valid_scores.mean() if len(valid_scores) > 0 else 0
        st.metric("Avg Score", f"{avg_score:.1f}")
    
    with col3:
        accelerating = ((filtered_df['acceleration_score'] > 80).sum() if 'acceleration_score' in filtered_df.columns and len(filtered_df) > 0 else 0)
        st.metric("Accelerating", f"{accelerating}")
    
    with col4:
        ready_breakout = ((filtered_df['breakout_score'] > 80).sum() if 'breakout_score' in filtered_df.columns and len(filtered_df) > 0 else 0)
        st.metric("Breakout Ready", f"{ready_breakout}")
    
    with col5:
        high_volume = (filtered_df['vol_ratio_30d_90d'] > 1.5).sum() if len(filtered_df) > 0 else 0
        st.metric("High Volume", f"{high_volume}")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Rankings", "📊 Analysis", "📈 Visualizations", "📥 Export"
    ])
    
    with tab1:
        st.markdown("### Top Ranked Stocks")
        
        # Display options
        col1, col2 = st.columns([1, 3])
        with col1:
            display_count = st.selectbox(
                "Show top",
                options=CONFIG.AVAILABLE_TOP_N,
                index=2
            )
        
        # Get top stocks
        display_df = filtered_df.head(display_count).copy()
        
        if len(display_df) > 0:
            # Format for display
            display_cols = [
                'rank', 'ticker', 'company_name', 'master_score',
                'position_score', 'volume_score', 'momentum_score',
                'acceleration_score', 'breakout_score', 'trend_quality',
                'patterns', 'price', 'from_low_pct', 'ret_30d',
                'category', 'sector', 'eps_tier', 'pe_tier'
            ]
            
            display_cols = [col for col in display_cols if col in display_df.columns]
            
            # Format numeric columns
            format_dict = {
                'master_score': '{:.1f}',
                'position_score': '{:.1f}',
                'volume_score': '{:.1f}',
                'momentum_score': '{:.1f}',
                'acceleration_score': '{:.1f}',
                'breakout_score': '{:.1f}',
                'trend_quality': '{:.1f}',
                'price': '₹{:,.2f}',
                'from_low_pct': '{:.1f}%',
                'ret_30d': '{:.1f}%'
            }
            
            for col, fmt in format_dict.items():
                if col in display_df.columns:
                    if '%' in fmt:
                        display_df[col] = display_df[col].apply(
                            lambda x: fmt.format(x) if pd.notna(x) else ''
                        )
                    elif '₹' in fmt:
                        display_df[col] = display_df[col].apply(
                            lambda x: fmt.format(x) if pd.notna(x) else ''
                        )
                    else:
                        display_df[col] = display_df[col].apply(
                            lambda x: fmt.format(x) if pd.notna(x) else ''
                        )
            
            st.dataframe(
                display_df[display_cols],
                use_container_width=True,
                height=600
            )
        else:
            st.warning("No stocks match the selected filters.")
    
    with tab2:
        st.markdown("### Market Analysis")
        
        if len(filtered_df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Score distribution
                fig_dist = Visualizer.create_score_distribution(filtered_df)
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                # Sector summary
                st.markdown("#### Sector Performance")
                valid_df = filtered_df[filtered_df['master_score'].notna()]
                
                if len(valid_df) > 0:
                    sector_summary = valid_df.groupby('sector').agg({
                        'master_score': ['mean', 'count']
                    }).round(2)
                    sector_summary.columns = ['Avg Score', 'Count']
                    sector_summary = sector_summary.sort_values('Avg Score', ascending=False)
                    st.dataframe(sector_summary, use_container_width=True)
                else:
                    st.info("No valid scores for sector analysis")
            
            # Category summary
            st.markdown("#### Category Performance")
            if len(valid_df) > 0:
                category_summary = valid_df.groupby('category').agg({
                    'master_score': ['mean', 'count']
                }).round(2)
                category_summary.columns = ['Avg Score', 'Count']
                category_summary = category_summary.sort_values('Avg Score', ascending=False)
                st.dataframe(category_summary, use_container_width=True)
            else:
                st.info("No valid scores for category analysis")
        else:
            st.info("No data available for analysis.")
    
    with tab3:
        st.markdown("### Visualizations")
        
        if len(filtered_df) > 0:
            # Top stocks chart
            st.markdown("#### Top Stocks Breakdown")
            n_stocks = st.slider("Number of stocks", 10, 50, 20)
            fig_top = Visualizer.create_top_stocks_chart(filtered_df, n_stocks)
            st.plotly_chart(fig_top, use_container_width=True)
            
            # Sector performance
            valid_df = filtered_df[filtered_df['master_score'].notna()]
            if len(valid_df.groupby('sector')) >= 3:
                st.markdown("#### Sector Performance")
                fig_sector = Visualizer.create_sector_performance(filtered_df)
                st.plotly_chart(fig_sector, use_container_width=True)
        else:
            st.info("No data available for visualization.")
    
    with tab4:
        st.markdown("### Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Excel Report")
            st.markdown("Comprehensive report with multiple sheets")
            
            if st.button("📊 Generate Excel Report"):
                with st.spinner("Generating report..."):
                    excel_file = ExportEngine.create_excel_report(filtered_df)
                    
                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_file,
                        file_name=f"wave_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        with col2:
            st.markdown("#### CSV Export")
            st.markdown("Raw data for further analysis")
            
            if st.button("📄 Generate CSV"):
                csv_data = filtered_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"wave_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
