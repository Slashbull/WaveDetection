"""
Wave Detection System 5.0 - True Data-Driven Trading
====================================================
Philosophy: Let Every Stock Tell Its Story

ZERO THRESHOLDS - 100% DATA DRIVEN
- No price cutoffs
- No volume minimums  
- No return limits
- Pure relative ranking

Smart Categorization:
- EPS Tiers (5↓, 5↑, 15↑, 35↑, 55↑, 75↑, 95↑)
- Price Tiers (100↓, 100↑, 200↑, 500↑, 1K↑, 2K↑, 5K↑)
- Category & Sector filters
- Let users choose their universe

Author: Data Liberation Front
Version: 5.0.0 FINAL
Status: Production Ready
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
from enum import Enum
import warnings
import time
from functools import wraps

# ============================================
# CONFIGURATION
# ============================================

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Page config
try:
    st.set_page_config(
        page_title="Wave Detection 5.0 | Pure Data",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except:
    pass

# Performance decorator
def monitor_performance(func):
    """Monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        if elapsed > 0.1:
            logger.warning(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper

# ============================================
# DATA MODELS
# ============================================

@dataclass
class Signal:
    """Trading signal information"""
    ticker: str
    company_name: str
    sector: str
    category: str
    price: float
    
    # Tiers
    eps_tier: str
    price_tier: str
    
    # Rankings (0-1 percentile)
    momentum_rank: float
    volume_rank: float
    acceleration_rank: float
    combined_rank: float
    
    # Raw scores
    momentum_score: float
    volume_score: float
    momentum_acceleration: float
    
    # Signal
    signal_type: str  # 'BUY', 'STRONG_BUY', 'WATCH', 'HOLD'
    signal_strength: float
    
    # Context
    days_up: int
    days_down: int
    trend_strength: float
    
    # Metadata
    raw_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MarketProfile:
    """Current market profile"""
    timestamp: datetime
    total_stocks: int
    
    # Distribution
    eps_tier_distribution: Dict[str, int]
    price_tier_distribution: Dict[str, int]
    category_distribution: Dict[str, int]
    sector_distribution: Dict[str, int]
    
    # Market character
    avg_momentum: float
    avg_volume: float
    breadth_positive: float
    breadth_negative: float
    
    # Top movers
    top_gainers: List[str]
    top_volume: List[str]
    top_momentum: List[str]

# ============================================
# TIER DEFINITIONS
# ============================================

def get_eps_tier(eps: float) -> str:
    """Categorize EPS into tiers"""
    if pd.isna(eps) or eps < 5:
        return "5↓"
    elif eps < 15:
        return "5↑"
    elif eps < 35:
        return "15↑"
    elif eps < 55:
        return "35↑"
    elif eps < 75:
        return "55↑"
    elif eps < 95:
        return "75↑"
    else:
        return "95↑"

def get_price_tier(price: float) -> str:
    """Categorize price into tiers"""
    if pd.isna(price) or price < 100:
        return "100↓"
    elif price < 200:
        return "100↑"
    elif price < 500:
        return "200↑"
    elif price < 1000:
        return "500↑"
    elif price < 2000:
        return "1K↑"
    elif price < 5000:
        return "2K↑"
    else:
        return "5K↑"

# ============================================
# CSS STYLING
# ============================================

CUSTOM_CSS = """
<style>
    /* Clean Modern Design */
    .main {
        padding: 0;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        text-align: center;
        font-size: 1.25rem;
        color: #6b7280;
        margin-bottom: 3rem;
    }
    
    /* Signal Cards */
    .signal-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        border-left: 5px solid;
    }
    
    .signal-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.12);
    }
    
    .signal-strong-buy {
        border-left-color: #10b981;
        background: linear-gradient(to right, #f0fdf4, white);
    }
    
    .signal-buy {
        border-left-color: #3b82f6;
        background: linear-gradient(to right, #eff6ff, white);
    }
    
    .signal-watch {
        border-left-color: #f59e0b;
        background: linear-gradient(to right, #fffbeb, white);
    }
    
    /* Tier Badges */
    .tier-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        margin: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .eps-tier {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
    }
    
    .price-tier {
        background: linear-gradient(135deg, #10b981, #34d399);
        color: white;
    }
    
    /* Metrics */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
        text-align: center;
        height: 100%;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Rank Visualization */
    .rank-bar {
        width: 100%;
        height: 12px;
        background: #e5e7eb;
        border-radius: 6px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .rank-fill {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        transition: width 0.5s ease;
    }
    
    /* Tables */
    .dataframe {
        font-size: 0.875rem;
    }
    
    .dataframe th {
        background: #f9fafb;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        position: sticky;
        top: 0;
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .signal-card, .metric-card {
            background: #1f2937;
            color: white;
        }
    }
</style>
"""

# ============================================
# CALCULATION ENGINES
# ============================================

class MomentumEngine:
    """Pure momentum calculations - no thresholds"""
    
    @staticmethod
    @monitor_performance
    def calculate_momentum_score(row: pd.Series) -> float:
        """Calculate momentum score - let any value speak"""
        try:
            # Time-weighted momentum
            momentum = (
                row.get('ret_1d', 0) * 4 +     # Today is most important
                row.get('ret_3d', 0) * 2 +     # Recent confirmation
                row.get('ret_7d', 0) * 1.5 +   # Weekly trend
                row.get('ret_30d', 0) * 0.5    # Monthly context
            ) / 8
            
            # Trend alignment multiplier
            trend_multiplier = 1.0
            if row.get('price', 0) > row.get('sma_20d', float('inf')):
                trend_multiplier += 0.1
            if row.get('sma_20d', 0) > row.get('sma_50d', float('inf')):
                trend_multiplier += 0.1
            if row.get('sma_50d', 0) > row.get('sma_200d', float('inf')):
                trend_multiplier += 0.1
            
            return momentum * trend_multiplier
            
        except Exception as e:
            logger.debug(f"Momentum calc error: {e}")
            return 0.0
    
    @staticmethod
    def calculate_acceleration(row: pd.Series) -> float:
        """Calculate momentum acceleration"""
        try:
            # Is momentum speeding up?
            if row.get('ret_7d', 0) != 0:
                current_speed = row.get('ret_1d', 0)
                average_speed = row.get('ret_7d', 0) / 7
                acceleration = (current_speed - average_speed) / abs(average_speed) if average_speed != 0 else current_speed
            else:
                acceleration = row.get('ret_1d', 0)
            
            return acceleration
            
        except Exception:
            return 0.0
    
    @staticmethod
    def calculate_trend_strength(row: pd.Series) -> float:
        """How many periods are positive?"""
        try:
            positive_periods = 0
            total_periods = 0
            
            for period in ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m']:
                if period in row and not pd.isna(row[period]):
                    total_periods += 1
                    if row[period] > 0:
                        positive_periods += 1
            
            return positive_periods / total_periods if total_periods > 0 else 0.5
            
        except Exception:
            return 0.5

class VolumeEngine:
    """Pure volume analysis - no thresholds"""
    
    @staticmethod
    @monitor_performance
    def calculate_volume_score(row: pd.Series) -> float:
        """Volume conviction score"""
        try:
            # Base: relative volume
            rvol = row.get('rvol', 1.0)
            
            # Volume trend
            vol_trend = 1.0
            if row.get('vol_ratio_30d_90d', 0) > 0:
                vol_trend = row.get('vol_ratio_30d_90d', 1.0)
            
            # Volume acceleration
            vol_accel = 1.0
            if row.get('vol_ratio_7d_90d', 0) > 0 and row.get('vol_ratio_1d_90d', 0) > 0:
                short_term = row.get('vol_ratio_1d_90d', 1.0)
                medium_term = row.get('vol_ratio_7d_90d', 1.0)
                vol_accel = short_term / medium_term if medium_term > 0 else 1.0
            
            # Combined score - no limits!
            return rvol * vol_trend * vol_accel
            
        except Exception as e:
            logger.debug(f"Volume calc error: {e}")
            return 1.0
    
    @staticmethod
    def calculate_volume_character(row: pd.Series) -> str:
        """What's the volume telling us?"""
        try:
            rvol = row.get('rvol', 1.0)
            price_move = row.get('ret_1d', 0)
            
            if rvol > 3 and abs(price_move) > 5:
                return "EXPLOSIVE"
            elif rvol > 2 and price_move > 0:
                return "ACCUMULATION"
            elif rvol > 2 and price_move < 0:
                return "DISTRIBUTION"
            elif rvol < 0.5:
                return "DORMANT"
            else:
                return "NORMAL"
                
        except Exception:
            return "NORMAL"

# ============================================
# SIGNAL GENERATION
# ============================================

class SignalGenerator:
    """Generate signals from pure rankings"""
    
    @staticmethod
    @monitor_performance
    def generate_signals(df: pd.DataFrame) -> List[Signal]:
        """Generate signals for ALL stocks"""
        signals = []
        
        try:
            for _, row in df.iterrows():
                # Determine signal type from pure rank
                signal_type = SignalGenerator._determine_signal_type(row)
                
                # Calculate signal strength
                signal_strength = row.get('combined_rank', 0) * 100
                
                # Count trend days
                days_up = sum(1 for period in ['ret_1d', 'ret_3d', 'ret_7d'] 
                             if row.get(period, 0) > 0)
                days_down = 3 - days_up
                
                # Create signal for EVERY stock
                signal = Signal(
                    ticker=row['ticker'],
                    company_name=row.get('company_name', row['ticker']),
                    sector=row.get('sector', 'Unknown'),
                    category=row.get('category', 'Unknown'),
                    price=row.get('price', 0),
                    eps_tier=row.get('eps_tier', 'Unknown'),
                    price_tier=row.get('price_tier', 'Unknown'),
                    momentum_rank=row.get('momentum_rank', 0),
                    volume_rank=row.get('volume_rank', 0),
                    acceleration_rank=row.get('acceleration_rank', 0),
                    combined_rank=row.get('combined_rank', 0),
                    momentum_score=row.get('momentum_score', 0),
                    volume_score=row.get('volume_score', 0),
                    momentum_acceleration=row.get('momentum_acceleration', 0),
                    signal_type=signal_type,
                    signal_strength=signal_strength,
                    days_up=days_up,
                    days_down=days_down,
                    trend_strength=row.get('trend_strength', 0),
                    raw_data=row.to_dict()
                )
                
                signals.append(signal)
            
            # Sort by combined rank
            signals.sort(key=lambda x: x.combined_rank, reverse=True)
            
            logger.info(f"Generated {len(signals)} signals")
            return signals
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return []
    
    @staticmethod
    def _determine_signal_type(row: pd.Series) -> str:
        """Determine signal type from rank"""
        combined_rank = row.get('combined_rank', 0)
        momentum_rank = row.get('momentum_rank', 0)
        volume_rank = row.get('volume_rank', 0)
        
        # Top 5% with volume = STRONG BUY
        if combined_rank >= 0.95 and volume_rank >= 0.8:
            return 'STRONG_BUY'
        # Top 10% = BUY
        elif combined_rank >= 0.90:
            return 'BUY'
        # Top 25% = WATCH
        elif combined_rank >= 0.75:
            return 'WATCH'
        # Rest = HOLD
        else:
            return 'HOLD'

# ============================================
# DATA PROCESSING PIPELINE
# ============================================

class DataPipeline:
    """Process all data without thresholds"""
    
    def __init__(self):
        self.momentum_engine = MomentumEngine()
        self.volume_engine = VolumeEngine()
        self.signal_generator = SignalGenerator()
    
    @monitor_performance
    def process_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal], MarketProfile]:
        """Process ALL data - no filtering!"""
        try:
            # Step 1: Clean only truly invalid data
            df = self._minimal_clean(df)
            
            if df.empty:
                logger.warning("No valid data")
                return df, [], self._empty_market_profile()
            
            # Step 2: Add tiers
            df = self._add_tiers(df)
            
            # Step 3: Calculate scores for ALL stocks
            df = self._calculate_scores(df)
            
            # Step 4: Rank everything
            df = self._calculate_ranks(df)
            
            # Step 5: Generate signals
            signals = self.signal_generator.generate_signals(df)
            
            # Step 6: Create market profile
            market_profile = self._create_market_profile(df)
            
            logger.info(f"Processed {len(df)} stocks, generated {len(signals)} signals")
            
            return df, signals, market_profile
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return pd.DataFrame(), [], self._empty_market_profile()
    
    def _minimal_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Minimal cleaning - only remove truly invalid"""
        try:
            initial_count = len(df)
            
            # Only remove if price is literally invalid
            df = df[df['price'].notna()]
            df = df[df['price'] > 0]  # Can't have negative price
            
            # Remove complete duplicates
            df = df.drop_duplicates(subset=['ticker'], keep='first')
            
            final_count = len(df)
            logger.info(f"Minimal clean: {initial_count} → {final_count} rows")
            
            return df
            
        except Exception as e:
            logger.error(f"Cleaning error: {e}")
            return pd.DataFrame()
    
    def _add_tiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add EPS and Price tiers"""
        try:
            # Add EPS tier
            df['eps_tier'] = df['eps_current'].apply(get_eps_tier)
            
            # Add Price tier  
            df['price_tier'] = df['price'].apply(get_price_tier)
            
            logger.info("Added EPS and Price tiers")
            return df
            
        except Exception as e:
            logger.error(f"Tier calculation error: {e}")
            return df
    
    def _calculate_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate scores for all stocks"""
        try:
            # Momentum scores
            df['momentum_score'] = df.apply(self.momentum_engine.calculate_momentum_score, axis=1)
            df['momentum_acceleration'] = df.apply(self.momentum_engine.calculate_acceleration, axis=1)
            df['trend_strength'] = df.apply(self.momentum_engine.calculate_trend_strength, axis=1)
            
            # Volume scores
            df['volume_score'] = df.apply(self.volume_engine.calculate_volume_score, axis=1)
            df['volume_character'] = df.apply(self.volume_engine.calculate_volume_character, axis=1)
            
            logger.info("Calculated all scores")
            return df
            
        except Exception as e:
            logger.error(f"Score calculation error: {e}")
            return df
    
    def _calculate_ranks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank everything - this is where the magic happens"""
        try:
            # Individual ranks (0-1 percentile)
            df['momentum_rank'] = df['momentum_score'].rank(pct=True, method='average')
            df['volume_rank'] = df['volume_score'].rank(pct=True, method='average')
            df['acceleration_rank'] = df['momentum_acceleration'].rank(pct=True, method='average')
            
            # Combined rank with weights
            df['combined_rank'] = (
                df['momentum_rank'] * 0.4 +
                df['volume_rank'] * 0.3 +
                df['acceleration_rank'] * 0.2 +
                df['trend_strength'] * 0.1
            )
            
            # Final overall rank
            df['overall_rank'] = df['combined_rank'].rank(pct=True, method='average')
            
            logger.info("Calculated all rankings")
            return df
            
        except Exception as e:
            logger.error(f"Ranking error: {e}")
            return df
    
    def _create_market_profile(self, df: pd.DataFrame) -> MarketProfile:
        """Create market profile"""
        try:
            # Distributions
            eps_dist = df['eps_tier'].value_counts().to_dict()
            price_dist = df['price_tier'].value_counts().to_dict()
            category_dist = df['category'].value_counts().to_dict()
            sector_dist = df['sector'].value_counts().head(10).to_dict()
            
            # Market character
            avg_momentum = df['momentum_score'].mean()
            avg_volume = df['volume_score'].mean()
            breadth_positive = (df['ret_1d'] > 0).mean() * 100
            breadth_negative = (df['ret_1d'] < 0).mean() * 100
            
            # Top movers
            top_gainers = df.nlargest(5, 'ret_1d')['ticker'].tolist()
            top_volume = df.nlargest(5, 'volume_score')['ticker'].tolist()
            top_momentum = df.nlargest(5, 'momentum_score')['ticker'].tolist()
            
            return MarketProfile(
                timestamp=datetime.now(),
                total_stocks=len(df),
                eps_tier_distribution=eps_dist,
                price_tier_distribution=price_dist,
                category_distribution=category_dist,
                sector_distribution=sector_dist,
                avg_momentum=avg_momentum,
                avg_volume=avg_volume,
                breadth_positive=breadth_positive,
                breadth_negative=breadth_negative,
                top_gainers=top_gainers,
                top_volume=top_volume,
                top_momentum=top_momentum
            )
            
        except Exception as e:
            logger.error(f"Market profile error: {e}")
            return self._empty_market_profile()
    
    def _empty_market_profile(self) -> MarketProfile:
        """Empty market profile"""
        return MarketProfile(
            timestamp=datetime.now(),
            total_stocks=0,
            eps_tier_distribution={},
            price_tier_distribution={},
            category_distribution={},
            sector_distribution={},
            avg_momentum=0,
            avg_volume=1,
            breadth_positive=0,
            breadth_negative=0,
            top_gainers=[],
            top_volume=[],
            top_momentum=[]
        )

# ============================================
# VISUALIZATION
# ============================================

class Visualizer:
    """Create visualizations"""
    
    @staticmethod
    def create_tier_distribution(market_profile: MarketProfile, tier_type: str) -> go.Figure:
        """Create tier distribution chart"""
        if tier_type == 'eps':
            data = market_profile.eps_tier_distribution
            title = "EPS Tier Distribution"
            colors = px.colors.sequential.Blues
        else:
            data = market_profile.price_tier_distribution
            title = "Price Tier Distribution"
            colors = px.colors.sequential.Greens
        
        if not data:
            return go.Figure()
        
        # Sort tiers properly
        tier_order = ['5↓', '5↑', '15↑', '35↑', '55↑', '75↑', '95↑'] if tier_type == 'eps' else ['100↓', '100↑', '200↑', '500↑', '1K↑', '2K↑', '5K↑']
        sorted_data = {k: data.get(k, 0) for k in tier_order if k in data}
        
        fig = go.Figure(go.Bar(
            x=list(sorted_data.keys()),
            y=list(sorted_data.values()),
            marker_color=colors[0:len(sorted_data)],
            text=list(sorted_data.values()),
            textposition='auto'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Tier",
            yaxis_title="Number of Stocks",
            height=300,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_rank_heatmap(df: pd.DataFrame, top_n: int = 50) -> go.Figure:
        """Create ranking heatmap"""
        if df.empty:
            return go.Figure()
        
        # Get top N stocks
        top_stocks = df.nlargest(top_n, 'combined_rank')
        
        if top_stocks.empty:
            return go.Figure()
        
        # Create heatmap data
        heatmap_data = []
        labels = []
        
        rank_columns = ['momentum_rank', 'volume_rank', 'acceleration_rank', 'combined_rank']
        
        for col in rank_columns:
            if col in top_stocks.columns:
                heatmap_data.append(top_stocks[col].values)
                labels.append(col.replace('_rank', '').title())
        
        fig = go.Figure(go.Heatmap(
            z=heatmap_data,
            x=top_stocks['ticker'].values,
            y=labels,
            colorscale='Viridis',
            text=[[f"{val:.2%}" for val in row] for row in heatmap_data],
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title=f"Top {top_n} Stocks - Ranking Heatmap",
            height=400,
            xaxis_title="",
            yaxis_title=""
        )
        
        return fig
    
    @staticmethod
    def create_scatter_matrix(df: pd.DataFrame) -> go.Figure:
        """Create interactive scatter matrix"""
        if df.empty or len(df) < 10:
            return go.Figure()
        
        # Get top 100 for clarity
        plot_df = df.nlargest(100, 'combined_rank')
        
        # Select columns for scatter matrix
        columns = ['momentum_score', 'volume_score', 'ret_7d', 'rvol']
        
        # Create scatter matrix
        fig = px.scatter_matrix(
            plot_df,
            dimensions=columns,
            color='combined_rank',
            color_continuous_scale='Viridis',
            title="Multi-Dimensional Analysis (Top 100)",
            height=800,
            hover_data=['ticker']
        )
        
        fig.update_traces(diagonal_visible=False)
        
        return fig

# ============================================
# REPORT GENERATION
# ============================================

def generate_excel_report(df: pd.DataFrame, signals: List[Signal], market_profile: MarketProfile) -> BytesIO:
    """Generate Excel report"""
    output = BytesIO()
    
    try:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Header format
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4A5568',
                'font_color': 'white',
                'border': 1
            })
            
            # Sheet 1: Top Signals
            if signals:
                signals_data = []
                for s in signals[:100]:  # Top 100
                    signals_data.append({
                        'Ticker': s.ticker,
                        'Company': s.company_name,
                        'Signal': s.signal_type,
                        'Rank': f"{s.combined_rank:.1%}",
                        'Price': s.price,
                        'EPS Tier': s.eps_tier,
                        'Price Tier': s.price_tier,
                        'Momentum Rank': f"{s.momentum_rank:.1%}",
                        'Volume Rank': f"{s.volume_rank:.1%}",
                        'Momentum Score': f"{s.momentum_score:.2f}",
                        'Volume Score': f"{s.volume_score:.2f}"
                    })
                
                signals_df = pd.DataFrame(signals_data)
                signals_df.to_excel(writer, sheet_name='Top 100 Signals', index=False)
            
            # Sheet 2: All Rankings
            rank_df = df[['ticker', 'company_name', 'price', 'eps_tier', 'price_tier',
                         'momentum_rank', 'volume_rank', 'combined_rank']].copy()
            rank_df = rank_df.sort_values('combined_rank', ascending=False)
            rank_df.to_excel(writer, sheet_name='All Rankings', index=False)
            
            # Sheet 3: Market Profile
            profile_data = {
                'Metric': [
                    'Total Stocks',
                    'Average Momentum',
                    'Average Volume',
                    'Breadth Positive %',
                    'Breadth Negative %'
                ],
                'Value': [
                    market_profile.total_stocks,
                    f"{market_profile.avg_momentum:.2f}",
                    f"{market_profile.avg_volume:.2f}",
                    f"{market_profile.breadth_positive:.1f}%",
                    f"{market_profile.breadth_negative:.1f}%"
                ]
            }
            
            pd.DataFrame(profile_data).to_excel(writer, sheet_name='Market Profile', index=False)
            
            # Format all sheets
            for sheet in writer.sheets.values():
                sheet.freeze_panes(1, 0)
        
        output.seek(0)
        return output
        
    except Exception as e:
        logger.error(f"Excel generation error: {e}")
        return output

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main application"""
    
    # Apply CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🌊 Wave Detection 5.0</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Let Every Stock Tell Its Story - No Thresholds, Pure Rankings</p>', unsafe_allow_html=True)
    
    # Initialize
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = DataPipeline()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Data Source")
        
        sheet_url = st.text_input(
            "Google Sheets URL",
            value="https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk/edit?usp=sharing"
        )
        
        gid = st.text_input(
            "Sheet GID",
            value="2026492216"
        )
        
        if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("## 🎯 Smart Filters")
        st.info("Filter by categories - don't remove data!")
        
        # EPS Tier Filter
        eps_tiers = ['All', '5↓', '5↑', '15↑', '35↑', '55↑', '75↑', '95↑']
        selected_eps_tiers = st.multiselect(
            "EPS Tiers",
            options=eps_tiers,
            default=['All'],
            help="Filter by EPS tiers"
        )
        
        # Price Tier Filter
        price_tiers = ['All', '100↓', '100↑', '200↑', '500↑', '1K↑', '2K↑', '5K↑']
        selected_price_tiers = st.multiselect(
            "Price Tiers",
            options=price_tiers,
            default=['All'],
            help="Filter by price tiers"
        )
        
        # Category Filter
        st.markdown("#### Category Filter")
        categories = ['All', 'Large Cap', 'Mid Cap', 'Small Cap', 'Micro Cap']
        selected_categories = st.multiselect(
            "Categories",
            options=categories,
            default=['All']
        )
        
        # Signal Type Filter
        st.markdown("#### Signal Filter")
        signal_types = ['All', 'STRONG_BUY', 'BUY', 'WATCH', 'HOLD']
        selected_signals = st.multiselect(
            "Signal Types",
            options=signal_types,
            default=['STRONG_BUY', 'BUY']
        )
    
    # Main content
    try:
        # Load data
        @st.cache_data(ttl=300)
        def load_data(url: str, gid_val: str) -> pd.DataFrame:
            """Load data from Google Sheets"""
            try:
                csv_url = f"{url.split('/edit')[0]}/export?format=csv&gid={gid_val}"
                df = pd.read_csv(csv_url)
                logger.info(f"Loaded {len(df)} rows")
                return df
            except Exception as e:
                st.error(f"Failed to load data: {e}")
                return pd.DataFrame()
        
        with st.spinner("Loading ALL stocks..."):
            raw_df = load_data(sheet_url, gid)
        
        if raw_df.empty:
            st.error("No data loaded!")
            st.stop()
        
        # Process ALL data
        with st.spinner(f"Processing {len(raw_df)} stocks..."):
            df, signals, market_profile = st.session_state.pipeline.process_data(raw_df)
        
        # Apply filters to signals (not data!)
        filtered_signals = signals
        
        if 'All' not in selected_eps_tiers:
            filtered_signals = [s for s in filtered_signals if s.eps_tier in selected_eps_tiers]
        
        if 'All' not in selected_price_tiers:
            filtered_signals = [s for s in filtered_signals if s.price_tier in selected_price_tiers]
        
        if 'All' not in selected_categories:
            filtered_signals = [s for s in filtered_signals if s.category in selected_categories]
        
        if 'All' not in selected_signals:
            filtered_signals = [s for s in filtered_signals if s.signal_type in selected_signals]
        
        # Market Overview
        st.markdown("## 📊 Market Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Stocks</div>
                <div class="metric-value">{market_profile.total_stocks}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            color = "#10b981" if market_profile.breadth_positive > 50 else "#ef4444"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Positive Breadth</div>
                <div class="metric-value" style="color: {color};">{market_profile.breadth_positive:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Momentum</div>
                <div class="metric-value">{market_profile.avg_momentum:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Volume</div>
                <div class="metric-value">{market_profile.avg_volume:.1f}x</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Filtered Signals</div>
                <div class="metric-value">{len(filtered_signals)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Tier Distributions
        col1, col2 = st.columns(2)
        
        with col1:
            eps_fig = Visualizer.create_tier_distribution(market_profile, 'eps')
            st.plotly_chart(eps_fig, use_container_width=True)
        
        with col2:
            price_fig = Visualizer.create_tier_distribution(market_profile, 'price')
            st.plotly_chart(price_fig, use_container_width=True)
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Top Signals",
            "📊 Rankings",
            "🔥 Analysis", 
            "📈 Visualizations",
            "📥 Reports"
        ])
        
        with tab1:
            st.markdown("### 🎯 Top Trading Signals")
            
            if filtered_signals:
                # Signal summary
                signal_summary = {}
                for s in filtered_signals:
                    signal_summary[s.signal_type] = signal_summary.get(s.signal_type, 0) + 1
                
                cols = st.columns(len(signal_summary))
                for i, (sig_type, count) in enumerate(signal_summary.items()):
                    with cols[i]:
                        st.metric(sig_type, count)
                
                st.markdown("---")
                
                # Display top signals
                for i, signal in enumerate(filtered_signals[:20]):
                    # Determine card style
                    if signal.signal_type == 'STRONG_BUY':
                        card_class = 'signal-strong-buy'
                    elif signal.signal_type == 'BUY':
                        card_class = 'signal-buy'
                    else:
                        card_class = 'signal-watch'
                    
                    st.markdown(f'<div class="signal-card {card_class}">', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4, col5 = st.columns([2.5, 1.5, 1.5, 1.5, 1])
                    
                    with col1:
                        st.markdown(f"### #{i+1} - {signal.ticker}")
                        st.caption(f"{signal.company_name[:50]}...")
                        st.markdown(
                            f'<span class="tier-badge eps-tier">{signal.eps_tier}</span>'
                            f'<span class="tier-badge price-tier">{signal.price_tier}</span>',
                            unsafe_allow_html=True
                        )
                        st.caption(f"**{signal.category}** | {signal.sector}")
                    
                    with col2:
                        st.metric("Price", f"₹{signal.price:,.2f}")
                        st.metric("Signal", signal.signal_type)
                    
                    with col3:
                        st.metric("Rank", f"{signal.combined_rank:.1%}")
                        st.markdown(
                            f'<div class="rank-bar">'
                            f'<div class="rank-fill" style="width: {signal.combined_rank*100}%"></div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    with col4:
                        st.metric("Momentum", f"{signal.momentum_rank:.0%}")
                        st.metric("Volume", f"{signal.volume_rank:.0%}")
                    
                    with col5:
                        st.metric("Score", f"{signal.signal_strength:.0f}")
                        if signal.days_up > signal.days_down:
                            st.success(f"↑ {signal.days_up}/{3}")
                        else:
                            st.error(f"↓ {signal.days_down}/{3}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No signals match your filters. Try adjusting the criteria.")
        
        with tab2:
            st.markdown("### 📊 Complete Rankings")
            
            # Ranking heatmap
            heatmap = Visualizer.create_rank_heatmap(df, 50)
            st.plotly_chart(heatmap, use_container_width=True)
            
            # Top 50 table
            st.markdown("#### Top 50 Stocks by Combined Rank")
            
            display_df = df.nlargest(50, 'combined_rank')[[
                'ticker', 'company_name', 'price', 'eps_tier', 'price_tier',
                'category', 'sector', 'combined_rank', 'momentum_rank', 'volume_rank'
            ]].copy()
            
            # Format ranks as percentages
            for col in ['combined_rank', 'momentum_rank', 'volume_rank']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(display_df, use_container_width=True, height=600)
        
        with tab3:
            st.markdown("### 🔥 Market Analysis")
            
            # Top movers
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 📈 Top Gainers")
                for ticker in market_profile.top_gainers:
                    stock_data = df[df['ticker'] == ticker].iloc[0]
                    st.write(f"**{ticker}**: +{stock_data['ret_1d']:.1f}%")
            
            with col2:
                st.markdown("#### 📊 Top Volume")
                for ticker in market_profile.top_volume:
                    stock_data = df[df['ticker'] == ticker].iloc[0]
                    st.write(f"**{ticker}**: {stock_data['rvol']:.1f}x")
            
            with col3:
                st.markdown("#### 🚀 Top Momentum")
                for ticker in market_profile.top_momentum:
                    stock_data = df[df['ticker'] == ticker].iloc[0]
                    st.write(f"**{ticker}**: Score {stock_data['momentum_score']:.1f}")
            
            # Sector performance
            st.markdown("#### 🏢 Top Sectors")
            sector_df = pd.DataFrame(
                list(market_profile.sector_distribution.items()),
                columns=['Sector', 'Count']
            ).sort_values('Count', ascending=False)
            
            st.bar_chart(sector_df.set_index('Sector')['Count'])
        
        with tab4:
            st.markdown("### 📈 Interactive Visualizations")
            
            # Scatter matrix
            scatter = Visualizer.create_scatter_matrix(df)
            st.plotly_chart(scatter, use_container_width=True)
        
        with tab5:
            st.markdown("### 📥 Download Reports")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Generate Excel Report", type="primary", use_container_width=True):
                    excel = generate_excel_report(df, signals, market_profile)
                    
                    st.download_button(
                        "📥 Download Excel",
                        data=excel,
                        file_name=f"wave_detection_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            with col2:
                if st.button("📝 Generate Summary", type="secondary", use_container_width=True):
                    summary = f"""
WAVE DETECTION 5.0 - MARKET SUMMARY
{datetime.now().strftime('%Y-%m-%d %H:%M')}

MARKET PROFILE:
- Total Stocks: {market_profile.total_stocks}
- Positive Breadth: {market_profile.breadth_positive:.1f}%
- Average Momentum: {market_profile.avg_momentum:.2f}
- Average Volume: {market_profile.avg_volume:.1f}x

TOP SIGNALS:
"""
                    for i, signal in enumerate(filtered_signals[:10], 1):
                        summary += f"""
{i}. {signal.ticker} ({signal.signal_type})
   Rank: {signal.combined_rank:.1%} | Price: ₹{signal.price:,.0f}
   EPS Tier: {signal.eps_tier} | Price Tier: {signal.price_tier}
"""
                    
                    st.text_area("Summary", summary, height=500)
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"Error: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <p style="text-align: center; color: #6b7280; font-size: 0.875rem;">
        Wave Detection 5.0 - Let Every Stock Tell Its Story<br>
        No Thresholds. Pure Rankings. Total Freedom.<br>
        © 2024
    </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
