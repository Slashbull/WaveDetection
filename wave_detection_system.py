"""
Wave Detection System 7.0 - Professional Trading Signal Platform
==============================================================
A production-ready stock screening system focused on actionable signals.

Architecture: Pipeline-based with lazy evaluation
Philosophy: One score, one rank, one clear action
Performance: Optimized for 2000+ stocks with sub-second response

Version: 7.0.1 (Bug Fix)
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
from io import BytesIO
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

# ============================================
# CONFIGURATION
# ============================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Wave Detection 7.0 | Trading Signals",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# CONSTANTS & CONFIGURATION
# ============================================

@dataclass
class Config:
    """System configuration"""
    # Data source
    DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk/edit?usp=sharing"
    DEFAULT_GID = "2026492216"
    
    # Cache settings
    CACHE_TTL = 300  # 5 minutes
    
    # Display settings
    TOP_STOCKS_DISPLAY = 20
    MIN_VOLUME_FILTER = 10000  # Minimum daily volume
    MIN_PRICE_FILTER = 10  # Minimum price ₹
    
    # Scoring thresholds
    BUY_THRESHOLD = 70
    SELL_THRESHOLD = 30
    STRONG_BUY_THRESHOLD = 85
    STRONG_SELL_THRESHOLD = 20
    
    # Risk settings
    DEFAULT_STOP_LOSS_PCT = 5
    DEFAULT_TARGET_PCT = 10

class SignalType(Enum):
    """Trading signal types"""
    STRONG_BUY = "🚀 STRONG BUY"
    BUY = "✅ BUY"
    HOLD = "⏸️ HOLD"
    SELL = "📉 SELL"
    STRONG_SELL = "🔴 STRONG SELL"
    NO_SIGNAL = "⚪ NO SIGNAL"

# ============================================
# DATA MODELS
# ============================================

@dataclass
class StockSignal:
    """Complete stock signal information"""
    ticker: str
    company_name: str
    category: str
    sector: str
    price: float
    
    # Core metrics
    power_score: float
    momentum_strength: float
    volume_surge: float
    position_health: float
    
    # Signal
    signal: SignalType
    confidence: float
    
    # Risk metrics
    stop_loss: float
    target: float
    risk_reward_ratio: float
    
    # Supporting data
    rank: int
    sector_rank: int
    peer_percentile: float

# ============================================
# DATA PIPELINE
# ============================================

class DataPipeline:
    """Handles all data operations with validation and optimization"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate dataframe has required columns"""
        required_cols = [
            'ticker', 'company_name', 'price', 'category', 'sector',
            'ret_1d', 'ret_7d', 'ret_30d', 'rvol', 'volume_1d',
            'from_low_pct', 'from_high_pct', 'sma_20d', 'sma_50d', 'sma_200d'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            return False, f"Missing columns: {', '.join(missing_cols)}"
        
        if df.empty:
            return False, "DataFrame is empty"
            
        return True, "Valid"
    
    @staticmethod
    def clean_numeric_value(value: Union[str, float]) -> float:
        """Clean Indian number format to float"""
        if pd.isna(value):
            return np.nan
            
        try:
            # Convert to string first
            value_str = str(value).strip()
            
            # Handle empty or invalid values
            if value_str in ['', '-', 'N/A', '#N/A', 'nan', 'None']:
                return np.nan
            
            # Remove currency and percentage symbols
            cleaned = value_str.replace('₹', '').replace('%', '').replace(',', '')
            
            # Convert to float
            return float(cleaned)
        except (ValueError, AttributeError):
            return np.nan
    
    @staticmethod
    def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Complete dataframe processing with proper type conversion"""
        try:
            # Create a copy to avoid modifying original
            processed_df = df.copy()
            
            # Define numeric columns to clean
            numeric_cols = [
                'price', 'prev_close', 'low_52w', 'high_52w',
                'sma_20d', 'sma_50d', 'sma_200d',
                'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y',
                'from_low_pct', 'from_high_pct',
                'volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d',
                'rvol', 'pe', 'eps_current', 'eps_change_pct'
            ]
            
            # Volume ratio columns
            vol_ratio_cols = [
                'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
                'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
                'vol_ratio_90d_180d'
            ]
            
            # Clean all numeric columns that exist
            for col in numeric_cols:
                if col in processed_df.columns:
                    processed_df[col] = processed_df[col].apply(DataPipeline.clean_numeric_value)
                    # Convert to numeric to ensure proper type
                    processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
            
            # Handle volume ratio columns specially
            # They come as percentages and need to be converted to ratios
            for col in vol_ratio_cols:
                if col in processed_df.columns:
                    processed_df[col] = processed_df[col].apply(DataPipeline.clean_numeric_value)
                    # Convert percentage to ratio: -56.61% becomes 0.4339
                    processed_df[col] = (100 + processed_df[col]) / 100
                    # Ensure numeric type
                    processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                    # Fill NaN with 1.0 (no change)
                    processed_df[col].fillna(1.0, inplace=True)
            
            # Clean categorical columns
            categorical_cols = ['ticker', 'company_name', 'category', 'sector']
            for col in categorical_cols:
                if col in processed_df.columns:
                    processed_df[col] = processed_df[col].astype(str).str.strip()
                    processed_df[col] = processed_df[col].replace(['nan', 'None', '', 'N/A'], 'Unknown')
            
            # Fill NaN values for critical numeric columns with appropriate defaults
            if 'rvol' in processed_df.columns:
                processed_df['rvol'].fillna(1.0, inplace=True)
            
            # Ensure price columns have no NaN for calculations
            price_cols = ['price', 'sma_20d', 'sma_50d', 'sma_200d']
            for col in price_cols:
                if col in processed_df.columns and col != 'price':
                    # Fill missing MA values with price
                    processed_df[col].fillna(processed_df['price'], inplace=True)
            
            return processed_df
            
        except Exception as e:
            logger.error(f"Error processing dataframe: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            raise
    
    @staticmethod
    def prefilter_stocks(df: pd.DataFrame) -> pd.DataFrame:
        """Apply performance filters before scoring"""
        initial_count = len(df)
        
        try:
            # Process the dataframe first
            df = DataPipeline.process_dataframe(df)
            
            # Apply filters with proper error handling
            valid_mask = (
                (df['price'].notna()) & 
                (df['price'] > Config.MIN_PRICE_FILTER) &
                (df['volume_1d'].notna()) &
                (df['volume_1d'] > Config.MIN_VOLUME_FILTER) &
                (df['ret_1d'].notna())
            )
            
            filtered_df = df[valid_mask].copy()
            
            final_count = len(filtered_df)
            logger.info(f"Prefiltered: {initial_count} → {final_count} stocks")
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error in prefilter: {str(e)}")
            raise

# ============================================
# SIGNAL ENGINE
# ============================================

class SignalEngine:
    """Core signal generation with vectorized operations"""
    
    @staticmethod
    def safe_numeric_operation(series: pd.Series, default: float = 0.0) -> pd.Series:
        """Safely convert series to numeric for operations"""
        return pd.to_numeric(series, errors='coerce').fillna(default)
    
    @staticmethod
    def calculate_momentum_strength(df: pd.DataFrame) -> pd.Series:
        """Calculate momentum strength (0-100) using multiple timeframes"""
        try:
            # Ensure numeric types for all return columns
            ret_1d = SignalEngine.safe_numeric_operation(df['ret_1d'])
            ret_7d = SignalEngine.safe_numeric_operation(df['ret_7d'])
            ret_30d = SignalEngine.safe_numeric_operation(df.get('ret_30d', 0))
            ret_3m = SignalEngine.safe_numeric_operation(df.get('ret_3m', 0))
            
            # Vectorized calculation for all stocks at once
            momentum_scores = pd.DataFrame()
            
            # Short-term momentum (40% weight)
            momentum_scores['short'] = (
                (ret_1d > 0).astype(int) * 10 +
                (ret_1d > 2).astype(int) * 10 +
                (ret_7d > 0).astype(int) * 10 +
                (ret_7d > 5).astype(int) * 10
            )
            
            # Medium-term momentum (40% weight)
            momentum_scores['medium'] = (
                (ret_30d > 0).astype(int) * 10 +
                (ret_30d > 10).astype(int) * 10 +
                (ret_3m > 0).astype(int) * 10 +
                (ret_3m > 20).astype(int) * 10
            )
            
            # Consistency bonus (20% weight)
            all_positive = (
                (ret_1d > 0) & 
                (ret_7d > 0) & 
                (ret_30d > 0)
            )
            momentum_scores['consistency'] = all_positive.astype(int) * 20
            
            return momentum_scores.sum(axis=1)
            
        except Exception as e:
            logger.error(f"Error calculating momentum: {str(e)}")
            return pd.Series(50, index=df.index)  # Return neutral score on error
    
    @staticmethod
    def calculate_volume_surge(df: pd.DataFrame) -> pd.Series:
        """Calculate volume surge score (0-100)"""
        try:
            # Ensure numeric types
            rvol = SignalEngine.safe_numeric_operation(df['rvol'], 1.0)
            
            # RVOL is the key indicator
            rvol_score = np.clip(rvol * 20, 0, 60)  # Max 60 points from RVOL
            
            # Volume trend - check if column exists and is numeric
            trend_score = 0
            if 'vol_ratio_30d_90d' in df.columns:
                vol_trend = SignalEngine.safe_numeric_operation(df['vol_ratio_30d_90d'], 1.0)
                trend_score = np.clip((vol_trend - 0.5) * 40, 0, 40)  # Max 40 points
            else:
                trend_score = pd.Series(20, index=df.index)  # Default neutral score
            
            return rvol_score + trend_score
            
        except Exception as e:
            logger.error(f"Error calculating volume surge: {str(e)}")
            return pd.Series(50, index=df.index)
    
    @staticmethod
    def calculate_position_health(df: pd.DataFrame) -> pd.Series:
        """Calculate position health score (0-100)"""
        try:
            scores = pd.DataFrame(index=df.index)
            
            # Ensure numeric types
            from_low_pct = SignalEngine.safe_numeric_operation(df['from_low_pct'])
            from_high_pct = SignalEngine.safe_numeric_operation(df['from_high_pct'])
            price = SignalEngine.safe_numeric_operation(df['price'])
            sma_20d = SignalEngine.safe_numeric_operation(df.get('sma_20d', price))
            sma_50d = SignalEngine.safe_numeric_operation(df.get('sma_50d', price))
            sma_200d = SignalEngine.safe_numeric_operation(df.get('sma_200d', price))
            
            # Distance from low (not too extended)
            # Optimal zone: 20-100% above low
            scores['low_distance'] = np.where(
                from_low_pct <= 100,
                from_low_pct * 0.4,  # Up to 40 points
                40 - (from_low_pct - 100) * 0.1  # Penalty for overextension
            )
            scores['low_distance'] = np.clip(scores['low_distance'], 0, 40)
            
            # Distance from high (room to grow)
            # Best: -40% to -10% from high
            scores['high_distance'] = np.where(
                from_high_pct > -10,
                30,  # Near highs = strong
                np.where(
                    from_high_pct > -40,
                    30 + from_high_pct * 0.5,  # Gradual score
                    10  # Too far from highs
                )
            )
            
            # Moving average alignment (30 points)
            ma_aligned = (
                (price > sma_20d) & 
                (sma_20d > sma_50d) & 
                (sma_50d > sma_200d)
            )
            scores['ma_score'] = ma_aligned.astype(int) * 30
            
            return scores.sum(axis=1)
            
        except Exception as e:
            logger.error(f"Error calculating position health: {str(e)}")
            return pd.Series(50, index=df.index)
    
    @staticmethod
    def calculate_power_score(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate unified power score and generate signals"""
        try:
            # Calculate component scores with error handling
            df['momentum_strength'] = SignalEngine.calculate_momentum_strength(df)
            df['volume_surge'] = SignalEngine.calculate_volume_surge(df)
            df['position_health'] = SignalEngine.calculate_position_health(df)
            
            # Unified power score
            df['power_score'] = (
                df['momentum_strength'] * 0.40 +
                df['volume_surge'] * 0.30 +
                df['position_health'] * 0.30
            )
            
            # Ensure power_score is numeric
            df['power_score'] = pd.to_numeric(df['power_score'], errors='coerce').fillna(50)
            
            # Generate trading signals
            df['signal'] = df['power_score'].apply(SignalEngine.get_signal)
            
            # Calculate confidence (0-100%)
            df['confidence'] = df.apply(SignalEngine.calculate_confidence, axis=1)
            
            # Risk metrics - ensure price is numeric
            price_numeric = pd.to_numeric(df['price'], errors='coerce')
            df['stop_loss'] = price_numeric * (1 - Config.DEFAULT_STOP_LOSS_PCT / 100)
            df['target'] = price_numeric * (1 + Config.DEFAULT_TARGET_PCT / 100)
            df['risk_reward_ratio'] = Config.DEFAULT_TARGET_PCT / Config.DEFAULT_STOP_LOSS_PCT
            
            # Rankings
            df['rank'] = df['power_score'].rank(ascending=False, method='min').astype(int)
            df['sector_rank'] = df.groupby('sector')['power_score'].rank(
                ascending=False, method='min'
            ).astype(int)
            df['peer_percentile'] = df.groupby('sector')['power_score'].rank(pct=True) * 100
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating power score: {str(e)}")
            logger.error(f"DataFrame shape: {df.shape}")
            logger.error(f"DataFrame columns: {list(df.columns)}")
            raise
    
    @staticmethod
    def get_signal(score: float) -> SignalType:
        """Convert score to trading signal"""
        try:
            score = float(score)
            if score >= Config.STRONG_BUY_THRESHOLD:
                return SignalType.STRONG_BUY
            elif score >= Config.BUY_THRESHOLD:
                return SignalType.BUY
            elif score <= Config.STRONG_SELL_THRESHOLD:
                return SignalType.STRONG_SELL
            elif score <= Config.SELL_THRESHOLD:
                return SignalType.SELL
            else:
                return SignalType.HOLD
        except:
            return SignalType.NO_SIGNAL
    
    @staticmethod
    def calculate_confidence(row: pd.Series) -> float:
        """Calculate signal confidence based on multiple factors"""
        try:
            confidence = 50  # Base confidence
            
            # Volume confirmation
            rvol = pd.to_numeric(row.get('rvol', 1), errors='coerce')
            if rvol > 2:
                confidence += 15
            elif rvol > 1.5:
                confidence += 10
            
            # Momentum consistency
            ret_1d = pd.to_numeric(row.get('ret_1d', 0), errors='coerce')
            ret_7d = pd.to_numeric(row.get('ret_7d', 0), errors='coerce')
            ret_30d = pd.to_numeric(row.get('ret_30d', 0), errors='coerce')
            
            if ret_1d > 0 and ret_7d > 0 and ret_30d > 0:
                confidence += 20
            
            # Position strength
            from_high_pct = pd.to_numeric(row.get('from_high_pct', -50), errors='coerce')
            from_low_pct = pd.to_numeric(row.get('from_low_pct', 50), errors='coerce')
            
            if from_high_pct > -20 and from_low_pct > 30:
                confidence += 15
            
            return min(confidence, 100)
            
        except:
            return 50

# ============================================
# UI COMPONENTS
# ============================================

class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def render_header():
        """Render application header"""
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }
        .metric-card {
            background: white;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .signal-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
        }
        .buy-signal { background: #27ae60; color: white; }
        .sell-signal { background: #e74c3c; color: white; }
        .hold-signal { background: #95a5a6; color: white; }
        </style>
        
        <div class="main-header">
            <h1>📊 Wave Detection 7.0</h1>
            <p>Professional Trading Signal Platform</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_metrics(df: pd.DataFrame):
        """Render key metrics"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_stocks = len(df)
        buy_signals = len(df[df['signal'].isin([SignalType.BUY, SignalType.STRONG_BUY])])
        avg_confidence = df[df['signal'] != SignalType.HOLD]['confidence'].mean()
        
        with col1:
            st.metric("Total Stocks", f"{total_stocks:,}")
        
        with col2:
            st.metric("Buy Signals", buy_signals)
        
        with col3:
            st.metric("Avg Confidence", f"{avg_confidence:.0f}%" if not pd.isna(avg_confidence) else "N/A")
        
        with col4:
            top_sector = df.groupby('sector')['power_score'].mean().idxmax() if len(df) > 0 else "N/A"
            st.metric("Top Sector", top_sector[:15] if top_sector != "N/A" else "N/A")
        
        with col5:
            market_breadth = (df['ret_1d'] > 0).mean() * 100 if len(df) > 0 else 0
            st.metric("Market Breadth", f"{market_breadth:.0f}%")
    
    @staticmethod
    def render_signals_table(df: pd.DataFrame, limit: int = 20):
        """Render signals table with formatting"""
        if df.empty:
            st.warning("No stocks to display")
            return
            
        # Get top stocks by power score
        top_stocks = df.nlargest(min(limit, len(df)), 'power_score')
        
        # Prepare display columns
        display_df = pd.DataFrame({
            'Rank': top_stocks['rank'],
            'Ticker': top_stocks['ticker'],
            'Company': top_stocks['company_name'],
            'Signal': top_stocks['signal'].apply(lambda x: x.value),
            'Score': top_stocks['power_score'].round(1),
            'Confidence': top_stocks['confidence'].apply(lambda x: f"{x:.0f}%"),
            'Price': top_stocks['price'].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "N/A"),
            'Change': top_stocks['ret_1d'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A"),
            'Volume': top_stocks['rvol'].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else "N/A"),
            'Stop Loss': top_stocks['stop_loss'].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "N/A"),
            'Target': top_stocks['target'].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "N/A"),
            'Sector': top_stocks['sector']
        })
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=600,
            hide_index=True
        )
    
    @staticmethod
    def render_sector_analysis(df: pd.DataFrame):
        """Render sector performance analysis"""
        if df.empty or 'sector' not in df.columns:
            st.warning("No sector data available")
            return
            
        sector_stats = df.groupby('sector').agg({
            'power_score': ['mean', 'std', 'count'],
            'signal': lambda x: (x.isin([SignalType.BUY, SignalType.STRONG_BUY])).sum()
        }).round(2)
        
        sector_stats.columns = ['Avg Score', 'Std Dev', 'Count', 'Buy Signals']
        sector_stats = sector_stats.sort_values('Avg Score', ascending=False)
        
        fig = px.bar(
            sector_stats.reset_index(),
            x='sector',
            y='Avg Score',
            color='Buy Signals',
            title='Sector Performance Analysis',
            labels={'sector': 'Sector', 'Avg Score': 'Average Power Score'},
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# EXPORT FUNCTIONS
# ============================================

def generate_excel_report(df: pd.DataFrame) -> BytesIO:
    """Generate Excel report with key insights"""
    output = BytesIO()
    
    try:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Sheet 1: Top Buy Signals
            buy_signals = df[df['signal'].isin([SignalType.BUY, SignalType.STRONG_BUY])].nlargest(100, 'power_score')
            
            if not buy_signals.empty:
                buy_signals_display = pd.DataFrame({
                    'Rank': buy_signals['rank'],
                    'Ticker': buy_signals['ticker'],
                    'Company': buy_signals['company_name'],
                    'Signal': buy_signals['signal'].apply(lambda x: x.value),
                    'Power Score': buy_signals['power_score'].round(1),
                    'Confidence %': buy_signals['confidence'].round(0),
                    'Price': buy_signals['price'],
                    'Stop Loss': buy_signals['stop_loss'],
                    'Target': buy_signals['target'],
                    'Risk/Reward': buy_signals['risk_reward_ratio'],
                    'Sector Rank': buy_signals['sector_rank'],
                    'Category': buy_signals['category'],
                    'Sector': buy_signals['sector']
                })
                
                buy_signals_display.to_excel(writer, sheet_name='Buy Signals', index=False)
            
            # Sheet 2: Sector Summary
            sector_summary = df.groupby('sector').agg({
                'power_score': ['mean', 'std', 'count'],
                'signal': lambda x: (x.isin([SignalType.BUY, SignalType.STRONG_BUY])).sum(),
                'ret_1d': 'mean',
                'rvol': 'mean'
            }).round(2)
            
            sector_summary.columns = ['Avg Score', 'Std Dev', 'Total Stocks', 'Buy Signals', 'Avg 1D Return', 'Avg RVol']
            sector_summary.to_excel(writer, sheet_name='Sector Analysis')
            
            # Sheet 3: All Stocks Ranked
            all_stocks = df[['rank', 'ticker', 'company_name', 'power_score', 'signal', 
                            'price', 'category', 'sector']].sort_values('rank')
            all_stocks['signal'] = all_stocks['signal'].apply(lambda x: x.value)
            all_stocks.to_excel(writer, sheet_name='All Stocks', index=False)
            
            # Format workbook
            workbook = writer.book
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#2a5298',
                'font_color': 'white',
                'border': 1
            })
            
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                worksheet.freeze_panes(1, 0)
        
        output.seek(0)
        return output
        
    except Exception as e:
        logger.error(f"Error generating Excel report: {str(e)}")
        output.seek(0)
        return output

# ============================================
# MAIN APPLICATION
# ============================================

@st.cache_data(ttl=Config.CACHE_TTL)
def load_and_process_data(sheet_url: str, gid: str) -> pd.DataFrame:
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
        
        logger.info(f"Loaded {len(df)} rows")
        
        # Validate
        is_valid, message = DataPipeline.validate_dataframe(df)
        if not is_valid:
            raise ValueError(f"Data validation failed: {message}")
        
        # Prefilter and process
        df = DataPipeline.prefilter_stocks(df)
        
        if df.empty:
            raise ValueError("No stocks passed filtering criteria")
        
        # Calculate signals
        df = SignalEngine.calculate_power_score(df)
        
        logger.info(f"Successfully processed {len(df)} stocks")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load/process data: {str(e)}")
        raise

def main():
    """Main application"""
    try:
        # Header
        UIComponents.render_header()
        
        # Sidebar configuration
        with st.sidebar:
            st.markdown("### ⚙️ Configuration")
            
            # Data source (collapsed by default)
            with st.expander("Data Source", expanded=False):
                sheet_url = st.text_input(
                    "Google Sheets URL",
                    value=Config.DEFAULT_SHEET_URL
                )
                gid = st.text_input(
                    "Sheet GID",
                    value=Config.DEFAULT_GID
                )
            
            # Quick filters
            st.markdown("### 🎯 Quick Filters")
            
            signal_filter = st.multiselect(
                "Signal Type",
                options=[s.value for s in SignalType if s != SignalType.NO_SIGNAL],
                default=[SignalType.STRONG_BUY.value, SignalType.BUY.value]
            )
            
            min_confidence = st.slider(
                "Min Confidence %",
                min_value=0,
                max_value=100,
                value=60,
                step=10
            )
            
            # Refresh button
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        # Load and process data
        try:
            with st.spinner("Loading market data..."):
                df = load_and_process_data(sheet_url, gid)
        except Exception as e:
            st.error(f"❌ Failed to load data: {str(e)}")
            st.info("Please check your data source and try refreshing.")
            st.stop()
        
        # Apply filters
        if not df.empty:
            filtered_df = df[
                df['signal'].apply(lambda x: x.value).isin(signal_filter) &
                (df['confidence'] >= min_confidence)
            ].copy()
        else:
            filtered_df = df
        
        # Main content
        UIComponents.render_metrics(filtered_df)
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Trading Signals", "📈 Analysis", "📥 Export"])
        
        with tab1:
            st.markdown("### 🎯 Top Trading Signals")
            
            # Display options
            col1, col2 = st.columns([3, 1])
            with col2:
                display_count = st.selectbox(
                    "Show top",
                    options=[10, 20, 50, 100],
                    index=1
                )
            
            # Signals table
            UIComponents.render_signals_table(filtered_df, display_count)
            
            # Key insights
            if not filtered_df.empty:
                st.markdown("### 💡 Key Insights")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    strong_buys = filtered_df[filtered_df['signal'] == SignalType.STRONG_BUY]
                    if not strong_buys.empty:
                        st.info(f"**{len(strong_buys)} Strong Buy signals** with avg confidence {strong_buys['confidence'].mean():.0f}%")
                
                with col2:
                    high_volume = filtered_df[filtered_df['rvol'] > 2]
                    if not high_volume.empty:
                        st.success(f"**{len(high_volume)} stocks** showing unusual volume (>2x average)")
                
                with col3:
                    momentum_stocks = filtered_df[filtered_df['momentum_strength'] > 80]
                    if not momentum_stocks.empty:
                        st.warning(f"**{len(momentum_stocks)} stocks** in strong momentum (>80 score)")
        
        with tab2:
            st.markdown("### 📊 Market Analysis")
            
            if not df.empty:
                # Sector analysis
                UIComponents.render_sector_analysis(df)
                
                # Signal distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    signal_dist = df['signal'].value_counts()
                    fig_pie = px.pie(
                        values=signal_dist.values,
                        names=[s.value for s in signal_dist.index],
                        title="Signal Distribution",
                        color_discrete_map={
                            SignalType.STRONG_BUY.value: '#27ae60',
                            SignalType.BUY.value: '#2ecc71',
                            SignalType.HOLD.value: '#95a5a6',
                            SignalType.SELL.value: '#e67e22',
                            SignalType.STRONG_SELL.value: '#e74c3c'
                        }
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Score distribution
                    fig_hist = px.histogram(
                        df,
                        x='power_score',
                        nbins=30,
                        title="Power Score Distribution",
                        labels={'power_score': 'Power Score', 'count': 'Number of Stocks'}
                    )
                    fig_hist.add_vline(x=Config.BUY_THRESHOLD, line_dash="dash", 
                                     annotation_text="Buy Threshold")
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Top movers
                st.markdown("### 🚀 Top Movers")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Biggest Gainers")
                    gainers = df.nlargest(10, 'ret_1d')[['ticker', 'company_name', 'ret_1d', 'rvol']]
                    gainers['ret_1d'] = gainers['ret_1d'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
                    gainers['rvol'] = gainers['rvol'].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else "N/A")
                    st.dataframe(gainers, use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("#### Highest Volume")
                    high_vol = df.nlargest(10, 'rvol')[['ticker', 'company_name', 'rvol', 'ret_1d']]
                    high_vol['rvol'] = high_vol['rvol'].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else "N/A")
                    high_vol['ret_1d'] = high_vol['ret_1d'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
                    st.dataframe(high_vol, use_container_width=True, hide_index=True)
        
        with tab3:
            st.markdown("### 📥 Export Reports")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Excel Report")
                st.markdown("Comprehensive report with:")
                st.markdown("- Top 100 buy signals with targets")
                st.markdown("- Sector-wise analysis")
                st.markdown("- All stocks ranked")
                
                if st.button("📊 Generate Excel Report", use_container_width=True):
                    with st.spinner("Generating report..."):
                        excel_file = generate_excel_report(df)
                    
                    st.download_button(
                        label="📥 Download Excel Report",
                        data=excel_file,
                        file_name=f"wave_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
            
            with col2:
                st.markdown("#### Buy Signals CSV")
                st.markdown("Quick list of buy signals for:")
                st.markdown("- Direct broker upload")
                st.markdown("- Watchlist creation")
                st.markdown("- Further analysis")
                
                if st.button("📄 Generate CSV", use_container_width=True):
                    buy_signals = filtered_df[
                        filtered_df['signal'].isin([SignalType.BUY, SignalType.STRONG_BUY])
                    ]
                    if not buy_signals.empty:
                        csv_data = buy_signals[
                            ['ticker', 'company_name', 'signal', 'power_score', 
                             'confidence', 'price', 'stop_loss', 'target']
                        ].to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_data,
                            file_name=f"buy_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.warning("No buy signals to export")
        
        # Footer
        st.markdown("---")
        last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.caption(f"Wave Detection 7.0 | Last Update: {last_update} | Data refreshes every 5 minutes")
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An error occurred. Please refresh the page.")

if __name__ == "__main__":
    main()
