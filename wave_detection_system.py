"""
Wave Detection Ultimate - Professional Stock Ranking System
=========================================================
A market-adaptive stock ranking system using pure relative strength.

Author: Professional Implementation
Version: 1.0.0
License: MIT
Python: 3.8+
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache, wraps
import time
import gc
from io import BytesIO
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================
# PROFESSIONAL LOGGING CONFIGURATION
# ============================================

class LoggerFactory:
    """Factory for creating configured loggers"""
    
    @staticmethod
    def create_logger(name: str, level: int = logging.INFO) -> logging.Logger:
        """Create a configured logger instance"""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

# Create main logger
logger = LoggerFactory.create_logger(__name__)

# ============================================
# CONFIGURATION MANAGEMENT
# ============================================

@dataclass(frozen=True)
class SystemConfig:
    """Immutable system configuration"""
    
    # Data source
    DEFAULT_SHEET_URL: str = "https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk/edit?usp=sharing"
    DEFAULT_GID: str = "2026492216"
    
    # Performance
    CACHE_TTL: int = 300  # 5 minutes
    MAX_DISPLAY_ROWS: int = 1000
    CHUNK_SIZE: int = 10000
    
    # Ranking weights (must sum to 1.0)
    POSITION_WEIGHT: float = 0.45
    VOLUME_WEIGHT: float = 0.35
    MOMENTUM_WEIGHT: float = 0.20
    
    # UI Settings
    DEFAULT_TOP_N: int = 50
    AVAILABLE_TOP_N: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200])
    
    # Numeric precision
    PRICE_PRECISION: int = 2
    PERCENT_PRECISION: int = 2
    SCORE_PRECISION: int = 1
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        weight_sum = self.POSITION_WEIGHT + self.VOLUME_WEIGHT + self.MOMENTUM_WEIGHT
        if not np.isclose(weight_sum, 1.0, rtol=1e-5):
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

# Create global config instance
CONFIG = SystemConfig()

# ============================================
# DATA MODELS
# ============================================

class MarketRegime(Enum):
    """Market regime types"""
    BULL = auto()
    BEAR = auto()
    SIDEWAYS = auto()
    VOLATILE = auto()
    UNKNOWN = auto()

@dataclass
class StockData:
    """Validated stock data model"""
    ticker: str
    company_name: str
    price: float
    from_low_pct: float
    from_high_pct: float
    vol_ratio_30d_90d: float
    ret_30d: float
    
    # Optional fields
    category: str = "Unknown"
    sector: str = "Unknown"
    vol_ratio_1d_90d: float = 0.0
    vol_ratio_7d_90d: float = 0.0
    ret_1d: float = 0.0
    ret_7d: float = 0.0
    sma_20d: Optional[float] = None
    sma_50d: Optional[float] = None
    sma_200d: Optional[float] = None
    
    def __post_init__(self):
        """Validate data after initialization"""
        if self.price <= 0:
            raise ValueError(f"Invalid price for {self.ticker}: {self.price}")
        if not -100 <= self.from_high_pct <= 0:
            raise ValueError(f"Invalid from_high_pct for {self.ticker}: {self.from_high_pct}")

# ============================================
# PERFORMANCE MONITORING
# ============================================

class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    @staticmethod
    def timer(func: Callable) -> Callable:
        """Decorator to time function execution"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_time = time.perf_counter() - start_time
            
            if elapsed_time > 1.0:
                logger.warning(
                    f"{func.__name__} took {elapsed_time:.2f}s"
                )
            else:
                logger.debug(
                    f"{func.__name__} completed in {elapsed_time:.3f}s"
                )
            
            return result
        return wrapper
    
    @staticmethod
    def memory_usage() -> float:
        """Get current memory usage in MB"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

# ============================================
# DATA VALIDATION
# ============================================

class DataValidator:
    """Validate and sanitize input data"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        Validate dataframe has required columns and data
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if df is None or df.empty:
            return False, "DataFrame is empty"
        
        required_columns = [
            'ticker', 'company_name', 'price', 
            'from_low_pct', 'from_high_pct',
            'vol_ratio_30d_90d', 'ret_30d'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
        
        # Check for minimum valid rows
        valid_rows = df['price'].notna() & (df['price'] > 0)
        if valid_rows.sum() < 10:
            return False, "Insufficient valid data rows (minimum 10 required)"
        
        return True, None
    
    @staticmethod
    def clean_numeric_value(value: Any) -> Optional[float]:
        """Clean and convert Indian number format to float"""
        if pd.isna(value) or value == '':
            return None
        
        try:
            # Convert to string
            cleaned = str(value)
            
            # Remove currency symbols and special characters
            for char in ['₹', '$', '%', ',']:
                cleaned = cleaned.replace(char, '')
            
            # Remove extra spaces
            cleaned = cleaned.strip()
            
            # Handle special cases
            if cleaned in ['', '-', 'N/A', 'n/a', '#N/A', 'nan', 'None']:
                return None
            
            # Convert to float
            return float(cleaned)
            
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to convert '{value}': {e}")
            return None

# ============================================
# DATA LOADER
# ============================================

class DataLoader:
    """Handle all data loading operations"""
    
    def __init__(self, cache_ttl: int = CONFIG.CACHE_TTL):
        self.cache_ttl = cache_ttl
        self.logger = LoggerFactory.create_logger(self.__class__.__name__)
    
    @st.cache_data(ttl=CONFIG.CACHE_TTL)
    def load_from_google_sheets(sheet_url: str, gid: str) -> pd.DataFrame:
        """
        Load data from Google Sheets with caching
        
        Args:
            sheet_url: Google Sheets URL
            gid: Sheet ID
            
        Returns:
            Raw dataframe
        """
        try:
            # Construct CSV export URL
            base_url = sheet_url.split('/edit')[0]
            csv_url = f"{base_url}/export?format=csv&gid={gid}"
            
            logger.info(f"Loading data from Google Sheets")
            
            # Load with pandas
            df = pd.read_csv(csv_url)
            
            if df.empty:
                raise ValueError("Loaded empty dataframe")
            
            logger.info(f"Successfully loaded {len(df):,} rows")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

# ============================================
# DATA PROCESSOR
# ============================================

class DataProcessor:
    """Process and clean raw data"""
    
    def __init__(self):
        self.logger = LoggerFactory.create_logger(self.__class__.__name__)
        self.validator = DataValidator()
    
    @PerformanceMonitor.timer
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete data processing pipeline
        
        Args:
            df: Raw dataframe
            
        Returns:
            Processed dataframe ready for ranking
        """
        if df.empty:
            return pd.DataFrame()
        
        # Validate input
        is_valid, error_msg = self.validator.validate_dataframe(df)
        if not is_valid:
            raise ValueError(f"Data validation failed: {error_msg}")
        
        # Create copy to avoid modifying original
        df = df.copy()
        
        # Process in steps
        df = self._clean_numeric_columns(df)
        df = self._clean_categorical_columns(df)
        df = self._fix_volume_ratios(df)
        df = self._remove_invalid_rows(df)
        df = self._add_derived_columns(df)
        
        self.logger.info(f"Processed {len(df):,} valid stocks")
        return df
    
    def _clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean all numeric columns"""
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
                df[col] = df[col].apply(self.validator.clean_numeric_value)
        
        return df
    
    def _clean_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean categorical columns"""
        categorical_columns = ['ticker', 'company_name', 'category', 'sector']
        
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(['nan', 'None', '', 'N/A'], 'Unknown')
        
        return df
    
    def _fix_volume_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert volume ratio percentages to multipliers"""
        volume_ratio_columns = [
            'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
            'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
            'vol_ratio_90d_180d'
        ]
        
        for col in volume_ratio_columns:
            if col in df.columns:
                # Convert from percentage change to multiplier
                # -56.61% becomes 0.4339
                df[col] = (100 + df[col]) / 100
        
        return df
    
    def _remove_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with invalid data"""
        initial_count = len(df)
        
        # Must have valid price
        df = df[df['price'].notna() & (df['price'] > 0)]
        
        # Must have position data
        df = df[df['from_low_pct'].notna()]
        df = df[df['from_high_pct'].notna()]
        
        final_count = len(df)
        if initial_count != final_count:
            self.logger.info(
                f"Removed {initial_count - final_count} invalid rows"
            )
        
        return df
    
    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add useful derived columns"""
        # Position in 52-week range (0-100)
        df['range_position'] = (
            df['from_low_pct'] / 
            (df['from_low_pct'] - df['from_high_pct'])
        ) * 100
        
        # Momentum acceleration
        df['momentum_acceleration'] = (
            df['ret_7d'] - (df['ret_30d'] / 4.3)
        )
        
        return df

# ============================================
# RANKING ENGINE
# ============================================

class RankingEngine:
    """Core ranking algorithm implementation"""
    
    def __init__(self, config: SystemConfig = CONFIG):
        self.config = config
        self.logger = LoggerFactory.create_logger(self.__class__.__name__)
    
    @PerformanceMonitor.timer
    def calculate_rankings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all rankings using pure relative strength
        
        Args:
            df: Processed dataframe
            
        Returns:
            Dataframe with rankings and scores
        """
        if df.empty:
            return df
        
        # Calculate percentile ranks
        df = self._calculate_percentile_ranks(df)
        
        # Calculate component scores
        df = self._calculate_component_scores(df)
        
        # Calculate master score
        df = self._calculate_master_score(df)
        
        # Add final rankings
        df = self._add_final_rankings(df)
        
        # Detect patterns
        df = self._detect_patterns(df)
        
        return df
    
    def _calculate_percentile_ranks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate percentile ranks for all metrics"""
        # Position ranks
        df['rank_from_low'] = df['from_low_pct'].rank(pct=True) * 100
        df['rank_from_high'] = (100 + df['from_high_pct']).rank(pct=True) * 100
        
        # Volume ranks
        df['rank_vol_1d'] = df['vol_ratio_1d_90d'].rank(pct=True) * 100
        df['rank_vol_7d'] = df['vol_ratio_7d_90d'].rank(pct=True) * 100
        df['rank_vol_30d'] = df['vol_ratio_30d_90d'].rank(pct=True) * 100
        
        # Momentum ranks
        df['rank_ret_1d'] = df['ret_1d'].rank(pct=True) * 100
        df['rank_ret_7d'] = df['ret_7d'].rank(pct=True) * 100
        df['rank_ret_30d'] = df['ret_30d'].rank(pct=True) * 100
        
        return df
    
    def _calculate_component_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate component scores"""
        # Position score (0-100)
        df['position_score'] = (
            df['rank_from_low'] * 0.6 +
            df['rank_from_high'] * 0.4
        )
        
        # Volume score (0-100)
        df['volume_score'] = (
            df['rank_vol_30d'] * 0.5 +
            df['rank_vol_7d'] * 0.3 +
            df['rank_vol_1d'] * 0.2
        )
        
        # Momentum score (0-100)
        df['momentum_score'] = df['rank_ret_30d']
        
        return df
    
    def _calculate_master_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate master score using configured weights"""
        df['master_score'] = (
            df['position_score'] * self.config.POSITION_WEIGHT +
            df['volume_score'] * self.config.VOLUME_WEIGHT +
            df['momentum_score'] * self.config.MOMENTUM_WEIGHT
        )
        
        return df
    
    def _add_final_rankings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add final rankings"""
        df['rank'] = df['master_score'].rank(
            ascending=False, 
            method='min'
        ).astype(int)
        
        df['percentile'] = df['master_score'].rank(pct=True) * 100
        
        return df
    
    def _detect_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect trading patterns"""
        patterns = []
        
        for idx, row in df.iterrows():
            stock_patterns = []
            
            # Breakout imminent
            if row['rank_from_high'] > 90 and row['rank_vol_7d'] > 70:
                stock_patterns.append("🚀 BREAKOUT")
            
            # Accumulation
            if row['rank_vol_30d'] > 80 and row['rank_from_low'] < 50:
                stock_patterns.append("🏦 ACCUMULATION")
            
            # Momentum leader
            if row['percentile'] > 95:
                stock_patterns.append("👑 LEADER")
            
            # Volume surge
            if row['rank_vol_1d'] > 95:
                stock_patterns.append("🔥 VOLUME")
            
            patterns.append(", ".join(stock_patterns) if stock_patterns else "")
        
        df['patterns'] = patterns
        return df

# ============================================
# MARKET ANALYZER
# ============================================

class MarketAnalyzer:
    """Analyze overall market conditions"""
    
    @staticmethod
    def detect_market_regime(df: pd.DataFrame) -> MarketRegime:
        """Detect current market regime"""
        if df.empty:
            return MarketRegime.UNKNOWN
        
        # Calculate market breadth
        advancing = (df['ret_1d'] > 0).sum()
        total = len(df)
        breadth = advancing / total if total > 0 else 0.5
        
        # Calculate average returns
        median_return_30d = df['ret_30d'].median()
        
        # Determine regime
        if breadth > 0.65 and median_return_30d > 10:
            return MarketRegime.BULL
        elif breadth < 0.35 and median_return_30d < -10:
            return MarketRegime.BEAR
        elif df['ret_1d'].std() > df['ret_30d'].std() / 4.3:
            return MarketRegime.VOLATILE
        else:
            return MarketRegime.SIDEWAYS
    
    @staticmethod
    def calculate_market_stats(df: pd.DataFrame) -> Dict[str, float]:
        """Calculate market statistics"""
        if df.empty:
            return {}
        
        return {
            'total_stocks': len(df),
            'average_score': df['master_score'].mean(),
            'median_return_30d': df['ret_30d'].median(),
            'high_volume_pct': (df['rank_vol_30d'] > 70).mean() * 100,
            'near_highs_pct': (df['rank_from_high'] > 80).mean() * 100,
            'strong_trend_pct': (df['rank_ret_30d'] > 70).mean() * 100
        }

# ============================================
# VISUALIZATION ENGINE
# ============================================

class VisualizationEngine:
    """Create all visualizations"""
    
    @staticmethod
    def create_score_distribution(df: pd.DataFrame) -> go.Figure:
        """Create score distribution chart"""
        fig = go.Figure()
        
        scores = ['position_score', 'volume_score', 'momentum_score']
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for score, color in zip(scores, colors):
            if score in df.columns:
                fig.add_trace(go.Box(
                    y=df[score],
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
        """Create top stocks bar chart"""
        top_df = df.nlargest(n, 'master_score')
        
        fig = go.Figure()
        
        # Add component scores
        fig.add_trace(go.Bar(
            name='Position',
            y=top_df['ticker'],
            x=top_df['position_score'] * CONFIG.POSITION_WEIGHT,
            orientation='h',
            marker_color='#3498db'
        ))
        
        fig.add_trace(go.Bar(
            name='Volume',
            y=top_df['ticker'],
            x=top_df['volume_score'] * CONFIG.VOLUME_WEIGHT,
            orientation='h',
            marker_color='#e74c3c'
        ))
        
        fig.add_trace(go.Bar(
            name='Momentum',
            y=top_df['ticker'],
            x=top_df['momentum_score'] * CONFIG.MOMENTUM_WEIGHT,
            orientation='h',
            marker_color='#2ecc71'
        ))
        
        fig.update_layout(
            title=f"Top {n} Stocks - Score Breakdown",
            xaxis_title="Weighted Score",
            barmode='stack',
            template='plotly_white',
            height=max(400, n * 25),
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_scatter_matrix(df: pd.DataFrame) -> go.Figure:
        """Create scatter plot of position vs volume"""
        # Limit to top 200 for performance
        plot_df = df.nlargest(min(200, len(df)), 'master_score')
        
        fig = px.scatter(
            plot_df,
            x='position_score',
            y='volume_score',
            size='master_score',
            color='momentum_score',
            hover_data=['ticker', 'company_name', 'patterns'],
            title='Position vs Volume Analysis',
            labels={
                'position_score': 'Position Score',
                'volume_score': 'Volume Score',
                'momentum_score': 'Momentum Score'
            },
            color_continuous_scale='viridis',
            template='plotly_white'
        )
        
        fig.update_layout(height=500)
        
        return fig

# ============================================
# EXPORT ENGINE
# ============================================

class ExportEngine:
    """Handle all data export operations"""
    
    @staticmethod
    def create_excel_report(df: pd.DataFrame) -> BytesIO:
        """Create comprehensive Excel report"""
        output = BytesIO()
        
        try:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Get workbook
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
                
                # Sheet 1: Top 100 Stocks
                top_100 = df.nlargest(100, 'master_score')
                export_columns = [
                    'rank', 'ticker', 'company_name', 'master_score',
                    'position_score', 'volume_score', 'momentum_score',
                    'price', 'from_low_pct', 'from_high_pct',
                    'ret_30d', 'vol_ratio_30d_90d', 'patterns',
                    'category', 'sector'
                ]
                
                # Filter to existing columns
                export_columns = [col for col in export_columns if col in top_100.columns]
                top_100[export_columns].to_excel(
                    writer, 
                    sheet_name='Top 100', 
                    index=False
                )
                
                # Sheet 2: All Stocks
                all_stocks_columns = [
                    'rank', 'ticker', 'company_name', 'master_score',
                    'category', 'sector'
                ]
                all_stocks_columns = [col for col in all_stocks_columns if col in df.columns]
                df[all_stocks_columns].to_excel(
                    writer,
                    sheet_name='All Stocks',
                    index=False
                )
                
                # Sheet 3: Market Analysis
                market_stats = MarketAnalyzer.calculate_market_stats(df)
                stats_df = pd.DataFrame([market_stats]).T
                stats_df.columns = ['Value']
                stats_df.to_excel(writer, sheet_name='Market Stats')
                
        except Exception as e:
            logger.error(f"Error creating Excel report: {e}")
            raise
        
        output.seek(0)
        return output

# ============================================
# USER INTERFACE
# ============================================

class UserInterface:
    """Main user interface handler"""
    
    def __init__(self):
        self.config = CONFIG
        self.data_loader = DataLoader()
        self.data_processor = DataProcessor()
        self.ranking_engine = RankingEngine()
        self.viz_engine = VisualizationEngine()
        self.export_engine = ExportEngine()
    
    def render(self):
        """Render the main application"""
        self._setup_page()
        self._render_header()
        
        # Sidebar
        with st.sidebar:
            sheet_url, gid = self._render_data_source_config()
            filters = self._render_filters()
        
        # Main content
        try:
            # Load and process data
            df = self._load_and_process_data(sheet_url, gid)
            
            if df.empty:
                st.error("No data available. Please check your data source.")
                return
            
            # Apply filters
            filtered_df = self._apply_filters(df, filters)
            
            # Display results
            self._render_main_content(filtered_df)
            
        except Exception as e:
            logger.error(f"Application error: {e}")
            st.error(f"An error occurred: {str(e)}")
    
    def _setup_page(self):
        """Configure Streamlit page"""
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
        </style>
        """, unsafe_allow_html=True)
    
    def _render_header(self):
        """Render application header"""
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; background: linear-gradient(90deg, #3498db, #2ecc71); color: white; border-radius: 10px;">
            <h1 style="margin: 0;">📊 Wave Detection Ultimate</h1>
            <p style="margin: 0;">Professional Stock Ranking System</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
    
    def _render_data_source_config(self) -> Tuple[str, str]:
        """Render data source configuration"""
        st.markdown("### 📁 Data Source")
        
        sheet_url = st.text_input(
            "Google Sheets URL",
            value=self.config.DEFAULT_SHEET_URL,
            help="Enter the Google Sheets URL"
        )
        
        gid = st.text_input(
            "Sheet ID (GID)",
            value=self.config.DEFAULT_GID,
            help="Enter the sheet ID"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", help="Clear cache and reload"):
                st.cache_data.clear()
                st.rerun()
        
        return sheet_url, gid
    
    def _render_filters(self) -> Dict[str, Any]:
        """Render filter controls"""
        st.markdown("---")
        st.markdown("### 🔍 Filters")
        
        filters = {}
        
        # Score filter
        filters['min_score'] = st.slider(
            "Minimum Score",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            help="Filter by minimum master score"
        )
        
        # Top N filter
        filters['top_n'] = st.selectbox(
            "Show Top",
            options=self.config.AVAILABLE_TOP_N,
            index=2,
            help="Number of top stocks to display"
        )
        
        return filters
    
    @PerformanceMonitor.timer
    def _load_and_process_data(self, sheet_url: str, gid: str) -> pd.DataFrame:
        """Load and process data with proper error handling"""
        with st.spinner("Loading data..."):
            raw_df = self.data_loader.load_from_google_sheets(sheet_url, gid)
        
        with st.spinner(f"Processing {len(raw_df):,} stocks..."):
            processed_df = self.data_processor.process_data(raw_df)
        
        with st.spinner("Calculating rankings..."):
            ranked_df = self.ranking_engine.calculate_rankings(processed_df)
        
        return ranked_df
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to dataframe"""
        filtered_df = df.copy()
        
        # Apply score filter
        if 'min_score' in filters:
            filtered_df = filtered_df[filtered_df['master_score'] >= filters['min_score']]
        
        # Sort by rank
        filtered_df = filtered_df.sort_values('rank')
        
        return filtered_df
    
    def _render_main_content(self, df: pd.DataFrame):
        """Render main content area"""
        # Summary metrics
        self._render_summary_metrics(df)
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🏆 Rankings", "📊 Analysis", "📈 Charts", "📥 Export"
        ])
        
        with tab1:
            self._render_rankings_tab(df)
        
        with tab2:
            self._render_analysis_tab(df)
        
        with tab3:
            self._render_charts_tab(df)
        
        with tab4:
            self._render_export_tab(df)
    
    def _render_summary_metrics(self, df: pd.DataFrame):
        """Render summary metrics"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        market_stats = MarketAnalyzer.calculate_market_stats(df)
        market_regime = MarketAnalyzer.detect_market_regime(df)
        
        with col1:
            st.metric("Total Stocks", f"{market_stats.get('total_stocks', 0):,}")
        
        with col2:
            st.metric("Avg Score", f"{market_stats.get('average_score', 0):.1f}")
        
        with col3:
            st.metric("Market Regime", market_regime.name)
        
        with col4:
            st.metric("Near Highs", f"{market_stats.get('near_highs_pct', 0):.1f}%")
        
        with col5:
            st.metric("High Volume", f"{market_stats.get('high_volume_pct', 0):.1f}%")
    
    def _render_rankings_tab(self, df: pd.DataFrame):
        """Render rankings tab"""
        st.markdown("### Top Ranked Stocks")
        
        # Get top N
        top_n = st.session_state.get('top_n', self.config.DEFAULT_TOP_N)
        display_df = df.head(top_n).copy()
        
        # Format display columns
        display_columns = [
            'rank', 'ticker', 'company_name', 'master_score',
            'position_score', 'volume_score', 'momentum_score',
            'patterns', 'price', 'from_low_pct', 'ret_30d'
        ]
        
        # Keep only existing columns
        display_columns = [col for col in display_columns if col in display_df.columns]
        
        # Format numeric columns
        format_dict = {
            'master_score': '{:.1f}',
            'position_score': '{:.1f}',
            'volume_score': '{:.1f}',
            'momentum_score': '{:.1f}',
            'price': '₹{:.2f}',
            'from_low_pct': '{:.1f}%',
            'ret_30d': '{:.1f}%'
        }
        
        # Apply formatting
        for col, fmt in format_dict.items():
            if col in display_df.columns:
                if '%' in fmt:
                    display_df[col] = display_df[col].apply(lambda x: fmt.format(x))
                elif '₹' in fmt:
                    display_df[col] = display_df[col].apply(lambda x: fmt.format(x))
                else:
                    display_df[col] = display_df[col].round(1)
        
        # Display dataframe
        st.dataframe(
            display_df[display_columns],
            use_container_width=True,
            height=600
        )
    
    def _render_analysis_tab(self, df: pd.DataFrame):
        """Render analysis tab"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Score Distribution")
            fig = self.viz_engine.create_score_distribution(df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Market Statistics")
            stats = MarketAnalyzer.calculate_market_stats(df)
            stats_df = pd.DataFrame([stats]).T
            stats_df.columns = ['Value']
            stats_df.index = stats_df.index.str.replace('_', ' ').str.title()
            st.dataframe(stats_df, use_container_width=True)
    
    def _render_charts_tab(self, df: pd.DataFrame):
        """Render charts tab"""
        # Top stocks chart
        st.markdown("#### Top Stocks Breakdown")
        n_stocks = st.slider("Number of stocks", 10, 50, 20)
        fig1 = self.viz_engine.create_top_stocks_chart(df, n_stocks)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Scatter plot
        st.markdown("#### Position vs Volume Analysis")
        fig2 = self.viz_engine.create_scatter_matrix(df)
        st.plotly_chart(fig2, use_container_width=True)
    
    def _render_export_tab(self, df: pd.DataFrame):
        """Render export tab"""
        st.markdown("### Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Excel Report")
            if st.button("Generate Excel Report"):
                with st.spinner("Generating report..."):
                    excel_file = self.export_engine.create_excel_report(df)
                    
                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_file,
                        file_name=f"wave_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        with col2:
            st.markdown("#### CSV Export")
            if st.button("Generate CSV"):
                csv_data = df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"wave_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# ============================================
# MAIN APPLICATION ENTRY POINT
# ============================================

def main():
    """Main application entry point"""
    try:
        # Create and run UI
        ui = UserInterface()
        ui.render()
        
    except Exception as e:
        logger.critical(f"Critical application error: {e}")
        st.error("A critical error occurred. Please refresh the page.")
        
        # Show debug info in development
        if st.secrets.get("debug", False):
            st.exception(e)

if __name__ == "__main__":
    main()
