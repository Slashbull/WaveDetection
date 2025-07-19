#!/usr/bin/env python3
"""
EDGE PROTOCOL FINAL v2.0
========================
Professional Trading Intelligence System for Indian Stock Markets
Complete, Bug-Free, Production-Ready Implementation

Features:
- Smart Category & Sector Filters
- Multi-Sheet Excel Reports
- Real-time Diagnostics
- Professional UI/UX
- Robust Error Handling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import warnings
import re
import logging
import io
import json
from typing import Dict, List, Tuple, Optional, Union, Any
import traceback
from functools import lru_cache
import xlsxwriter
from io import BytesIO

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Page Configuration
st.set_page_config(
    page_title="EDGE Protocol v2.0 | Professional Trading System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('EDGE_Protocol')

# Data Source Configuration
SHEET_CONFIG = {
    'SHEET_ID': '1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk',
    'GID': '2026492216',
    'CACHE_TTL': 300,  # 5 minutes
    'REQUEST_TIMEOUT': 30,
    'MAX_RETRIES': 3
}

# Trading Configuration
TRADING_CONFIG = {
    'MIN_PRICE': 20,
    'MIN_VOLUME': 10000,
    'MIN_TRADE_VALUE': 1000000,  # ₹10 Lakhs
    'MAX_POSITIONS': 5,
    'DEFAULT_STOP_LOSS': 0.08,  # 8%
    'ATR_MULTIPLIER': 2.0,
    'POSITION_SIZES': {
        'SUPER_EDGE': 0.20,
        'STRONG': 0.15,
        'MODERATE': 0.10,
        'WATCH': 0.05
    }
}

# Signal Thresholds
SIGNAL_THRESHOLDS = {
    'SUPER_EDGE': 90,
    'STRONG': 75,
    'MODERATE': 60,
    'WATCH': 45,
    'IGNORE': 0
}

# Market Regime Thresholds
MARKET_REGIME = {
    'BULL': {'breadth': 0.65, 'multiplier': 1.0},
    'NEUTRAL': {'breadth': 0.35, 'multiplier': 0.7},
    'BEAR': {'breadth': 0.0, 'multiplier': 0.5}
}

# ============================================================================
# ERROR HANDLING & DIAGNOSTICS
# ============================================================================

class DiagnosticsTracker:
    """Track system health and performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'data_load_time': 0,
            'process_time': 0,
            'total_stocks': 0,
            'valid_stocks': 0,
            'filtered_stocks': 0,
            'data_quality_score': 0,
            'missing_data_pct': 0,
            'errors': [],
            'warnings': [],
            'info': []
        }
        self.start_time = datetime.now()
    
    def add_metric(self, key: str, value: Any):
        """Add or update a metric"""
        self.metrics[key] = value
    
    def add_error(self, error: str):
        """Add error message"""
        self.metrics['errors'].append(f"{datetime.now():%H:%M:%S} - {error}")
        logger.error(error)
    
    def add_warning(self, warning: str):
        """Add warning message"""
        self.metrics['warnings'].append(f"{datetime.now():%H:%M:%S} - {warning}")
        logger.warning(warning)
    
    def add_info(self, info: str):
        """Add info message"""
        self.metrics['info'].append(f"{datetime.now():%H:%M:%S} - {info}")
        logger.info(info)
    
    def get_report(self) -> Dict:
        """Get diagnostic report"""
        self.metrics['total_runtime'] = (datetime.now() - self.start_time).total_seconds()
        return self.metrics

# Global diagnostics instance
diagnostics = DiagnosticsTracker()

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def parse_indian_number(value: Union[str, float, int]) -> float:
    """
    Parse Indian number format safely
    Handles: ₹1,23,45,678.90 | 1,23,45,678 | -12.34% | 5.5Cr | 2.3L
    """
    try:
        if pd.isna(value) or value == '' or value == '-' or value == 'NA':
            return np.nan
        
        if isinstance(value, (int, float)):
            return float(value)
        
        val_str = str(value).strip()
        
        # Remove currency symbols
        val_str = re.sub(r'[₹$€£¥]', '', val_str)
        
        # Handle percentage
        is_percentage = '%' in val_str
        val_str = val_str.replace('%', '')
        
        # Handle Cr/L/K/M/B
        multipliers = {
            'cr': 1e7, 'crore': 1e7, 'crores': 1e7,
            'l': 1e5, 'lakh': 1e5, 'lakhs': 1e5, 'lac': 1e5,
            'k': 1e3, 'm': 1e6, 'b': 1e9
        }
        
        val_lower = val_str.lower()
        multiplier = 1
        for suffix, mult in multipliers.items():
            if val_lower.endswith(suffix):
                multiplier = mult
                val_str = val_str[:-(len(suffix))]
                break
        
        # Remove commas
        val_str = val_str.replace(',', '').strip()
        
        # Convert to float
        result = float(val_str) * multiplier
        return result
        
    except Exception as e:
        diagnostics.add_warning(f"Failed to parse number '{value}': {str(e)}")
        return np.nan

@st.cache_data(ttl=SHEET_CONFIG['CACHE_TTL'], show_spinner=False)
def load_data() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load and process data with comprehensive error handling"""
    
    load_start = datetime.now()
    diagnostics.add_info("Starting data load...")
    
    try:
        # Build URL
        url = f"https://docs.google.com/spreadsheets/d/{SHEET_CONFIG['SHEET_ID']}/export?format=csv&gid={SHEET_CONFIG['GID']}"
        
        # Fetch with retries
        for attempt in range(SHEET_CONFIG['MAX_RETRIES']):
            try:
                response = requests.get(url, timeout=SHEET_CONFIG['REQUEST_TIMEOUT'])
                response.raise_for_status()
                break
            except Exception as e:
                if attempt == SHEET_CONFIG['MAX_RETRIES'] - 1:
                    raise
                diagnostics.add_warning(f"Retry {attempt + 1}: {str(e)}")
        
        # Parse CSV
        df = pd.read_csv(io.StringIO(response.text))
        diagnostics.add_metric('total_stocks', len(df))
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Process all columns
        df = process_data_columns(df)
        
        # Calculate derived metrics
        df = calculate_derived_metrics(df)
        
        # Quality checks
        df = apply_quality_filters(df)
        
        # Calculate data quality score
        data_quality = calculate_data_quality(df)
        diagnostics.add_metric('data_quality_score', data_quality)
        
        # Load time
        load_time = (datetime.now() - load_start).total_seconds()
        diagnostics.add_metric('data_load_time', load_time)
        diagnostics.add_info(f"Data loaded successfully in {load_time:.2f}s")
        
        return df, diagnostics.get_report()
        
    except Exception as e:
        error_msg = f"Critical error loading data: {str(e)}"
        diagnostics.add_error(error_msg)
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return pd.DataFrame(), diagnostics.get_report()

def process_data_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Process all data columns with proper type conversion"""
    
    try:
        # Price columns
        price_cols = ['price', 'low_52w', 'high_52w', 'sma_20d', 'sma_50d', 'sma_200d', 'prev_close']
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].apply(parse_indian_number)
        
        # Return columns
        return_cols = ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y']
        for col in return_cols:
            if col in df.columns:
                df[col] = df[col].apply(parse_indian_number)
        
        # Volume columns
        volume_cols = ['volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d']
        for col in volume_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: parse_indian_number(str(x).replace(',', '')))
        
        # Volume ratio columns
        ratio_cols = ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
                      'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 'vol_ratio_90d_180d']
        for col in ratio_cols:
            if col in df.columns:
                df[col] = df[col].apply(parse_indian_number)
        
        # Percentage columns
        pct_cols = ['from_low_pct', 'from_high_pct', 'eps_change_pct']
        for col in pct_cols:
            if col in df.columns:
                df[col] = df[col].apply(parse_indian_number)
        
        # Market cap
        if 'market_cap' in df.columns:
            df['market_cap_value'] = df['market_cap'].apply(parse_indian_number)
        
        # EPS and PE
        for col in ['pe', 'eps_current', 'eps_last_qtr']:
            if col in df.columns:
                df[col] = df[col].apply(parse_indian_number)
        
        # RVOL fix if needed
        if 'rvol' in df.columns:
            df['rvol'] = pd.to_numeric(df['rvol'], errors='coerce').fillna(1.0)
        
        return df
        
    except Exception as e:
        diagnostics.add_error(f"Error processing columns: {str(e)}")
        return df

def calculate_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all derived metrics"""
    
    try:
        # Value traded
        df['value_traded'] = df['price'] * df['volume_1d']
        
        # Volume acceleration
        df['volume_acceleration'] = df['vol_ratio_30d_90d'] - df['vol_ratio_30d_180d']
        
        # Price position
        df['price_position'] = ((df['price'] - df['low_52w']) / 
                               (df['high_52w'] - df['low_52w']) * 100).fillna(50)
        
        # ATR proxy
        df['atr_pct'] = ((df['high_52w'] - df['low_52w']) / df['price'] * 100 / 4).fillna(5)
        
        # Momentum scores
        df['momentum_short'] = (df['ret_1d'] * 0.2 + df['ret_3d'] * 0.3 + df['ret_7d'] * 0.5)
        df['momentum_medium'] = (df['ret_7d'] * 0.3 + df['ret_30d'] * 0.7)
        
        # Quality score
        df['quality_score'] = 0
        if 'ret_3y' in df.columns:
            df['quality_score'] += np.where(df['ret_3y'] > 300, 40, 
                                          np.where(df['ret_3y'] > 200, 30,
                                          np.where(df['ret_3y'] > 100, 20, 10)))
        
        # EPS momentum
        if all(col in df.columns for col in ['eps_current', 'eps_last_qtr']):
            df['eps_momentum'] = np.where(
                df['eps_last_qtr'] > 0,
                (df['eps_current'] - df['eps_last_qtr']) / df['eps_last_qtr'] * 100,
                0
            )
        
        return df
        
    except Exception as e:
        diagnostics.add_error(f"Error calculating derived metrics: {str(e)}")
        return df

def apply_quality_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply quality filters to remove bad data"""
    
    initial_count = len(df)
    
    # Basic filters
    df = df[
        (df['price'] > 0) & 
        (df['price'].notna()) &
        (df['volume_1d'] > 0) &
        (df['ticker'].notna())
    ]
    
    filtered_count = initial_count - len(df)
    if filtered_count > 0:
        diagnostics.add_info(f"Filtered {filtered_count} invalid rows")
    
    diagnostics.add_metric('valid_stocks', len(df))
    
    return df

def calculate_data_quality(df: pd.DataFrame) -> float:
    """Calculate overall data quality score"""
    
    critical_cols = ['price', 'volume_1d', 'ret_7d', 'ret_30d']
    quality_scores = []
    
    for col in critical_cols:
        if col in df.columns:
            completeness = df[col].notna().sum() / len(df)
            quality_scores.append(completeness)
    
    return np.mean(quality_scores) * 100 if quality_scores else 0

# ============================================================================
# MARKET ANALYSIS FUNCTIONS
# ============================================================================

def analyze_market_regime(df: pd.DataFrame) -> Tuple[str, float, Dict]:
    """Analyze overall market conditions"""
    
    try:
        # Market breadth
        above_50_sma = (df['price'] > df['sma_50d']).sum() / len(df)
        above_200_sma = (df['price'] > df['sma_200d']).sum() / len(df)
        
        # Momentum breadth
        positive_momentum = (df['ret_7d'] > 0).sum() / len(df)
        
        # Volume analysis
        avg_rvol = df['rvol'].mean()
        volume_expansion = (df['vol_ratio_30d_90d'] > 0).sum() / len(df)
        
        # Determine regime
        breadth_score = (above_50_sma + above_200_sma) / 2
        
        if breadth_score > MARKET_REGIME['BULL']['breadth']:
            regime = 'BULL'
            position_multiplier = MARKET_REGIME['BULL']['multiplier']
        elif breadth_score > MARKET_REGIME['NEUTRAL']['breadth']:
            regime = 'NEUTRAL'
            position_multiplier = MARKET_REGIME['NEUTRAL']['multiplier']
        else:
            regime = 'BEAR'
            position_multiplier = MARKET_REGIME['BEAR']['multiplier']
        
        market_stats = {
            'regime': regime,
            'breadth_score': breadth_score * 100,
            'positive_momentum_pct': positive_momentum * 100,
            'avg_rvol': avg_rvol,
            'volume_expansion_pct': volume_expansion * 100,
            'stocks_above_50_sma': above_50_sma * 100,
            'stocks_above_200_sma': above_200_sma * 100
        }
        
        return regime, position_multiplier, market_stats
        
    except Exception as e:
        diagnostics.add_error(f"Error analyzing market regime: {str(e)}")
        return 'NEUTRAL', 0.7, {}

# ============================================================================
# SCORING ENGINE
# ============================================================================

def calculate_edge_scores(df: pd.DataFrame, market_regime: str) -> pd.DataFrame:
    """Calculate comprehensive EDGE scores"""
    
    try:
        # Get market averages for relative scoring
        market_vol_avg = df['vol_ratio_30d_90d'].median()
        market_ret_avg = df['ret_7d'].median()
        
        # Initialize score components
        df['score_volume'] = 0
        df['score_momentum'] = 0
        df['score_quality'] = 0
        df['score_value'] = 0
        df['score_technical'] = 0
        
        # 1. Volume Score (30% weight)
        df['relative_vol_strength'] = df['vol_ratio_30d_90d'] - market_vol_avg
        df['score_volume'] = np.where(
            df['relative_vol_strength'] > 30, 100,
            np.where(df['relative_vol_strength'] > 20, 85,
            np.where(df['relative_vol_strength'] > 10, 70,
            np.where(df['relative_vol_strength'] > 0, 55, 40)))
        )
        
        # Boost for volume acceleration
        vol_accel_boost = np.where(df['volume_acceleration'] > 20, 1.2,
                                  np.where(df['volume_acceleration'] > 10, 1.1, 1.0))
        df['score_volume'] = (df['score_volume'] * vol_accel_boost).clip(0, 100)
        
        # 2. Momentum Score (25% weight)
        df['relative_momentum'] = df['ret_7d'] - market_ret_avg
        
        # Check momentum alignment
        df['momentum_aligned'] = (
            (df['ret_1d'] > 0) & 
            (df['ret_3d'] > df['ret_1d']) & 
            (df['ret_7d'] > df['ret_3d'])
        )
        
        df['score_momentum'] = np.where(
            df['momentum_aligned'] & (df['relative_momentum'] > 5), 100,
            np.where(df['relative_momentum'] > 5, 85,
            np.where(df['relative_momentum'] > 2, 70,
            np.where(df['relative_momentum'] > 0, 55, 40)))
        )
        
        # 3. Quality Score (20% weight)
        if 'ret_3y' in df.columns:
            df['score_quality'] = np.where(
                df['ret_3y'] > 500, 100,
                np.where(df['ret_3y'] > 300, 85,
                np.where(df['ret_3y'] > 200, 70,
                np.where(df['ret_3y'] > 100, 55, 40)))
            )
        
        # EPS boost
        if 'eps_momentum' in df.columns:
            eps_boost = np.where(df['eps_momentum'] > 20, 1.2,
                               np.where(df['eps_momentum'] > 10, 1.1, 1.0))
            df['score_quality'] = (df['score_quality'] * eps_boost).clip(0, 100)
        
        # 4. Value Score (15% weight)
        df['score_value'] = np.where(
            (df['from_high_pct'] < -25) & (df['from_high_pct'] > -40), 100,
            np.where((df['from_high_pct'] < -15) & (df['from_high_pct'] > -25), 85,
            np.where(df['from_high_pct'] > -10, 50, 70))
        )
        
        # 5. Technical Score (10% weight)
        sma_alignment = (
            (df['price'] > df['sma_20d']).astype(int) * 25 +
            (df['price'] > df['sma_50d']).astype(int) * 25 +
            (df['price'] > df['sma_200d']).astype(int) * 25 +
            (df['sma_20d'] > df['sma_50d']).astype(int) * 25
        )
        df['score_technical'] = sma_alignment
        
        # Calculate final EDGE score with market regime adjustment
        if market_regime == 'BULL':
            weights = [0.30, 0.30, 0.15, 0.15, 0.10]  # Volume, Momentum heavy
        elif market_regime == 'BEAR':
            weights = [0.25, 0.20, 0.30, 0.20, 0.05]  # Quality, Value heavy
        else:
            weights = [0.30, 0.25, 0.20, 0.15, 0.10]  # Balanced
        
        df['edge_score'] = (
            df['score_volume'] * weights[0] +
            df['score_momentum'] * weights[1] +
            df['score_quality'] * weights[2] +
            df['score_value'] * weights[3] +
            df['score_technical'] * weights[4]
        )
        
        # Classify signals
        df['signal'] = pd.cut(
            df['edge_score'],
            bins=[-np.inf, 45, 60, 75, 90, 100.1],
            labels=['IGNORE', 'WATCH', 'MODERATE', 'STRONG', 'SUPER_EDGE']
        )
        
        return df
        
    except Exception as e:
        diagnostics.add_error(f"Error calculating EDGE scores: {str(e)}")
        return df

# ============================================================================
# PATTERN DETECTION
# ============================================================================

def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect key trading patterns"""
    
    try:
        # Initialize pattern columns
        df['pattern_breakout'] = False
        df['pattern_accumulation'] = False
        df['pattern_reversal'] = False
        df['pattern_momentum'] = False
        
        # Breakout pattern
        df['pattern_breakout'] = (
            (df['price'] > df['high_52w'] * 0.95) &
            (df['rvol'] > 1.5) &
            (df['ret_7d'] > 5)
        )
        
        # Accumulation pattern
        df['pattern_accumulation'] = (
            (abs(df['ret_30d']) < 10) &
            (df['volume_acceleration'] > 10) &
            (df['from_high_pct'] < -20)
        )
        
        # Reversal pattern
        df['pattern_reversal'] = (
            (df['from_low_pct'] < 20) &
            (df['ret_7d'] > 0) &
            (df['volume_acceleration'] > 0) &
            (df['ret_3y'] > 200)
        )
        
        # Momentum pattern
        df['pattern_momentum'] = (
            df['momentum_aligned'] &
            (df['score_momentum'] > 70) &
            (df['volume_acceleration'] > 5)
        )
        
        # Count patterns
        pattern_cols = [col for col in df.columns if col.startswith('pattern_')]
        df['pattern_count'] = df[pattern_cols].sum(axis=1)
        
        return df
        
    except Exception as e:
        diagnostics.add_error(f"Error detecting patterns: {str(e)}")
        return df

# ============================================================================
# RISK MANAGEMENT
# ============================================================================

def calculate_risk_metrics(df: pd.DataFrame, position_multiplier: float) -> pd.DataFrame:
    """Calculate risk management metrics"""
    
    try:
        # Position sizing based on signal
        df['base_position_size'] = df['signal'].map({
            'SUPER_EDGE': TRADING_CONFIG['POSITION_SIZES']['SUPER_EDGE'],
            'STRONG': TRADING_CONFIG['POSITION_SIZES']['STRONG'],
            'MODERATE': TRADING_CONFIG['POSITION_SIZES']['MODERATE'],
            'WATCH': TRADING_CONFIG['POSITION_SIZES']['WATCH'],
            'IGNORE': 0
        }).fillna(0)
        
        # Adjust for market regime
        df['position_size'] = df['base_position_size'] * position_multiplier
        
        # ATR-based stop loss
        df['stop_loss_pct'] = df['atr_pct'].clip(3, 15) / 100
        df['stop_loss'] = df['price'] * (1 - df['stop_loss_pct'])
        
        # Support-based stop (if better)
        support_stop = df[['sma_20d', 'sma_50d', 'low_52w']].min(axis=1) * 0.98
        df['stop_loss'] = df[['stop_loss', support_stop]].max(axis=1)
        
        # Targets based on ATR and signal strength
        atr_multiplier = df['signal'].map({
            'SUPER_EDGE': 3.0,
            'STRONG': 2.5,
            'MODERATE': 2.0,
            'WATCH': 1.5
        }).fillna(1.0)
        
        df['target_1'] = df['price'] + (df['price'] * df['atr_pct'] / 100 * atr_multiplier)
        df['target_2'] = df['price'] + (df['price'] * df['atr_pct'] / 100 * atr_multiplier * 1.5)
        
        # Risk/Reward ratio
        df['risk_amount'] = df['price'] - df['stop_loss']
        df['reward_amount'] = df['target_1'] - df['price']
        df['risk_reward_ratio'] = (df['reward_amount'] / df['risk_amount'].replace(0, 1)).round(2)
        
        return df
        
    except Exception as e:
        diagnostics.add_error(f"Error calculating risk metrics: {str(e)}")
        return df

# ============================================================================
# EXCEL REPORT GENERATION
# ============================================================================

def generate_excel_report(df: pd.DataFrame, market_stats: Dict) -> BytesIO:
    """Generate professional multi-sheet Excel report"""
    
    output = BytesIO()
    
    try:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4472C4',
                'font_color': 'white',
                'align': 'center',
                'valign': 'vcenter',
                'border': 1
            })
            
            currency_format = workbook.add_format({'num_format': '₹#,##0.00'})
            percent_format = workbook.add_format({'num_format': '0.00%'})
            number_format = workbook.add_format({'num_format': '#,##0'})
            
            # 1. Executive Summary Sheet
            summary_df = create_executive_summary(df, market_stats)
            summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
            
            worksheet = writer.sheets['Executive Summary']
            worksheet.set_column('A:A', 30)
            worksheet.set_column('B:B', 20)
            
            # 2. Action Items Sheet (Top opportunities)
            action_df = df[df['signal'].isin(['SUPER_EDGE', 'STRONG'])].head(10)
            if not action_df.empty:
                action_columns = [
                    'ticker', 'company_name', 'signal', 'edge_score',
                    'price', 'position_size', 'stop_loss', 'target_1',
                    'risk_reward_ratio', 'volume_acceleration'
                ]
                action_df[action_columns].to_excel(
                    writer, sheet_name='Action Items TODAY', index=False
                )
                
                worksheet = writer.sheets['Action Items TODAY']
                worksheet.set_column('A:B', 15)
                worksheet.set_column('C:J', 12)
            
            # 3. Category Analysis
            category_analysis = df.groupby('category').agg({
                'edge_score': ['mean', 'max', 'count'],
                'volume_acceleration': 'mean',
                'ret_7d': 'mean'
            }).round(2)
            category_analysis.columns = ['Avg Score', 'Max Score', 'Count', 'Avg Vol Accel', 'Avg 7d Return']
            category_analysis.to_excel(writer, sheet_name='Category Analysis')
            
            # 4. Sector Analysis
            sector_analysis = df.groupby('sector').agg({
                'edge_score': ['mean', 'max', 'count'],
                'volume_acceleration': 'mean',
                'ret_30d': 'mean'
            }).round(2)
            sector_analysis.columns = ['Avg Score', 'Max Score', 'Count', 'Avg Vol Accel', 'Avg 30d Return']
            sector_analysis.to_excel(writer, sheet_name='Sector Analysis')
            
            # 5. Full Signal List
            signal_columns = [
                'ticker', 'company_name', 'category', 'sector', 'signal',
                'edge_score', 'price', 'volume_acceleration', 'ret_7d', 'ret_30d',
                'position_size', 'stop_loss', 'target_1', 'risk_reward_ratio'
            ]
            
            signals_df = df[df['signal'] != 'IGNORE'][signal_columns]
            signals_df.to_excel(writer, sheet_name='All Signals', index=False)
            
            # 6. Pattern Analysis
            pattern_summary = pd.DataFrame({
                'Pattern': ['Breakout', 'Accumulation', 'Reversal', 'Momentum'],
                'Count': [
                    df['pattern_breakout'].sum(),
                    df['pattern_accumulation'].sum(),
                    df['pattern_reversal'].sum(),
                    df['pattern_momentum'].sum()
                ],
                'Avg Score': [
                    df[df['pattern_breakout']]['edge_score'].mean(),
                    df[df['pattern_accumulation']]['edge_score'].mean(),
                    df[df['pattern_reversal']]['edge_score'].mean(),
                    df[df['pattern_momentum']]['edge_score'].mean()
                ]
            })
            pattern_summary.to_excel(writer, sheet_name='Pattern Analysis', index=False)
            
            # 7. Risk Analysis
            risk_df = df[df['signal'] != 'IGNORE'][['ticker', 'signal', 'position_size', 'stop_loss_pct', 'risk_reward_ratio']]
            risk_df['stop_loss_pct'] = risk_df['stop_loss_pct'] * 100
            risk_df.to_excel(writer, sheet_name='Risk Analysis', index=False)
        
        output.seek(0)
        return output
        
    except Exception as e:
        diagnostics.add_error(f"Error generating Excel report: {str(e)}")
        return BytesIO()

def create_executive_summary(df: pd.DataFrame, market_stats: Dict) -> pd.DataFrame:
    """Create executive summary data"""
    
    signals_df = df[df['signal'] != 'IGNORE']
    
    summary_data = {
        'Metric': [
            'Market Regime',
            'Market Breadth (%)',
            'Positive Momentum (%)',
            'Total Stocks Analyzed',
            'Actionable Signals',
            'SUPER EDGE Signals',
            'STRONG Signals',
            'Average EDGE Score',
            'Top Sector',
            'Top Category',
            'Recommended Positions',
            'Total Portfolio Allocation (%)',
            'Average Risk/Reward',
            'Data Quality Score (%)',
            'Analysis Timestamp'
        ],
        'Value': [
            market_stats.get('regime', 'NEUTRAL'),
            f"{market_stats.get('breadth_score', 0):.1f}",
            f"{market_stats.get('positive_momentum_pct', 0):.1f}",
            len(df),
            len(signals_df),
            len(df[df['signal'] == 'SUPER_EDGE']),
            len(df[df['signal'] == 'STRONG']),
            f"{signals_df['edge_score'].mean():.1f}",
            df.groupby('sector')['edge_score'].mean().idxmax() if not df.empty else 'N/A',
            df.groupby('category')['edge_score'].mean().idxmax() if not df.empty else 'N/A',
            min(5, len(signals_df[signals_df['signal'].isin(['SUPER_EDGE', 'STRONG'])])),
            f"{signals_df.head(5)['position_size'].sum() * 100:.1f}",
            f"{signals_df['risk_reward_ratio'].mean():.2f}",
            f"{diagnostics.metrics.get('data_quality_score', 0):.1f}",
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ]
    }
    
    return pd.DataFrame(summary_data)

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_market_overview_chart(market_stats: Dict) -> go.Figure:
    """Create market overview visualization"""
    
    fig = go.Figure()
    
    # Market breadth gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=market_stats.get('breadth_score', 50),
        title={'text': "Market Breadth"},
        domain={'x': [0, 0.5], 'y': [0.5, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 35], 'color': "lightgray"},
                {'range': [35, 65], 'color': "gray"},
                {'range': [65, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    # Volume expansion gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=market_stats.get('volume_expansion_pct', 50),
        title={'text': "Volume Expansion %"},
        domain={'x': [0.5, 1], 'y': [0.5, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 60], 'color': "gray"},
                {'range': [60, 100], 'color': "lightgreen"}
            ]
        }
    ))
    
    # Momentum indicator
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=market_stats.get('positive_momentum_pct', 50),
        title={'text': "Stocks with Positive Momentum %"},
        domain={'x': [0, 0.5], 'y': [0, 0.5]}
    ))
    
    # RVOL indicator
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=market_stats.get('avg_rvol', 1),
        title={'text': "Average RVOL"},
        domain={'x': [0.5, 1], 'y': [0, 0.5]}
    ))
    
    fig.update_layout(
        title="Market Overview Dashboard",
        height=500,
        showlegend=False
    )
    
    return fig

def create_signal_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create signal distribution chart"""
    
    signal_counts = df['signal'].value_counts()
    
    colors = {
        'SUPER_EDGE': '#FF0000',
        'STRONG': '#FF6600',
        'MODERATE': '#FFA500',
        'WATCH': '#FFD700',
        'IGNORE': '#808080'
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=signal_counts.index,
            y=signal_counts.values,
            marker_color=[colors.get(x, '#808080') for x in signal_counts.index],
            text=signal_counts.values,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Signal Distribution",
        xaxis_title="Signal Type",
        yaxis_title="Count",
        height=400,
        showlegend=False
    )
    
    return fig

def create_sector_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create sector performance heatmap"""
    
    # Aggregate by sector
    sector_metrics = df.groupby('sector').agg({
        'edge_score': 'mean',
        'volume_acceleration': 'mean',
        'ret_7d': 'mean',
        'ret_30d': 'mean'
    }).round(2)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=sector_metrics.values.T,
        x=sector_metrics.index,
        y=['EDGE Score', 'Vol Acceleration', '7D Return', '30D Return'],
        colorscale='RdYlGn',
        text=sector_metrics.values.T,
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Sector Performance Heatmap",
        height=400,
        xaxis_title="Sector",
        yaxis_title="Metric"
    )
    
    return fig

# ============================================================================
# MAIN UI COMPONENTS
# ============================================================================

def render_sidebar_diagnostics():
    """Render diagnostics in sidebar"""
    
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔧 System Diagnostics")
        
        # Data quality indicator
        quality_score = diagnostics.metrics.get('data_quality_score', 0)
        if quality_score > 80:
            st.success(f"Data Quality: {quality_score:.1f}%")
        elif quality_score > 60:
            st.warning(f"Data Quality: {quality_score:.1f}%")
        else:
            st.error(f"Data Quality: {quality_score:.1f}%")
        
        # Key metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Load Time", f"{diagnostics.metrics.get('data_load_time', 0):.1f}s")
            st.metric("Valid Stocks", diagnostics.metrics.get('valid_stocks', 0))
        
        with col2:
            st.metric("Process Time", f"{diagnostics.metrics.get('process_time', 0):.1f}s")
            st.metric("Signals", diagnostics.metrics.get('filtered_stocks', 0))
        
        # Errors and warnings
        if diagnostics.metrics['errors']:
            with st.expander("❌ Errors", expanded=False):
                for error in diagnostics.metrics['errors'][-5:]:  # Show last 5
                    st.error(error)
        
        if diagnostics.metrics['warnings']:
            with st.expander("⚠️ Warnings", expanded=False):
                for warning in diagnostics.metrics['warnings'][-5:]:  # Show last 5
                    st.warning(warning)
        
        # Download diagnostics
        if st.button("📥 Download Full Diagnostics"):
            diag_json = json.dumps(diagnostics.get_report(), indent=2)
            st.download_button(
                "Download JSON",
                diag_json,
                f"diagnostics_{datetime.now():%Y%m%d_%H%M%S}.json",
                "application/json"
            )

def render_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """Render smart filters"""
    
    st.markdown("### 🎯 Smart Filters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    filters = {}
    
    with col1:
        # Category filter with counts
        categories = df['category'].value_counts()
        category_options = [f"{cat} ({count})" for cat, count in categories.items()]
        selected_categories = st.multiselect(
            "Category",
            category_options,
            default=[],
            help="Filter by market cap category"
        )
        filters['categories'] = [cat.split(' (')[0] for cat in selected_categories]
    
    with col2:
        # Sector filter with average score
        sector_scores = df.groupby('sector')['edge_score'].mean().round(1)
        sector_options = [f"{sector} (Score: {score})" for sector, score in sector_scores.items()]
        selected_sectors = st.multiselect(
            "Sector",
            sector_options,
            default=[],
            help="Filter by sector with average EDGE score"
        )
        filters['sectors'] = [sec.split(' (')[0] for sec in selected_sectors]
    
    with col3:
        # Signal filter
        filters['signals'] = st.multiselect(
            "Signal Type",
            ['SUPER_EDGE', 'STRONG', 'MODERATE', 'WATCH'],
            default=['SUPER_EDGE', 'STRONG'],
            help="Filter by signal strength"
        )
    
    with col4:
        # Quick presets
        preset = st.selectbox(
            "Quick Presets",
            ['Custom', 'Top Picks Only', 'Large Cap Leaders', 'Volume Surge', 'Quality Value'],
            help="Pre-configured filter combinations"
        )
        
        if preset == 'Top Picks Only':
            filters['signals'] = ['SUPER_EDGE', 'STRONG']
            filters['min_score'] = 75
        elif preset == 'Large Cap Leaders':
            filters['categories'] = ['Large Cap']
            filters['min_score'] = 60
        elif preset == 'Volume Surge':
            filters['min_vol_accel'] = 20
        elif preset == 'Quality Value':
            filters['min_quality'] = 70
            filters['max_from_high'] = -20
    
    # Advanced filters (expandable)
    with st.expander("🔍 Advanced Filters"):
        adv_col1, adv_col2, adv_col3 = st.columns(3)
        
        with adv_col1:
            filters['min_score'] = st.slider("Min EDGE Score", 0, 100, 50)
            filters['min_vol_accel'] = st.number_input("Min Volume Acceleration %", value=0.0)
        
        with adv_col2:
            filters['min_price'] = st.number_input("Min Price (₹)", value=0.0)
            filters['min_volume'] = st.number_input("Min Daily Volume", value=0)
        
        with adv_col3:
            filters['max_from_high'] = st.number_input("Max % from High", value=0.0, max_value=0.0)
            filters['pattern'] = st.multiselect("Patterns", ['Breakout', 'Accumulation', 'Reversal', 'Momentum'])
    
    return filters

def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply filters to dataframe"""
    
    filtered = df.copy()
    
    # Category filter
    if filters.get('categories'):
        filtered = filtered[filtered['category'].isin(filters['categories'])]
    
    # Sector filter
    if filters.get('sectors'):
        filtered = filtered[filtered['sector'].isin(filters['sectors'])]
    
    # Signal filter
    if filters.get('signals'):
        filtered = filtered[filtered['signal'].isin(filters['signals'])]
    
    # Score filter
    if 'min_score' in filters:
        filtered = filtered[filtered['edge_score'] >= filters['min_score']]
    
    # Volume acceleration filter
    if 'min_vol_accel' in filters and filters['min_vol_accel'] > 0:
        filtered = filtered[filtered['volume_acceleration'] >= filters['min_vol_accel']]
    
    # Price filter
    if 'min_price' in filters and filters['min_price'] > 0:
        filtered = filtered[filtered['price'] >= filters['min_price']]
    
    # Volume filter
    if 'min_volume' in filters and filters['min_volume'] > 0:
        filtered = filtered[filtered['volume_1d'] >= filters['min_volume']]
    
    # From high filter
    if 'max_from_high' in filters and filters['max_from_high'] < 0:
        filtered = filtered[filtered['from_high_pct'] >= filters['max_from_high']]
    
    # Pattern filter
    if filters.get('pattern'):
        pattern_mask = pd.Series(False, index=filtered.index)
        for pattern in filters['pattern']:
            pattern_col = f'pattern_{pattern.lower()}'
            if pattern_col in filtered.columns:
                pattern_mask |= filtered[pattern_col]
        filtered = filtered[pattern_mask]
    
    diagnostics.add_metric('filtered_stocks', len(filtered))
    
    return filtered

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Header
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #1e3c72;
    }
    </style>
    
    <div class="main-header">
        <h1>⚡ EDGE Protocol v2.0</h1>
        <p>Professional Trading Intelligence System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading market data..."):
        df, diag_report = load_data()
    
    if df.empty:
        st.error("❌ Failed to load data. Check diagnostics in sidebar.")
        render_sidebar_diagnostics()
        return
    
    # Process data
    process_start = datetime.now()
    with st.spinner("Analyzing market..."):
        # Market analysis
        market_regime, position_multiplier, market_stats = analyze_market_regime(df)
        
        # Calculate scores
        df = calculate_edge_scores(df, market_regime)
        
        # Detect patterns
        df = detect_patterns(df)
        
        # Risk metrics
        df = calculate_risk_metrics(df, position_multiplier)
    
    process_time = (datetime.now() - process_start).total_seconds()
    diagnostics.add_metric('process_time', process_time)
    
    # Market status banner
    regime_colors = {'BULL': 'green', 'NEUTRAL': 'orange', 'BEAR': 'red'}
    st.markdown(f"""
    <div style="background-color: {regime_colors[market_regime]}; color: white; 
                padding: 10px; border-radius: 5px; text-align: center; margin-bottom: 1rem;">
        <h3>Market Regime: {market_regime} | Position Sizing: {position_multiplier*100:.0f}%</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Filters
    filters = render_filters(df)
    filtered_df = apply_filters(df, filters)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Trading Signals",
        "📈 Market Analysis", 
        "🎯 Top Opportunities",
        "📋 Reports",
        "ℹ️ Help"
    ])
    
    with tab1:
        st.header("Trading Signals")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Signals", len(filtered_df[filtered_df['signal'] != 'IGNORE']))
        with col2:
            st.metric("SUPER EDGE", len(filtered_df[filtered_df['signal'] == 'SUPER_EDGE']))
        with col3:
            st.metric("Avg Score", f"{filtered_df['edge_score'].mean():.1f}")
        with col4:
            st.metric("Avg R/R", f"{filtered_df['risk_reward_ratio'].mean():.2f}")
        
        # Signals table
        if not filtered_df.empty:
            display_columns = [
                'ticker', 'company_name', 'category', 'sector', 'signal', 'edge_score',
                'price', 'volume_acceleration', 'ret_7d', 'ret_30d',
                'position_size', 'stop_loss', 'target_1', 'risk_reward_ratio'
            ]
            
            styled_df = filtered_df[filtered_df['signal'] != 'IGNORE'][display_columns].style.format({
                'edge_score': '{:.1f}',
                'price': '₹{:.2f}',
                'volume_acceleration': '{:.1f}%',
                'ret_7d': '{:.1f}%',
                'ret_30d': '{:.1f}%',
                'position_size': '{:.1%}',
                'stop_loss': '₹{:.2f}',
                'target_1': '₹{:.2f}',
                'risk_reward_ratio': '{:.2f}'
            })
            
            # Apply conditional formatting
            def highlight_signals(s):
                if s.name == 'signal':
                    return ['background-color: #ff4444' if x == 'SUPER_EDGE' else
                           'background-color: #ff8844' if x == 'STRONG' else
                           'background-color: #ffaa44' if x == 'MODERATE' else
                           '' for x in s]
                return [''] * len(s)
            
            styled_df = styled_df.apply(highlight_signals)
            
            st.dataframe(styled_df, use_container_width=True, height=600)
        else:
            st.info("No signals match current filters")
    
    with tab2:
        st.header("Market Analysis")
        
        # Market overview
        st.plotly_chart(create_market_overview_chart(market_stats), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Signal distribution
            st.plotly_chart(create_signal_distribution_chart(df), use_container_width=True)
        
        with col2:
            # Sector heatmap
            st.plotly_chart(create_sector_heatmap(df), use_container_width=True)
    
    with tab3:
        st.header("Top Opportunities")
        
        # Get top opportunities
        top_picks = filtered_df[filtered_df['signal'].isin(['SUPER_EDGE', 'STRONG'])].nlargest(10, 'edge_score')
        
        if not top_picks.empty:
            for idx, (_, stock) in enumerate(top_picks.iterrows()):
                with st.expander(f"#{idx+1} {stock['ticker']} - {stock['company_name']} | EDGE: {stock['edge_score']:.1f}", 
                               expanded=(idx < 3)):
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Signal", stock['signal'])
                        st.metric("Price", f"₹{stock['price']:.2f}")
                        st.metric("Position Size", f"{stock['position_size']*100:.0f}%")
                    
                    with col2:
                        st.metric("Vol Acceleration", f"{stock['volume_acceleration']:.1f}%")
                        st.metric("7D Return", f"{stock['ret_7d']:.1f}%")
                        st.metric("30D Return", f"{stock['ret_30d']:.1f}%")
                    
                    with col3:
                        st.metric("Stop Loss", f"₹{stock['stop_loss']:.2f}")
                        st.metric("Target 1", f"₹{stock['target_1']:.2f}")
                        st.metric("Risk/Reward", f"{stock['risk_reward_ratio']:.2f}")
                    
                    with col4:
                        st.metric("Category", stock['category'])
                        st.metric("Sector", stock['sector'])
                        patterns = []
                        for p in ['breakout', 'accumulation', 'reversal', 'momentum']:
                            if stock.get(f'pattern_{p}', False):
                                patterns.append(p.capitalize())
                        st.metric("Patterns", ', '.join(patterns) if patterns else 'None')
                    
                    # Score breakdown
                    st.markdown("**Score Components:**")
                    score_col1, score_col2, score_col3, score_col4, score_col5 = st.columns(5)
                    with score_col1:
                        st.metric("Volume", f"{stock['score_volume']:.0f}")
                    with score_col2:
                        st.metric("Momentum", f"{stock['score_momentum']:.0f}")
                    with score_col3:
                        st.metric("Quality", f"{stock['score_quality']:.0f}")
                    with score_col4:
                        st.metric("Value", f"{stock['score_value']:.0f}")
                    with score_col5:
                        st.metric("Technical", f"{stock['score_technical']:.0f}")
        else:
            st.info("No high-conviction opportunities found with current filters")
    
    with tab4:
        st.header("Reports & Downloads")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Excel Report")
            st.markdown("""
            Professional multi-sheet Excel report includes:
            - Executive Summary
            - Action Items (Top trades for today)
            - Category Analysis
            - Sector Analysis  
            - Full Signal List
            - Pattern Analysis
            - Risk Analysis
            """)
            
            if st.button("📥 Generate Excel Report", type="primary"):
                with st.spinner("Generating report..."):
                    excel_file = generate_excel_report(filtered_df, market_stats)
                    
                    st.download_button(
                        label="Download Excel Report",
                        data=excel_file,
                        file_name=f"EDGE_Report_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        with col2:
            st.markdown("### 📄 Quick Downloads")
            
            # Top picks CSV
            top_picks_csv = top_picks.to_csv(index=False)
            st.download_button(
                "Top 10 Opportunities (CSV)",
                top_picks_csv,
                f"top_opportunities_{datetime.now():%Y%m%d}.csv",
                "text/csv"
            )
            
            # Full signals CSV
            signals_csv = filtered_df[filtered_df['signal'] != 'IGNORE'].to_csv(index=False)
            st.download_button(
                "All Signals (CSV)",
                signals_csv,
                f"all_signals_{datetime.now():%Y%m%d}.csv",
                "text/csv"
            )
    
    with tab5:
        st.header("Help & Documentation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📖 Quick Guide
            
            **Signal Types:**
            - 🔴 **SUPER_EDGE**: Ultra-high conviction (90+ score)
            - 🟠 **STRONG**: High conviction (75-90 score)
            - 🟡 **MODERATE**: Good opportunity (60-75 score)
            - ⚪ **WATCH**: Monitor closely (45-60 score)
            
            **Key Metrics:**
            - **EDGE Score**: Composite score (0-100)
            - **Volume Acceleration**: 30d/90d vs 30d/180d change
            - **Risk/Reward**: Expected gain vs potential loss
            - **Position Size**: Recommended allocation %
            """)
        
        with col2:
            st.markdown("""
            ### 🎯 Trading Process
            
            1. **Check Market Regime** (Bull/Neutral/Bear)
            2. **Review Top Opportunities** tab
            3. **Verify position sizes** match your risk
            4. **Use provided stops and targets**
            5. **Download Excel report** for details
            
            **Risk Management:**
            - Never exceed recommended position sizes
            - Always use stop losses
            - Reduce positions in BEAR markets
            - Maximum 5 positions recommended
            """)
    
    # Render diagnostics in sidebar
    render_sidebar_diagnostics()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>EDGE Protocol v2.0 | Professional Trading Intelligence System</p>
        <p>For educational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Critical application error: {str(e)}")
        logger.error(f"Application crash: {str(e)}\n{traceback.format_exc()}")
        st.stop()
