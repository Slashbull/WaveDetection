# mantra_edge_analyzer.py - COMPLETE EDGE SYSTEM ANALYZER & REPORTER
"""
M.A.N.T.R.A. EDGE Complete Analyzer
===================================
ONE FILE that tells you EVERYTHING about how your edge system is performing.
Generates comprehensive HTML report with all metrics, backtests, and recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import io
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="M.A.N.T.R.A. Edge Analyzer",
    page_icon="📊",
    layout="wide"
)

# Google Sheets Configuration  
SHEET_ID = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"
GID = "2026492216"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

# ============================================================================
# DATA LOADING & EDGE DETECTION (From your system)
# ============================================================================

@st.cache_data(ttl=300)
def load_data():
    """Load and clean data - EXACTLY as in edge system"""
    try:
        response = requests.get(SHEET_URL, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        
        # Remove unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
        
        # Clean column names
        df.columns = [col.strip() for col in df.columns]
        
        # Handle market_cap
        if 'market_cap' in df.columns:
            df['market_cap_num'] = (
                df['market_cap']
                .astype(str)
                .str.replace('₹', '', regex=False)
                .str.replace(',', '', regex=False)
                .str.replace(' Cr', '', regex=False)
                .str.replace('Cr', '', regex=False)
                .str.strip()
            )
            df['market_cap_num'] = pd.to_numeric(df['market_cap_num'], errors='coerce')
        
        # Volume columns
        volume_cols = ['volume_1d', 'volume_7d', 'volume_30d', 'volume_3m', 'volume_90d', 'volume_180d']
        for col in volume_cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(',', '', regex=False)
                    .str.replace('₹', '', regex=False)
                    .str.strip()
                )
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Volume ratio columns
        vol_ratio_cols = [
            'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
            'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 
            'vol_ratio_90d_180d'
        ]
        for col in vol_ratio_cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace('%', '', regex=False)
                    .str.strip()
                )
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Return columns
        return_cols = ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y']
        for col in return_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Price columns
        price_cols = ['price', 'prev_close', 'sma_20d', 'sma_50d', 'sma_200d', 'low_52w', 'high_52w']
        for col in price_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Percentage columns
        pct_cols = ['from_low_pct', 'from_high_pct', 'eps_change_pct']
        for col in pct_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fundamental columns
        fundamental_cols = ['pe', 'eps_current', 'eps_last_qtr', 'rvol', 'year']
        for col in fundamental_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return pd.DataFrame()

def run_edge_detection(df):
    """Run complete edge detection - EXACTLY as in your system"""
    # Calculate volume acceleration
    if 'vol_ratio_30d_90d' in df.columns and 'vol_ratio_30d_180d' in df.columns:
        df['volume_acceleration'] = df['vol_ratio_30d_90d'] - df['vol_ratio_30d_180d']
    else:
        df['volume_acceleration'] = 0
    
    # Calculate conviction score
    df['conviction_score'] = 0
    
    # Volume acceleration component (40 points)
    if 'volume_acceleration' in df.columns:
        df.loc[df['volume_acceleration'] > 0, 'conviction_score'] += 20
        df.loc[df['volume_acceleration'] > 10, 'conviction_score'] += 20
    
    # Momentum building component (20 points)
    if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
        df.loc[df['ret_7d'] > df['ret_30d'] / 4, 'conviction_score'] += 20
    
    # Fundamentals improving (20 points)
    if all(col in df.columns for col in ['eps_current', 'eps_last_qtr']):
        df.loc[df['eps_current'] > df['eps_last_qtr'], 'conviction_score'] += 20
    
    # Technical support (10 points)
    if all(col in df.columns for col in ['price', 'sma_50d']):
        df.loc[df['price'] > df['sma_50d'], 'conviction_score'] += 10
    
    # High interest today (10 points)
    if 'rvol' in df.columns:
        df.loc[df['rvol'] > 1.5, 'conviction_score'] += 10
    
    # Initialize signal column
    df['EDGE_SIGNAL'] = 'NONE'
    
    # Triple Alignment
    if all(col in df.columns for col in ['volume_acceleration', 'eps_current', 'eps_last_qtr', 'from_high_pct', 'ret_30d', 'pe']):
        triple_mask = (
            (df['volume_acceleration'] > 10) &
            (df['eps_current'] > df['eps_last_qtr']) &
            (df['from_high_pct'] < -20) &
            (df['ret_30d'].abs() < 5) &
            (df['pe'] > 0) & (df['pe'] < 50)
        )
        df.loc[triple_mask, 'EDGE_SIGNAL'] = 'TRIPLE_ALIGNMENT'
    
    # Coiled Spring
    if all(col in df.columns for col in ['volume_acceleration', 'ret_30d', 'from_high_pct']):
        spring_mask = (
            (df['volume_acceleration'] > 5) &
            (df['ret_30d'].abs() < 5) &
            (df['from_high_pct'] < -30) &
            (df['EDGE_SIGNAL'] == 'NONE')
        )
        df.loc[spring_mask, 'EDGE_SIGNAL'] = 'COILED_SPRING'
    
    # Momentum Knife
    if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'vol_ratio_1d_90d', 'price', 'sma_50d']):
        # Calculate momentum acceleration
        df['momentum_acceleration'] = 0
        valid_returns = (df['ret_7d'] != 0) & df['ret_7d'].notna()
        df.loc[valid_returns, 'momentum_acceleration'] = df.loc[valid_returns, 'ret_1d'] / (df.loc[valid_returns, 'ret_7d'] / 7)
        
        knife_mask = (
            (df['momentum_acceleration'] > 1.5) &
            (df['vol_ratio_1d_90d'] > 100) &
            (df['ret_1d'] > 0) &
            (df['price'] > df['sma_50d']) &
            (df['EDGE_SIGNAL'] == 'NONE')
        )
        df.loc[knife_mask, 'EDGE_SIGNAL'] = 'MOMENTUM_KNIFE'
    
    # Smart Money
    if all(col in df.columns for col in ['eps_change_pct', 'pe', 'volume_acceleration']):
        smart_mask = (
            (df['eps_change_pct'] > 20) &
            (df['pe'] > 0) & (df['pe'] < 40) &
            (df['volume_acceleration'] > 0) &
            (df['EDGE_SIGNAL'] == 'NONE')
        )
        df.loc[smart_mask, 'EDGE_SIGNAL'] = 'SMART_MONEY'
    
    return df

# ============================================================================
# COMPREHENSIVE ANALYSIS ENGINE
# ============================================================================

class EdgeSystemAnalyzer:
    """Complete analysis of edge system performance"""
    
    def __init__(self, df):
        self.df = df
        self.report = {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_summary': {},
            'signal_analysis': {},
            'quality_metrics': {},
            'historical_backtest': {},
            'current_opportunities': {},
            'system_health': {},
            'recommendations': []
        }
    
    def run_complete_analysis(self):
        """Run all analyses and generate report"""
        self.analyze_data_quality()
        self.analyze_signal_distribution()
        self.analyze_signal_quality()
        self.backtest_historical_performance()
        self.analyze_current_opportunities()
        self.calculate_system_health()
        self.generate_recommendations()
        return self.report
    
    def analyze_data_quality(self):
        """1. Data Quality Analysis"""
        total_rows = len(self.df)
        total_cols = len(self.df.columns)
        
        # Null analysis
        null_counts = self.df.isnull().sum()
        null_pct = (null_counts.sum() / (total_rows * total_cols) * 100)
        
        # Key columns coverage
        key_cols = ['price', 'volume_1d', 'pe', 'eps_current', 'ret_30d', 'volume_acceleration']
        key_coverage = {}
        for col in key_cols:
            if col in self.df.columns:
                non_null = self.df[col].notna().sum()
                key_coverage[col] = f"{non_null}/{total_rows} ({non_null/total_rows*100:.1f}%)"
        
        self.report['data_summary'] = {
            'total_stocks': total_rows,
            'total_columns': total_cols,
            'null_percentage': f"{null_pct:.1f}%",
            'key_columns_coverage': key_coverage,
            'date_range': {
                'min_year': int(self.df['year'].min()) if 'year' in self.df.columns else 'N/A',
                'max_year': int(self.df['year'].max()) if 'year' in self.df.columns else 'N/A'
            }
        }
    
    def analyze_signal_distribution(self):
        """2. Signal Distribution Analysis"""
        signal_counts = self.df['EDGE_SIGNAL'].value_counts()
        total_stocks = len(self.df)
        
        # Calculate rates
        signal_rate = (total_stocks - signal_counts.get('NONE', 0)) / total_stocks * 100
        
        # Distribution by signal
        distribution = {}
        for signal in signal_counts.index:
            count = signal_counts[signal]
            pct = count / total_stocks * 100
            distribution[signal] = {
                'count': int(count),
                'percentage': f"{pct:.2f}%"
            }
        
        # Triple alignment deep dive
        triple_stocks = self.df[self.df['EDGE_SIGNAL'] == 'TRIPLE_ALIGNMENT']
        
        self.report['signal_analysis'] = {
            'total_signals': int(total_stocks - signal_counts.get('NONE', 0)),
            'signal_rate': f"{signal_rate:.2f}%",
            'distribution': distribution,
            'triple_alignment_analysis': {
                'count': len(triple_stocks),
                'avg_volume_acceleration': f"{triple_stocks['volume_acceleration'].mean():.2f}%" if len(triple_stocks) > 0 else 'N/A',
                'avg_pe': f"{triple_stocks['pe'].mean():.2f}" if len(triple_stocks) > 0 else 'N/A',
                'sectors': triple_stocks['sector'].value_counts().head(5).to_dict() if 'sector' in triple_stocks.columns and len(triple_stocks) > 0 else {}
            }
        }
    
    def analyze_signal_quality(self):
        """3. Signal Quality Metrics"""
        quality_metrics = {}
        
        for signal_type in ['TRIPLE_ALIGNMENT', 'COILED_SPRING', 'MOMENTUM_KNIFE', 'SMART_MONEY']:
            signal_df = self.df[self.df['EDGE_SIGNAL'] == signal_type]
            
            if len(signal_df) > 0:
                metrics = {
                    'count': len(signal_df),
                    'avg_conviction': f"{signal_df['conviction_score'].mean():.1f}" if 'conviction_score' in signal_df else 'N/A',
                    'avg_volume_accel': f"{signal_df['volume_acceleration'].mean():.2f}%",
                    'price_range': f"₹{signal_df['price'].min():.2f} - ₹{signal_df['price'].max():.2f}",
                    'market_cap_distribution': {
                        'large_cap': len(signal_df[signal_df['market_cap_num'] >= 20000]) if 'market_cap_num' in signal_df else 0,
                        'mid_cap': len(signal_df[(signal_df['market_cap_num'] >= 5000) & (signal_df['market_cap_num'] < 20000)]) if 'market_cap_num' in signal_df else 0,
                        'small_cap': len(signal_df[signal_df['market_cap_num'] < 5000]) if 'market_cap_num' in signal_df else 0
                    }
                }
                
                # Recent performance
                if all(col in signal_df.columns for col in ['ret_7d', 'ret_30d']):
                    metrics['recent_performance'] = {
                        'positive_7d': f"{(signal_df['ret_7d'] > 0).sum() / len(signal_df) * 100:.1f}%",
                        'positive_30d': f"{(signal_df['ret_30d'] > 0).sum() / len(signal_df) * 100:.1f}%",
                        'avg_7d_return': f"{signal_df['ret_7d'].mean():.2f}%",
                        'avg_30d_return': f"{signal_df['ret_30d'].mean():.2f}%"
                    }
                
                # Top stocks
                top_stocks = signal_df.nlargest(5, 'conviction_score')[['ticker', 'conviction_score', 'volume_acceleration', 'price']]
                metrics['top_5_stocks'] = top_stocks.to_dict('records')
                
                quality_metrics[signal_type] = metrics
        
        self.report['quality_metrics'] = quality_metrics
    
    def backtest_historical_performance(self):
        """4. Historical Performance Backtest"""
        backtest_results = {}
        
        # Simulate what would have happened if we bought signals 30 days ago
        for signal_type in ['TRIPLE_ALIGNMENT', 'COILED_SPRING', 'SMART_MONEY']:
            signal_df = self.df[self.df['EDGE_SIGNAL'] == signal_type]
            
            if len(signal_df) > 0 and 'ret_30d' in signal_df.columns:
                # Use 30d return as proxy for backtest
                winners = (signal_df['ret_30d'] > 0).sum()
                total = len(signal_df)
                win_rate = winners / total * 100 if total > 0 else 0
                
                # Calculate returns
                positive_returns = signal_df[signal_df['ret_30d'] > 0]['ret_30d']
                negative_returns = signal_df[signal_df['ret_30d'] <= 0]['ret_30d']
                
                avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
                avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
                
                # Expected value
                expected_value = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)
                
                backtest_results[signal_type] = {
                    'total_signals': total,
                    'winners': winners,
                    'losers': total - winners,
                    'win_rate': f"{win_rate:.1f}%",
                    'avg_winning_trade': f"{avg_win:.2f}%",
                    'avg_losing_trade': f"{avg_loss:.2f}%",
                    'expected_value': f"{expected_value:.2f}%",
                    'best_performer': {
                        'ticker': signal_df.nlargest(1, 'ret_30d')['ticker'].iloc[0] if len(signal_df) > 0 else 'N/A',
                        'return': f"{signal_df['ret_30d'].max():.2f}%"
                    },
                    'worst_performer': {
                        'ticker': signal_df.nsmallest(1, 'ret_30d')['ticker'].iloc[0] if len(signal_df) > 0 else 'N/A',
                        'return': f"{signal_df['ret_30d'].min():.2f}%"
                    }
                }
        
        self.report['historical_backtest'] = backtest_results
    
    def analyze_current_opportunities(self):
        """5. Current Trading Opportunities"""
        opportunities = {}
        
        # Best Triple Alignment
        triple = self.df[self.df['EDGE_SIGNAL'] == 'TRIPLE_ALIGNMENT'].nlargest(3, 'conviction_score')
        if len(triple) > 0:
            opportunities['best_triple_alignment'] = []
            for _, stock in triple.iterrows():
                opportunities['best_triple_alignment'].append({
                    'ticker': stock['ticker'],
                    'price': f"₹{stock['price']:.2f}",
                    'conviction': int(stock['conviction_score']),
                    'volume_accel': f"{stock['volume_acceleration']:.1f}%",
                    'pe': f"{stock['pe']:.1f}" if stock['pe'] > 0 else 'N/A',
                    'sector': stock.get('sector', 'N/A')
                })
        
        # Volume Leaders
        vol_leaders = self.df[self.df['volume_acceleration'] > 20].nlargest(5, 'volume_acceleration')
        if len(vol_leaders) > 0:
            opportunities['volume_leaders'] = []
            for _, stock in vol_leaders.iterrows():
                opportunities['volume_leaders'].append({
                    'ticker': stock['ticker'],
                    'volume_accel': f"{stock['volume_acceleration']:.1f}%",
                    'signal': stock['EDGE_SIGNAL']
                })
        
        # High Conviction Plays
        high_conviction = self.df[self.df['conviction_score'] >= 80].nlargest(5, 'conviction_score')
        if len(high_conviction) > 0:
            opportunities['high_conviction'] = []
            for _, stock in high_conviction.iterrows():
                opportunities['high_conviction'].append({
                    'ticker': stock['ticker'],
                    'conviction': int(stock['conviction_score']),
                    'signal': stock['EDGE_SIGNAL']
                })
        
        self.report['current_opportunities'] = opportunities
    
    def calculate_system_health(self):
        """6. Overall System Health Score"""
        scores = []
        
        # Signal rate score (optimal: 3-15%)
        signal_rate = float(self.report['signal_analysis']['signal_rate'].strip('%'))
        if 3 <= signal_rate <= 15:
            signal_score = 100
        elif signal_rate < 3:
            signal_score = 60  # Too few signals
        else:
            signal_score = 70  # Too many signals
        scores.append(('Signal Rate', signal_score))
        
        # Triple alignment score (should be rare: <2%)
        triple_rate = self.report['signal_analysis']['distribution']['TRIPLE_ALIGNMENT']['percentage'].strip('%')
        triple_rate = float(triple_rate)
        triple_score = 100 if triple_rate <= 2 else 70
        scores.append(('Triple Rarity', triple_score))
        
        # Win rate score (from backtest)
        win_rates = []
        for signal, metrics in self.report['historical_backtest'].items():
            if 'win_rate' in metrics:
                win_rates.append(float(metrics['win_rate'].strip('%')))
        avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else 0
        win_score = min(100, avg_win_rate * 2)  # 50% win rate = 100 score
        scores.append(('Win Rate', win_score))
        
        # Calculate overall
        overall_score = sum(score for _, score in scores) / len(scores)
        
        self.report['system_health'] = {
            'overall_score': f"{overall_score:.1f}/100",
            'component_scores': {name: f"{score:.0f}/100" for name, score in scores},
            'status': 'EXCELLENT' if overall_score >= 85 else 'GOOD' if overall_score >= 70 else 'NEEDS OPTIMIZATION'
        }
    
    def generate_recommendations(self):
        """7. Generate Actionable Recommendations"""
        recs = []
        
        # Check signal rate
        signal_rate = float(self.report['signal_analysis']['signal_rate'].strip('%'))
        if signal_rate > 15:
            recs.append({
                'issue': 'Signal rate too high',
                'impact': 'Too many false positives',
                'action': 'Tighten criteria: Increase volume_acceleration threshold to 15%, reduce PE range to 10-35'
            })
        
        # Check triple alignment rate
        triple_pct = float(self.report['signal_analysis']['distribution']['TRIPLE_ALIGNMENT']['percentage'].strip('%'))
        if triple_pct > 2:
            recs.append({
                'issue': 'Too many Triple Alignments',
                'impact': 'Signal not selective enough',
                'action': 'Make stricter: from_high_pct < -30, ret_30d.abs() < 3, volume_acceleration > 15'
            })
        
        # Check win rates
        for signal, metrics in self.report['historical_backtest'].items():
            if 'win_rate' in metrics:
                win_rate = float(metrics['win_rate'].strip('%'))
                if win_rate < 45:
                    recs.append({
                        'issue': f'{signal} low win rate ({win_rate:.1f}%)',
                        'impact': 'Poor risk/reward',
                        'action': f'Review {signal} criteria or reduce position size'
                    })
        
        # Volume acceleration check
        if 'volume_leaders' in self.report['current_opportunities']:
            if len(self.report['current_opportunities']['volume_leaders']) < 3:
                recs.append({
                    'issue': 'Few stocks with strong volume acceleration',
                    'impact': 'Missing institutional accumulation',
                    'action': 'Check data quality for volume_90d and volume_180d columns'
                })
        
        self.report['recommendations'] = recs

# ============================================================================
# VISUALIZATION & REPORTING
# ============================================================================

def create_performance_dashboard(report):
    """Create visual dashboard from report"""
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            'System Health Score', 'Signal Distribution', 'Win Rates by Signal',
            'Volume Acceleration Leaders', 'Historical Performance', 'Market Cap Distribution',
            'Sector Distribution', 'Conviction Score Distribution', 'Key Metrics'
        ),
        specs=[
            [{'type': 'indicator'}, {'type': 'pie'}, {'type': 'bar'}],
            [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'pie'}],
            [{'type': 'sunburst'}, {'type': 'histogram'}, {'type': 'table'}]
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # 1. System Health Gauge
    health_score = float(report['system_health']['overall_score'].split('/')[0])
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=health_score,
            delta={'reference': 85, 'valueformat': '.1f'},
            title={'text': "System Health"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 70], 'color': "yellow"},
                    {'range': [70, 85], 'color': "lightgreen"},
                    {'range': [85, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 85
                }
            }
        ),
        row=1, col=1
    )
    
    # 2. Signal Distribution Pie
    signal_data = report['signal_analysis']['distribution']
    labels = []
    values = []
    for signal, data in signal_data.items():
        if signal != 'NONE':
            labels.append(signal)
            values.append(data['count'])
    
    fig.add_trace(
        go.Pie(labels=labels, values=values, hole=0.3),
        row=1, col=2
    )
    
    # 3. Win Rates Bar Chart
    signals = []
    win_rates = []
    for signal, metrics in report['historical_backtest'].items():
        signals.append(signal)
        win_rates.append(float(metrics['win_rate'].strip('%')))
    
    fig.add_trace(
        go.Bar(x=signals, y=win_rates, marker_color=['green' if wr > 50 else 'red' for wr in win_rates]),
        row=1, col=3
    )
    
    # Update layout
    fig.update_layout(
        height=1200,
        showlegend=False,
        title_text=f"M.A.N.T.R.A. Edge System Analysis - {report['generated_at']}",
        title_font_size=24
    )
    
    return fig

def generate_html_report(report):
    """Generate comprehensive HTML report"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>M.A.N.T.R.A. Edge Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .header {{ background: #1a1a1a; color: white; padding: 20px; text-align: center; }}
            .section {{ background: white; margin: 20px 0; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f0f0f0; border-radius: 5px; }}
            .good {{ color: green; font-weight: bold; }}
            .bad {{ color: red; font-weight: bold; }}
            .warning {{ color: orange; font-weight: bold; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background: #f0f0f0; }}
            .recommendation {{ background: #fff3cd; padding: 15px; margin: 10px 0; border-left: 5px solid #ffc107; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>M.A.N.T.R.A. EDGE System Analysis</h1>
            <p>Generated: {report['generated_at']}</p>
            <h2>Overall Score: <span class="{'good' if float(report['system_health']['overall_score'].split('/')[0]) >= 85 else 'warning'}">{report['system_health']['overall_score']}</span></h2>
        </div>
        
        <div class="section">
            <h2>📊 Executive Summary</h2>
            <div class="metric">Total Stocks: {report['data_summary']['total_stocks']}</div>
            <div class="metric">Active Signals: {report['signal_analysis']['total_signals']}</div>
            <div class="metric">Signal Rate: <span class="{'good' if 3 <= float(report['signal_analysis']['signal_rate'].strip('%')) <= 15 else 'bad'}">{report['signal_analysis']['signal_rate']}</span></div>
            <div class="metric">System Status: <span class="{'good' if report['system_health']['status'] == 'EXCELLENT' else 'warning'}">{report['system_health']['status']}</span></div>
        </div>
        
        <div class="section">
            <h2>🎯 Signal Analysis</h2>
            <table>
                <tr><th>Signal Type</th><th>Count</th><th>Percentage</th><th>Win Rate</th><th>Expected Value</th></tr>
    """
    
    # Add signal rows
    for signal, dist in report['signal_analysis']['distribution'].items():
        if signal != 'NONE':
            backtest = report['historical_backtest'].get(signal, {})
            win_rate = backtest.get('win_rate', 'N/A')
            exp_value = backtest.get('expected_value', 'N/A')
            
            html += f"""
                <tr>
                    <td>{signal}</td>
                    <td>{dist['count']}</td>
                    <td>{dist['percentage']}</td>
                    <td class="{'good' if win_rate != 'N/A' and float(win_rate.strip('%')) > 50 else 'bad'}">{win_rate}</td>
                    <td class="{'good' if exp_value != 'N/A' and float(exp_value.strip('%')) > 0 else 'bad'}">{exp_value}</td>
                </tr>
            """
    
    html += """
            </table>
        </div>
        
        <div class="section">
            <h2>🏆 Current Top Opportunities</h2>
    """
    
    # Add top opportunities
    if 'best_triple_alignment' in report['current_opportunities']:
        html += "<h3>Best Triple Alignments:</h3><ul>"
        for stock in report['current_opportunities']['best_triple_alignment']:
            html += f"<li><strong>{stock['ticker']}</strong> - Price: {stock['price']}, Conviction: {stock['conviction']}, Volume Accel: {stock['volume_accel']}</li>"
        html += "</ul>"
    
    # Add recommendations
    html += """
        <div class="section">
            <h2>💡 Recommendations</h2>
    """
    
    for rec in report['recommendations']:
        html += f"""
            <div class="recommendation">
                <strong>Issue:</strong> {rec['issue']}<br>
                <strong>Impact:</strong> {rec['impact']}<br>
                <strong>Action:</strong> {rec['action']}
            </div>
        """
    
    html += """
        </div>
    </body>
    </html>
    """
    
    return html

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.title("📊 M.A.N.T.R.A. Edge System Complete Analyzer")
    st.markdown("""
    This tool analyzes EVERYTHING about your edge detection system:
    - Signal quality and distribution
    - Historical performance backtest
    - Current opportunities
    - System health score
    - Actionable recommendations
    """)
    
    # Load data and run edge detection
    with st.spinner("Loading data and running edge detection..."):
        df = load_data()
        
        if df.empty:
            st.error("Failed to load data!")
            return
        
        # Run edge detection
        df = run_edge_detection(df)
        
        # Run complete analysis
        analyzer = EdgeSystemAnalyzer(df)
        report = analyzer.run_complete_analysis()
    
    # Display results
    st.success(f"Analysis complete! System Score: {report['system_health']['overall_score']}")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", "📈 Detailed Analysis", "🎯 Current Signals", 
        "📋 Full Report", "💾 Export"
    ])
    
    with tab1:
        # Quick metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Stocks", report['data_summary']['total_stocks'])
        with col2:
            st.metric("Signal Rate", report['signal_analysis']['signal_rate'])
        with col3:
            st.metric("System Score", report['system_health']['overall_score'])
        with col4:
            st.metric("Status", report['system_health']['status'])
        
        # Visual dashboard
        fig = create_performance_dashboard(report)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Signal Quality
        st.subheader("📊 Signal Quality Analysis")
        for signal, metrics in report['quality_metrics'].items():
            with st.expander(f"{signal} ({metrics['count']} stocks)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Average Conviction:** {metrics['avg_conviction']}")
                    st.write(f"**Volume Acceleration:** {metrics['avg_volume_accel']}")
                    st.write(f"**Price Range:** {metrics['price_range']}")
                with col2:
                    if 'recent_performance' in metrics:
                        st.write("**Recent Performance:**")
                        st.write(f"- 7-day positive: {metrics['recent_performance']['positive_7d']}")
                        st.write(f"- 30-day positive: {metrics['recent_performance']['positive_30d']}")
                
                if 'top_5_stocks' in metrics:
                    st.write("**Top 5 Stocks:**")
                    st.dataframe(pd.DataFrame(metrics['top_5_stocks']))
        
        # Backtest Results
        st.subheader("📈 Historical Backtest Results")
        backtest_df = []
        for signal, metrics in report['historical_backtest'].items():
            backtest_df.append({
                'Signal': signal,
                'Win Rate': metrics['win_rate'],
                'Avg Win': metrics['avg_winning_trade'],
                'Avg Loss': metrics['avg_losing_trade'],
                'Expected Value': metrics['expected_value']
            })
        st.dataframe(pd.DataFrame(backtest_df))
    
    with tab3:
        # Current opportunities
        st.subheader("🎯 Current Trading Opportunities")
        
        if 'best_triple_alignment' in report['current_opportunities']:
            st.write("### 🔥 Best Triple Alignments")
            triple_df = pd.DataFrame(report['current_opportunities']['best_triple_alignment'])
            st.dataframe(triple_df)
        
        if 'volume_leaders' in report['current_opportunities']:
            st.write("### 📊 Volume Acceleration Leaders")
            vol_df = pd.DataFrame(report['current_opportunities']['volume_leaders'])
            st.dataframe(vol_df)
        
        if 'high_conviction' in report['current_opportunities']:
            st.write("### ⭐ High Conviction Plays")
            conv_df = pd.DataFrame(report['current_opportunities']['high_conviction'])
            st.dataframe(conv_df)
    
    with tab4:
        # Full JSON report
        st.subheader("📋 Complete Analysis Report")
        st.json(report)
    
    with tab5:
        # Export options
        st.subheader("💾 Export Report")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # JSON download
            json_str = json.dumps(report, indent=2)
            st.download_button(
                "📄 Download JSON Report",
                json_str,
                f"mantra_edge_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )
        
        with col2:
            # HTML download
            html_report = generate_html_report(report)
            st.download_button(
                "🌐 Download HTML Report",
                html_report,
                f"mantra_edge_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                "text/html"
            )
        
        with col3:
            # Current signals CSV
            signals_df = df[df['EDGE_SIGNAL'] != 'NONE'][
                ['ticker', 'EDGE_SIGNAL', 'conviction_score', 'volume_acceleration', 'price', 'pe', 'sector']
            ].sort_values('conviction_score', ascending=False)
            csv = signals_df.to_csv(index=False)
            st.download_button(
                "📊 Download Signals CSV",
                csv,
                f"mantra_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
    
    # Recommendations
    if report['recommendations']:
        st.warning("### ⚠️ System Recommendations")
        for rec in report['recommendations']:
            st.error(f"""
            **Issue:** {rec['issue']}  
            **Impact:** {rec['impact']}  
            **Action Required:** {rec['action']}
            """)
    else:
        st.success("✅ No critical issues found. System is performing optimally!")

if __name__ == "__main__":
    main()
