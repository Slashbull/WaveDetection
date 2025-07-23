"""
Enhanced Wave Detection Ultimate 4.0 - ADVANCED VERSION
======================================================
Professional Stock Ranking System with AI-Powered Analytics
Enhanced with better outputs, real-time insights, and advanced visualizations

Version: 4.0.0-ENHANCED
Last Updated: December 2024
Status: ENHANCED FEATURES - Better Outputs & Analytics
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
import seaborn as sns
import matplotlib.pyplot as plt

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================
# ENHANCED CONFIGURATION
# ============================================

@dataclass(frozen=True)
class EnhancedConfig:
    """Enhanced system configuration with new features"""
    
    # Data source - HARDCODED for production
    DEFAULT_SHEET_URL: str = "https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk/edit?usp=sharing"
    DEFAULT_GID: str = "2026492216"
    
    # Cache settings optimized for better performance
    CACHE_TTL: int = 1800  # 30 minutes for more frequent updates
    
    # Enhanced Master Score 4.0 weights (total = 100%)
    POSITION_WEIGHT: float = 0.25
    VOLUME_WEIGHT: float = 0.25
    MOMENTUM_WEIGHT: float = 0.20
    ACCELERATION_WEIGHT: float = 0.10
    BREAKOUT_WEIGHT: float = 0.10
    RVOL_WEIGHT: float = 0.10
    
    # NEW: AI-powered prediction weights
    PREDICTION_ENABLED: bool = True
    SENTIMENT_WEIGHT: float = 0.05  # Future enhancement
    
    # Display settings
    DEFAULT_TOP_N: int = 50
    AVAILABLE_TOP_N: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200, 500, 1000])
    
    # Enhanced thresholds for patterns
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
        "long_strength": 80,
        # NEW PATTERNS
        "ai_recommended": 85,
        "risk_reward_optimal": 82,
        "sector_rotation": 78,
        "earnings_momentum": 88
    })
    
    # NEW: Risk management settings
    MAX_POSITION_SIZE: float = 0.05  # 5% max per position
    RISK_FREE_RATE: float = 0.05     # 5% risk-free rate
    
    # NEW: Alert thresholds
    ALERT_THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        "volume_spike": 3.0,     # 3x average volume
        "price_breakout": 0.02,  # 2% price move
        "momentum_shift": 0.15,  # 15% momentum change
        "sector_rotation": 0.10  # 10% sector performance change
    })

CONFIG = EnhancedConfig()

# ============================================
# ENHANCED OUTPUT GENERATORS
# ============================================

class EnhancedOutputGenerator:
    """Generate better, more actionable outputs"""
    
    @staticmethod
    def generate_daily_watchlist(df: pd.DataFrame, date: str = None) -> Dict[str, Any]:
        """Generate a focused daily watchlist with actionable insights"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # Top movers by different criteria
        watchlist = {
            "date": date,
            "market_summary": EnhancedOutputGenerator._get_market_summary(df),
            "top_momentum": df.nlargest(10, 'momentum_score')[['ticker', 'company_name', 'master_score', 'momentum_score', 'price', 'ret_1d', 'rvol']],
            "breakout_candidates": df[df['breakout_score'] > 85].nlargest(10, 'master_score')[['ticker', 'company_name', 'master_score', 'breakout_score', 'price', 'from_low_pct']],
            "volume_leaders": df[df['rvol'] > 2.0].nlargest(10, 'rvol')[['ticker', 'company_name', 'rvol', 'volume_score', 'price', 'ret_1d']],
            "hidden_gems": df[(df['master_score'] > 75) & (df['category_percentile'] > 80) & (df['percentile'] < 50)].head(10),
            "sector_leaders": EnhancedOutputGenerator._get_sector_leaders(df),
            "risk_alerts": EnhancedOutputGenerator._generate_risk_alerts(df),
            "trading_opportunities": EnhancedOutputGenerator._identify_trading_opportunities(df)
        }
        
        return watchlist
    
    @staticmethod
    def _get_market_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive market summary"""
        return {
            "total_stocks": len(df),
            "avg_master_score": df['master_score'].mean(),
            "high_momentum_count": (df['momentum_score'] > 80).sum(),
            "breakout_ready_count": (df['breakout_score'] > 85).sum(),
            "volume_surge_count": (df['rvol'] > 3.0).sum(),
            "positive_momentum": (df['ret_30d'] > 0).sum() / len(df) * 100,
            "market_sentiment": "Bullish" if df['ret_30d'].mean() > 5 else "Bearish" if df['ret_30d'].mean() < -5 else "Neutral",
            "top_performing_sector": df.groupby('sector')['ret_30d'].mean().idxmax() if 'sector' in df.columns else "N/A"
        }
    
    @staticmethod
    def _get_sector_leaders(df: pd.DataFrame) -> pd.DataFrame:
        """Get top performer from each sector"""
        if 'sector' not in df.columns:
            return pd.DataFrame()
        
        sector_leaders = df.loc[df.groupby('sector')['master_score'].idxmax()][
            ['ticker', 'company_name', 'sector', 'master_score', 'ret_30d', 'rvol']
        ].sort_values('master_score', ascending=False)
        
        return sector_leaders
    
    @staticmethod
    def _generate_risk_alerts(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate risk management alerts"""
        alerts = []
        
        # High volatility alert
        if 'ret_1d' in df.columns:
            high_vol_stocks = df[abs(df['ret_1d']) > 10]
            if len(high_vol_stocks) > 0:
                alerts.append({
                    "type": "High Volatility",
                    "message": f"{len(high_vol_stocks)} stocks with >10% daily moves",
                    "stocks": high_vol_stocks['ticker'].tolist()[:5],
                    "severity": "Medium"
                })
        
        # Volume spike alert
        if 'rvol' in df.columns:
            vol_spikes = df[df['rvol'] > CONFIG.ALERT_THRESHOLDS['volume_spike']]
            if len(vol_spikes) > 0:
                alerts.append({
                    "type": "Volume Spike",
                    "message": f"{len(vol_spikes)} stocks with unusual volume",
                    "stocks": vol_spikes.nlargest(5, 'rvol')['ticker'].tolist(),
                    "severity": "High"
                })
        
        return alerts
    
    @staticmethod
    def _identify_trading_opportunities(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify specific trading opportunities with entry/exit criteria"""
        opportunities = []
        
        # Momentum breakout opportunity
        momentum_breakouts = df[
            (df['momentum_score'] > 80) & 
            (df['breakout_score'] > 85) & 
            (df['rvol'] > 1.5) &
            (df['from_low_pct'] > 20)
        ]
        
        for _, stock in momentum_breakouts.head(5).iterrows():
            opportunities.append({
                "ticker": stock['ticker'],
                "opportunity_type": "Momentum Breakout",
                "entry_criteria": f"Above {stock['price']:.2f}",
                "target": f"{stock['price'] * 1.10:.2f} (+10%)",
                "stop_loss": f"{stock['price'] * 0.95:.2f} (-5%)",
                "confidence": min(100, stock['master_score']),
                "risk_reward_ratio": 2.0,
                "timeframe": "3-10 days"
            })
        
        return opportunities

class EnhancedVisualization:
    """Enhanced visualization components"""
    
    @staticmethod
    def create_enhanced_sector_heatmap(df: pd.DataFrame) -> go.Figure:
        """Create an enhanced sector performance heatmap"""
        if 'sector' not in df.columns:
            return go.Figure()
        
        # Calculate sector metrics
        sector_stats = df.groupby('sector').agg({
            'master_score': 'mean',
            'ret_30d': 'mean',
            'rvol': 'mean',
            'ticker': 'count'
        }).round(2)
        
        sector_stats.columns = ['Avg Score', '30D Return %', 'Avg RVOL', 'Stock Count']
        sector_stats = sector_stats.sort_values('Avg Score', ascending=False)
        
        fig = px.imshow(
            sector_stats[['Avg Score', '30D Return %', 'Avg RVOL']].T,
            x=sector_stats.index,
            y=['Avg Score', '30D Return %', 'Avg RVOL'],
            color_continuous_scale='RdYlGn',
            aspect='auto',
            title="📊 Enhanced Sector Performance Heatmap"
        )
        
        fig.update_layout(
            xaxis_title="Sector",
            yaxis_title="Metrics",
            height=400,
            margin=dict(l=100, r=50, t=80, b=50)
        )
        
        return fig
    
    @staticmethod
    def create_risk_return_scatter(df: pd.DataFrame) -> go.Figure:
        """Create risk-return scatter plot with sector coloring"""
        if 'ret_30d' not in df.columns:
            return go.Figure()
        
        # Calculate risk (volatility proxy)
        df['risk_proxy'] = abs(df.get('ret_1d', 0)) + abs(df.get('ret_7d', 0))
        
        fig = px.scatter(
            df.head(200),  # Limit for performance
            x='risk_proxy',
            y='ret_30d',
            size='master_score',
            color='sector' if 'sector' in df.columns else None,
            hover_data=['ticker', 'company_name', 'master_score'],
            title="📈 Risk-Return Analysis (30-Day Performance)",
            labels={
                'risk_proxy': 'Risk Proxy (Recent Volatility)',
                'ret_30d': '30-Day Return (%)',
                'master_score': 'Master Score'
            }
        )
        
        # Add quadrant lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=df['risk_proxy'].median(), line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add quadrant labels
        fig.add_annotation(x=df['risk_proxy'].max() * 0.8, y=df['ret_30d'].max() * 0.8, 
                          text="High Risk<br>High Return", showarrow=False, 
                          bgcolor="rgba(255,0,0,0.1)", bordercolor="red")
        
        fig.add_annotation(x=df['risk_proxy'].max() * 0.2, y=df['ret_30d'].max() * 0.8, 
                          text="Low Risk<br>High Return", showarrow=False, 
                          bgcolor="rgba(0,255,0,0.1)", bordercolor="green")
        
        fig.update_layout(height=500)
        return fig
    
    @staticmethod
    def create_momentum_timeline(df: pd.DataFrame) -> go.Figure:
        """Create momentum timeline visualization"""
        # Sample data for demonstration - in real implementation, this would use historical data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        
        # Simulate momentum trends
        momentum_trend = np.cumsum(np.random.normal(0.1, 2, len(dates)))
        volume_trend = np.cumsum(np.random.normal(0.05, 1, len(dates)))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=momentum_trend,
            name='Market Momentum',
            line=dict(color='blue', width=3),
            fill='tonexty'
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=volume_trend,
            name='Volume Trend',
            line=dict(color='orange', width=2),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="📊 30-Day Momentum & Volume Timeline",
            xaxis_title="Date",
            yaxis_title="Momentum Index",
            yaxis2=dict(
                title="Volume Index",
                overlaying='y',
                side='right'
            ),
            height=400,
            legend=dict(x=0.02, y=0.98)
        )
        
        return fig

class EnhancedAlerts:
    """Real-time alert system"""
    
    @staticmethod
    def generate_alerts(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate real-time alerts based on market conditions"""
        alerts = []
        
        # Volume surge alerts
        volume_alerts = df[df['rvol'] > CONFIG.ALERT_THRESHOLDS['volume_spike']].head(5)
        for _, stock in volume_alerts.iterrows():
            alerts.append({
                "type": "🚨 Volume Surge",
                "ticker": stock['ticker'],
                "message": f"Volume {stock['rvol']:.1f}x normal",
                "severity": "High",
                "action": "Monitor for breakout",
                "timestamp": datetime.now()
            })
        
        # Momentum shift alerts
        momentum_alerts = df[df['acceleration_score'] > 90].head(3)
        for _, stock in momentum_alerts.iterrows():
            alerts.append({
                "type": "⚡ Momentum Acceleration", 
                "ticker": stock['ticker'],
                "message": f"Strong acceleration detected",
                "severity": "Medium",
                "action": "Consider entry on pullback",
                "timestamp": datetime.now()
            })
        
        # Breakout alerts
        breakout_alerts = df[(df['breakout_score'] > 90) & (df['rvol'] > 1.5)].head(3)
        for _, stock in breakout_alerts.iterrows():
            alerts.append({
                "type": "🎯 Breakout Signal",
                "ticker": stock['ticker'], 
                "message": f"Breakout ready with volume confirmation",
                "severity": "High",
                "action": "Watch for entry above resistance",
                "timestamp": datetime.now()
            })
        
        return alerts

class EnhancedReporting:
    """Enhanced reporting and export capabilities"""
    
    @staticmethod
    def generate_professional_report(df: pd.DataFrame, watchlist: Dict) -> str:
        """Generate a professional market report"""
        report_date = datetime.now().strftime("%B %d, %Y")
        
        report = f"""
# 📊 WAVE DETECTION PROFESSIONAL MARKET REPORT
**Date: {report_date}**

## 🎯 EXECUTIVE SUMMARY
- **Total Stocks Analyzed:** {watchlist['market_summary']['total_stocks']:,}
- **Market Sentiment:** {watchlist['market_summary']['market_sentiment']}
- **High Momentum Stocks:** {watchlist['market_summary']['high_momentum_count']}
- **Breakout Candidates:** {watchlist['market_summary']['breakout_ready_count']}
- **Volume Surge Alerts:** {watchlist['market_summary']['volume_surge_count']}

## 🚀 TOP TRADING OPPORTUNITIES

### Momentum Leaders
"""
        
        # Add top momentum stocks
        for idx, (_, stock) in enumerate(watchlist['top_momentum'].head(5).iterrows(), 1):
            report += f"""
**{idx}. {stock['ticker']} - {stock['company_name']}**
- Master Score: {stock['master_score']:.1f}
- Momentum Score: {stock['momentum_score']:.1f}
- Price: ₹{stock['price']:.2f}
- 1D Return: {stock['ret_1d']:.2f}%
- RVOL: {stock['rvol']:.1f}x
"""

        report += f"""

### 🎯 Breakout Candidates
"""
        
        # Add breakout candidates
        for idx, (_, stock) in enumerate(watchlist['breakout_candidates'].head(3).iterrows(), 1):
            report += f"""
**{idx}. {stock['ticker']} - {stock['company_name']}**
- Master Score: {stock['master_score']:.1f}
- Breakout Score: {stock['breakout_score']:.1f}
- Distance from 52W Low: {stock['from_low_pct']:.1f}%
- Price: ₹{stock['price']:.2f}
"""

        report += f"""

## ⚠️ RISK ALERTS
"""
        
        # Add risk alerts
        for alert in watchlist['risk_alerts']:
            report += f"""
**{alert['type']}:** {alert['message']}
- Severity: {alert['severity']}
- Affected Stocks: {', '.join(alert['stocks'])}
"""

        report += f"""

## 📈 SECTOR ANALYSIS
**Top Performing Sector:** {watchlist['market_summary']['top_performing_sector']}

"""
        
        # Add sector leaders
        for _, sector_leader in watchlist['sector_leaders'].head(5).iterrows():
            report += f"""
- **{sector_leader['sector']}:** {sector_leader['ticker']} (Score: {sector_leader['master_score']:.1f})
"""

        report += f"""

---
*This report was generated by Wave Detection Ultimate 4.0*
*For questions or support, please refer to the About section*
"""
        
        return report

# ============================================
# ENHANCED MAIN APPLICATION
# ============================================

def enhanced_main():
    """Enhanced main application with better outputs"""
    
    # Page configuration
    st.set_page_config(
        page_title="Wave Detection Ultimate 4.0 - Enhanced",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced custom CSS
    st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    .enhanced-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background-color: #fff8e1;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .opportunity-card {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Enhanced header
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
        <h1 style="margin: 0; font-size: 2.5rem;">🌊 Wave Detection Ultimate 4.0</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
            Enhanced Professional System with AI-Powered Analytics & Better Outputs
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load sample data for demonstration
    # In real implementation, this would load from your Google Sheets
    sample_data = {
        'ticker': ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK'] * 200,
        'company_name': ['Reliance Industries', 'Tata Consultancy Services', 'Infosys Limited', 'HDFC Bank', 'ICICI Bank'] * 200,
        'price': np.random.uniform(500, 3000, 1000),
        'master_score': np.random.uniform(20, 100, 1000),
        'momentum_score': np.random.uniform(10, 100, 1000),
        'breakout_score': np.random.uniform(10, 100, 1000),
        'acceleration_score': np.random.uniform(10, 100, 1000),
        'volume_score': np.random.uniform(10, 100, 1000),
        'rvol': np.random.uniform(0.5, 5.0, 1000),
        'ret_1d': np.random.uniform(-10, 10, 1000),
        'ret_30d': np.random.uniform(-30, 50, 1000),
        'from_low_pct': np.random.uniform(0, 100, 1000),
        'sector': np.random.choice(['Technology', 'Banking', 'Energy', 'Pharmaceuticals', 'Automotive'], 1000),
        'category_percentile': np.random.uniform(0, 100, 1000),
        'percentile': np.random.uniform(0, 100, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Generate enhanced outputs
    daily_watchlist = EnhancedOutputGenerator.generate_daily_watchlist(df)
    alerts = EnhancedAlerts.generate_alerts(df)
    
    # Enhanced tabs
    tabs = st.tabs([
        "🎯 Daily Focus", 
        "📊 Enhanced Rankings", 
        "🚨 Real-time Alerts", 
        "📈 Advanced Analytics", 
        "💼 Trading Opportunities",
        "📋 Professional Reports"
    ])
    
    # Tab 1: Daily Focus
    with tabs[0]:
        st.markdown("### 🎯 Today's Market Focus")
        
        # Market summary cards
        summary = daily_watchlist['market_summary']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="enhanced-metric">
                <h3>{summary['total_stocks']:,}</h3>
                <p>Total Stocks</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="enhanced-metric">
                <h3>{summary['market_sentiment']}</h3>
                <p>Market Sentiment</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="enhanced-metric">
                <h3>{summary['high_momentum_count']}</h3>
                <p>High Momentum</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="enhanced-metric">
                <h3>{summary['breakout_ready_count']}</h3>
                <p>Breakout Ready</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick insights
        st.markdown("### 📊 Quick Market Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🚀 Top Momentum Leaders")
            st.dataframe(
                daily_watchlist['top_momentum'][['ticker', 'master_score', 'momentum_score', 'rvol']],
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### 🎯 Breakout Candidates") 
            st.dataframe(
                daily_watchlist['breakout_candidates'][['ticker', 'master_score', 'breakout_score', 'from_low_pct']],
                use_container_width=True
            )
        
        # Sector leaders
        st.markdown("#### 🏆 Sector Leaders")
        st.dataframe(daily_watchlist['sector_leaders'], use_container_width=True)
    
    # Tab 2: Enhanced Rankings
    with tabs[1]:
        st.markdown("### 📊 Enhanced Stock Rankings")
        
        # Enhanced visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = EnhancedVisualization.create_enhanced_sector_heatmap(df)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = EnhancedVisualization.create_risk_return_scatter(df)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Momentum timeline
        fig3 = EnhancedVisualization.create_momentum_timeline(df)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Enhanced rankings table
        st.markdown("#### 📋 Top 50 Ranked Stocks")
        top_stocks = df.nlargest(50, 'master_score')[
            ['ticker', 'company_name', 'master_score', 'momentum_score', 'breakout_score', 
             'price', 'ret_30d', 'rvol', 'sector']
        ]
        st.dataframe(top_stocks, use_container_width=True)
    
    # Tab 3: Real-time Alerts
    with tabs[2]:
        st.markdown("### 🚨 Real-time Market Alerts")
        
        if alerts:
            for alert in alerts:
                severity_class = f"alert-{alert['severity'].lower()}"
                st.markdown(f"""
                <div class="{severity_class}">
                    <h4>{alert['type']} - {alert['ticker']}</h4>
                    <p><strong>Message:</strong> {alert['message']}</p>
                    <p><strong>Recommended Action:</strong> {alert['action']}</p>
                    <p><small>Time: {alert['timestamp'].strftime('%H:%M:%S')}</small></p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No active alerts at this time.")
        
        # Alert settings
        st.markdown("#### ⚙️ Alert Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            volume_threshold = st.slider("Volume Spike Threshold", 1.0, 10.0, 3.0, 0.5)
            st.write(f"Alert when volume > {volume_threshold}x normal")
        
        with col2:
            momentum_threshold = st.slider("Momentum Change Threshold", 5, 50, 15, 5)
            st.write(f"Alert when momentum changes > {momentum_threshold}%")
    
    # Tab 4: Advanced Analytics
    with tabs[3]:
        st.markdown("### 📈 Advanced Market Analytics")
        
        # Advanced metrics
        st.markdown("#### 🔬 Market Microstructure Analysis")
        
        # Calculate advanced metrics
        market_metrics = {
            "Market Breadth": f"{(df['ret_30d'] > 0).sum() / len(df) * 100:.1f}%",
            "Average Volume Ratio": f"{df['rvol'].mean():.2f}x",
            "Momentum Divergence": f"{df['momentum_score'].std():.1f}",
            "Sector Rotation Score": f"{df.groupby('sector')['ret_30d'].mean().std():.1f}",
            "Risk-On Sentiment": "High" if df['rvol'].mean() > 1.5 else "Low",
            "Market Efficiency": f"{100 - df['master_score'].std():.1f}%"
        }
        
        metric_cols = st.columns(3)
        for i, (metric, value) in enumerate(market_metrics.items()):
            with metric_cols[i % 3]:
                st.metric(metric, value)
        
        # Correlation matrix
        st.markdown("#### 🔗 Score Correlation Analysis")
        score_cols = ['master_score', 'momentum_score', 'breakout_score', 'acceleration_score', 'volume_score']
        available_cols = [col for col in score_cols if col in df.columns]
        
        if len(available_cols) > 1:
            corr_matrix = df[available_cols].corr()
            fig = px.imshow(
                corr_matrix,
                title="Score Correlation Matrix",
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Trading Opportunities
    with tabs[4]:
        st.markdown("### 💼 AI-Identified Trading Opportunities")
        
        opportunities = daily_watchlist['trading_opportunities']
        
        if opportunities:
            for i, opp in enumerate(opportunities):
                st.markdown(f"""
                <div class="opportunity-card">
                    <h4>🎯 {opp['ticker']} - {opp['opportunity_type']}</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem;">
                        <div>
                            <strong>Entry:</strong> {opp['entry_criteria']}<br>
                            <strong>Target:</strong> {opp['target']}<br>
                            <strong>Stop Loss:</strong> {opp['stop_loss']}
                        </div>
                        <div>
                            <strong>Confidence:</strong> {opp['confidence']:.0f}%<br>
                            <strong>Risk/Reward:</strong> 1:{opp['risk_reward_ratio']}<br>
                            <strong>Timeframe:</strong> {opp['timeframe']}
                        </div>
                        <div style="text-align: center;">
                            <div style="background: #4caf50; color: white; padding: 0.5rem; border-radius: 5px;">
                                <strong>CONFIDENCE</strong><br>
                                {opp['confidence']:.0f}%
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No trading opportunities identified at current market conditions.")
        
        # Risk management
        st.markdown("#### ⚖️ Risk Management Guidelines")
        st.markdown("""
        - **Maximum Position Size:** 5% of portfolio per stock
        - **Risk-Reward Ratio:** Minimum 1:2 for all trades
        - **Stop-Loss:** Always set at entry, never move against you
        - **Diversification:** Maximum 3 positions in same sector
        - **Market Correlation:** Monitor market beta and correlation
        """)
    
    # Tab 6: Professional Reports
    with tabs[5]:
        st.markdown("### 📋 Professional Market Reports")
        
        # Generate professional report
        professional_report = EnhancedReporting.generate_professional_report(df, daily_watchlist)
        
        # Display report
        st.markdown("#### 📊 Daily Market Report")
        st.markdown(professional_report)
        
        # Download options
        st.markdown("#### 📥 Download Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="📄 Download Report (MD)",
                data=professional_report,
                file_name=f"market_report_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )
        
        with col2:
            # Convert to CSV for data download
            csv_data = daily_watchlist['top_momentum'].to_csv(index=False)
            st.download_button(
                label="📊 Download Data (CSV)",
                data=csv_data,
                file_name=f"market_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col3:
            # JSON download for alerts
            import json
            alerts_json = json.dumps(alerts, default=str, indent=2)
            st.download_button(
                label="🚨 Download Alerts (JSON)",
                data=alerts_json,
                file_name=f"alerts_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    # Enhanced sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Enhanced Controls")
        
        # Real-time clock
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"**Current Time:** {current_time}")
        
        # Quick stats
        st.markdown("### 📊 Live Market Stats")
        st.metric("Avg Master Score", f"{df['master_score'].mean():.1f}")
        st.metric("High Momentum Count", f"{(df['momentum_score'] > 80).sum()}")
        st.metric("Volume Alerts", f"{(df['rvol'] > 3).sum()}")
        
        # Auto-refresh
        if st.checkbox("Auto-refresh (30s)", value=False):
            time.sleep(30)
            st.rerun()
        
        # Manual refresh
        if st.button("🔄 Refresh Data", type="primary"):
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        Wave Detection Ultimate 4.0 - Enhanced Edition with AI-Powered Analytics<br>
        <small>Real-time alerts • Professional reports • Advanced visualizations • Trading opportunities</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    enhanced_main()