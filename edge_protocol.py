"""
Wave Detection System - Professional Stock Analysis Platform
Author: AI Assistant
Version: 1.0.0
Last Updated: December 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import logging
from io import BytesIO
import xlsxwriter
from typing import Dict, List, Tuple, Optional
import requests

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Wave Detection System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .wave-active {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
    .wave-forming {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1e3c72;
        color: white;
    }
    .diagnostic-info {
        font-size: 0.8rem;
        color: #6c757d;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA LOADING AND CLEANING FUNCTIONS
# ============================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_google_sheets_data(sheet_url: str, gid: str) -> pd.DataFrame:
    """Load data from Google Sheets with error handling"""
    try:
        # Convert to CSV export URL
        csv_url = f"{sheet_url.split('/edit')[0]}/export?format=csv&gid={gid}"
        
        # Load data
        df = pd.read_csv(csv_url)
        
        # Log success
        logger.info(f"Successfully loaded {len(df)} rows from Google Sheets")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error(f"Failed to load data: {str(e)}")
        return pd.DataFrame()

def clean_numeric_column(series: pd.Series, column_type: str) -> pd.Series:
    """Clean numeric columns based on type"""
    if series.dtype == 'object':
        if column_type == 'currency':
            # Remove ₹ symbol and commas
            series = series.str.replace('₹', '').str.replace(',', '')
        elif column_type == 'percentage':
            # Remove % symbol
            series = series.str.replace('%', '')
        elif column_type == 'volume':
            # Remove commas
            series = series.str.replace(',', '')
            
    return pd.to_numeric(series, errors='coerce')

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare the data"""
    try:
        # Define column types
        currency_cols = ['price', 'low_52w', 'high_52w', 'sma_20d', 'sma_50d', 'sma_200d', 'prev_close']
        percentage_cols = ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
                          'from_low_pct', 'from_high_pct', 'eps_change_pct',
                          'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
                          'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 'vol_ratio_90d_180d']
        volume_cols = ['volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d']
        
        # Clean numeric columns
        for col in currency_cols:
            if col in df.columns:
                df[col] = clean_numeric_column(df[col], 'currency')
                
        for col in percentage_cols:
            if col in df.columns:
                df[col] = clean_numeric_column(df[col], 'percentage')
                
        for col in volume_cols:
            if col in df.columns:
                df[col] = clean_numeric_column(df[col], 'volume')
        
        # Clean market cap
        if 'market_cap' in df.columns:
            df['market_cap_clean'] = df['market_cap'].str.replace('₹', '').str.replace(' Cr', '').str.replace(',', '')
            df['market_cap_clean'] = pd.to_numeric(df['market_cap_clean'], errors='coerce')
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['ticker'], keep='first')
        
        # Remove extreme outliers
        if 'rvol' in df.columns:
            df = df[df['rvol'] < 100]  # Remove data errors
            
        logger.info(f"Data cleaned successfully. Final shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        st.error(f"Data cleaning failed: {str(e)}")
        return df

# ============================================
# WAVE DETECTION ALGORITHMS
# ============================================

def calculate_wave_score(row: pd.Series) -> float:
    """Calculate comprehensive wave score"""
    try:
        score = 0
        
        # Volume Power (25%)
        if pd.notna(row.get('rvol', 0)):
            vol_score = (
                min(row['rvol'], 5) / 5 * 0.2 +
                (row.get('vol_ratio_1d_90d', 0) / 100 * 0.15 if row.get('vol_ratio_1d_90d', 0) > 0 else 0) +
                (row.get('vol_ratio_7d_90d', 0) / 100 * 0.15 if row.get('vol_ratio_7d_90d', 0) > 0 else 0) +
                (row.get('vol_ratio_30d_90d', 0) / 100 * 0.15 if row.get('vol_ratio_30d_90d', 0) > 0 else 0) +
                (row.get('vol_ratio_90d_180d', 0) / 100 * 0.35 if row.get('vol_ratio_90d_180d', 0) > 0 else 0)
            )
            score += vol_score * 25
        
        # Position Opportunity (25%)
        if pd.notna(row.get('from_low_pct', 0)) and pd.notna(row.get('from_high_pct', 0)):
            pos_score = (
                max(0, (100 - row['from_low_pct']) / 100) * 0.6 +
                max(0, (100 + row['from_high_pct']) / 100) * 0.4
            )
            score += pos_score * 25
        
        # Momentum Cascade (25%)
        mom_score = 0
        if row.get('ret_1d', 0) > 0: mom_score += 5
        if row.get('ret_7d', 0) > row.get('ret_30d', 0) / 4: mom_score += 10
        if row.get('ret_30d', 0) > row.get('ret_3m', 0) / 3: mom_score += 10
        score += mom_score
        
        # Technical Alignment (25%)
        tech_score = 0
        if pd.notna(row.get('price', 0)) and pd.notna(row.get('sma_20d', 0)):
            if row['price'] > row['sma_20d']: tech_score += 8
            if pd.notna(row.get('sma_50d', 0)) and row.get('sma_20d', 0) > row['sma_50d']: tech_score += 8
            if pd.notna(row.get('sma_200d', 0)) and row.get('sma_50d', 0) > row['sma_200d']: tech_score += 9
        score += tech_score
        
        # Bonus factors
        if pd.notna(row.get('eps_change_pct', 0)) and row['eps_change_pct'] > 20: score += 5
        if pd.notna(row.get('pe', 0)) and 0 < row['pe'] < 25: score += 5
        
        return min(score, 100)  # Cap at 100
        
    except Exception as e:
        logger.error(f"Error calculating wave score: {str(e)}")
        return 0

def detect_wave_stage(row: pd.Series) -> Tuple[str, str, float]:
    """Detect current wave stage"""
    try:
        # Check for active waves
        if (row.get('rvol', 0) > 2 and 
            row.get('ret_7d', 0) > 3 and 
            row.get('from_low_pct', 0) < 50):
            
            momentum_strength = row.get('ret_7d', 0) / 7
            if momentum_strength > 1:
                return "🚀 Explosive Wave", "Active", 90
            else:
                return "🏄 Riding Wave", "Active", 75
        
        # Check for forming waves
        elif (row.get('vol_ratio_90d_180d', 0) > 0.9 and 
              row.get('from_low_pct', 0) < 30 and
              abs(row.get('ret_7d', 0)) < 2):
            
            pressure = row.get('vol_ratio_90d_180d', 0) * 100
            if pressure > 110:
                return "⚡ High Pressure", "Forming", 80
            else:
                return "🌊 Building Wave", "Forming", 60
        
        # Check for exhausted waves
        elif (row.get('from_high_pct', 0) > -10 and 
              row.get('rvol', 0) < 0.5):
            return "⚠️ Exhausted", "Danger", 20
        
        # Default
        else:
            return "😴 Dormant", "Inactive", 10
            
    except Exception as e:
        logger.error(f"Error detecting wave stage: {str(e)}")
        return "❓ Unknown", "Error", 0

# ============================================
# FILTERING AND ANALYSIS FUNCTIONS
# ============================================

def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Apply user-selected filters"""
    filtered_df = df.copy()
    
    # Category filter
    if filters['categories'] and 'All' not in filters['categories']:
        filtered_df = filtered_df[filtered_df['category'].isin(filters['categories'])]
    
    # Sector filter
    if filters['sectors'] and 'All' not in filters['sectors']:
        filtered_df = filtered_df[filtered_df['sector'].isin(filters['sectors'])]
    
    # Volume filter
    if filters['min_volume'] > 0:
        filtered_df = filtered_df[filtered_df['volume_30d'] >= filters['min_volume']]
    
    # Price filter
    if filters['min_price'] > 0:
        filtered_df = filtered_df[filtered_df['price'] >= filters['min_price']]
    
    # Wave stage filter
    if filters['wave_stages'] and 'All' not in filters['wave_stages']:
        filtered_df['wave_type'] = filtered_df.apply(lambda x: detect_wave_stage(x)[1], axis=1)
        filtered_df = filtered_df[filtered_df['wave_type'].isin(filters['wave_stages'])]
    
    return filtered_df

def get_top_opportunities(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Get top opportunities based on wave score"""
    # Calculate scores and stages
    df['wave_score'] = df.apply(calculate_wave_score, axis=1)
    df[['wave_stage', 'wave_type', 'confidence']] = df.apply(
        lambda x: pd.Series(detect_wave_stage(x)), axis=1
    )
    
    # Sort by score and get top n
    top_df = df.nlargest(n, 'wave_score')
    
    return top_df

# ============================================
# REPORT GENERATION FUNCTIONS
# ============================================

def generate_excel_report(df: pd.DataFrame, analysis_df: pd.DataFrame) -> BytesIO:
    """Generate comprehensive Excel report"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#1e3c72',
            'font_color': 'white',
            'border': 1
        })
        
        number_format = workbook.add_format({'num_format': '#,##0.00'})
        percent_format = workbook.add_format({'num_format': '0.00%'})
        currency_format = workbook.add_format({'num_format': '₹#,##0.00'})
        
        # Sheet 1: Executive Summary
        summary_data = {
            'Metric': ['Total Stocks Analyzed', 'Active Waves', 'Forming Waves', 
                      'Average Wave Score', 'Top Opportunity Score'],
            'Value': [
                len(df),
                len(analysis_df[analysis_df['wave_type'] == 'Active']),
                len(analysis_df[analysis_df['wave_type'] == 'Forming']),
                f"{analysis_df['wave_score'].mean():.2f}",
                f"{analysis_df['wave_score'].max():.2f}"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
        
        # Sheet 2: Top Opportunities
        analysis_df.to_excel(writer, sheet_name='Top Opportunities', index=False)
        
        # Sheet 3: Active Waves
        active_waves = analysis_df[analysis_df['wave_type'] == 'Active']
        active_waves.to_excel(writer, sheet_name='Active Waves', index=False)
        
        # Sheet 4: Forming Waves
        forming_waves = analysis_df[analysis_df['wave_type'] == 'Forming']
        forming_waves.to_excel(writer, sheet_name='Forming Waves', index=False)
        
        # Sheet 5: Full Data
        df.to_excel(writer, sheet_name='Full Data', index=False)
        
        # Format sheets
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            worksheet.set_zoom(90)
            
            # Auto-fit columns
            for i, col in enumerate(df.columns):
                column_width = max(df[col].astype(str).str.len().max(), len(col)) + 2
                worksheet.set_column(i, i, min(column_width, 50))
    
    output.seek(0)
    return output

def generate_diagnostic_report(df: pd.DataFrame) -> Dict:
    """Generate system diagnostic information"""
    diagnostics = {
        'data_quality': {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_tickers': df['ticker'].duplicated().sum(),
            'data_freshness': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'coverage': {
            'categories': df['category'].value_counts().to_dict(),
            'sectors': df['sector'].nunique(),
            'price_range': f"₹{df['price'].min():.2f} - ₹{df['price'].max():.2f}",
            'market_cap_range': f"₹{df['market_cap_clean'].min():.0f} Cr - ₹{df['market_cap_clean'].max():.0f} Cr"
        },
        'wave_analysis': {
            'high_rvol_stocks': len(df[df['rvol'] > 2]),
            'accumulation_candidates': len(df[(df['vol_ratio_90d_180d'] > 0.9) & (df['from_low_pct'] < 30)]),
            'momentum_stocks': len(df[(df['ret_7d'] > 5) & (df['ret_30d'] > 10)]),
            'oversold_stocks': len(df[df['from_low_pct'] < 20])
        }
    }
    
    return diagnostics

# ============================================
# VISUALIZATION FUNCTIONS
# ============================================

def create_wave_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create wave stage distribution chart"""
    wave_counts = df['wave_type'].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=wave_counts.index,
            y=wave_counts.values,
            marker_color=['#28a745', '#ffc107', '#dc3545', '#6c757d']
        )
    ])
    
    fig.update_layout(
        title="Wave Stage Distribution",
        xaxis_title="Wave Stage",
        yaxis_title="Number of Stocks",
        showlegend=False,
        height=400
    )
    
    return fig

def create_sector_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create sector performance heatmap"""
    sector_perf = df.groupby('sector')['ret_30d'].mean().sort_values(ascending=False).head(15)
    
    fig = go.Figure(data=[
        go.Bar(
            x=sector_perf.values,
            y=sector_perf.index,
            orientation='h',
            marker_color=px.colors.diverging.RdYlGn[::1],
            text=[f"{x:.1f}%" for x in sector_perf.values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Top 15 Sectors by 30-Day Return",
        xaxis_title="Average 30-Day Return (%)",
        yaxis_title="Sector",
        height=600,
        margin=dict(l=200)
    )
    
    return fig

def create_wave_score_scatter(df: pd.DataFrame) -> go.Figure:
    """Create wave score vs return scatter plot"""
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=df['wave_score'],
        y=df['ret_30d'],
        mode='markers',
        marker=dict(
            size=df['rvol'] * 5,
            color=df['confidence'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Confidence")
        ),
        text=df['ticker'] + '<br>' + df['company_name'],
        hovertemplate='%{text}<br>Wave Score: %{x:.1f}<br>30d Return: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Wave Score vs 30-Day Return",
        xaxis_title="Wave Score",
        yaxis_title="30-Day Return (%)",
        height=500
    )
    
    return fig

# ============================================
# MAIN STREAMLIT APPLICATION
# ============================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🌊 Wave Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6c757d;">Advanced Stock Analysis Platform</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Data Source")
        sheet_url = st.text_input(
            "Google Sheets URL",
            value="https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk/edit?usp=sharing",
            help="Enter your Google Sheets URL"
        )
        gid = st.text_input("Sheet GID", value="2026492216", help="Sheet identifier")
        
        if st.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Load data
        df = load_google_sheets_data(sheet_url, gid)
        
        if not df.empty:
            df = clean_data(df)
            
            # Filters
            st.markdown("## 🔍 Filters")
            
            # Category filter
            categories = ['All'] + sorted(df['category'].dropna().unique().tolist())
            selected_categories = st.multiselect("Category", categories, default=['All'])
            
            # Sector filter
            sectors = ['All'] + sorted(df['sector'].dropna().unique().tolist())
            selected_sectors = st.multiselect("Sector", sectors, default=['All'])
            
            # Volume filter
            min_volume = st.number_input("Min 30-day Volume", min_value=0, value=50000, step=10000)
            
            # Price filter
            min_price = st.number_input("Min Price (₹)", min_value=0.0, value=0.0, step=10.0)
            
            # Wave stage filter
            wave_stages = ['All', 'Active', 'Forming', 'Danger', 'Inactive']
            selected_wave_stages = st.multiselect("Wave Stages", wave_stages, default=['Active', 'Forming'])
            
            # Analysis depth
            st.markdown("---")
            st.markdown("## ⚙️ Settings")
            top_n = st.slider("Top N Opportunities", min_value=5, max_value=50, value=20, step=5)
    
    # Main content
    if not df.empty:
        # Apply filters
        filters = {
            'categories': selected_categories,
            'sectors': selected_sectors,
            'min_volume': min_volume,
            'min_price': min_price,
            'wave_stages': selected_wave_stages
        }
        
        filtered_df = apply_filters(df, filters)
        
        # Get top opportunities
        analysis_df = get_top_opportunities(filtered_df, top_n)
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏄 Active Waves", "🌊 Forming Waves", "📊 Analytics", 
            "📈 Reports", "🔧 Diagnostics"
        ])
        
        with tab1:
            st.markdown("## 🏄 Active Waves - Ride These Now!")
            
            active_waves = analysis_df[analysis_df['wave_type'] == 'Active'].sort_values('wave_score', ascending=False)
            
            if not active_waves.empty:
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Active Waves", len(active_waves))
                with col2:
                    st.metric("Avg Score", f"{active_waves['wave_score'].mean():.1f}")
                with col3:
                    st.metric("Best Score", f"{active_waves['wave_score'].max():.1f}")
                with col4:
                    st.metric("Avg 7d Return", f"{active_waves['ret_7d'].mean():.1f}%")
                
                st.markdown("---")
                
                # Display active waves
                for _, row in active_waves.iterrows():
                    with st.container():
                        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{row['ticker']}** - {row['company_name'][:40]}...")
                            st.markdown(f"<small>{row['sector']}</small>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"**Score: {row['wave_score']:.1f}**")
                            st.markdown(f"{row['wave_stage']}")
                        
                        with col3:
                            st.markdown(f"**₹{row['price']:.2f}**")
                            st.markdown(f"Vol: {row['rvol']:.1f}x")
                        
                        with col4:
                            color = "green" if row['ret_7d'] > 0 else "red"
                            st.markdown(f"<span style='color:{color}'>7d: {row['ret_7d']:.1f}%</span>", 
                                      unsafe_allow_html=True)
                            st.markdown(f"30d: {row['ret_30d']:.1f}%")
                        
                        with col5:
                            st.markdown(f"**Target: +{max(40 - row['from_low_pct'], 10):.0f}%**")
                            st.markdown(f"Confidence: {row['confidence']:.0f}%")
                        
                        st.markdown("---")
            else:
                st.info("No active waves found with current filters")
        
        with tab2:
            st.markdown("## 🌊 Forming Waves - Prepare for Entry")
            
            forming_waves = analysis_df[analysis_df['wave_type'] == 'Forming'].sort_values('wave_score', ascending=False)
            
            if not forming_waves.empty:
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Forming Waves", len(forming_waves))
                with col2:
                    st.metric("Avg Score", f"{forming_waves['wave_score'].mean():.1f}")
                with col3:
                    st.metric("Best Setup", f"{forming_waves['wave_score'].max():.1f}")
                with col4:
                    st.metric("Avg Position", f"{forming_waves['from_low_pct'].mean():.0f}%")
                
                st.markdown("---")
                
                # Display forming waves
                for _, row in forming_waves.iterrows():
                    with st.container():
                        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{row['ticker']}** - {row['company_name'][:40]}...")
                            st.markdown(f"<small>{row['sector']}</small>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"**Score: {row['wave_score']:.1f}**")
                            st.markdown(f"{row['wave_stage']}")
                        
                        with col3:
                            st.markdown(f"**₹{row['price']:.2f}**")
                            st.markdown(f"90/180: {row['vol_ratio_90d_180d']:.2f}")
                        
                        with col4:
                            st.markdown(f"From Low: {row['from_low_pct']:.0f}%")
                            st.markdown(f"Room: {100 - row['from_low_pct']:.0f}%")
                        
                        with col5:
                            pressure = row['vol_ratio_90d_180d'] * 100
                            st.markdown(f"**Pressure: {pressure:.0f}%**")
                            eta = "1-3 days" if pressure > 100 else "3-7 days"
                            st.markdown(f"ETA: {eta}")
                        
                        st.markdown("---")
            else:
                st.info("No forming waves found with current filters")
        
        with tab3:
            st.markdown("## 📊 Market Analytics")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_wave_distribution_chart(analysis_df), use_container_width=True)
            
            with col2:
                st.plotly_chart(create_sector_heatmap(filtered_df), use_container_width=True)
            
            # Wave Score Analysis
            st.plotly_chart(create_wave_score_scatter(analysis_df), use_container_width=True)
            
            # Market Statistics
            st.markdown("### 📈 Market Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Stocks Above SMA20", 
                         f"{len(filtered_df[filtered_df['price'] > filtered_df['sma_20d']])} "
                         f"({len(filtered_df[filtered_df['price'] > filtered_df['sma_20d']]) / len(filtered_df) * 100:.1f}%)")
            
            with col2:
                st.metric("High Volume Stocks (rvol > 2)", 
                         f"{len(filtered_df[filtered_df['rvol'] > 2])}")
            
            with col3:
                st.metric("Near 52w Low (<20%)", 
                         f"{len(filtered_df[filtered_df['from_low_pct'] < 20])}")
            
            with col4:
                st.metric("Near 52w High (>-10%)", 
                         f"{len(filtered_df[filtered_df['from_high_pct'] > -10])}")
        
        with tab4:
            st.markdown("## 📈 Download Reports")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Generate Excel Report", type="primary"):
                    with st.spinner("Generating report..."):
                        excel_file = generate_excel_report(filtered_df, analysis_df)
                        
                        st.download_button(
                            label="📥 Download Excel Report",
                            data=excel_file,
                            file_name=f"wave_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            
            with col2:
                if st.button("📄 Generate PDF Summary", type="secondary"):
                    st.info("PDF generation coming soon!")
            
            # Quick summary
            st.markdown("### 📋 Quick Summary")
            
            summary_text = f"""
            **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
            
            **Stocks Analyzed:** {len(filtered_df)}
            
            **Top 5 Opportunities:**
            """
            
            for i, (_, row) in enumerate(analysis_df.head(5).iterrows(), 1):
                summary_text += f"\n{i}. **{row['ticker']}** (Score: {row['wave_score']:.1f}) - {row['wave_stage']}"
            
            st.text_area("Summary", summary_text, height=300)
        
        with tab5:
            st.markdown("## 🔧 System Diagnostics")
            
            diagnostics = generate_diagnostic_report(df)
            
            # Data Quality
            st.markdown("### 📊 Data Quality")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Rows", f"{diagnostics['data_quality']['total_rows']:,}")
            with col2:
                st.metric("Total Columns", diagnostics['data_quality']['total_columns'])
            with col3:
                st.metric("Missing Values", f"{diagnostics['data_quality']['missing_values']:,}")
            with col4:
                st.metric("Duplicate Tickers", diagnostics['data_quality']['duplicate_tickers'])
            
            # Coverage
            st.markdown("### 📈 Market Coverage")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Unique Sectors", diagnostics['coverage']['sectors'])
                st.metric("Price Range", diagnostics['coverage']['price_range'])
            
            with col2:
                categories_text = "\n".join([f"{k}: {v}" for k, v in diagnostics['coverage']['categories'].items()])
                st.text_area("Category Distribution", categories_text, height=150)
            
            # Wave Analysis Stats
            st.markdown("### 🌊 Wave Analysis Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("High Volume Stocks", diagnostics['wave_analysis']['high_rvol_stocks'])
            with col2:
                st.metric("Accumulation Candidates", diagnostics['wave_analysis']['accumulation_candidates'])
            with col3:
                st.metric("Momentum Stocks", diagnostics['wave_analysis']['momentum_stocks'])
            with col4:
                st.metric("Oversold Stocks", diagnostics['wave_analysis']['oversold_stocks'])
            
            # System Info
            st.markdown("### 💻 System Information")
            st.markdown(f"<p class='diagnostic-info'>Last Update: {diagnostics['data_quality']['data_freshness']}</p>", 
                       unsafe_allow_html=True)
            st.markdown(f"<p class='diagnostic-info'>Data Source: Google Sheets (GID: {gid})</p>", 
                       unsafe_allow_html=True)
            st.markdown(f"<p class='diagnostic-info'>Version: 1.0.0</p>", 
                       unsafe_allow_html=True)
    
    else:
        st.error("No data loaded. Please check your Google Sheets URL and GID.")

if __name__ == "__main__":
    main()
