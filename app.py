# ultimate_data_explorer.py - DEEP DATA UNDERSTANDING TOOL
"""
Ultimate Data Explorer for M.A.N.T.R.A.
======================================
This tool will analyze EVERY aspect of your data to build
the most powerful trading system ever created.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import io
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="M.A.N.T.R.A. Data Explorer", layout="wide")

st.title("🔬 M.A.N.T.R.A. Ultimate Data Explorer")
st.caption("Understanding every pattern, every correlation, every opportunity")

# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data
def load_raw_data():
    """Load data exactly as is"""
    url = "https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk/export?format=csv&gid=2026492216"
    response = requests.get(url)
    response.encoding = 'utf-8'
    df = pd.read_csv(io.StringIO(response.text))
    return df

@st.cache_data
def clean_data(df):
    """Clean and convert all data properly"""
    df_clean = df.copy()
    
    # Remove unnamed columns
    df_clean = df_clean.loc[:, ~df_clean.columns.str.contains('^Unnamed')]
    
    # Price columns
    price_cols = ['price', 'prev_close', 'low_52w', 'high_52w', 'sma_20d', 'sma_50d', 'sma_200d']
    for col in price_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Percentage columns
    pct_cols = ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
                'from_low_pct', 'from_high_pct', 'eps_change_pct',
                'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d']
    for col in pct_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.replace('%', '')
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Volume columns
    vol_cols = ['volume_1d', 'volume_7d', 'volume_30d', 'volume_3m']
    for col in vol_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.replace(',', '')
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Other numeric
    other_cols = ['pe', 'eps_current', 'eps_last_qtr', 'eps_duplicate', 'rvol', 'year']
    for col in other_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col].astype(str).str.replace(',', ''), errors='coerce')
    
    # Market cap
    if 'market_cap' in df_clean.columns:
        df_clean['market_cap_num'] = df_clean['market_cap'].str.extract(r'([\d,]+\.?\d*)')[0].str.replace(',', '').astype(float)
    
    return df_clean

# Load data
df_raw = load_raw_data()
df = clean_data(df_raw)

# ============================================================================
# MAIN ANALYSIS TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview", "🔍 Patterns", "🧮 Correlations", 
    "📈 Distributions", "🎯 Opportunities", "💡 Insights", "❓ Questions"
])

with tab1:
    st.header("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Stocks", f"{len(df):,}")
    with col2:
        st.metric("Data Columns", len(df.columns))
    with col3:
        st.metric("Sectors", df['sector'].nunique() if 'sector' in df.columns else 0)
    with col4:
        st.metric("Categories", df['category'].nunique() if 'category' in df.columns else 0)
    
    # Data Quality
    st.subheader("🔍 Data Quality Analysis")
    
    quality_metrics = []
    for col in df.columns:
        null_count = df[col].isna().sum()
        null_pct = (null_count / len(df) * 100)
        unique_count = df[col].nunique()
        dtype = str(df[col].dtype)
        
        quality_metrics.append({
            'Column': col,
            'Data Type': dtype,
            'Non-Null': len(df) - null_count,
            'Null %': f"{null_pct:.1f}%",
            'Unique Values': unique_count,
            'Sample Values': str(df[col].dropna().head(3).tolist())[:50] + "..."
        })
    
    quality_df = pd.DataFrame(quality_metrics)
    st.dataframe(quality_df, use_container_width=True, height=400)
    
    # Missing data heatmap
    st.subheader("🗺️ Missing Data Heatmap")
    fig_missing = go.Figure(data=go.Heatmap(
        z=df.isna().astype(int).T,
        y=df.columns,
        colorscale='RdYlGn_r',
        showscale=True
    ))
    fig_missing.update_layout(height=800, title="Missing Data Pattern (Red = Missing)")
    st.plotly_chart(fig_missing, use_container_width=True)

with tab2:
    st.header("🔍 Data Patterns Discovery")
    
    # Volume Patterns
    st.subheader("📊 Volume Ratio Patterns")
    
    vol_ratios = ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d']
    if all(col in df.columns for col in vol_ratios):
        
        # Distribution of volume ratios
        fig_vol = make_subplots(rows=1, cols=3, subplot_titles=vol_ratios)
        
        for i, col in enumerate(vol_ratios):
            fig_vol.add_trace(
                go.Histogram(x=df[col].dropna(), nbinsx=50, name=col),
                row=1, col=i+1
            )
        
        fig_vol.update_layout(height=400, title="Volume Ratio Distributions")
        st.plotly_chart(fig_vol, use_container_width=True)
        
        # Volume pattern combinations
        st.subheader("🎯 Smart Volume Patterns")
        
        # Pattern 1: Accumulation
        accumulation = df[
            (df['vol_ratio_30d_90d'] > 20) & 
            (df['vol_ratio_7d_90d'] > 10) & 
            (df['vol_ratio_1d_90d'] < 0)
        ]
        
        # Pattern 2: Distribution
        distribution = df[
            (df['vol_ratio_1d_90d'] > 100) & 
            (df['ret_30d'] > 20)
        ]
        
        # Pattern 3: Breakout
        breakout = df[
            (df['vol_ratio_1d_90d'] > 100) & 
            (df['vol_ratio_7d_90d'] > 50) & 
            (df['vol_ratio_30d_90d'] > 30)
        ]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accumulation Pattern", len(accumulation))
            st.caption("High 30d volume, low today")
        with col2:
            st.metric("Distribution Pattern", len(distribution))
            st.caption("High volume + high returns")
        with col3:
            st.metric("Breakout Pattern", len(breakout))
            st.caption("All volume ratios elevated")
    
    # Return Patterns
    st.subheader("📈 Return Patterns Across Timeframes")
    
    return_cols = ['ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y']
    available_returns = [col for col in return_cols if col in df.columns]
    
    if available_returns:
        # Momentum consistency
        momentum_quality = pd.DataFrame()
        for col in available_returns:
            momentum_quality[col] = (df[col] > 0).astype(int)
        
        momentum_quality['consistency_score'] = momentum_quality.sum(axis=1) / len(available_returns)
        
        # Show distribution
        fig_momentum = go.Figure(data=[
            go.Histogram(x=momentum_quality['consistency_score'], nbinsx=20)
        ])
        fig_momentum.update_layout(
            title="Momentum Consistency Score Distribution",
            xaxis_title="Consistency Score (0-1)",
            yaxis_title="Number of Stocks"
        )
        st.plotly_chart(fig_momentum, use_container_width=True)
        
        # Best momentum stocks
        best_momentum = df[momentum_quality['consistency_score'] == 1.0]
        st.write(f"**Perfect Momentum Stocks** (positive across all timeframes): {len(best_momentum)}")
        if len(best_momentum) > 0:
            st.dataframe(best_momentum[['ticker', 'company_name'] + available_returns].head(10))

with tab3:
    st.header("🧮 Correlation Analysis")
    
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Key correlations to explore
    st.subheader("🔑 Key Correlation Insights")
    
    if 'alpha_score' not in df.columns:
        # Create a simple alpha score for correlation analysis
        score_components = []
        if 'ret_30d' in df.columns:
            score_components.append(df['ret_30d'].rank(pct=True))
        if 'vol_ratio_30d_90d' in df.columns:
            score_components.append((df['vol_ratio_30d_90d'] > 20).astype(float))
        if 'eps_change_pct' in df.columns:
            score_components.append(df['eps_change_pct'].rank(pct=True))
        
        if score_components:
            df['alpha_score'] = sum(score_components) / len(score_components)
    
    # Calculate correlations with returns
    if 'ret_30d' in df.columns:
        correlations = []
        for col in numeric_cols:
            if col != 'ret_30d' and df[col].notna().sum() > 100:
                corr = df['ret_30d'].corr(df[col])
                if not np.isnan(corr):
                    correlations.append({
                        'Feature': col,
                        'Correlation with 30D Return': corr,
                        'Abs Correlation': abs(corr)
                    })
        
        corr_df = pd.DataFrame(correlations).sort_values('Abs Correlation', ascending=False)
        
        st.subheader("🎯 Features Most Correlated with 30D Returns")
        st.dataframe(corr_df.head(20))
        
        # Visualization
        fig_corr = go.Figure(data=[
            go.Bar(
                x=corr_df.head(15)['Correlation with 30D Return'],
                y=corr_df.head(15)['Feature'],
                orientation='h',
                marker_color=corr_df.head(15)['Correlation with 30D Return'],
                marker_colorscale='RdBu'
            )
        ])
        fig_corr.update_layout(
            title="Top Features Correlated with 30D Returns",
            xaxis_title="Correlation",
            height=500
        )
        st.plotly_chart(fig_corr, use_container_width=True)

with tab4:
    st.header("📈 Distribution Analysis")
    
    # PE Distribution
    if 'pe' in df.columns:
        st.subheader("📊 P/E Ratio Distribution")
        
        # Remove outliers for better visualization
        pe_clean = df[(df['pe'] > 0) & (df['pe'] < 100)]['pe']
        
        fig_pe = go.Figure()
        fig_pe.add_trace(go.Histogram(x=pe_clean, nbinsx=50, name='P/E Distribution'))
        fig_pe.add_vline(x=pe_clean.median(), line_dash="dash", 
                        annotation_text=f"Median: {pe_clean.median():.1f}")
        fig_pe.update_layout(title="P/E Ratio Distribution (0-100)")
        st.plotly_chart(fig_pe, use_container_width=True)
    
    # EPS Tier Distribution
    if 'eps_tier' in df.columns:
        st.subheader("📊 EPS Tier Distribution")
        
        eps_tier_counts = df['eps_tier'].value_counts()
        fig_eps = go.Figure(data=[
            go.Bar(x=eps_tier_counts.index, y=eps_tier_counts.values)
        ])
        fig_eps.update_layout(title="EPS Tier Distribution", xaxis_title="EPS Tier")
        st.plotly_chart(fig_eps, use_container_width=True)
    
    # Return distributions by timeframe
    st.subheader("📈 Return Distributions by Timeframe")
    
    return_cols = ['ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y']
    available = [col for col in return_cols if col in df.columns]
    
    if available:
        fig_returns = go.Figure()
        for col in available:
            fig_returns.add_trace(go.Box(y=df[col].dropna(), name=col))
        
        fig_returns.update_layout(
            title="Return Distribution by Timeframe",
            yaxis_title="Return %",
            showlegend=False
        )
        st.plotly_chart(fig_returns, use_container_width=True)

with tab5:
    st.header("🎯 Hidden Opportunities")
    
    opportunities = {}
    
    # 1. Oversold Quality Stocks
    if all(col in df.columns for col in ['from_low_pct', 'eps_tier', 'pe']):
        oversold_quality = df[
            (df['from_low_pct'] < 20) &  # Near 52w low
            (df['eps_tier'].isin(['35↑', '55↑', '75↑', '95↑'])) &  # Good EPS
            (df['pe'] > 0) & (df['pe'] < 25)  # Reasonable PE
        ]
        opportunities['Oversold Quality'] = oversold_quality
    
    # 2. Momentum Building
    if all(col in df.columns for col in ['ret_7d', 'ret_30d', 'vol_ratio_7d_90d']):
        momentum_building = df[
            (df['ret_7d'] > 5) &  # Recent positive
            (df['ret_30d'] < 5) & (df['ret_30d'] > -5) &  # Was consolidating
            (df['vol_ratio_7d_90d'] > 30)  # Volume picking up
        ]
        opportunities['Momentum Building'] = momentum_building
    
    # 3. Volume Surge Value
    if all(col in df.columns for col in ['pe', 'vol_ratio_1d_90d', 'ret_1d']):
        volume_value = df[
            (df['pe'] > 0) & (df['pe'] < 20) &
            (df['vol_ratio_1d_90d'] > 100) &
            (df['ret_1d'] > 2)
        ]
        opportunities['Volume Surge Value'] = volume_value
    
    # Display opportunities
    for opp_name, opp_df in opportunities.items():
        st.subheader(f"🎯 {opp_name} ({len(opp_df)} stocks)")
        if len(opp_df) > 0:
            display_cols = ['ticker', 'company_name', 'price', 'pe', 'ret_30d', 'eps_tier']
            display_cols = [col for col in display_cols if col in opp_df.columns]
            st.dataframe(opp_df[display_cols].head(10))

with tab6:
    st.header("💡 Data-Driven Insights")
    
    insights = []
    
    # Insight 1: Volume patterns
    if all(col in df.columns for col in ['vol_ratio_30d_90d', 'ret_30d']):
        high_vol_returns = df[df['vol_ratio_30d_90d'] > 50]['ret_30d'].mean()
        low_vol_returns = df[df['vol_ratio_30d_90d'] < -50]['ret_30d'].mean()
        
        insights.append(f"""
        **Volume-Return Relationship**: 
        - Stocks with 50%+ volume increase: Avg 30D return = {high_vol_returns:.1f}%
        - Stocks with 50%+ volume decrease: Avg 30D return = {low_vol_returns:.1f}%
        """)
    
    # Insight 2: EPS tier performance
    if all(col in df.columns for col in ['eps_tier', 'ret_1y']):
        eps_performance = df.groupby('eps_tier')['ret_1y'].agg(['mean', 'count'])
        best_eps_tier = eps_performance['mean'].idxmax()
        
        insights.append(f"""
        **EPS Tier Performance**:
        - Best performing tier: {best_eps_tier} with {eps_performance.loc[best_eps_tier, 'mean']:.1f}% avg 1Y return
        - Number of stocks: {eps_performance.loc[best_eps_tier, 'count']}
        """)
    
    # Insight 3: Sector analysis
    if all(col in df.columns for col in ['sector', 'ret_30d']):
        sector_perf = df.groupby('sector')['ret_30d'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        
        insights.append(f"""
        **Top 5 Performing Sectors (30D)**:
        {sector_perf.head(5).to_string()}
        """)
    
    # Display all insights
    for insight in insights:
        st.info(insight)
    
    # Statistical summary
    st.subheader("📊 Statistical Summary of Key Metrics")
    
    key_metrics = ['price', 'pe', 'ret_30d', 'ret_1y', 'vol_ratio_30d_90d', 'eps_change_pct']
    available_metrics = [m for m in key_metrics if m in df.columns]
    
    if available_metrics:
        summary_stats = df[available_metrics].describe()
        st.dataframe(summary_stats.round(2))

with tab7:
    st.header("❓ Critical Questions for Ultimate System Design")
    
    st.markdown("""
    Based on the data analysis, here are the critical questions to build the most powerful system:
    
    ### 🎯 **Pattern Recognition Questions**
    
    1. **Volume Intelligence**
       - When vol_ratio_30d_90d > 20% but vol_ratio_1d_90d < 0%, is this accumulation?
       - Should we weight sustained volume (30d) more than spike volume (1d)?
       - How do we handle negative volume ratios (declining volume)?
    
    2. **Momentum Dynamics**
       - Is momentum acceleration (ret_1d > ret_7d/7) more important than absolute momentum?
       - Should we prioritize stocks with consistent positive returns across timeframes?
       - How much should we penalize momentum divergence (price up but volume down)?
    
    3. **Valuation Context**
       - Should PE < 20 with EPS growth > 25% be our value sweet spot?
       - How do we handle negative PE stocks?
       - Is EPS tier transition (e.g., from 15↑ to 35↑) a strong signal?
    
    ### 🔬 **Algorithm Design Questions**
    
    4. **Signal Combination**
       - Should we use geometric mean (multiplicative) or arithmetic mean for scores?
       - How do we handle missing data - neutral score (0.5) or skip the factor?
       - Should regime (trending/ranging) change our scoring weights?
    
    5. **Risk Management**
       - Should high volatility (std of returns) reduce alpha score?
       - How much should we penalize stocks near 52-week highs?
       - Is low volume (rvol < 0.5) a disqualifying factor?
    
    6. **Special Situations**
       - What defines a "stealth breakout" in your market?
       - Should sector relative strength matter more than absolute performance?
       - How do we detect institutional accumulation patterns?
    
    ### 🚀 **Implementation Questions**
    
    7. **Thresholds**
       - Should ALPHA_EXTREME be top 5% or top 10%?
       - Do you want conservative (fewer signals) or aggressive (more signals)?
       - Should we have hard filters (automatic disqualification)?
    
    8. **Sector/Category Handling**
       - Should we score within sectors or across the entire market?
       - Do small-cap and large-cap need different algorithms?
       - Should we have sector-specific thresholds?
    
    ### 💎 **Your Specific Preferences**
    
    9. **What matters most to you?**
       - Catching big moves early (momentum focus)?
       - Finding undervalued gems (value focus)?
       - Following smart money (volume focus)?
       - Technical precision (chart patterns)?
    
    10. **Risk Tolerance**
       - Maximum acceptable PE ratio?
       - Minimum acceptable liquidity (volume)?
       - Preferred holding period (affects timeframe weights)?
    """)
    
    st.divider()
    
    st.markdown("""
    ### 📊 **Data-Specific Observations**
    
    From your data, I notice:
    - Volume ratios can be highly negative (down to -84%)
    - Many stocks have strong long-term returns (3Y, 5Y) but weak short-term
    - EPS tiers are well distributed across growth levels
    - Some sectors significantly outperform others
    
    **Should we build multiple strategies for different market conditions?**
    """)

# Download analysis results
st.sidebar.header("💾 Download Analysis")

analysis_data = {
    'total_stocks': len(df),
    'columns': df.columns.tolist(),
    'data_types': df.dtypes.astype(str).to_dict(),
    'missing_data': df.isnull().sum().to_dict(),
    'basic_stats': df.describe().to_dict() if len(df) > 0 else {}
}

if st.sidebar.button("Generate Analysis Report"):
    report = f"""
M.A.N.T.R.A. Data Analysis Report
=================================

Total Stocks: {len(df)}
Total Columns: {len(df.columns)}

Column Details:
{pd.DataFrame(quality_metrics).to_string()}

Key Findings:
- Stocks with perfect momentum (all periods positive): {len(df[momentum_quality['consistency_score'] == 1.0]) if 'momentum_quality' in locals() else 'N/A'}
- Accumulation pattern stocks: {len(accumulation) if 'accumulation' in locals() else 'N/A'}
- Oversold quality stocks: {len(opportunities.get('Oversold Quality', [])) if 'opportunities' in locals() else 'N/A'}

This data contains {len(df)} stocks with comprehensive price, volume, fundamental, and technical indicators.
The richness of this dataset allows for sophisticated multi-factor analysis and pattern recognition.
"""
    
    st.sidebar.download_button(
        "📥 Download Report",
        report,
        "mantra_analysis_report.txt",
        "text/plain"
    )
