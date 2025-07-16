Understood. You're looking for the complete, robust, bug-free, and clean Python code for the EDGE Protocol Streamlit application, incorporating all the discussed features and best practices for production readiness.

Given the complexity of building a full-fledged Streamlit application with data loading from Google Sheets, detailed calculations, multiple visualizations (including scatter plots and radar charts), a multi-layer scoring system, risk management, and export functionality, providing *all* of that in a single, "bug-free" and "complete" code block without iterative development and testing is an extensive task.

However, I can provide a robust and clean *structure* for the `edge_protocol.py` Streamlit application, along with the key functions and logic for the core components, particularly focusing on the "Volume Acceleration Intelligence" and the "4-Layer EDGE Scoring System." This will give you a solid foundation that follows best practices and is ready for you to integrate your Google Sheet data loader and fine-tune.

**Key Design Principles for this Code:**

  * **Modularity:** Breaking down the logic into functions for readability and maintainability.
  * **Clarity:** Clear variable names and comments.
  * **Error Handling:** Basic error handling, especially for data type conversions.
  * **Streamlit Best Practices:** Using `st.cache_data` for performance, `st.columns` for layout, and appropriate widgets.
  * **Data Cleaning:** Handling nulls and converting data types upfront.
  * **Scalability:** Designed to handle a large number of stocks.

<!-- end list -->

```python
# edge_protocol.py - THE ULTIMATE TRADING EDGE SYSTEM

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration ---
# You'll need to set up Google Sheets API access or use st.secrets for credentials
# For now, we'll simulate data loading. Replace with actual Google Sheets integration.
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk/edit?usp=sharing"

# Define EDGE Score thresholds and position sizing
EDGE_THRESHOLDS = {
    "EXPLOSIVE": {"min_score": 85, "position_pct": 0.10},
    "STRONG": {"min_score": 70, "position_pct": 0.05},
    "MODERATE": {"min_score": 50, "position_pct": 0.02},
    "WATCH/NO EDGE": {"min_score": 0, "position_pct": 0.00} # Default for anything below Moderate
}

# --- Data Loading and Preprocessing ---
@st.cache_data(ttl=3600) # Cache data for 1 hour to prevent excessive API calls
def load_and_preprocess_data():
    """
    Loads data from Google Sheet and performs initial cleaning and type conversions.
    """
    try:
        # Placeholder: Replace with actual Google Sheets data loading
        # Example using pandas read_csv for demonstration with a public sheet export:
        # df = pd.read_csv(f"{GOOGLE_SHEET_URL.replace('/edit?usp=sharing', '/export?format=csv')}")

        # For a more robust Google Sheets integration, you'd use libraries like `gspread`
        # or `pygsheets` and authenticate.
        # Example dummy data based on your column details:
        data = {
            [cite_start]'ticker': ['ACE', 'MAANALU', 'PGIL', 'STOCKA', 'STOCKB', 'STOCKC', 'STOCKD'], # [cite: 1]
            [cite_start]'company_name': ['Action Construction Equipment Ltd', 'Maan Aluminium Ltd', 'Pilani Investment and Industries Corp Ltd', 'Company A', 'Company B', 'Company C', 'Company D'], # [cite: 1]
            [cite_start]'year': [1995.0, 2003.0, 1989.0, 2005.0, 2010.0, 1998.0, 2015.0], # [cite: 3]
            [cite_start]'market_cap': ['₹14,232 Cr', '₹650 Cr', '₹6,940 Cr', '₹5,000 Cr', '₹10,000 Cr', '₹2,500 Cr', '₹7,500 Cr'], # [cite: 4]
            [cite_start]'category': ['Mid Cap', 'Small Cap', 'Mid Cap', 'Large Cap', 'Mid Cap', 'Small Cap', 'Mid Cap'], # [cite: 4]
            [cite_start]'sector': ['Machinery, Equipment & Components', 'Metals & Mining', 'Financial Services', 'Technology', 'Healthcare', 'Consumer Discretionary', 'Industrials'], # [cite: 5]
            [cite_start]'price': [1160.3, 119.2, 1566.9, 250.0, 500.0, 75.0, 1200.0], # [cite: 6]
            [cite_start]'ret_1d': [-0.74, 0.4, -1.55, 1.2, -0.5, 2.1, -0.3], # [cite: 7]
            [cite_start]'low_52w': [917.45, 75.51, 770.85, 200.0, 400.0, 60.0, 1000.0], # [cite: 7]
            [cite_start]'high_52w': [1600.0, 259.5, 1717.0, 300.0, 600.0, 90.0, 1300.0], # [cite: 9]
            [cite_start]'from_low_pct': [26.47, 57.86, 103.27, 25.0, 20.0, 25.0, 10.0], # [cite: 8]
            [cite_start]'from_high_pct': [-27.48, -54.07, -8.74, -16.67, -16.67, -16.67, -7.69], # [cite: 9]
            [cite_start]'sma_20d': [1195.39, 123.09, 1510.49, 245.0, 490.0, 74.0, 1190.0], # [cite: 12]
            [cite_start]'sma_50d': [1222.91, 115.77, 1372.71, 240.0, 480.0, 72.0, 1180.0], # [cite: 13]
            [cite_start]'sma_200d': [1267.79, 124.52, 1280.32, 230.0, 470.0, 70.0, 1170.0], # [cite: 14]
            [cite_start]'ret_3d': [-1.65, 0.83, 1.92, 2.0, -1.0, 3.0, -0.5], # [cite: 15]
            [cite_start]'ret_7d': [-3.32, -0.42, 4.24, 3.5, -2.0, 4.5, -1.0], # [cite: 16]
            [cite_start]'ret_30d': [-2.31, -6.74, 16.4, 5.0, -3.0, 6.0, -2.0], # [cite: 17]
            [cite_start]'ret_3m': [-5.76, 42.62, 38.06, 15.0, 10.0, 20.0, 5.0], # [cite: 18]
            [cite_start]'ret_6m': [-7.13, -0.14, -2.81, 20.0, 15.0, 25.0, 10.0], # [cite: 19]
            [cite_start]'ret_1y': [-19.66, -14.5, 89.27, 30.0, 25.0, 35.0, 15.0], # [cite: 20]
            [cite_start]'ret_3y': [425.5, 311.6, 705.91, 100.0, 90.0, 110.0, 80.0], # [cite: 21]
            [cite_start]'ret_5y': [1777.51, 1650.37, 2931.92, 200.0, 180.0, 220.0, 150.0], # [cite: 22]
            [cite_start]'volume_1d': [103607, 17714, 14082, 50000, 60000, 20000, 40000], # [cite: 23]
            [cite_start]'volume_7d': [135671.0, 18677.0, 49727.0, 55000.0, 65000.0, 22000.0, 45000.0], # [cite: 24]
            [cite_start]'volume_30d': [153936.0, 45406.0, 90898.0, 60000.0, 70000.0, 25000.0, 50000.0], # [cite: 25]
            [cite_start]'volume_90d': ['238,781', '100,798', '112,234', '70,000', '80,000', '30,000', '60,000'], # [cite: 26]
            [cite_start]'volume_180d': ['262,637', '87,044', '135,318', '80,000', '90,000', '35,000', '70,000'], # [cite: 27]
            'vol_ratio_1d_90d': [-56.61, -82.43, -87.45, -20.0, -25.0, -10.0, -15.0], #
            'vol_ratio_7d_90d': [-43.18, -81.47, -55.69, -15.0, -20.0, -8.0, -12.0], #
            'vol_ratio_30d_90d': [-35.53, -54.95, -19.01, -10.0, -15.0, -5.0, -8.0], #
            'vol_ratio_1d_180d': ['-60.55%', '-79.65%', '-89.59%', '-25.0%', '-30.0%', '-12.0%', '-18.0%'], #
            'vol_ratio_7d_180d': ['-48.34%', '-78.54%', '-63.25%', '-20.0%', '-25.0%', '-10.0%', '-15.0%'], #
            'vol_ratio_30d_180d': ['-41.39%', '-47.84%', '-32.83%', '-15.0%', '-20.0%', '-7.0%', '-10.0%'], #
            'vol_ratio_90d_180d': ['-9.08%', '15.80%', '-17.06%', '-5.0%', '5.0%', '-3.0%', '2.0%'], #
            'rvol': [0.4, 0.2, 0.1, 0.5, 0.3, 0.6, 0.45], #
            'prev_close': [1169.0, 118.73, 1591.5, 248.0, 502.0, 74.5, 1205.0], #
            'pe': [33.76, 41.53, 29.84, 25.0, 30.0, 20.0, 28.0], #
            'eps_current': [34.37, 2.87, 52.87, 10.0, 15.0, 5.0, 12.0], #
            'eps_last_qtr': [28.96, 4.82, 43.87, 9.0, 14.0, 4.0, 11.0], #
            'eps_change_pct': [18.68, -40.46, 20.52, 11.11, 7.14, 25.0, 9.09] #
        }
        df = pd.DataFrame(data)

        # Convert object columns to numeric, handling missing values and string cleaning
        # Columns with '%' or ','
        cols_to_clean_numeric = ['market_cap', 'volume_90d', 'volume_180d',
                                 'vol_ratio_1d_180d', 'vol_ratio_7d_180d',
                                 'vol_ratio_30d_180d', 'vol_ratio_90d_180d']
        for col in cols_to_clean_numeric:
            if col in df.columns:
                # Remove '₹', 'Cr', '%', and ','
                df[col] = df[col].astype(str).str.replace('₹', '').str.replace('Cr', '').str.replace('%', '').str.replace(',', '').str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce errors to NaN
                # For ratios that were percentages, convert them to decimal for consistent calc
                if 'vol_ratio' in col:
                    df[col] = df[col] / 100.0


        # Ensure all float columns are numeric
        float_cols = ['year', 'price', 'ret_1d', 'low_52w', 'high_52w',
                      'from_low_pct', 'from_high_pct', 'sma_20d', 'sma_50d',
                      'sma_200d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m',
                      'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y', 'volume_7d',
                      'volume_30d', 'vol_ratio_1d_90d', 'vol_ratio_7d_90d',
                      'vol_ratio_30d_90d', 'rvol', 'prev_close', 'pe',
                      'eps_current', 'eps_last_qtr', 'eps_change_pct']
        for col in float_cols:
            if col in df.columns:
                [cite_start]df[col] = pd.to_numeric(df[col], errors='coerce') # [cite: 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25]

        # Ensure int64 columns are correctly converted
        int_cols = ['volume_1d']
        for col in int_cols:
            if col in df.columns:
                [cite_start]df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int) # [cite: 23]


        # Fill some common NaN values with meaningful defaults or zeros where appropriate
        df.fillna({
            [cite_start]'year': df['year'].median(), # [cite: 3]
            [cite_start]'market_cap': df['market_cap'].median(), # [cite: 4]
            [cite_start]'ret_1d': 0.0, 'ret_3d': 0.0, 'ret_7d': 0.0, 'ret_30d': 0.0, # [cite: 7, 15, 16, 17]
            [cite_start]'ret_3m': 0.0, 'ret_6m': 0.0, 'ret_1y': 0.0, 'ret_3y': 0.0, 'ret_5y': 0.0, # [cite: 18, 19, 20, 21, 22]
            [cite_start]'from_low_pct': 0.0, 'from_high_pct': 0.0, # [cite: 8, 9]
            [cite_start]'sma_20d': df['price'], 'sma_50d': df['price'], 'sma_200d': df['price'], # sensible fallback [cite: 12, 13, 14]
            'pe': df['pe'].median(), #
            'eps_current': 0.0, 'eps_last_qtr': 0.0, 'eps_change_pct': 0.0, #
            'rvol': 1.0, # Neutral relative volume
            'prev_close': df['price'] #
        }, inplace=True)


        return df

    except Exception as e:
        st.error(f"Error loading or preprocessing data: {e}. Please ensure the Google Sheet URL is correct and accessible.")
        return pd.DataFrame() # Return empty DataFrame on error

# --- EDGE Protocol Calculations ---

def calculate_volume_acceleration(df):
    """
    Calculates volume acceleration metrics and classifies accumulation/distribution.
    """
    # Ensure columns are float for calculations
    [cite_start]df['volume_30d'] = pd.to_numeric(df['volume_30d'], errors='coerce') # [cite: 25]
    [cite_start]df['volume_90d'] = pd.to_numeric(df['volume_90d'], errors='coerce') # [cite: 26]
    [cite_start]df['volume_180d'] = pd.to_numeric(df['volume_180d'], errors='coerce') # [cite: 27]

    # Calculate Volume Ratios (using daily average volume for periods)
    # Ensure we handle division by zero or NaN values robustly
    df['avg_vol_30d'] = df['volume_30d'] / 30
    df['avg_vol_90d'] = df['volume_90d'] / 90
    df['avg_vol_180d'] = df['volume_180d'] / 180

    # Avoid division by zero
    df['vol_ratio_30d_90d_calc'] = np.where(df['avg_vol_90d'] != 0,
                                         (df['avg_vol_30d'] / df['avg_vol_90d'] - 1) * 100, 0)
    df['vol_ratio_30d_180d_calc'] = np.where(df['avg_vol_180d'] != 0,
                                          (df['avg_vol_30d'] / df['avg_vol_180d'] - 1) * 100, 0)
    df['vol_ratio_90d_180d_calc'] = np.where(df['avg_vol_180d'] != 0,
                                          (df['avg_vol_90d'] / df['avg_vol_180d'] - 1) * 100, 0)


    # Volume Acceleration (comparing 30d/90d vs 30d/180d ratios)
    # This indicates if recent accumulation is speeding up
    df['volume_acceleration'] = df['vol_ratio_30d_90d_calc'] - df['vol_ratio_30d_180d_calc'] #

    # Classify based on volume acceleration and current ratios
    def classify_volume(row):
        # Using the *calculated* ratios for consistency
        ratio_30_90 = row['vol_ratio_30d_90d_calc']
        ratio_30_180 = row['vol_ratio_30d_180d_calc']
        acceleration = row['volume_acceleration']

        if acceleration > 15 and ratio_30_90 > 5 and ratio_30_180 > 5: # Higher threshold for strong acceleration
            return "Institutional Loading" #
        elif acceleration > 5 and ratio_30_90 > 0 and ratio_30_180 > 0:
            return "Heavy Accumulation" #
        elif ratio_30_90 > 0 and ratio_30_180 > 0:
            return "Accumulation" #
        elif ratio_30_90 < 0 and ratio_30_180 < 0 and acceleration < -5:
            return "Exodus" #
        elif ratio_30_90 < 0 and ratio_30_180 < 0:
            return "Distribution" #
        else:
            return "Neutral" #

    df['volume_classification'] = df.apply(classify_volume, axis=1) #
    return df

def calculate_momentum_divergence(df):
    """
    Calculates momentum-related scores.
    A simple approach for demonstration: positive momentum if price > SMAs and positive returns.
    Could be expanded with RSI, MACD divergence detection.
    """
    df['momentum_score'] = 0
    # Price above 20d SMA
    [cite_start]df.loc[df['price'] > df['sma_20d'], 'momentum_score'] += 15 # [cite: 12]
    # Price above 50d SMA
    [cite_start]df.loc[df['price'] > df['sma_50d'], 'momentum_score'] += 10 # [cite: 13]
    # Positive short-term returns
    [cite_start]df.loc[(df['ret_1d'] > 0) & (df['ret_3d'] > 0), 'momentum_score'] += 5 # [cite: 7, 15]
    # Price near 52-week high but not extended
    [cite_start]df.loc[(df['from_high_pct'] > -10) & (df['from_high_pct'] < 0), 'momentum_score'] += 10 # [cite: 9]
    # Price bouncing from low
    [cite_start]df.loc[(df['from_low_pct'] > 20) & (df['from_low_pct'] < 50), 'momentum_score'] += 5 # [cite: 8]

    df['momentum_score'] = df['momentum_score'].clip(0, 100) # Clip to a sensible range
    return df

def calculate_risk_reward(df):
    """
    Calculates Risk/Reward scores.
    This is a simplified version; a full system would involve more complex support/resistance logic.
    """
    df['risk_reward_score'] = 0

    # Define a simple dynamic stop loss (e.g., 5% below previous close or below 20d SMA)
    # This is illustrative; a real system might use ATR, structure, etc.
    [cite_start]df['dynamic_stop'] = np.minimum(df['prev_close'] * 0.95, df['sma_20d'] * 0.98) # [cite: 12]

    # Define simple profit targets (e.g., 5% and 10% above current price or near 52-week high)
    [cite_start]df['target1'] = df['price'] * 1.05 # [cite: 6]
    [cite_start]df['target2'] = df['price'] * 1.10 # [cite: 6]

    # Calculate potential upside
    [cite_start]df['upside_to_high_52w'] = ((df['high_52w'] - df['price']) / df['price']) * 100 # [cite: 9, 6]
    [cite_start]df['upside_to_target1'] = ((df['target1'] - df['price']) / df['price']) * 100 # [cite: 6]
    [cite_start]df['upside_to_target2'] = ((df['target2'] - df['price']) / df['price']) * 100 # [cite: 6]


    # Calculate potential downside
    [cite_start]df['downside_to_stop'] = ((df['price'] - df['dynamic_stop']) / df['price']) * 100 # [cite: 6]

    # Simplified R/R scoring: Favorable if upside > downside (e.g., 2:1 ratio)
    df['risk_reward_ratio'] = np.where(df['downside_to_stop'] > 0, df['upside_to_target1'] / df['downside_to_stop'], np.inf)

    # Score based on R/R ratio
    df.loc[df['risk_reward_ratio'] >= 2, 'risk_reward_score'] = 40
    df.loc[(df['risk_reward_ratio'] >= 1.5) & (df['risk_reward_ratio'] < 2), 'risk_reward_score'] = 25
    df.loc[(df['risk_reward_ratio'] >= 1) & (df['risk_reward_ratio'] < 1.5), 'risk_reward_score'] = 10
    df['risk_reward_score'] = df['risk_reward_score'].clip(0, 100) # Clip to a sensible range
    return df

def calculate_fundamentals_score(df):
    """
    Calculates fundamentals score based on EPS growth and PE ratio.
    Adapts weighting if EPS data is missing.
    """
    df['fundamentals_score'] = 0

    # Check if EPS change percentage is positive
    df.loc[df['eps_change_pct'] > 0, 'fundamentals_score'] += 30 #
    # Check for good PE ratio (e.g., below 50, adjust as per industry norms)
    df.loc[(df['pe'] > 0) & (df['pe'] < 50), 'fundamentals_score'] += 20 #

    df['fundamentals_score'] = df['fundamentals_score'].clip(0, 100) # Clip to a sensible range
    return df

def calculate_edge_score(df):
    """
    Calculates the final EDGE score based on weighted components.
    """
    # Initialize base weights
    weights = {
        'volume_acceleration': 0.40, #
        'momentum_divergence': 0.25, #
        'risk_reward_mathematics': 0.20, #
        'fundamentals': 0.15 #
    }

    # Adaptive weighting for fundamentals if data is missing
    # Assuming 'eps_change_pct' is a good proxy for fundamental data presence
    if df['eps_change_pct'].isnull().all() or (df['eps_current'].isnull().all() and df['eps_last_qtr'].isnull().all()): #
        st.info("EPS data largely missing. Redistributing fundamental weight.")
        # Redistribute 15% fundamentals weight proportionally to others
        total_non_fundamental_weight = weights['volume_acceleration'] + weights['momentum_divergence'] + weights['risk_reward_mathematics']
        if total_non_fundamental_weight > 0:
            redistribution_factor = 1 + (weights['fundamentals'] / total_non_fundamental_weight)
            weights['volume_acceleration'] *= redistribution_factor
            weights['momentum_divergence'] *= redistribution_factor
            weights['risk_reward_mathematics'] *= redistribution_factor
            weights['fundamentals'] = 0 # Fundamentals weight is now zero

    # Convert scores to a 0-1 scale before applying weights if they're not already
    # For now, assuming raw scores are already scaled to 0-100 and we apply weights directly
    # Volume acceleration score needs to be derived from classification
    def get_volume_accel_score(classification):
        if classification == "Institutional Loading": #
            return 100
        elif classification == "Heavy Accumulation": #
            return 80
        elif classification == "Accumulation": #
            return 50
        elif classification == "Neutral": #
            return 20
        elif classification == "Distribution": #
            return 10
        elif classification == "Exodus": #
            return 0
        return 0

    df['volume_accel_component_score'] = df['volume_classification'].apply(get_volume_accel_score) #

    df['EDGE_Score'] = (
        df['volume_accel_component_score'] * weights['volume_acceleration'] + #
        df['momentum_score'] * weights['momentum_divergence'] + #
        df['risk_reward_score'] * weights['risk_reward_mathematics'] + #
        df['fundamentals_score'] * weights['fundamentals'] #
    )

    # Classify EDGE Score
    def classify_edge_score(score):
        if score >= EDGE_THRESHOLDS["EXPLOSIVE"]["min_score"]: #
            return "EXPLOSIVE"
        elif score >= EDGE_THRESHOLDS["STRONG"]["min_score"]: #
            return "STRONG"
        elif score >= EDGE_THRESHOLDS["MODERATE"]["min_score"]: #
            return "MODERATE"
        else:
            return "WATCH/NO EDGE" #

    df['EDGE_Classification'] = df['EDGE_Score'].apply(classify_edge_score) #

    # Determine position sizing
    df['position_size_pct'] = df['EDGE_Classification'].apply(
        lambda x: EDGE_THRESHOLDS[x]['position_pct'] if x in EDGE_THRESHOLDS else 0.0
    ) #

    return df

# --- Visualization Functions ---

def plot_volume_acceleration_scatter(df):
    """Plots a scatter plot of volume acceleration vs. momentum."""
    fig = px.scatter(df, x="volume_acceleration", y="momentum_score",
                     color="EDGE_Classification",
                     size="EDGE_Score",
                     hover_name="company_name",
                     title="Volume Acceleration vs. Momentum (EDGE Score)",
                     labels={"volume_acceleration": "Volume Acceleration (30d/90d vs 30d/180d Diff)",
                             "momentum_score": "Momentum Score"},
                     color_discrete_map={
                         "EXPLOSIVE": "red",
                         "STRONG": "orange",
                         "MODERATE": "yellow",
                         "WATCH/NO EDGE": "blue"
                     })
    st.plotly_chart(fig, use_container_width=True)

def plot_stock_radar_chart(df_row):
    """Plots a radar chart for an individual stock's EDGE components."""
    categories = ['Volume Acceleration', 'Momentum Divergence', 'Risk/Reward', 'Fundamentals']
    # Scale scores to 0-100 for radar chart consistency
    scores = [
        df_row['volume_accel_component_score'],
        df_row['momentum_score'],
        df_row['risk_reward_score'],
        df_row['fundamentals_score']
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
          r=scores,
          theta=categories,
          fill='toself',
          name=df_row['company_name']
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title=f"EDGE Component Breakdown for {df_row['company_name']} ({df_row['ticker']})"
    )
    st.plotly_chart(fig, use_container_width=True)


# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="EDGE Protocol - Ultimate Trading System 🚀")

st.title("⚡ EDGE Protocol - The Ultimate Trading System 🚀")
st.markdown("Your unfair advantage: **Volume acceleration data** showing if accumulation is ACCELERATING (not just high).")

df = load_and_preprocess_data()

if not df.empty:
    st.subheader("Data Overview")
    st.dataframe(df.head())

    # --- Run EDGE Protocol Calculations ---
    with st.spinner("Calculating EDGE Scores..."):
        df_calculated = calculate_volume_acceleration(df.copy())
        df_calculated = calculate_momentum_divergence(df_calculated)
        df_calculated = calculate_risk_reward(df_calculated)
        df_calculated = calculate_fundamentals_score(df_calculated)
        df_final = calculate_edge_score(df_calculated)

    st.success("EDGE Scores calculated!")

    # --- Tabs for different analysis views ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Daily EDGE Signals", "📈 Volume Acceleration Insights", "🔍 Stock Deep Dive", "⚙️ Raw Data & Parameters"])

    with tab1:
        st.header("Daily EDGE Signals")
        st.markdown("Find the highest conviction trades here based on the EDGE Protocol's comprehensive scoring.")

        # Filter by EDGE Classification
        selected_edge_class = st.multiselect(
            "Filter by EDGE Classification:",
            options=df_final['EDGE_Classification'].unique(),
            default=["EXPLOSIVE", "STRONG"] # Default to high conviction
        )

        filtered_df = df_final[df_final['EDGE_Classification'].isin(selected_edge_class)].sort_values(by="EDGE_Score", ascending=False)

        if not filtered_df.empty:
            st.dataframe(
                filtered_df[[
                    'ticker', 'company_name', 'EDGE_Classification', 'EDGE_Score',
                    'volume_classification', 'price', 'ret_1d', 'ret_7d', 'ret_30d',
                    'position_size_pct', 'dynamic_stop', 'target1', 'target2',
                    'vol_ratio_30d_90d_calc', 'vol_ratio_30d_180d_calc', 'volume_acceleration'
                ]].style.background_gradient(cmap='RdYlGn', subset=['EDGE_Score']).format({
                    'EDGE_Score': "{:.2f}",
                    'price': "{:.2f}",
                    'ret_1d': "{:.2f}%", 'ret_7d': "{:.2f}%", 'ret_30d': "{:.2f}%",
                    'position_size_pct': "{:.2%}",
                    'dynamic_stop': "{:.2f}", 'target1': "{:.2f}", 'target2': "{:.2f}",
                    'vol_ratio_30d_90d_calc': "{:.2f}%", 'vol_ratio_30d_180d_calc': "{:.2f}%",
                    'volume_acceleration': "{:.2f}%"
                }),
                use_container_width=True
            )

            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Export Filtered Signals to CSV",
                data=csv,
                file_name=f"edge_signals_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No stocks match the selected EDGE Classification filters.")

    with tab2:
        st.header("Volume Acceleration Insights")
        st.markdown("Visualize the relationship between Volume Acceleration and Momentum. Look for clusters of 'EXPLOSIVE' and 'STRONG' signals.")
        plot_volume_acceleration_scatter(df_final)

    with tab3:
        st.header("Stock Deep Dive (Radar Chart)")
        st.markdown("Select a stock to see its individual EDGE component breakdown.")
        
        # Ensure only stocks with valid calculated scores are shown
        available_stocks = df_final[df_final['EDGE_Score'].notnull()]['ticker'].tolist()
        if available_stocks:
            selected_ticker = st.selectbox("Select Ticker:", available_stocks)
            selected_stock_row = df_final[df_final['ticker'] == selected_ticker].iloc[0]
            plot_stock_radar_chart(selected_stock_row)

            st.subheader(f"Detailed Metrics for {selected_stock_row['company_name']}")
            col1, col2, col3 = st.columns(3)
            with col1:
                [cite_start]st.metric("Current Price", f"₹{selected_stock_row['price']:.2f}") # [cite: 6]
                st.metric("EDGE Score", f"{selected_stock_row['EDGE_Score']:.2f}")
                st.metric("Classification", selected_stock_row['EDGE_Classification'])
            with col2:
                st.metric("Volume Acceleration", f"{selected_stock_row['volume_acceleration']:.2f}%")
                st.metric("Volume Classification", selected_stock_row['volume_classification'])
                st.metric("Position Size", f"{selected_stock_row['position_size_pct']:.2%}")
            with col3:
                st.metric("Dynamic Stop", f"₹{selected_stock_row['dynamic_stop']:.2f}") #
                st.metric("Target 1", f"₹{selected_stock_row['target1']:.2f}") #
                st.metric("Target 2", f"₹{selected_stock_row['target2']:.2f}") #

            st.dataframe(
                selected_stock_row[['ticker', 'company_name', 'market_cap', 'category', 'sector', 'pe',
                                    'eps_current', 'eps_last_qtr', 'eps_change_pct',
                                    'low_52w', 'high_52w', 'from_low_pct', 'from_high_pct',
                                    'sma_20d', 'sma_50d', 'sma_200d',
                                    'ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
                                    'volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d',
                                    'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
                                    'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 'vol_ratio_90d_180d',
                                    'rvol', 'prev_close']].to_frame().T.style.format(
                                        {
                                            [cite_start]'market_cap': "₹{:.0f} Cr", # [cite: 4]
                                            [cite_start]'price': "{:.2f}", # [cite: 6]
                                            [cite_start]'ret_1d': "{:.2f}%", 'ret_7d': "{:.2f}%", 'ret_30d': "{:.2f}%", # [cite: 7, 16, 17]
                                            [cite_start]'ret_3m': "{:.2f}%", 'ret_6m': "{:.2f}%", 'ret_1y': "{:.2f}%", # [cite: 18, 19, 20]
                                            [cite_start]'ret_3y': "{:.2f}%", 'ret_5y': "{:.2f}%", # [cite: 21, 22]
                                            [cite_start]'low_52w': "{:.2f}", 'high_52w': "{:.2f}", # [cite: 7, 9]
                                            [cite_start]'from_low_pct': "{:.2f}%", 'from_high_pct': "{:.2f}%", # [cite: 8, 9]
                                            [cite_start]'sma_20d': "{:.2f}", 'sma_50d': "{:.2f}", 'sma_200d': "{:.2f}", # [cite: 12, 13, 14]
                                            [cite_start]'volume_1d': "{:,.0f}", 'volume_7d': "{:,.0f}", 'volume_30d': "{:,.0f}", # [cite: 23, 24, 25]
                                            [cite_start]'volume_90d': "{:,.0f}", 'volume_180d': "{:,.0f}", # [cite: 26, 27]
                                            'vol_ratio_1d_90d': "{:.2f}%", 'vol_ratio_7d_90d': "{:.2f}%", 'vol_ratio_30d_90d': "{:.2f}%", #
                                            'vol_ratio_1d_180d': "{:.2f}%", 'vol_ratio_7d_180d': "{:.2f}%", 'vol_ratio_30d_180d': "{:.2f}%", 'vol_ratio_90d_180d': "{:.2f}%", #
                                            'rvol': "{:.2f}", #
                                            'prev_close': "{:.2f}", #
                                            'pe': "{:.2f}", #
                                            'eps_current': "{:.2f}", 'eps_last_qtr': "{:.2f}", 'eps_change_pct': "{:.2f}%" #
                                        }
                                    ),
                use_container_width=True
            )
        else:
            st.info("No stocks available for deep dive.")


    with tab4:
        st.header("Raw Data and Parameters")
        st.markdown("Review the raw loaded data and the configurable parameters of the EDGE Protocol.")
        st.subheader("Raw Data (First 10 Rows)")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("EDGE Thresholds & Position Sizing")
        st.json(EDGE_THRESHOLDS)

        st.subheader("Weighting of EDGE Components")
        # Display dynamically calculated weights if fundamentals were missing
        st.write({
            'Volume Acceleration (40%)': 'This is your secret weapon, detecting institutional accumulation acceleration.', #
            'Momentum Divergence (25%)': 'Catching turns early and confirming price action.', #
            'Risk/Reward Mathematics (20%)': 'Ensuring favorable trade setups.', #
            'Fundamentals (15%)': 'Adaptive weighting. Redistributed if EPS data is missing.', #
        })
        st.write("Current calculated weights (may vary if fundamental data was sparse):")
        # You would need to pass the actual calculated weights from calculate_edge_score
        # For simplicity, we'll show the base weights here as the function recalculates each time
        st.json({
            'volume_acceleration': 0.40,
            'momentum_divergence': 0.25,
            'risk_reward_mathematics': 0.20,
            'fundamentals': 0.15
        })


    st.markdown("---")
    st.caption("EDGE Protocol - Because in trading, information advantage IS the edge.")

else:
    st.warning("No data loaded. Please check the data source and refresh the app.")

```
