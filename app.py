# edge_protocol.py - THE ULTIMATE TRADING EDGE SYSTEM

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration ---
# You'll need to set up Google Sheets API access or use st.secrets for credentials
# For now, we'll simulate data loading. Replace with actual Google Sheets integration.
# To connect to Google Sheets, you'd typically use `gspread` or `pygsheets` with a service account.
# For a simple public sheet, you might use:
# GOOGLE_SHEET_EXPORT_URL = "https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk/export?format=csv"
# df = pd.read_csv(GOOGLE_SHEET_EXPORT_URL)
# For this example, we use hardcoded dummy data for immediate execution.

# Define EDGE Score thresholds and position sizing
EDGE_THRESHOLDS = {
    "EXPLOSIVE": {"min_score": 85, "position_pct": 0.10}, #
    "STRONG": {"min_score": 70, "position_pct": 0.05}, #
    "MODERATE": {"min_score": 50, "position_pct": 0.02}, #
    "WATCH/NO EDGE": {"min_score": 0, "position_pct": 0.00} # Default for anything below Moderate
}

# --- Data Loading and Preprocessing ---
@st.cache_data(ttl=3600) # Cache data for 1 hour to prevent excessive API calls
def load_and_preprocess_data():
    """
    Loads data (simulated) and performs initial cleaning and type conversions.
    In a real application, replace this with Google Sheets API integration.
    """
    try:
        # [cite_start]Example dummy data based on your column details [cite: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
        data = {
            [cite_start]'ticker': ['ACE', 'MAANALU', 'PGIL', 'STOCKA', 'STOCKB', 'STOCKC', 'STOCKD', 'STOCKE', 'STOCKF'], # [cite: 1]
            [cite_start]'company_name': ['Action Construction Equipment Ltd', 'Maan Aluminium Ltd', 'Pilani Investment and Industries Corp Ltd', 'Company A', 'Company B', 'Company C', 'Company D', 'Company E', 'Company F'], # [cite: 2]
            [cite_start]'year': [1995.0, 2003.0, 1989.0, 2005.0, 2010.0, 1998.0, 2015.0, np.nan, 2007.0], # [cite: 3]
            [cite_start]'market_cap': ['₹14,232 Cr', '₹650 Cr', '₹6,940 Cr', '₹5,000 Cr', '₹10,000 Cr', '₹2,500 Cr', '₹7,500 Cr', '₹1,500 Cr', '₹9,000 Cr'], # [cite: 4]
            [cite_start]'category': ['Mid Cap', 'Small Cap', 'Mid Cap', 'Large Cap', 'Mid Cap', 'Small Cap', 'Mid Cap', 'Small Cap', 'Large Cap'], # [cite: 5]
            [cite_start]'sector': ['Machinery, Equipment & Components', 'Metals & Mining', 'Financial Services', 'Technology', 'Healthcare', 'Consumer Discretionary', 'Industrials', 'Energy', 'Financial Services'], # [cite: 5]
            [cite_start]'price': [1160.3, 119.2, 1566.9, 250.0, 500.0, 75.0, 1200.0, 50.0, 300.0], # [cite: 6]
            [cite_start]'ret_1d': [-0.74, 0.4, -1.55, 1.2, -0.5, 2.1, -0.3, 0.8, -0.1], # [cite: 7]
            [cite_start]'low_52w': [917.45, 75.51, 770.85, 200.0, 400.0, 60.0, 1000.0, 40.0, 250.0], # [cite: 8]
            [cite_start]'high_52w': [1600.0, 259.5, 1717.0, 300.0, 600.0, 90.0, 1300.0, 70.0, 350.0], # [cite: 9]
            [cite_start]'from_low_pct': [26.47, 57.86, 103.27, 25.0, 20.0, 25.0, 10.0, 15.0, 18.0], # [cite: 10]
            [cite_start]'from_high_pct': [-27.48, -54.07, -8.74, -16.67, -16.67, -16.67, -7.69, -28.57, -14.28], # [cite: 11]
            [cite_start]'sma_20d': [1195.39, 123.09, 1510.49, 245.0, 490.0, 74.0, 1190.0, 52.0, 290.0], # [cite: 12]
            [cite_start]'sma_50d': [1222.91, 115.77, 1372.71, 240.0, 480.0, 72.0, 1180.0, 55.0, 280.0], # [cite: 13]
            [cite_start]'sma_200d': [1267.79, 124.52, 1280.32, 230.0, 470.0, 70.0, 1170.0, 58.0, 270.0], # [cite: 14]
            [cite_start]'ret_3d': [-1.65, 0.83, 1.92, 2.0, -1.0, 3.0, -0.5, 1.5, -0.8], # [cite: 15]
            [cite_start]'ret_7d': [-3.32, -0.42, 4.24, 3.5, -2.0, 4.5, -1.0, 2.0, -1.5], # [cite: 16]
            [cite_start]'ret_30d': [-2.31, -6.74, 16.4, 5.0, -3.0, 6.0, -2.0, 3.0, -2.5], # [cite: 17]
            [cite_start]'ret_3m': [-5.76, 42.62, 38.06, 15.0, 10.0, 20.0, 5.0, 8.0, 12.0], # [cite: 18]
            [cite_start]'ret_6m': [-7.13, -0.14, -2.81, 20.0, 15.0, 25.0, 10.0, 12.0, 18.0], # [cite: 19]
            [cite_start]'ret_1y': [-19.66, -14.5, 89.27, 30.0, 25.0, 35.0, 15.0, 20.0, 28.0], # [cite: 20]
            [cite_start]'ret_3y': [425.5, 311.6, 705.91, 100.0, 90.0, 110.0, 80.0, 70.0, 95.0], # [cite: 21]
            [cite_start]'ret_5y': [1777.51, 1650.37, 2931.92, 200.0, 180.0, 220.0, 150.0, 130.0, 190.0], # [cite: 22]
            [cite_start]'volume_1d': [103607, 17714, 14082, 50000, 60000, 20000, 40000, 10000, 35000], # [cite: 23]
            [cite_start]'volume_7d': [135671.0, 18677.0, 49727.0, 55000.0, 65000.0, 22000.0, 45000.0, 12000.0, 38000.0], # [cite: 24]
            [cite_start]'volume_30d': [153936.0, 45406.0, 90898.0, 60000.0, 70000.0, 25000.0, 50000.0, 15000.0, 40000.0], # [cite: 25]
            [cite_start]'volume_90d': ['238,781', '100,798', '112,234', '70,000', '80,000', '30,000', '60,000', '20,000', '45,000'], # [cite: 26]
            [cite_start]'volume_180d': ['262,637', '87,044', '135,318', '80,000', '90,000', '35,000', '70,000', '25,000', '50,000'], # [cite: 27]
            'vol_ratio_1d_90d': [-56.61, -82.43, -87.45, -20.0, -25.0, -10.0, -15.0, -5.0, -18.0], #
            'vol_ratio_7d_90d': [-43.18, -81.47, -55.69, -15.0, -20.0, -8.0, -12.0, -3.0, -16.0], #
            'vol_ratio_30d_90d': [-35.53, -54.95, -19.01, -10.0, -15.0, -5.0, -8.0, -2.0, -12.0], #
            'vol_ratio_1d_180d': ['-60.55%', '-79.65%', '-89.59%', '-25.0%', '-30.0%', '-12.0%', '-18.0%', '-8.0%', '-20.0%'], #
            'vol_ratio_7d_180d': ['-48.34%', '-78.54%', '-63.25%', '-20.0%', '-25.0%', '-10.0%', '-15.0%', '-5.0%', '-18.0%'], #
            'vol_ratio_30d_180d': ['-41.39%', '-47.84%', '-32.83%', '-15.0%', '-20.0%', '-7.0%', '-10.0%', '-4.0%', '-14.0%'], #
            'vol_ratio_90d_180d': ['-9.08%', '15.80%', '-17.06%', '-5.0%', '5.0%', '-3.0%', '2.0%', '1.0%', '3.0%'], #
            'rvol': [0.4, 0.2, 0.1, 0.5, 0.3, 0.6, 0.45, 0.7, 0.35], #
            'prev_close': [1169.0, 118.73, 1591.5, 248.0, 502.0, 74.5, 1205.0, 49.0, 298.0], #
            'pe': [33.76, 41.53, 29.84, 25.0, 30.0, 20.0, 28.0, 35.0, np.nan], #
            'eps_current': [34.37, 2.87, 52.87, 10.0, 15.0, 5.0, 12.0, 1.5, 8.0], #
            'eps_last_qtr': [28.96, 4.82, 43.87, 9.0, 14.0, 4.0, 11.0, 1.2, 7.5], #
            'eps_change_pct': [18.68, -40.46, 20.52, 11.11, 7.14, 25.0, 9.09, 25.0, np.nan] #
        }
        df = pd.DataFrame(data)

        # --- Data Cleaning and Type Conversion ---

        # 1. Clean 'market_cap'
        if 'market_cap' in df.columns:
            df['market_cap'] = df['market_cap'].astype(str).str.replace('₹', '').str.replace('Cr', '').str.replace(',', '').str.strip()
            [cite_start]df['market_cap'] = pd.to_numeric(df['market_cap'], errors='coerce') # [cite: 4]

        # [cite_start]2. Clean volume_XXd columns that are strings with commas [cite: 26, 27]
        cols_to_clean_volume_str = ['volume_90d', 'volume_180d']
        for col in cols_to_clean_volume_str:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '').str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 3. Clean percentage string columns and convert to float (decimal)
        cols_to_clean_pct = ['vol_ratio_1d_180d', 'vol_ratio_7d_180d',
                             'vol_ratio_30d_180d', 'vol_ratio_90d_180d']
        for col in cols_to_clean_pct:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('%', '').str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce') / 100.0 # Convert to decimal


        # [cite_start]4. Ensure all specified float columns are numeric [cite: 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25]
        float_cols = ['year', 'price', 'ret_1d', 'low_52w', 'high_52w',
                      'from_low_pct', 'from_high_pct', 'sma_20d', 'sma_50d',
                      'sma_200d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m',
                      'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y', 'volume_7d',
                      'volume_30d', 'vol_ratio_1d_90d', 'vol_ratio_7d_90d',
                      'vol_ratio_30d_90d', 'rvol', 'prev_close', 'pe',
                      'eps_current', 'eps_last_qtr', 'eps_change_pct']
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # [cite_start]5. Ensure int64 columns are correctly converted [cite: 23]
        int_cols = ['volume_1d']
        for col in int_cols:
            if col in df.columns:
                # Use float first to handle NaNs, then convert to Int64 (pandas nullable integer)
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                df[col] = df[col].fillna(0).astype(int) # Fill NaN with 0 then convert to non-nullable int


        # 6. Fill remaining NaN values with meaningful defaults
        # For SMA values, if they are NaN, use the current price as a neutral fallback
        [cite_start]df['sma_20d'] = df['sma_20d'].fillna(df['price']) # [cite: 12]
        [cite_start]df['sma_50d'] = df['sma_50d'].fillna(df['price']) # [cite: 13]
        [cite_start]df['sma_200d'] = df['sma_200d'].fillna(df['price']) # [cite: 14]

        # Other numerical columns
        df.fillna({
            [cite_start]'year': df['year'].median(), # [cite: 3]
            [cite_start]'market_cap': df['market_cap'].median(), # [cite: 4]
            [cite_start]'ret_1d': 0.0, 'ret_3d': 0.0, 'ret_7d': 0.0, 'ret_30d': 0.0, # [cite: 7, 15, 16, 17]
            [cite_start]'ret_3m': 0.0, 'ret_6m': 0.0, 'ret_1y': 0.0, 'ret_3y': 0.0, 'ret_5y': 0.0, # [cite: 18, 19, 20, 21, 22]
            [cite_start]'from_low_pct': 0.0, 'from_high_pct': 0.0, # [cite: 10, 11]
            'pe': df['pe'].median(), #
            'eps_current': 0.0, 'eps_last_qtr': 0.0, 'eps_change_pct': 0.0, #
            'rvol': 1.0, # Neutral relative volume
            'prev_close': df['price'], #
            [cite_start]'volume_7d': 0.0, 'volume_30d': 0.0, 'volume_90d': 0.0, 'volume_180d': 0.0, # [cite: 24, 25, 26, 27]
            'vol_ratio_1d_90d': 0.0, 'vol_ratio_7d_90d': 0.0, 'vol_ratio_30d_90d': 0.0, #
            'vol_ratio_1d_180d': 0.0, 'vol_ratio_7d_180d': 0.0, 'vol_ratio_30d_180d': 0.0, 'vol_ratio_90d_180d': 0.0 #
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
    # Calculate Average Daily Volume for respective periods
    df['avg_vol_30d'] = df['volume_30d'] / 30.0
    df['avg_vol_90d'] = df['volume_90d'] / 90.0
    df['avg_vol_180d'] = df['volume_180d'] / 180.0

    # Calculate Volume Ratios (percentage change)
    # Handle division by zero by using np.where or replacing zero denominators with NaN then filling
    df['vol_ratio_30d_90d_calc'] = np.where(df['avg_vol_90d'] != 0,
                                         (df['avg_vol_30d'] / df['avg_vol_90d'] - 1) * 100, 0) #
    df['vol_ratio_30d_180d_calc'] = np.where(df['avg_vol_180d'] != 0,
                                          (df['avg_vol_30d'] / df['avg_vol_180d'] - 1) * 100, 0) #
    df['vol_ratio_90d_180d_calc'] = np.where(df['avg_vol_180d'] != 0,
                                          (df['avg_vol_90d'] / df['avg_vol_180d'] - 1) * 100, 0) #


    # Volume Acceleration: Checks if recent accumulation (30d) is accelerating faster than longer periods (90d, 180d)
    # The described "secret sauce" is 30d/90d vs 30d/180d.
    # A positive value means 30d relative to 90d is stronger than 30d relative to 180d,
    # suggesting an acceleration in recent accumulation.
    df['volume_acceleration'] = df['vol_ratio_30d_90d_calc'] - df['vol_ratio_30d_180d_calc'] #

    # Classify based on volume acceleration and current ratios
    def classify_volume(row):
        ratio_30_90 = row['vol_ratio_30d_90d_calc']
        ratio_30_180 = row['vol_ratio_30d_180d_calc']
        acceleration = row['volume_acceleration']

        # for classifications
        if acceleration > 20 and ratio_30_90 > 5 and ratio_30_180 > 5: # Higher threshold for strong acceleration, matching "secret sauce"
            return "Institutional Loading"
        elif acceleration > 5 and ratio_30_90 > 0 and ratio_30_180 > 0:
            return "Heavy Accumulation"
        elif ratio_30_90 > 0 and ratio_30_180 > 0:
            return "Accumulation"
        elif ratio_30_90 < 0 and ratio_30_180 < 0 and acceleration < -5:
            return "Exodus"
        elif ratio_30_90 < 0 and ratio_30_180 < 0:
            return "Distribution"
        else:
            return "Neutral"

    df['volume_classification'] = df.apply(classify_volume, axis=1)
    return df

def calculate_momentum_divergence(df):
    """
    Calculates momentum-related scores.
    A simple approach for demonstration: positive momentum if price > SMAs and positive returns.
    Could be expanded with RSI, MACD divergence detection for "catching turns early".
    """
    df['momentum_score'] = 0

    # [cite_start]Price above short-term SMAs [cite: 12, 13]
    df.loc[df['price'] > df['sma_20d'], 'momentum_score'] += 15
    df.loc[df['price'] > df['sma_50d'], 'momentum_score'] += 10

    # [cite_start]Positive short-term returns [cite: 7, 15]
    df.loc[(df['ret_1d'] > 0) & (df['ret_3d'] > 0), 'momentum_score'] += 5

    # [cite_start]Price near 52-week high but not overly extended (could signal breakout potential) [cite: 9, 11]
    df.loc[(df['from_high_pct'] >= -10) & (df['from_high_pct'] < 0), 'momentum_score'] += 10
    # [cite_start]Price bouncing strongly from 52-week low [cite: 8, 10]
    df.loc[(df['from_low_pct'] > 20) & (df['from_low_pct'] < 50), 'momentum_score'] += 5

    df['momentum_score'] = df['momentum_score'].clip(0, 100) # Ensure score is within 0-100
    return df

def calculate_risk_reward(df):
    """
    Calculates Risk/Reward scores based on dynamic stops and targets.
    Simplified logic for demonstration; real-world would be more complex.
    """
    df['risk_reward_score'] = 0

    # Dynamic Stop Losses based on support levels
    # Option 1: A percentage below current price
    stop_loss_pct = 0.05
    df['dynamic_stop_pct'] = df['price'] * (1 - stop_loss_pct)

    # Option 2: Below a key moving average (e.g., SMA_20d)
    df['dynamic_stop_sma'] = df['sma_20d'] * 0.98 # A little below SMA

    # Combine for a conservative stop: The lower of the two or a fixed percentage below
    df['dynamic_stop'] = np.minimum(df['dynamic_stop_pct'], df['dynamic_stop_sma']) #
    df['dynamic_stop'] = np.maximum(df['dynamic_stop'], df['low_52w'] * 0.95) # Ensure it's not too far below 52w low

    # Two-tier profit targets
    df['target1'] = df['price'] * 1.05 # Simple 5% gain
    df['target2'] = df['price'] * 1.10 # Simple 10% gain

    # [cite_start]Also consider 52-week high as a potential target [cite: 9]
    df['target1'] = np.maximum(df['target1'], df['price'] + (df['high_52w'] - df['price']) * 0.3) # At least 30% towards 52w high
    df['target2'] = np.maximum(df['target2'], df['price'] + (df['high_52w'] - df['price']) * 0.6) # At least 60% towards 52w high

    # Calculate potential upside and downside
    df['potential_upside'] = np.maximum(0, df['target1'] - df['price']) # Using target1 for R/R
    df['potential_downside'] = np.maximum(0, df['price'] - df['dynamic_stop'])

    # Calculate Risk/Reward Ratio
    df['risk_reward_ratio'] = np.where(df['potential_downside'] > 0,
                                     df['potential_upside'] / df['potential_downside'],
                                     np.inf) # Avoid division by zero

    # Score based on R/R ratio
    df.loc[df['risk_reward_ratio'] >= 3, 'risk_reward_score'] = 100 # Excellent R/R
    df.loc[(df['risk_reward_ratio'] >= 2) & (df['risk_reward_ratio'] < 3), 'risk_reward_score'] = 80
    df.loc[(df['risk_reward_ratio'] >= 1.5) & (df['risk_reward_ratio'] < 2), 'risk_reward_score'] = 60
    df.loc[(df['risk_reward_ratio'] >= 1) & (df['risk_reward_ratio'] < 1.5), 'risk_reward_score'] = 30
    df['risk_reward_score'] = df['risk_reward_score'].clip(0, 100)
    return df

def calculate_fundamentals_score(df):
    """
    Calculates fundamentals score based on EPS growth and PE ratio.
    """
    df['fundamentals_score'] = 0

    # Robustly check for positive EPS change
    df.loc[(df['eps_change_pct'].notnull()) & (df['eps_change_pct'] > 0), 'fundamentals_score'] += 50
    # Check for reasonable PE ratio (e.g., between 10 and 60, adjust as per industry norms)
    df.loc[(df['pe'].notnull()) & (df['pe'] > 0) & (df['pe'] < 60), 'fundamentals_score'] += 50

    df['fundamentals_score'] = df['fundamentals_score'].clip(0, 100) # Ensure score is within 0-100
    return df

def calculate_edge_score(df):
    """
    Calculates the final EDGE score based on weighted components.
    Handles adaptive weighting if fundamental data is missing.
    """
    # Initialize base weights
    weights = {
        'volume_acceleration_raw': 0.40,
        'momentum_score': 0.25,
        'risk_reward_score': 0.20,
        'fundamentals_score': 0.15
    }

    # Adaptive weighting for fundamentals if data is missing
    # Check if fundamental columns are substantially null
    has_eps_data = df['eps_change_pct'].notnull().any() or \
                   (df['eps_current'].notnull().any() and df['eps_last_qtr'].notnull().any()) #
    has_pe_data = df['pe'].notnull().any() #

    if not (has_eps_data and has_pe_data): # If both EPS and PE data are largely missing
        st.info("Fundamental data (EPS/PE) is substantially missing. Redistributing fundamental weight to other components.") #
        weights_sum_before_redist = weights['volume_acceleration_raw'] + weights['momentum_score'] + weights['risk_reward_score']
        if weights_sum_before_redist > 0:
            redistribution_factor = (weights['fundamentals_score'] / weights_sum_before_redist)
            weights['volume_acceleration_raw'] += weights['volume_acceleration_raw'] * redistribution_factor
            weights['momentum_score'] += weights['momentum_score'] * redistribution_factor
            weights['risk_reward_score'] += weights['risk_reward_score'] * redistribution_factor
            weights['fundamentals_score'] = 0 # Fundamentals weight is now zero

    # Convert volume classification to a numerical score for weighting
    def get_volume_accel_component_score(classification):
        # Scale these to reflect importance for a 0-100 component score
        if classification == "Institutional Loading": return 100 #
        elif classification == "Heavy Accumulation": return 80 #
        elif classification == "Accumulation": return 50 #
        elif classification == "Neutral": return 20 #
        elif classification == "Distribution": return 10 #
        elif classification == "Exodus": return 0 #
        return 0

    df['volume_accel_component_score'] = df['volume_classification'].apply(get_volume_accel_component_score)

    # Calculate final EDGE Score
    df['EDGE_Score'] = (
        df['volume_accel_component_score'] * weights['volume_acceleration_raw'] + #
        df['momentum_score'] * weights['momentum_score'] + #
        df['risk_reward_score'] * weights['risk_reward_score'] + #
        df['fundamentals_score'] * weights['fundamentals_score'] #
    )
    df['EDGE_Score'] = df['EDGE_Score'].round(2) # Round for cleaner display

    # Classify EDGE Score
    def classify_edge_score(score):
        if score >= EDGE_THRESHOLDS["EXPLOSIVE"]["min_score"]: return "EXPLOSIVE" #
        elif score >= EDGE_THRESHOLDS["STRONG"]["min_score"]: return "STRONG" #
        elif score >= EDGE_THRESHOLDS["MODERATE"]["min_score"]: return "MODERATE" #
        else: return "WATCH/NO EDGE" #

    df['EDGE_Classification'] = df['EDGE_Score'].apply(classify_edge_score)

    # Determine position sizing
    df['position_size_pct'] = df['EDGE_Classification'].apply(
        lambda x: EDGE_THRESHOLDS[x]['position_pct'] if x in EDGE_THRESHOLDS else 0.0
    )

    return df, weights # Return weights for display purposes

# --- Visualization Functions ---

def plot_volume_acceleration_scatter(df):
    """Plots a scatter plot of volume acceleration vs. momentum."""
    # Ensure EDGE_Classification order for consistent coloring
    order = ["EXPLOSIVE", "STRONG", "MODERATE", "WATCH/NO EDGE"]
    df['EDGE_Classification'] = pd.Categorical(df['EDGE_Classification'], categories=order, ordered=True)
    df = df.sort_values('EDGE_Classification')

    fig = px.scatter(df, x="volume_acceleration", y="momentum_score",
                     color="EDGE_Classification",
                     size="EDGE_Score",
                     hover_name="company_name",
                     title="Volume Acceleration vs. Momentum (EDGE Score)",
                     labels={"volume_acceleration": "Volume Accel. (30d/90d - 30d/180d % Diff)", #
                             "momentum_score": "Momentum Score (0-100)"},
                     color_discrete_map={ # Consistent colors
                         "EXPLOSIVE": "#FF4B4B", # Red
                         "STRONG": "#FFA500",    # Orange
                         "MODERATE": "#FFD700",  # Gold/Yellow
                         "WATCH/NO EDGE": "#1F77B4" # Blue
                     },
                     category_orders={"EDGE_Classification": order}
                    )
    fig.update_layout(xaxis_title="Volume Acceleration (%)", yaxis_title="Momentum Score")
    st.plotly_chart(fig, use_container_width=True)

def plot_stock_radar_chart(df_row):
    """Plots a radar chart for an individual stock's EDGE components."""
    categories = ['Volume Acceleration', 'Momentum Divergence', 'Risk/Reward', 'Fundamentals']
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
          name=df_row['company_name'],
          line_color='darkblue'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100] # Ensure consistent scale
            )),
        showlegend=False, # Legend is redundant for single stock
        title=f"EDGE Component Breakdown for {df_row['company_name']} ({df_row['ticker']})",
        font_size=16
    )
    st.plotly_chart(fig, use_container_width=True)


# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="EDGE Protocol - Ultimate Trading System 🚀")

st.title("⚡ EDGE Protocol - The Ultimate Trading System 🚀")
st.markdown("Your unfair advantage: **Volume acceleration data** showing if accumulation is ACCELERATING (not just high).") #

df_raw = load_and_preprocess_data()

if not df_raw.empty:
    # --- Run EDGE Protocol Calculations ---
    with st.spinner("Calculating EDGE Scores..."):
        df_calculated, final_weights = calculate_edge_score( # Pass the df_raw, as all calculations are chained within it
            calculate_fundamentals_score(
                calculate_risk_reward(
                    calculate_momentum_divergence(
                        calculate_volume_acceleration(df_raw.copy()) # Use a copy to avoid modifying original cached df
                    )
                )
            )
        )

    st.success("EDGE Scores calculated! Last updated: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S IST'))

    # --- Tabs for different analysis views ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Daily EDGE Signals", "📈 Volume Acceleration Insights", "🔍 Stock Deep Dive", "⚙️ Raw Data & Parameters"])

    with tab1:
        st.header("Daily EDGE Signals")
        st.markdown("Find the highest conviction trades here based on the EDGE Protocol's comprehensive scoring.")

        # Filter by EDGE Classification
        selected_edge_class = st.multiselect(
            "Filter by EDGE Classification:",
            options=["EXPLOSIVE", "STRONG", "MODERATE", "WATCH/NO EDGE"], # Explicit order
            default=["EXPLOSIVE", "STRONG"] # Default to high conviction
        )

        filtered_df = df_calculated[df_calculated['EDGE_Classification'].isin(selected_edge_class)].sort_values(by="EDGE_Score", ascending=False)

        if not filtered_df.empty:
            st.dataframe(
                filtered_df[[
                    'ticker', 'company_name', 'EDGE_Classification', 'EDGE_Score',
                    'volume_classification', 'price', 'ret_1d', 'ret_7d', 'ret_30d',
                    'position_size_pct', 'dynamic_stop', 'target1', 'target2',
                    'vol_ratio_30d_90d_calc', 'vol_ratio_30d_180d_calc', 'volume_acceleration'
                ]].style.background_gradient(cmap='RdYlGn', subset=['EDGE_Score']).format({
                    'EDGE_Score': "{:.2f}",
                    [cite_start]'price': "₹{:.2f}", # [cite: 6]
                    [cite_start]'ret_1d': "{:.2f}%", 'ret_7d': "{:.2f}%", 'ret_30d': "{:.2f}%", # [cite: 7, 16, 17]
                    'position_size_pct': "{:.2%}", #
                    'dynamic_stop': "₹{:.2f}", 'target1': "₹{:.2f}", 'target2': "₹{:.2f}", #
                    'vol_ratio_30d_90d_calc': "{:.2f}%", 'vol_ratio_30d_180d_calc': "{:.2f}%", #
                    'volume_acceleration': "{:.2f}%" #
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
        plot_volume_acceleration_scatter(df_calculated)

    with tab3:
        st.header("Stock Deep Dive (Radar Chart)")
        st.markdown("Select a stock to see its individual EDGE component breakdown.")

        # Ensure only stocks with valid calculated scores are shown
        available_stocks = df_calculated[df_calculated['EDGE_Score'].notnull()]['ticker'].tolist()
        if available_stocks:
            selected_ticker = st.selectbox("Select Ticker:", available_stocks)
            selected_stock_row = df_calculated[df_calculated['ticker'] == selected_ticker].iloc[0]
            plot_stock_radar_chart(selected_stock_row)

            st.subheader(f"Detailed Metrics for {selected_stock_row['company_name']}")
            col1, col2, col3 = st.columns(3)
            with col1:
                [cite_start]st.metric("Current Price", f"₹{selected_stock_row['price']:.2f}") # [cite: 6]
                st.metric("EDGE Score", f"{selected_stock_row['EDGE_Score']:.2f}") #
                st.metric("Classification", selected_stock_row['EDGE_Classification']) #
            with col2:
                st.metric("Volume Acceleration", f"{selected_stock_row['volume_acceleration']:.2f}%") #
                st.metric("Volume Classification", selected_stock_row['volume_classification']) #
                st.metric("Position Size", f"{selected_stock_row['position_size_pct']:.2%}") #
            with col3:
                st.metric("Dynamic Stop", f"₹{selected_stock_row['dynamic_stop']:.2f}") #
                st.metric("Target 1", f"₹{selected_stock_row['target1']:.2f}") #
                st.metric("Target 2", f"₹{selected_stock_row['target2']:.2f}") #

            st.dataframe(
                selected_stock_row[[
                    [cite_start]'ticker', 'company_name', 'market_cap', 'category', 'sector', 'pe', # [cite: 1, 2, 4, 5]
                    'eps_current', 'eps_last_qtr', 'eps_change_pct', #
                    [cite_start]'low_52w', 'high_52w', 'from_low_pct', 'from_high_pct', # [cite: 8, 9, 10, 11]
                    [cite_start]'sma_20d', 'sma_50d', 'sma_200d', # [cite: 12, 13, 14]
                    [cite_start]'ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y', # [cite: 7, 16, 17, 18, 19, 20, 21, 22]
                    [cite_start]'volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d', # [cite: 23, 24, 25, 26, 27]
                    'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d', #
                    'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 'vol_ratio_90d_180d', #
                    'rvol', 'prev_close' #
                ]].to_frame().T.style.format(
                                        {
                                            [cite_start]'market_cap': "₹{:,.0f} Cr", # [cite: 4]
                                            [cite_start]'price': "₹{:.2f}", # [cite: 6]
                                            [cite_start]'ret_1d': "{:.2f}%", 'ret_7d': "{:.2f}%", 'ret_30d': "{:.2f}%", # [cite: 7, 16, 17]
                                            [cite_start]'ret_3m': "{:.2f}%", 'ret_6m': "{:.2f}%", 'ret_1y': "{:.2f}%", # [cite: 18, 19, 20]
                                            [cite_start]'ret_3y': "{:.2f}%", 'ret_5y': "{:.2f}%", # [cite: 21, 22]
                                            [cite_start]'low_52w': "₹{:.2f}", 'high_52w': "₹{:.2f}", # [cite: 8, 9]
                                            [cite_start]'from_low_pct': "{:.2f}%", 'from_high_pct': "{:.2f}%", # [cite: 10, 11]
                                            [cite_start]'sma_20d': "₹{:.2f}", 'sma_50d': "₹{:.2f}", 'sma_200d': "₹{:.2f}", # [cite: 12, 13, 14]
                                            [cite_start]'volume_1d': "{:,.0f}", 'volume_7d': "{:,.0f}", 'volume_30d': "{:,.0f}", # [cite: 23, 24, 25]
                                            [cite_start]'volume_90d': "{:,.0f}", 'volume_180d': "{:,.0f}", # [cite: 26, 27]
                                            # These are already ratios (e.g., 0.5 for 50%) but were in % string in source.
                                            # Display as percentages for user clarity
                                            'vol_ratio_1d_90d': "{:.2%}", 'vol_ratio_7d_90d': "{:.2%}", 'vol_ratio_30d_90d': "{:.2%}", #
                                            'vol_ratio_1d_180d': "{:.2%}", 'vol_ratio_7d_180d': "{:.2%}", 'vol_ratio_30d_180d': "{:.2%}", 'vol_ratio_90d_180d': "{:.2%}", #
                                            'rvol': "{:.2f}", #
                                            'prev_close': "₹{:.2f}", #
                                            'pe': "{:.2f}", #
                                            'eps_current': "{:.2f}", 'eps_last_qtr': "{:.2f}", 'eps_change_pct': "{:.2f}%" #
                                        }
                                    ),
                use_container_width=True
            )
        else:
            st.info("No stocks available for deep dive or data issues. Please check data loading.")


    with tab4:
        st.header("Raw Data and Parameters")
        st.markdown("Review the raw loaded data and the configurable parameters of the EDGE Protocol.")
        st.subheader("Raw Data (First 10 Rows)")
        st.dataframe(df_raw.head(10), use_container_width=True)

        st.subheader("EDGE Thresholds & Position Sizing")
        st.json(EDGE_THRESHOLDS)

        st.subheader("Weighting of EDGE Components")
        st.write("These weights determine the influence of each component on the final EDGE Score:")
        st.markdown(f"""
            * **Volume Acceleration ({final_weights['volume_acceleration_raw'] * 100:.0f}%)**: Your secret weapon, detecting institutional accumulation acceleration.
            * **Momentum Divergence ({final_weights['momentum_score'] * 100:.0f}%)**: Catching turns early and confirming price action.
            * **Risk/Reward Mathematics ({final_weights['risk_reward_score'] * 100:.0f}%)**: Ensuring favorable trade setups.
            * **Fundamentals ({final_weights['fundamentals_score'] * 100:.0f}%)**: Adaptive weighting. Redistributed if EPS/PE data is missing.
        """)
        if final_weights['fundamentals_score'] == 0:
            st.warning("Note: Fundamental (EPS/PE) data was largely missing, so its weight has been redistributed.")


    st.markdown("---")
    st.caption("EDGE Protocol - Because in trading, information advantage IS the edge.")

else:
    st.error("Application could not load data. Please check the `load_and_preprocess_data` function and its data source.")

```
