#!/usr/bin/env python3
"""
EDGE PROTOCOL MASTER SYSTEM v1.0
================================
The Ultimate Trading Intelligence System for Indian Stock Markets
Built after 24-hour deep analysis - Everything in ONE file

Author: EDGE Protocol Systems
Date: January 2025
Market: Indian Cash Equities
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import sys

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Master Configuration
CONFIG = {
    'MIN_PRICE': 20,                    # Avoid penny stocks
    'MIN_VOLUME': 50000,                # Minimum daily volume
    'MIN_VALUE_TRADED': 10000000,       # ₹1 Crore minimum
    'MAX_POSITIONS': 5,                 # Concentrated portfolio
    'POSITION_SIZE_MAX': 0.25,          # 25% max per position
    'STOP_LOSS_PCT': 8,                 # Universal stop loss
    'TARGET_1_PCT': 15,                 # First target
    'TARGET_2_PCT': 30,                 # Second target
    'MIN_EDGE_SCORE': 70,               # Minimum score to consider
}

# Signal Thresholds
SIGNAL_LEVELS = {
    'SUPER_EDGE': 90,     # Ultra high conviction
    'STRONG_BUY': 80,     # High conviction
    'BUY': 70,            # Good opportunity
    'WATCH': 60,          # Monitor closely
    'IGNORE': 0           # Skip
}

# ============================================================================
# DATA PROCESSING - INDIAN FORMAT HANDLER
# ============================================================================

def parse_indian_number(val):
    """Parse Indian number formats including ₹, Cr, L, commas, %"""
    if pd.isna(val) or val == '' or val == '-':
        return np.nan
    
    # Convert to string and clean
    val = str(val).strip()
    
    # Remove currency symbols
    val = val.replace('₹', '').replace('Rs', '').replace('$', '')
    
    # Handle percentage
    if '%' in val:
        val = val.replace('%', '')
        try:
            return float(val)
        except:
            return np.nan
    
    # Handle Cr and L
    multiplier = 1
    if 'Cr' in val:
        multiplier = 10000000
        val = val.replace('Cr', '')
    elif 'L' in val:
        multiplier = 100000
        val = val.replace('L', '')
    
    # Remove commas
    val = val.replace(',', '').strip()
    
    try:
        return float(val) * multiplier
    except:
        return np.nan

def load_and_clean_data(filepath):
    """Load CSV and fix all Indian format issues"""
    
    print("📊 Loading data...")
    df = pd.read_csv(filepath)
    
    print("🔧 Cleaning Indian number formats...")
    
    # Price columns
    price_cols = ['price', 'low_52w', 'high_52w', 'sma_20d', 'sma_50d', 'sma_200d', 'prev_close']
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_indian_number)
    
    # Return columns (already in decimal, just remove % if present)
    return_cols = ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
                   'from_low_pct', 'from_high_pct']
    for col in return_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_indian_number)
    
    # Volume columns
    volume_cols = ['volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d']
    for col in volume_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: parse_indian_number(str(x).replace(',', '')))
    
    # Volume ratio columns (convert to percentage)
    ratio_cols = ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
                  'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d', 'vol_ratio_90d_180d']
    for col in ratio_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_indian_number)
    
    # Market cap
    if 'market_cap' in df.columns:
        df['market_cap_value'] = df['market_cap'].apply(parse_indian_number)
    
    # EPS and PE
    for col in ['pe', 'eps_current', 'eps_last_qtr', 'eps_change_pct']:
        if col in df.columns:
            df[col] = df[col].apply(parse_indian_number)
    
    # Calculate derived columns
    df['value_traded'] = df['price'] * df['volume_1d']
    df['volume_acceleration'] = df['vol_ratio_30d_90d'] - df['vol_ratio_30d_180d']
    
    # Fix RVOL if needed
    if df['rvol'].max() < 10:  # Seems to be in decimal
        df['rvol_fixed'] = df['volume_1d'] / df['volume_30d'].replace(0, 1)
    else:
        df['rvol_fixed'] = df['rvol']
    
    print(f"✅ Loaded {len(df)} stocks")
    return df

# ============================================================================
# PATTERN DETECTION ENGINE
# ============================================================================

def detect_phoenix_rising(row):
    """Dead stock coming back to life"""
    score = 0
    
    # Was dead (negative returns)
    if row['ret_30d'] < -20 and row['ret_7d'] > 0:
        score += 40
    
    # Volume returning
    if row['volume_acceleration'] > -10:  # Less negative = good
        score += 30
    
    # Quality stock (high historical returns)
    if row['ret_3y'] > 200:
        score += 30
    
    return score

def detect_stealth_accumulation(row):
    """Smart money loading quietly"""
    score = 0
    
    # Price flat but volume improving
    if abs(row['ret_30d']) < 10 and row['volume_acceleration'] > 0:
        score += 50
    
    # Near lows but holding
    if row['from_low_pct'] < 30 and row['price'] > row['sma_200d']:
        score += 30
    
    # Relative volume strength
    if row['rvol_fixed'] > 1.2:
        score += 20
    
    return score

def detect_momentum_explosion(row):
    """Ready to explode higher"""
    score = 0
    
    # Momentum building
    if row['ret_7d'] > row['ret_30d'] / 4 and row['ret_7d'] > 3:
        score += 40
    
    # Volume confirmation
    if row['volume_acceleration'] > 20:
        score += 40
    
    # Not overextended
    if row['from_high_pct'] > -15:
        score += 20
    
    return score

def detect_breakout_loading(row):
    """Preparing for breakout"""
    score = 0
    
    # Near 52w high with volume
    if row['from_high_pct'] > -10 and row['rvol_fixed'] > 1.5:
        score += 50
    
    # Price above all SMAs
    if row['price'] > row['sma_20d'] and row['price'] > row['sma_50d'] and row['price'] > row['sma_200d']:
        score += 30
    
    # Consistent buying
    if row['ret_1d'] > 0 and row['ret_3d'] > 0 and row['ret_7d'] > 0:
        score += 20
    
    return score

def detect_quality_discount(row):
    """Great company on sale"""
    score = 0
    
    # Quality metrics
    if row['ret_3y'] > 300 and row['from_high_pct'] < -25:
        score += 40
    
    # Fundamentals improving
    if row['eps_change_pct'] > 15:
        score += 30
    
    # Reasonable valuation
    if 0 < row['pe'] < 25:
        score += 30
    
    return score

# ============================================================================
# MASTER SCORING ALGORITHM - THE BRAIN
# ============================================================================

def calculate_edge_score(row, market_context):
    """The master algorithm combining everything"""
    
    # 1. RELATIVE VOLUME STRENGTH (35%)
    # Compare to market average
    market_vol_avg = market_context['avg_vol_ratio']
    relative_vol = row['vol_ratio_30d_90d'] - market_vol_avg
    
    if relative_vol > 30:
        vol_score = 100
    elif relative_vol > 20:
        vol_score = 85
    elif relative_vol > 10:
        vol_score = 70
    elif relative_vol > 0:
        vol_score = 55
    else:
        vol_score = 40
    
    # Boost for volume acceleration
    if row['volume_acceleration'] > 20:
        vol_score = min(vol_score * 1.2, 100)
    
    # 2. MOMENTUM INTELLIGENCE (30%)
    # Multi-timeframe momentum
    short_momentum = row['ret_7d'] if row['ret_7d'] > 0 else 0
    medium_momentum = row['ret_30d'] if row['ret_30d'] > 0 else 0
    
    # Momentum alignment bonus
    if row['ret_1d'] > 0 and row['ret_3d'] > row['ret_1d'] and row['ret_7d'] > row['ret_3d']:
        momentum_aligned = True
        momentum_bonus = 20
    else:
        momentum_aligned = False
        momentum_bonus = 0
    
    momentum_score = min(short_momentum * 3 + medium_momentum + momentum_bonus, 100)
    
    # 3. POSITION VALUE (20%)
    # Where in the range?
    if row['from_high_pct'] > -15 and row['ret_7d'] > 0:
        position_score = 90  # Breaking out
    elif row['from_high_pct'] < -30 and row['ret_7d'] > 0 and row['ret_3y'] > 200:
        position_score = 85  # Quality recovery
    elif row['from_low_pct'] < 20 and row['price'] > row['sma_200d']:
        position_score = 80  # Bottom reversal
    elif row['price'] > row['sma_50d'] and row['price'] > row['sma_200d']:
        position_score = 70  # Healthy trend
    else:
        position_score = 50
    
    # 4. QUALITY FILTER (15%)
    # Long-term performance and fundamentals
    quality_score = 0
    
    if row['ret_3y'] > 500:
        quality_score += 40
    elif row['ret_3y'] > 300:
        quality_score += 30
    elif row['ret_3y'] > 200:
        quality_score += 20
    
    if row['eps_change_pct'] > 20:
        quality_score += 30
    elif row['eps_change_pct'] > 10:
        quality_score += 20
    
    if 0 < row['pe'] < 25:
        quality_score += 30
    elif 25 <= row['pe'] < 40:
        quality_score += 20
    
    # FINAL EDGE SCORE
    edge_score = (
        vol_score * 0.35 +
        momentum_score * 0.30 +
        position_score * 0.20 +
        quality_score * 0.15
    )
    
    # Pattern bonus (up to 10 points)
    patterns = {
        'phoenix': detect_phoenix_rising(row),
        'stealth': detect_stealth_accumulation(row),
        'momentum': detect_momentum_explosion(row),
        'breakout': detect_breakout_loading(row),
        'quality': detect_quality_discount(row)
    }
    
    best_pattern = max(patterns.values())
    if best_pattern > 80:
        edge_score += 10
    elif best_pattern > 60:
        edge_score += 5
    
    # Store pattern info
    best_pattern_name = max(patterns, key=patterns.get)
    
    return edge_score, momentum_aligned, best_pattern_name, patterns[best_pattern_name]

# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def generate_signals(df):
    """Generate trading signals"""
    
    # Calculate market context
    market_context = {
        'avg_vol_ratio': df['vol_ratio_30d_90d'].median(),
        'positive_momentum_pct': (df['ret_7d'] > 0).sum() / len(df) * 100,
        'avg_rvol': df['rvol_fixed'].mean()
    }
    
    # Determine market mode
    if market_context['positive_momentum_pct'] > 60 and market_context['avg_rvol'] > 1.2:
        market_mode = "BULLISH"
    elif market_context['positive_momentum_pct'] < 30:
        market_mode = "BEARISH"
    else:
        market_mode = "NEUTRAL"
    
    print(f"\n📈 Market Mode: {market_mode}")
    print(f"   Positive Momentum: {market_context['positive_momentum_pct']:.1f}%")
    print(f"   Average RVOL: {market_context['avg_rvol']:.2f}")
    
    # Apply quality filters
    quality_universe = df[
        (df['price'] >= CONFIG['MIN_PRICE']) &
        (df['volume_1d'] >= CONFIG['MIN_VOLUME']) &
        (df['value_traded'] >= CONFIG['MIN_VALUE_TRADED']) &
        (df['price'].notna()) &
        (df['volume_1d'].notna())
    ].copy()
    
    print(f"\n✅ Quality Universe: {len(quality_universe)} stocks (from {len(df)})")
    
    # Calculate scores
    results = []
    
    for idx, row in quality_universe.iterrows():
        edge_score, momentum_aligned, pattern, pattern_score = calculate_edge_score(row, market_context)
        
        # Determine signal
        if edge_score >= SIGNAL_LEVELS['SUPER_EDGE']:
            signal = "SUPER_EDGE"
            position_size = 0.25  # 25%
        elif edge_score >= SIGNAL_LEVELS['STRONG_BUY']:
            signal = "STRONG_BUY"
            position_size = 0.20  # 20%
        elif edge_score >= SIGNAL_LEVELS['BUY']:
            signal = "BUY"
            position_size = 0.15  # 15%
        elif edge_score >= SIGNAL_LEVELS['WATCH']:
            signal = "WATCH"
            position_size = 0.10  # 10%
        else:
            signal = "IGNORE"
            position_size = 0
        
        # Risk management
        stop_loss = row['price'] * (1 - CONFIG['STOP_LOSS_PCT'] / 100)
        target_1 = row['price'] * (1 + CONFIG['TARGET_1_PCT'] / 100)
        target_2 = row['price'] * (1 + CONFIG['TARGET_2_PCT'] / 100)
        
        # Adjust for market mode
        if market_mode == "BEARISH":
            position_size *= 0.5  # Half position in bear market
            stop_loss = row['price'] * 0.95  # Tighter stop
        
        results.append({
            'ticker': row['ticker'],
            'company_name': row['company_name'],
            'sector': row['sector'],
            'price': row['price'],
            'edge_score': edge_score,
            'signal': signal,
            'pattern': pattern,
            'pattern_score': pattern_score,
            'momentum_aligned': momentum_aligned,
            'volume_acceleration': row['volume_acceleration'],
            'rvol': row['rvol_fixed'],
            'ret_7d': row['ret_7d'],
            'ret_30d': row['ret_30d'],
            'from_high_pct': row['from_high_pct'],
            'position_size_pct': position_size * 100,
            'stop_loss': stop_loss,
            'target_1': target_1,
            'target_2': target_2,
            'risk_reward': (target_1 - row['price']) / (row['price'] - stop_loss)
        })
    
    # Convert to DataFrame and sort
    signals_df = pd.DataFrame(results)
    signals_df = signals_df[signals_df['signal'] != 'IGNORE'].sort_values('edge_score', ascending=False)
    
    return signals_df, market_mode

# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def display_signals(signals_df, market_mode):
    """Display signals in a beautiful format"""
    
    print("\n" + "="*100)
    print("🚀 EDGE PROTOCOL TRADING SIGNALS")
    print("="*100)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Market Mode: {market_mode}")
    print("="*100)
    
    # Separate by signal type
    super_edge = signals_df[signals_df['signal'] == 'SUPER_EDGE']
    strong_buy = signals_df[signals_df['signal'] == 'STRONG_BUY']
    buy = signals_df[signals_df['signal'] == 'BUY']
    watch = signals_df[signals_df['signal'] == 'WATCH']
    
    # Display each category
    if len(super_edge) > 0:
        print("\n⭐ SUPER EDGE SIGNALS (Highest Conviction)")
        print("-" * 100)
        for _, row in super_edge.iterrows():
            print(f"\n{row['ticker']} - {row['company_name']} ({row['sector']})")
            print(f"  Price: ₹{row['price']:.2f} | EDGE Score: {row['edge_score']:.1f}")
            print(f"  Pattern: {row['pattern'].upper()} | Volume Accel: {row['volume_acceleration']:.1f}%")
            print(f"  Position: {row['position_size_pct']:.0f}% | Stop: ₹{row['stop_loss']:.2f} | Target: ₹{row['target_1']:.2f}")
            print(f"  7D Return: {row['ret_7d']:.1f}% | From High: {row['from_high_pct']:.1f}%")
            if row['momentum_aligned']:
                print(f"  ✅ MOMENTUM ALIGNED - Higher probability!")
    
    if len(strong_buy) > 0:
        print("\n💪 STRONG BUY SIGNALS")
        print("-" * 100)
        for _, row in strong_buy.head(5).iterrows():
            print(f"{row['ticker']:10} | ₹{row['price']:8.2f} | Score: {row['edge_score']:5.1f} | "
                  f"Pattern: {row['pattern']:10} | Position: {row['position_size_pct']:.0f}%")
    
    if len(buy) > 0:
        print("\n📈 BUY SIGNALS")
        print("-" * 100)
        for _, row in buy.head(5).iterrows():
            print(f"{row['ticker']:10} | ₹{row['price']:8.2f} | Score: {row['edge_score']:5.1f} | "
                  f"Pattern: {row['pattern']:10} | Position: {row['position_size_pct']:.0f}%")
    
    if len(watch) > 0:
        print("\n👀 WATCH LIST")
        print("-" * 100)
        for _, row in watch.head(10).iterrows():
            print(f"{row['ticker']:10} | ₹{row['price']:8.2f} | Score: {row['edge_score']:5.1f} | "
                  f"Pattern: {row['pattern']:10}")
    
    # Summary
    print("\n" + "="*100)
    print("📊 SUMMARY")
    print(f"Total Signals: {len(signals_df)}")
    print(f"Super Edge: {len(super_edge)} | Strong Buy: {len(strong_buy)} | Buy: {len(buy)} | Watch: {len(watch)}")
    
    # Risk warning for bear market
    if market_mode == "BEARISH":
        print("\n⚠️  WARNING: Market in BEARISH mode. Position sizes reduced by 50%.")
        print("   Consider waiting for market to stabilize before taking large positions.")

def save_signals(signals_df, market_mode):
    """Save signals to CSV"""
    
    filename = f"edge_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    signals_df.to_csv(filename, index=False)
    print(f"\n💾 Signals saved to: {filename}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "="*100)
    print("⚡ EDGE PROTOCOL MASTER SYSTEM v1.0")
    print("="*100)
    
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python edge_protocol_master.py <data_file.csv>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    try:
        # Load and process data
        df = load_and_clean_data(filepath)
        
        # Generate signals
        signals_df, market_mode = generate_signals(df)
        
        # Display results
        display_signals(signals_df, market_mode)
        
        # Save results
        save_signals(signals_df, market_mode)
        
        # Final recommendations
        print("\n" + "="*100)
        print("🎯 RECOMMENDED ACTIONS")
        print("="*100)
        
        top_5 = signals_df.head(5)
        if len(top_5) > 0:
            print("\nTOP 5 POSITIONS TO CONSIDER:")
            total_allocation = 0
            for i, row in top_5.iterrows():
                print(f"{i+1}. {row['ticker']} - Allocate {row['position_size_pct']:.0f}% "
                      f"(₹{row['price']:.2f} → ₹{row['target_1']:.2f})")
                total_allocation += row['position_size_pct']
            
            print(f"\nTotal Portfolio Allocation: {total_allocation:.0f}%")
            print(f"Keep {100-total_allocation:.0f}% in cash for opportunities")
        else:
            print("\n⚠️  No high-conviction signals today. Stay in cash and wait for better setups.")
        
        print("\n✅ Analysis Complete!")
        print("="*100)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please check your data file and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
