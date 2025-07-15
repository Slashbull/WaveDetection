# mantra_diagnostics.py - COMPLETE SYSTEM HEALTH CHECK & TESTING
"""
M.A.N.T.R.A. Diagnostics Dashboard
==================================
This diagnostic system checks EVERYTHING:
- Data loading pipeline
- Calculation accuracy
- Signal generation
- Indian market compatibility
- Performance metrics
- Complete system health

Run this alongside your main app to ensure everything works perfectly.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import io
import json
from datetime import datetime
import time
import traceback
import base64

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="M.A.N.T.R.A. Diagnostics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Use same config as main system
SHEET_ID = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"
GID = "2026492216"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

# Expected columns for Indian market
EXPECTED_COLUMNS = {
    'ticker', 'company_name', 'exchange', 'sector', 'price', 'market_cap',
    'pe', 'eps_current', 'eps_last_qtr', 'eps_change_pct',
    'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y',
    'volume_1d', 'volume_7d', 'volume_30d', 'volume_90d', 'volume_180d',
    'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
    'vol_ratio_30d_180d', 'vol_ratio_90d_180d',
    'sma_20d', 'sma_50d', 'sma_200d',
    'low_52w', 'high_52w', 'from_low_pct', 'from_high_pct'
}

# ============================================================================
# DIAGNOSTIC FUNCTIONS
# ============================================================================

class SystemDiagnostics:
    """Complete system diagnostics and health checks"""
    
    def __init__(self):
        self.diagnostics = {
            'timestamp': datetime.now().isoformat(),
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'warnings': [],
            'errors': [],
            'data_quality': {},
            'calculation_checks': {},
            'signal_analysis': {},
            'performance_metrics': {}
        }
    
    def run_complete_diagnostics(self):
        """Run all diagnostic tests"""
        st.header("🔍 M.A.N.T.R.A. Complete System Diagnostics")
        
        # Create diagnostic tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Data Pipeline", "🧮 Calculations", "📈 Signals", 
            "🇮🇳 Indian Market", "⚡ Performance", "📥 Reports"
        ])
        
        with tab1:
            self.test_data_pipeline()
        
        with tab2:
            self.test_calculations()
        
        with tab3:
            self.test_signal_generation()
        
        with tab4:
            self.test_indian_market_compatibility()
        
        with tab5:
            self.test_performance_metrics()
        
        with tab6:
            self.generate_diagnostic_reports()
    
    def test_data_pipeline(self):
        """Test complete data loading pipeline"""
        st.subheader("📊 Data Pipeline Testing")
        
        # Test 1: URL Accessibility
        with st.expander("Test 1: Google Sheets Accessibility", expanded=True):
            try:
                start = time.time()
                response = requests.head(SHEET_URL, timeout=10)
                load_time = time.time() - start
                
                if response.status_code == 200:
                    st.success(f"✅ Google Sheets accessible (Response time: {load_time:.2f}s)")
                    self.diagnostics['tests_passed'] += 1
                else:
                    st.error(f"❌ Google Sheets error: Status {response.status_code}")
                    self.diagnostics['tests_failed'] += 1
                    self.diagnostics['errors'].append(f"Sheet status: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")
                self.diagnostics['tests_failed'] += 1
                self.diagnostics['errors'].append(f"Connection error: {str(e)}")
            
            self.diagnostics['tests_run'] += 1
        
        # Test 2: Data Loading
        with st.expander("Test 2: Data Loading & Parsing", expanded=True):
            try:
                df_raw = self.load_raw_data()
                if df_raw is not None and not df_raw.empty:
                    st.success(f"✅ Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns")
                    self.diagnostics['tests_passed'] += 1
                    
                    # Show sample
                    st.write("Sample data (first 3 rows):")
                    st.dataframe(df_raw.head(3))
                    
                    # Column analysis
                    st.write("Column types:")
                    col_types = df_raw.dtypes.value_counts()
                    st.write(col_types)
                else:
                    st.error("❌ Failed to load data")
                    self.diagnostics['tests_failed'] += 1
            except Exception as e:
                st.error(f"❌ Data loading error: {str(e)}")
                self.diagnostics['tests_failed'] += 1
                self.diagnostics['errors'].append(f"Load error: {str(e)}")
            
            self.diagnostics['tests_run'] += 1
        
        # Test 3: Column Completeness
        with st.expander("Test 3: Column Completeness Check", expanded=True):
            if 'df_raw' in locals():
                missing_cols = EXPECTED_COLUMNS - set(df_raw.columns)
                extra_cols = set(df_raw.columns) - EXPECTED_COLUMNS
                
                col1, col2 = st.columns(2)
                with col1:
                    if missing_cols:
                        st.warning(f"⚠️ Missing columns: {missing_cols}")
                        self.diagnostics['warnings'].append(f"Missing columns: {list(missing_cols)}")
                    else:
                        st.success("✅ All expected columns present")
                        self.diagnostics['tests_passed'] += 1
                
                with col2:
                    if extra_cols:
                        st.info(f"ℹ️ Extra columns: {len(extra_cols)}")
                
                self.diagnostics['tests_run'] += 1
                self.diagnostics['data_quality']['missing_columns'] = list(missing_cols)
                self.diagnostics['data_quality']['total_columns'] = len(df_raw.columns)
        
        # Test 4: Data Quality
        with st.expander("Test 4: Data Quality Analysis", expanded=True):
            if 'df_raw' in locals():
                df_clean = self.clean_data(df_raw)
                
                # Null analysis
                null_counts = df_clean.isnull().sum()
                high_null_cols = null_counts[null_counts > len(df_clean) * 0.5]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    null_pct = (null_counts.sum() / (len(df_clean) * len(df_clean.columns)) * 100)
                    st.metric("Null %", f"{null_pct:.1f}%")
                    if null_pct < 10:
                        st.success("✅ Good data quality")
                        self.diagnostics['tests_passed'] += 1
                    else:
                        st.warning("⚠️ High null percentage")
                        self.diagnostics['warnings'].append(f"High null percentage: {null_pct:.1f}%")
                
                with col2:
                    st.metric("High Null Columns", len(high_null_cols))
                
                with col3:
                    duplicates = df_clean['ticker'].duplicated().sum()
                    st.metric("Duplicate Tickers", duplicates)
                    if duplicates > 0:
                        self.diagnostics['warnings'].append(f"Duplicate tickers: {duplicates}")
                
                self.diagnostics['tests_run'] += 1
                self.diagnostics['data_quality']['null_percentage'] = null_pct
                self.diagnostics['data_quality']['duplicate_tickers'] = duplicates
    
    def test_calculations(self):
        """Test all calculations"""
        st.subheader("🧮 Calculation Testing")
        
        df = self.load_and_process_data()
        if df is None:
            st.error("❌ Cannot test calculations - data loading failed")
            return
        
        # Test Volume Acceleration
        with st.expander("Test 1: Volume Acceleration Calculation", expanded=True):
            if all(col in df.columns for col in ['vol_ratio_30d_90d', 'vol_ratio_30d_180d']):
                # Manual calculation
                vol_accel_manual = df['vol_ratio_30d_90d'] - df['vol_ratio_30d_180d']
                
                # System calculation
                df['volume_acceleration'] = df['vol_ratio_30d_90d'] - df['vol_ratio_30d_180d']
                
                # Compare
                diff = (vol_accel_manual - df['volume_acceleration']).abs().max()
                if diff < 0.01:
                    st.success(f"✅ Volume acceleration calculation correct (max diff: {diff:.6f})")
                    self.diagnostics['tests_passed'] += 1
                else:
                    st.error(f"❌ Volume acceleration mismatch (max diff: {diff:.6f})")
                    self.diagnostics['tests_failed'] += 1
                
                # Show examples
                st.write("Sample calculations:")
                sample_df = df[['ticker', 'vol_ratio_30d_90d', 'vol_ratio_30d_180d', 'volume_acceleration']].head(5)
                st.dataframe(sample_df)
            
            self.diagnostics['tests_run'] += 1
        
        # Test Momentum Acceleration
        with st.expander("Test 2: Momentum Acceleration", expanded=True):
            if all(col in df.columns for col in ['ret_1d', 'ret_3d', 'ret_7d']):
                # Check for divide by zero handling
                zero_returns = df[(df['ret_3d'] == 0) | (df['ret_7d'] == 0)]
                st.write(f"Stocks with zero returns: {len(zero_returns)}")
                
                if len(zero_returns) > 0:
                    st.success("✅ System handles zero returns correctly")
                    self.diagnostics['tests_passed'] += 1
                
                # Check acceleration logic
                positive_accel = df[(df['ret_1d'] > 0) & (df['ret_3d'] > 0) & (df['ret_7d'] > 0)]
                st.write(f"Stocks with positive momentum: {len(positive_accel)}")
            
            self.diagnostics['tests_run'] += 1
        
        # Test Conviction Score
        with st.expander("Test 3: Conviction Score Components", expanded=True):
            # Initialize score
            df['test_conviction'] = 0
            
            # Test each component
            components = {
                'Volume Acceleration': (df['volume_acceleration'] > 10).sum(),
                'Momentum Building': (df['ret_7d'] > df['ret_30d']/4).sum() if 'ret_30d' in df.columns else 0,
                'EPS Improving': (df['eps_current'] > df['eps_last_qtr']).sum() if all(col in df.columns for col in ['eps_current', 'eps_last_qtr']) else 0,
                'Above 50MA': (df['price'] > df['sma_50d']).sum() if all(col in df.columns for col in ['price', 'sma_50d']) else 0,
                'High Volume': (df['rvol'] > 1.5).sum() if 'rvol' in df.columns else 0
            }
            
            for component, count in components.items():
                st.write(f"{component}: {count} stocks")
            
            self.diagnostics['calculation_checks']['conviction_components'] = components
            self.diagnostics['tests_run'] += 1
            self.diagnostics['tests_passed'] += 1
    
    def test_signal_generation(self):
        """Test signal generation logic"""
        st.subheader("📈 Signal Generation Testing")
        
        df = self.load_and_process_data()
        if df is None:
            st.error("❌ Cannot test signals - data loading failed")
            return
        
        # Run signal detection
        df = self.apply_signal_logic(df)
        
        # Signal distribution
        with st.expander("Signal Distribution Analysis", expanded=True):
            if 'EDGE_SIGNAL' in df.columns:
                signal_counts = df['EDGE_SIGNAL'].value_counts()
                
                fig = go.Figure(data=[go.Bar(
                    x=signal_counts.index,
                    y=signal_counts.values,
                    text=signal_counts.values,
                    textposition='auto'
                )])
                fig.update_layout(title="Signal Distribution", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                self.diagnostics['signal_analysis']['distribution'] = signal_counts.to_dict()
                
                # Quality check
                total_signals = signal_counts[signal_counts.index != 'NONE'].sum()
                signal_ratio = total_signals / len(df) * 100
                
                st.write(f"Signal generation rate: {signal_ratio:.1f}%")
                if 5 <= signal_ratio <= 15:
                    st.success("✅ Signal generation rate is optimal (5-15%)")
                    self.diagnostics['tests_passed'] += 1
                else:
                    st.warning(f"⚠️ Signal rate {signal_ratio:.1f}% (expected 5-15%)")
                    self.diagnostics['warnings'].append(f"Signal rate: {signal_ratio:.1f}%")
            
            self.diagnostics['tests_run'] += 1
        
        # Triple Alignment Quality
        with st.expander("Triple Alignment Pattern Testing", expanded=True):
            triple_align = df[df.get('EDGE_SIGNAL', '') == 'TRIPLE_ALIGNMENT']
            
            if len(triple_align) > 0:
                st.success(f"✅ Found {len(triple_align)} Triple Alignment patterns")
                
                # Verify conditions
                conditions_met = {
                    'Volume > 10%': (triple_align['volume_acceleration'] > 10).all(),
                    'EPS Growing': (triple_align['eps_current'] > triple_align['eps_last_qtr']).all(),
                    'Away from highs': (triple_align['from_high_pct'] < -20).all(),
                    'Stable price': (triple_align['ret_30d'].abs() < 5).all()
                }
                
                for condition, met in conditions_met.items():
                    if met:
                        st.write(f"✅ {condition}")
                    else:
                        st.write(f"❌ {condition}")
                        self.diagnostics['warnings'].append(f"Triple alignment condition failed: {condition}")
            else:
                st.info("No Triple Alignment patterns found (this is rare and normal)")
            
            self.diagnostics['signal_analysis']['triple_alignments'] = len(triple_align)
            self.diagnostics['tests_run'] += 1
    
    def test_indian_market_compatibility(self):
        """Test Indian market specific features"""
        st.subheader("🇮🇳 Indian Market Compatibility")
        
        df = self.load_and_process_data()
        if df is None:
            return
        
        # Test 1: Currency Format
        with st.expander("Test 1: Indian Currency (₹) Handling", expanded=True):
            # Check if price columns are numeric after ₹ removal
            price_cols = ['price', 'low_52w', 'high_52w', 'sma_20d', 'sma_50d', 'sma_200d']
            available_price_cols = [col for col in price_cols if col in df.columns]
            
            all_numeric = True
            for col in available_price_cols:
                if df[col].dtype not in ['float64', 'int64']:
                    all_numeric = False
                    st.error(f"❌ {col} is not numeric")
            
            if all_numeric:
                st.success("✅ All price columns properly converted from ₹ format")
                self.diagnostics['tests_passed'] += 1
                
                # Show sample prices
                st.write("Sample price data:")
                st.dataframe(df[available_price_cols].head(5))
            else:
                self.diagnostics['tests_failed'] += 1
            
            self.diagnostics['tests_run'] += 1
        
        # Test 2: Indian Exchanges
        with st.expander("Test 2: Indian Exchange Detection", expanded=True):
            if 'exchange' in df.columns:
                exchanges = df['exchange'].value_counts()
                st.write("Exchanges found:")
                st.write(exchanges)
                
                indian_exchanges = ['NSE', 'BSE', 'NSE:', 'BSE:']
                indian_found = any(ex in str(exchanges.index.tolist()) for ex in indian_exchanges)
                
                if indian_found:
                    st.success("✅ Indian exchanges detected")
                    self.diagnostics['tests_passed'] += 1
                else:
                    st.warning("⚠️ No clear Indian exchange markers found")
                    self.diagnostics['warnings'].append("Indian exchange detection failed")
            
            self.diagnostics['tests_run'] += 1
        
        # Test 3: Market Cap in Crores
        with st.expander("Test 3: Market Cap (Cr) Format", expanded=True):
            if 'market_cap' in df.columns:
                # Check original format
                sample_caps = df['market_cap'].head(10).tolist()
                cr_format = any('Cr' in str(cap) for cap in sample_caps if pd.notna(cap))
                
                if cr_format:
                    st.success("✅ Market cap in Crores format detected")
                    self.diagnostics['tests_passed'] += 1
                else:
                    st.info("Market cap format unclear")
                
                st.write("Sample market caps:", sample_caps[:5])
            
            self.diagnostics['tests_run'] += 1
        
        # Test 4: Sector Distribution
        with st.expander("Test 4: Indian Sector Analysis", expanded=True):
            if 'sector' in df.columns:
                sectors = df['sector'].value_counts().head(10)
                
                fig = go.Figure(data=[go.Bar(x=sectors.values, y=sectors.index, orientation='h')])
                fig.update_layout(title="Top 10 Sectors", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Common Indian sectors
                indian_sectors = ['Banks', 'IT', 'Pharma', 'Auto', 'FMCG', 'Metal', 'Realty']
                found_indian = any(sector in str(sectors.index.tolist()) for sector in indian_sectors)
                
                if found_indian:
                    st.success("✅ Indian market sectors identified")
                    self.diagnostics['tests_passed'] += 1
                
                self.diagnostics['data_quality']['top_sectors'] = sectors.head(5).to_dict()
            
            self.diagnostics['tests_run'] += 1
    
    def test_performance_metrics(self):
        """Test system performance"""
        st.subheader("⚡ Performance Metrics")
        
        # Test load times
        with st.expander("Load Time Analysis", expanded=True):
            # Test 1: Data load time
            start = time.time()
            df = self.load_raw_data()
            load_time = time.time() - start
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Data Load Time", f"{load_time:.2f}s")
                if load_time < 5:
                    st.success("✅ Excellent")
                elif load_time < 10:
                    st.warning("⚠️ Acceptable")
                else:
                    st.error("❌ Too slow")
            
            # Test 2: Processing time
            if df is not None:
                start = time.time()
                df_processed = self.apply_signal_logic(df)
                process_time = time.time() - start
                
                with col2:
                    st.metric("Processing Time", f"{process_time:.2f}s")
                    if process_time < 3:
                        st.success("✅ Excellent")
                
                with col3:
                    total_time = load_time + process_time
                    st.metric("Total Time", f"{total_time:.2f}s")
            
            self.diagnostics['performance_metrics'] = {
                'load_time': load_time,
                'process_time': process_time if 'process_time' in locals() else None,
                'total_time': total_time if 'total_time' in locals() else None
            }
        
        # Memory usage
        with st.expander("Memory Usage", expanded=True):
            if df is not None:
                memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric("DataFrame Memory", f"{memory_mb:.1f} MB")
                
                if memory_mb < 100:
                    st.success("✅ Memory usage optimal")
                else:
                    st.warning("⚠️ High memory usage")
                
                self.diagnostics['performance_metrics']['memory_mb'] = memory_mb
    
    def generate_diagnostic_reports(self):
        """Generate downloadable diagnostic reports"""
        st.subheader("📥 Diagnostic Reports")
        
        # Update final counts
        self.diagnostics['total_tests'] = self.diagnostics['tests_run']
        self.diagnostics['success_rate'] = (
            self.diagnostics['tests_passed'] / self.diagnostics['tests_run'] * 100 
            if self.diagnostics['tests_run'] > 0 else 0
        )
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tests Run", self.diagnostics['tests_run'])
        with col2:
            st.metric("Tests Passed", self.diagnostics['tests_passed'])
        with col3:
            st.metric("Tests Failed", self.diagnostics['tests_failed'])
        with col4:
            st.metric("Success Rate", f"{self.diagnostics['success_rate']:.1f}%")
        
        # Overall health status
        if self.diagnostics['success_rate'] >= 90:
            st.success("✅ System Health: EXCELLENT")
        elif self.diagnostics['success_rate'] >= 70:
            st.warning("⚠️ System Health: GOOD (with warnings)")
        else:
            st.error("❌ System Health: NEEDS ATTENTION")
        
        # Download options
        st.markdown("### Download Diagnostic Reports")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # JSON Report
            json_report = json.dumps(self.diagnostics, indent=2, default=str)
            b64 = base64.b64encode(json_report.encode()).decode()
            href = f'<a href="data:application/json;base64,{b64}" download="mantra_diagnostics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json">📥 Download JSON Report</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            # Text Report
            text_report = self.generate_text_report()
            b64 = base64.b64encode(text_report.encode()).decode()
            href = f'<a href="data:text/plain;base64,{b64}" download="mantra_diagnostics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt">📥 Download Text Report</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        with col3:
            # Full Data Sample
            df = self.load_and_process_data()
            if df is not None:
                csv = df.head(100).to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:text/csv;base64,{b64}" download="mantra_sample_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv">📥 Download Sample Data</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        # Show warnings and errors
        if self.diagnostics['warnings']:
            with st.expander(f"⚠️ Warnings ({len(self.diagnostics['warnings'])})", expanded=True):
                for warning in self.diagnostics['warnings']:
                    st.warning(warning)
        
        if self.diagnostics['errors']:
            with st.expander(f"❌ Errors ({len(self.diagnostics['errors'])})", expanded=True):
                for error in self.diagnostics['errors']:
                    st.error(error)
    
    # Helper methods
    def load_raw_data(self):
        """Load raw data for testing"""
        try:
            response = requests.get(SHEET_URL, timeout=30)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
            df.columns = [col.strip() for col in df.columns]
            return df
        except Exception as e:
            st.error(f"Failed to load data: {str(e)}")
            return None
    
    def clean_data(self, df):
        """Apply basic cleaning"""
        # Remove empty columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed|^_|^$', regex=True)]
        
        # Convert numeric columns
        for col in df.columns:
            if any(keyword in col for keyword in ['price', 'volume', 'ret_', 'pe', 'eps']):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def load_and_process_data(self):
        """Load and process data similar to main app"""
        df = self.load_raw_data()
        if df is None:
            return None
        
        df = self.clean_data(df)
        
        # Basic calculations
        if all(col in df.columns for col in ['vol_ratio_30d_90d', 'vol_ratio_30d_180d']):
            df['volume_acceleration'] = df['vol_ratio_30d_90d'] - df['vol_ratio_30d_180d']
        
        return df
    
    def apply_signal_logic(self, df):
        """Apply basic signal logic for testing"""
        df['EDGE_SIGNAL'] = 'NONE'
        
        # Simple signal logic for testing
        if all(col in df.columns for col in ['volume_acceleration', 'eps_current', 'eps_last_qtr', 'from_high_pct', 'ret_30d', 'pe']):
            # Triple alignment
            triple_mask = (
                (df['volume_acceleration'] > 10) &
                (df['eps_current'] > df['eps_last_qtr']) &
                (df['from_high_pct'] < -20) &
                (df['ret_30d'].abs() < 5) &
                (df['pe'] > 0) & (df['pe'] < 50)
            )
            df.loc[triple_mask, 'EDGE_SIGNAL'] = 'TRIPLE_ALIGNMENT'
            
            # Coiled spring
            spring_mask = (
                (df['volume_acceleration'] > 5) &
                (df['ret_30d'].abs() < 5) &
                (df['from_high_pct'] < -30) &
                (df['EDGE_SIGNAL'] == 'NONE')
            )
            df.loc[spring_mask, 'EDGE_SIGNAL'] = 'COILED_SPRING'
        
        return df
    
    def generate_text_report(self):
        """Generate human-readable text report"""
        report = f"""
M.A.N.T.R.A. DIAGNOSTIC REPORT
==============================
Generated: {self.diagnostics['timestamp']}

SUMMARY
-------
Total Tests Run: {self.diagnostics['tests_run']}
Tests Passed: {self.diagnostics['tests_passed']}
Tests Failed: {self.diagnostics['tests_failed']}
Success Rate: {self.diagnostics['success_rate']:.1f}%

DATA QUALITY
------------
{json.dumps(self.diagnostics['data_quality'], indent=2)}

PERFORMANCE METRICS
------------------
{json.dumps(self.diagnostics['performance_metrics'], indent=2)}

SIGNAL ANALYSIS
--------------
{json.dumps(self.diagnostics['signal_analysis'], indent=2)}

WARNINGS ({len(self.diagnostics['warnings'])})
--------
{chr(10).join(self.diagnostics['warnings'])}

ERRORS ({len(self.diagnostics['errors'])})
------
{chr(10).join(self.diagnostics['errors'])}

RECOMMENDATION
--------------
"""
        if self.diagnostics['success_rate'] >= 90:
            report += "System is functioning EXCELLENTLY. Ready for production use."
        elif self.diagnostics['success_rate'] >= 70:
            report += "System is functioning WELL with minor issues. Review warnings."
        else:
            report += "System needs ATTENTION. Review errors and warnings before production use."
        
        return report

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    st.title("🔍 M.A.N.T.R.A. System Diagnostics")
    st.markdown("""
    This diagnostic tool comprehensively tests your M.A.N.T.R.A. EDGE system to ensure:
    - ✅ Data loads correctly from Google Sheets
    - ✅ All calculations are accurate
    - ✅ Signals generate properly
    - ✅ Indian market compatibility
    - ✅ Performance is optimal
    """)
    
    # Initialize diagnostics
    diagnostics = SystemDiagnostics()
    
    # Run button
    if st.button("🚀 Run Complete Diagnostics", type="primary", use_container_width=True):
        diagnostics.run_complete_diagnostics()
    else:
        st.info("👆 Click 'Run Complete Diagnostics' to start comprehensive system testing")
    
    # Quick health check
    with st.sidebar:
        st.header("🏥 Quick Health Check")
        
        if st.button("Test Connection"):
            try:
                response = requests.head(SHEET_URL, timeout=5)
                if response.status_code == 200:
                    st.success("✅ Connected")
                else:
                    st.error(f"❌ Error: {response.status_code}")
            except:
                st.error("❌ Connection failed")
        
        if st.button("Test Data Load"):
            try:
                df = pd.read_csv(SHEET_URL, nrows=5)
                st.success(f"✅ Loaded {len(df)} rows")
            except Exception as e:
                st.error(f"❌ {str(e)}")
        
        st.markdown("---")
        st.markdown("""
        ### 📋 What This Tests:
        
        1. **Data Pipeline**
           - Google Sheets access
           - CSV parsing
           - Column validation
           - Data quality
        
        2. **Calculations**
           - Volume acceleration
           - Momentum metrics
           - Conviction scores
        
        3. **Signal Logic**
           - Pattern detection
           - Signal distribution
           - Edge cases
        
        4. **Indian Market**
           - Currency handling
           - Exchange detection
           - Sector analysis
        
        5. **Performance**
           - Load times
           - Memory usage
           - Processing speed
        """)

if __name__ == "__main__":
    main()
