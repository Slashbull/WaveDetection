# debug_test.py - Quick data check
import streamlit as st
import pandas as pd
import requests
import io

st.set_page_config(page_title="Data Debug Test", layout="wide")

st.title("🔍 M.A.N.T.R.A. Data Debug Test")

# Load data
@st.cache_data
def load_data():
    url = "https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk/export?format=csv&gid=2026492216"
    response = requests.get(url)
    df = pd.read_csv(io.StringIO(response.text))
    return df

df = load_data()

# Show basic info
st.header("1. Basic Data Info")
st.write(f"- Total rows: {len(df)}")
st.write(f"- Total columns: {len(df.columns)}")

# Show column names
st.header("2. Column Names (first 50)")
cols = df.columns.tolist()
st.write(cols[:50])

# Show data types
st.header("3. Data Types")
st.dataframe(df.dtypes.astype(str).to_frame('Type'))

# Show first few rows
st.header("4. First 5 Rows (raw data)")
st.dataframe(df.head())

# Check specific columns
st.header("5. Specific Column Values")

# Check volume ratio columns
vol_cols = [col for col in df.columns if 'vol_ratio' in col.lower()]
if vol_cols:
    st.write("Volume ratio columns found:", vol_cols)
    for col in vol_cols:
        st.write(f"\n{col} sample values:")
        st.write(df[col].head(10).tolist())

# Check return columns  
ret_cols = [col for col in df.columns if col.lower().startswith('ret_')]
if ret_cols:
    st.write("\nReturn columns found:", ret_cols)
    for col in ret_cols[:3]:  # Show first 3
        st.write(f"\n{col} sample values:")
        st.write(df[col].head(10).tolist())

# Check EPS tier
eps_tier_cols = [col for col in df.columns if 'eps_tier' in col.lower()]
if eps_tier_cols:
    st.write("\nEPS tier column:", eps_tier_cols)
    st.write("Unique values:", df[eps_tier_cols[0]].unique()[:20])

# Check numeric conversion
st.header("6. Numeric Conversion Test")
test_col = 'price' if 'price' in df.columns else df.columns[0]
st.write(f"Testing column: {test_col}")
st.write(f"Original values: {df[test_col].head().tolist()}")

# Try converting
try:
    converted = pd.to_numeric(
        df[test_col].astype(str).str.replace('[₹$,€£%CrLKMB]', '', regex=True),
        errors='coerce'
    )
    st.write(f"Converted values: {converted.head().tolist()}")
    st.write(f"Nulls after conversion: {converted.isna().sum()}")
except Exception as e:
    st.error(f"Conversion error: {e}")
