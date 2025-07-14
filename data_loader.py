# data_loader.py - FINAL
import io, re, time, logging, warnings
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry
from typing import Tuple, Dict, Any, Optional
from datetime import datetime

from config import CONFIG

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [DATA] %(levelname)s - %(message)s"))
    logger.addHandler(handler)
warnings.filterwarnings('ignore')

def create_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=CONFIG.MAX_RETRIES, backoff_factor=0.3, status_forcelist=[500,502,503,504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))
    return session
_session = None
def get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = create_session()
    return _session

class SimpleCache:
    def __init__(self): self._cache = {}
    def get(self, key, ttl=300): 
        if key in self._cache:
            value, ts = self._cache[key]
            if time.time() - ts < ttl: return value
            else: del self._cache[key]
        return None
    def set(self, key, value): self._cache[key] = (value, time.time())
    def clear(self): self._cache.clear()
_cache = SimpleCache()

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.match(r"Unnamed")]
    df.columns = [
        re.sub(r"\s+", "_", re.sub(r"[^\w\s]", "", col.strip().lower()))
        for col in df.columns
    ]
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace("\u00A0", " ", regex=False)
    return df

def clean_numeric_series(series: pd.Series, col_name: str = "") -> pd.Series:
    s = series.astype(str)
    for symbol in ['₹','$','€','£','Cr','L','K','M','B','%',',','↑','↓']:
        s = s.str.replace(symbol,'',regex=False)
    s = s.str.replace(r'[^\x00-\x7F]+','',regex=True).str.strip()
    s = s.replace('', 'NaN')
    numeric_series = pd.to_numeric(s, errors='coerce')
    if col_name.endswith('_pct') or '%' in series.astype(str).str.cat():
        return numeric_series
    non_null = numeric_series.dropna()
    if len(non_null) > 0 and non_null.max() < 1 and non_null.min() >= 0:
        return numeric_series * 100
    return numeric_series

def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    numeric_pattern = re.compile(r"^(price|prev_close|ret_|avg_ret|volume|vol_ratio|low_52w|high_52w|from_low_pct|from_high_pct|pe|eps|rvol|market_cap|sma_|dma_|sector_ret_|sector_avg_)")
    for col in df.columns:
        if numeric_pattern.match(col):
            df[col] = clean_numeric_series(df[col], col)
    return df

def load_sheet(name: str, use_cache: bool = True) -> pd.DataFrame:
    url = CONFIG.get_sheet_url(name)
    cache_key = f"sheet_{name}_{CONFIG.SCHEMA_VERSION}"
    if use_cache:
        cached = _cache.get(cache_key, CONFIG.CACHE_TTL)
        if cached is not None:
            logger.info(f"✅ Cache hit for sheet '{name}'")
            return cached
    logger.info(f"📥 Loading sheet '{name}'...")
    try:
        session = get_session()
        response = session.get(url, timeout=CONFIG.REQUEST_TIMEOUT)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        df = clean_dataframe(df)
        if use_cache: _cache.set(cache_key, df)
        return df
    except Exception as e:
        logger.error(f"❌ Failed to load sheet '{name}': {e}")
        return pd.DataFrame()

def validate_schema(df: pd.DataFrame, required_cols: set, sheet_name: str) -> None:
    missing = required_cols - set(df.columns)
    if missing:
        logger.warning(f"⚠️ Missing columns in {sheet_name}: {missing}")

def merge_datasets(watchlist_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
    returns_df = returns_df.drop(columns=['company_name'], errors='ignore')
    merged = watchlist_df.merge(returns_df, on='ticker', how='left', validate='one_to_one')
    if merged['ticker'].duplicated().any():
        merged = merged.drop_duplicates(subset='ticker', keep='last')
    merged['ticker'] = merged['ticker'].astype(str).str.upper().str.strip()
    return merged

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    # (add any custom features here as needed, omitted for brevity)
    return df

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in ['sector','category','exchange','price_tier','eps_tier']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype('category')
    return df

def analyze_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    analysis = {
        'timestamp': datetime.utcnow().isoformat(),
        'row_count': len(df),
        'column_count': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    null_counts = df.isnull().sum()
    null_percentage = (null_counts.sum() / (len(df) * len(df.columns)) * 100).round(2) if len(df) else 0
    analysis['null_percentage'] = float(null_percentage)
    analysis['columns_with_nulls'] = int((null_counts > 0).sum())
    analysis['duplicate_tickers'] = int(df['ticker'].duplicated().sum()) if 'ticker' in df else 0
    quality_score = 100.0 - min(50, null_percentage * 2) - min(20, analysis['duplicate_tickers'] * 0.5)
    analysis['quality_score'] = max(0, quality_score)
    analysis['quality_grade'] = (
        'A' if quality_score >= 90 else
        'B' if quality_score >= 80 else
        'C' if quality_score >= 70 else
        'D' if quality_score >= 60 else
        'F'
    )
    return analysis

def load_and_process(use_cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    start_time = time.time()
    try:
        watchlist_df = load_sheet('watchlist', use_cache)
        returns_df = load_sheet('returns', use_cache)
        sector_df = load_sheet('sector', use_cache)
        validate_schema(watchlist_df, CONFIG.REQUIRED_WATCHLIST, 'watchlist')
        validate_schema(returns_df, CONFIG.REQUIRED_RETURNS, 'returns')
        validate_schema(sector_df, CONFIG.REQUIRED_SECTOR, 'sector')
        stocks_df = merge_datasets(watchlist_df, returns_df)
        stocks_df = clean_numeric_columns(stocks_df)
        sector_df = clean_numeric_columns(sector_df)
        stocks_df = add_derived_features(stocks_df)
        stocks_df = optimize_dtypes(stocks_df)
        sector_df = optimize_dtypes(sector_df)
        quality_analysis = analyze_data_quality(stocks_df)
        health = {
            'processing_time_s': time.time() - start_time,
            'timestamp': datetime.utcnow().isoformat(),
            'quality_analysis': quality_analysis,
            'total_stocks': len(stocks_df),
            'total_sectors': len(sector_df),
            'cache_used': use_cache
        }
        return stocks_df, sector_df, health
    except Exception as e:
        logger.error(f"❌ Data loading pipeline failed: {e}")
        return pd.DataFrame(), pd.DataFrame(), {}
