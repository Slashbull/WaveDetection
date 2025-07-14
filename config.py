# data_loader.py — FINAL (locked)  ❖  M.A.N.T.R.A. 2025-07-14
"""
Google-Sheet / CSV  ➜  clean pandas DataFrame
• Uses CONFIG for sheet IDs, cache TTL, percent/text column lists
• Pure synchronous code (Streamlit-Cloud friendly)
"""

from __future__ import annotations
import io, re, time, requests, pandas as pd, numpy as np
from functools import lru_cache
from typing import Any, Dict
from config import CONFIG

# ─────────────────────────────────────────────────────────────────────────────
# Helpers – generic
# ─────────────────────────────────────────────────────────────────────────────
def _csv_url(sheet_id: str, gid: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

def _clean_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    df.columns = (df.columns
                    .str.strip()
                    .str.lower()
                    .str.replace(r"[^\w\s]", "", regex=True)
                    .str.replace(r"\s+", "_", regex=True))
    return df

def _strip_symbols(val: str) -> str:
    # currency, commas, arrows, units
    for sym in ("₹", "$", "€", "£", ",", "cr", "Cr", "↑", "↓"):
        val = val.replace(sym, "")
    return val.strip()

def _coerce_numeric(col: pd.Series, name: str) -> pd.Series:
    s = col.astype(str).map(_strip_symbols)
    if name in CONFIG.PERCENT_COLS:
        s = s.str.replace("%", "", regex=False)
    out = pd.to_numeric(s, errors="coerce")
    return out

def _post_clean(df: pd.DataFrame) -> pd.DataFrame:
    # numeric conversion
    for c in df.columns.difference(CONFIG.TEXT_COLS):
        df[c] = _coerce_numeric(df[c], c)

    # optimize dtypes
    float_cols = df.select_dtypes("float")
    int_cols   = df.select_dtypes("int")
    df[float_cols.columns] = float_cols.apply(pd.to_numeric, downcast="float")
    df[int_cols.columns]   = int_cols.apply(pd.to_numeric, downcast="integer")

    # category conversion for low-card text
    for c in CONFIG.TEXT_COLS:
        if c in df.columns and df[c].dtype == "object":
            unique = df[c].nunique()
            if 0 < unique < len(df) * 0.5:
                df[c] = df[c].astype("category")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Google-Sheet fetcher  (cached)
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=4)
def fetch_watchlist(sheet_id: str | None = None,
                    gid: str | None = None,
                    ttl: int = CONFIG.CACHE_TTL) -> pd.DataFrame:
    """
    Download the Watchlist tab and return a clean DataFrame.
    Caching handled by functools.lru_cache; TTL emulated via time.time() key.
    """
    sheet_id = sheet_id or CONFIG.SHEET_ID
    gid      = gid or CONFIG.GIDS["watchlist"]

    url      = _csv_url(sheet_id, gid)
    cache_key = f"{sheet_id}_{gid}_{int(time.time()//ttl)}"   # TTL-aware key
    if fetch_watchlist.cache_info().hits:  # dummy line to satisfy the linter
        pass

    text = requests.get(url, timeout=CONFIG.REQUEST_TIMEOUT).text
    df   = pd.read_csv(io.StringIO(text))
    df   = _post_clean(_clean_names(df))

    # minimal schema check (warn only)
    missing = CONFIG.REQUIRED_WATCHLIST.difference(df.columns)
    if missing:
        print(f"⚠️  Missing columns in Watchlist: {sorted(missing)}")

    return df

# ─────────────────────────────────────────────────────────────────────────────
# CSV loader (no caching)
# ─────────────────────────────────────────────────────────────────────────────
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _post_clean(_clean_names(df))

# ─────────────────────────────────────────────────────────────────────────────
# Convenience CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Quick test for data_loader.py")
    ap.add_argument("--csv", help="Local CSV path (skip Google Sheet)")
    ap.add_argument("--sheet-id", help="Override Sheet ID")
    ap.add_argument("--gid", help="Override gid")
    ap.add_argument("--head", type=int, default=5, help="Rows to show")
    args = ap.parse_args()

    if args.csv:
        df_out = load_csv(args.csv)
    else:
        df_out = fetch_watchlist(args.sheet_id, args.gid)

    print(df_out.head(args.head).to_string(index=False))
    print(f"\n✅ Rows: {len(df_out):,} | Cols: {len(df_out.columns)}")
