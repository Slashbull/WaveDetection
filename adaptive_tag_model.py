# adaptive_tag_model.py — FINAL (locked)  ❖  M.A.N.T.R.A. 2025‑07‑14
"""
Pure‑logic BUY / WATCH / AVOID engine.
• Consumes a cleaned Watchlist DataFrame (see data_loader.py)
• Produces `edge_prob` (0‑1) and `tag` columns
• All tunable constants pulled from CONFIG (no magic numbers hard‑coded)
"""

from __future__ import annotations
import numpy as np, pandas as pd
from scipy import stats
from config import CONFIG

# ─────────────────────────────────────────────────────────────────────────────
# Constants (only EPS tier map local — everything else in CONFIG)
# ─────────────────────────────────────────────────────────────────────────────
_EPS_MAP = {"5↓":0,"5↑":1,"15↑":2,"35↑":3,"55↑":4,"75↑":5,"95↑":6}
_MAX_EPS = max(_EPS_MAP.values())

# Shortcuts
W  = CONFIG.FACTOR_WEIGHTS
TH = CONFIG.TAG_THRESHOLDS
FL = CONFIG.HARD_FILTERS

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
_pct_rank = lambda s: s.rank(pct=True, method="min")
_z        = lambda s: stats.zscore(s.fillna(s.median()), nan_policy="omit")


# ─────────────────────────────────────────────────────────────────────────────
# Factor construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_factors(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)

    # Momentum (three speeds)
    f["mom_fast"] = _pct_rank(df["ret_1d"] + df["ret_3d"] + df["ret_7d"])
    f["mom_mid"]  = _pct_rank(df["ret_30d"] + df["ret_3m"])
    f["mom_long"] = _pct_rank(df["ret_6m"] + df["ret_1y"])

    # Trend
    f["trend"] = _pct_rank(df["price"] / df["sma_20d"] +
                            df["price"] / df["sma_50d"] +
                            df["price"] / df["sma_200d"])

    # Value (inverse PE)
    f["value"] = _pct_rank(-df["pe"].replace([np.inf, -np.inf], np.nan))

    # EPS (growth + tier)
    eps_scaled = df["eps_tier"].map(_EPS_MAP).fillna(0) / _MAX_EPS
    f["eps"] = 0.5 * _pct_rank(df["eps_change_pct"]) + 0.5 * eps_scaled

    # Volume conviction
    f["volume"] = _pct_rank(df["vol_ratio_1d_90d"] +
                             df["vol_ratio_7d_90d"] +
                             df["vol_ratio_30d_90d"] +
                             df["rvol"])

    # Risk (lower → better)
    stretch = df["from_high_pct"].clip(lower=-25)
    deep    = (-df["from_low_pct"]).clip(lower=-75)
    f["risk"] = 1 - _pct_rank(stretch + deep)

    # Liquidity
    f["liquidity"] = _pct_rank(_pct_rank(df["market_cap"]) * _pct_rank(df["price"]))

    return f


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _blend(f: pd.DataFrame) -> pd.Series:
    # Exponent‑weighted geometric mean (prevents one weak pillar hiding)
    f_clipped = f.clip(lower=1e-6, upper=1)
    exps = pd.Series(W, index=f.columns)
    return np.exp((np.log(f_clipped) * exps).sum(axis=1))


def _probability(raw: pd.Series) -> pd.Series:
    return 1 / (1 + np.exp(-_z(raw)))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def tag_watchlist(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    # Build composite edge probability
    factors     = _build_factors(work)
    edge_raw    = _blend(factors)
    work["edge_prob"] = _probability(edge_raw)

    # Initial tag assignment
    work["tag"] = "AVOID"
    work.loc[work.edge_prob >= TH["buy"],   "tag"] = "BUY"
    mask_watch = (work.edge_prob >= TH["watch"]) & (work.edge_prob < TH["buy"])
    work.loc[mask_watch, "tag"] = "WATCH"

    # Hard‑filter demotions
    fail = (
        (work["pe"] > FL["pe_max"]) |
        (work["eps_change_pct"] < FL["eps_change_min"]) |
        (work["vol_ratio_30d_90d"] < FL["vol_ratio_30d_min"]) |
        (work["price"] <= work["sma_50d"]) |
        (work["price"] <= work["sma_200d"])
    )
    work.loc[(work.tag == "BUY")   & fail, "tag"] = "WATCH"
    work.loc[(work.tag == "WATCH") & fail, "tag"] = "AVOID"

    return work[[
        "ticker", "category", "sector", "price_tier", "tag", "edge_prob"
    ]].sort_values("edge_prob", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# CLI test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, data_loader as dl

    ap = argparse.ArgumentParser(description="Run adaptive tag model on watchlist")
    ap.add_argument("--csv", help="Local CSV instead of Google Sheet")
    ap.add_argument("--head", type=int, default=25)
    args = ap.parse_args()

    df_in = dl.load_csv(args.csv) if args.csv else dl.fetch_watchlist()
    out   = tag_watchlist(df_in)
    pd.set_option("display.float_format", lambda x: f"{x:0.3f}")
    print(out.head(args.head).to_string(index=False))
