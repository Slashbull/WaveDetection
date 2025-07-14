# engine.py (ALL-TIME BEST, FINAL VERSION)
"""
M.A.N.T.R.A. Analytics & Scoring Engine
=======================================
Contains ALL scoring, tagging, anomaly, edge, sector, and regime logic.
Each function is pure, stateless, and can be imported individually.
Optimized for integration with core.py data pipeline and Streamlit dashboard.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any

# ============================================================================
# REGIME WEIGHTS & DETECTION
# ============================================================================

REGIME_WEIGHTS = {
    "balanced":  {"momentum": 0.22, "value": 0.21, "eps": 0.19, "volume": 0.19, "sector": 0.19},
    "momentum":  {"momentum": 0.40, "value": 0.10, "eps": 0.15, "volume": 0.20, "sector": 0.15},
    "value":     {"momentum": 0.10, "value": 0.45, "eps": 0.15, "volume": 0.15, "sector": 0.15},
    "growth":    {"momentum": 0.17, "value": 0.08, "eps": 0.40, "volume": 0.15, "sector": 0.20},
    "volume":    {"momentum": 0.15, "value": 0.10, "eps": 0.10, "volume": 0.50, "sector": 0.15},
}

def get_regime_weights(regime: str = "balanced") -> Dict[str, float]:
    """Returns the weight dict for the selected regime."""
    return REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["balanced"]).copy()

def auto_detect_regime(df: pd.DataFrame) -> str:
    """
    Auto-detects regime based on *current* data only.
    Returns: regime key ("momentum", "value", "growth", "volume", "balanced")
    """
    try:
        if "ret_3m" in df and (df["ret_3m"] > 6).mean() > 0.6:
            return "momentum"
        if "pe" in df and "ret_1y" in df:
            if df["pe"].median() < 16 and df["ret_1y"].median() < 3:
                return "value"
        if "eps_change_pct" in df and (df["eps_change_pct"] > 15).mean() > 0.5:
            return "growth"
        if "vol_ratio_1d_90d" in df and (df["vol_ratio_1d_90d"] > 2.2).mean() > 0.35:
            return "volume"
        return "balanced"
    except Exception:
        return "balanced"

# ============================================================================
# SIGNAL SCORING ENGINE
# ============================================================================

class SignalEngine:
    """
    Flexible, robust signal engine.
    Uses 5 factors: Momentum, Value, EPS, Volume, Sector.
    Weights are passed at init (regime support).
    """

    def __init__(self, weights: Dict[str, float]):
        # All factor keys: momentum, value, eps, volume, sector
        self.weights = weights
        self.momentum_cols = ["ret_3d", "ret_7d", "ret_30d", "ret_3m"]

    def _percentile(self, series: pd.Series, ascending=True) -> pd.Series:
        # Ranks from 0-100, fallback to mean if empty/constant
        if series.notna().sum() == 0:
            return pd.Series(50.0, index=series.index)
        ranked = series.rank(method="average", ascending=ascending, pct=True) * 100
        return ranked.fillna(50.0)

    def compute_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.momentum_cols:
            if col not in df: df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        arr = np.vstack([self._percentile(df[c]) for c in self.momentum_cols])
        weights = np.array([0.10, 0.20, 0.30, 0.40])
        df["momentum_score"] = np.average(arr, axis=0, weights=weights)
        return df

    def compute_value(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["pe"] = pd.to_numeric(df.get("pe", 0), errors="coerce").fillna(0)
        df["eps_current"] = pd.to_numeric(df.get("eps_current", 0), errors="coerce").fillna(0)
        value_ratio = pd.Series(0.0, index=df.index)
        mask = (df["pe"] > 0) & (df["eps_current"] > 0)
        value_ratio[mask] = df.loc[mask, "eps_current"] / df.loc[mask, "pe"]
        df["value_score"] = self._percentile(value_ratio, ascending=True)
        return df

    def compute_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ["vol_ratio_1d_90d", "vol_ratio_7d_90d", "vol_ratio_30d_90d"]:
            if col not in df: df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        combo = (0.5 * df["vol_ratio_1d_90d"] +
                 0.3 * df["vol_ratio_7d_90d"] +
                 0.2 * df["vol_ratio_30d_90d"])
        df["volume_score"] = self._percentile(combo, ascending=True)
        return df

    def compute_eps(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["eps_change_pct"] = pd.to_numeric(df.get("eps_change_pct", 0), errors="coerce").fillna(0)
        df["eps_score"] = self._percentile(df["eps_change_pct"].clip(lower=-50, upper=200), ascending=True)
        return df

    def compute_sector(self, df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "sector" not in df: df["sector"] = "Unknown"
        df["sector"] = df["sector"].astype(str).fillna("Unknown")
        sector_map = {}
        if sector_df is not None and not sector_df.empty and "sector" in sector_df and "sector_avg_3m" in sector_df:
            sector_map = pd.Series(
                pd.to_numeric(sector_df.set_index("sector")["sector_avg_3m"], errors="coerce"),
                index=sector_df["sector"]
            )
            mean_score = sector_map.mean() if len(sector_map) else 50.0
            df["sector_score"] = df["sector"].map(sector_map).fillna(mean_score)
            df["sector_score"] = self._percentile(df["sector_score"], ascending=True)
        else:
            df["sector_score"] = 50.0
        return df

    def fit_transform(self, df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
        df = self.compute_momentum(df)
        df = self.compute_value(df)
        df = self.compute_volume(df)
        df = self.compute_eps(df)
        df = self.compute_sector(df, sector_df)
        # Defensive: ensure all exist, all numeric
        score_cols = ["momentum_score", "value_score", "volume_score", "eps_score", "sector_score"]
        for col in score_cols:
            if col not in df: df[col] = 50.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(50.0)
        w = np.array([
            self.weights["momentum"], self.weights["value"],
            self.weights["eps"], self.weights["volume"], self.weights["sector"]
        ])
        w = w / w.sum()
        arr = np.vstack([df[c].values for c in score_cols])
        df["final_score"] = np.average(arr, axis=0, weights=w)
        df["final_score"] = df["final_score"].clip(0, 100).round(2)
        df["final_rank"] = df["final_score"].rank(method="min", ascending=False).fillna(999999).astype(int)
        return df

def run_signal_engine(df: pd.DataFrame, sector_df: pd.DataFrame, regime: Optional[str] = "balanced") -> pd.DataFrame:
    """
    Runs signal scoring engine with regime weights.
    Returns DataFrame with factor and final scores, rank.
    """
    weights = get_regime_weights(regime)
    engine = SignalEngine(weights)
    return engine.fit_transform(df, sector_df)

# ============================================================================
# DECISION/TAGGING ENGINE
# ============================================================================

def run_decision_engine(
    df: pd.DataFrame,
    thresholds: Optional[Dict] = None,
    tag_date: Optional[str] = None,
    sort_by: str = "final_score",
    ascending: bool = False
) -> pd.DataFrame:
    """
    Tags each stock as 'Buy', 'Watch', or 'Avoid' based on final_score.
    Also adds color, confidence, band, and signal_strength fields for UI.
    """
    DEFAULT_THRESHOLDS = {"buy": 75, "watch": 60}
    thresholds = thresholds or DEFAULT_THRESHOLDS
    df = df.copy()
    if df.empty or "final_score" not in df:
        df["tag"] = "Avoid"
        df["tag_color"] = "red"
        df["confidence"] = 0
        df["confidence_band"] = "Low"
        df["signal_strength"] = "↓ Weak"
        df["tag_reason"] = ""
        df["tag_date"] = tag_date or pd.Timestamp.today().strftime("%Y-%m-%d")
        return df
    df["tag"] = "Avoid"
    df.loc[df["final_score"] >= thresholds["watch"], "tag"] = "Watch"
    df.loc[df["final_score"] >= thresholds["buy"], "tag"] = "Buy"
    tag_color_map = {"Buy": "green", "Watch": "orange", "Avoid": "red"}
    df["tag_color"] = df["tag"].map(tag_color_map).fillna("gray")
    df["confidence"] = df["final_score"].clip(0, 100)
    df["confidence_band"] = pd.cut(
        df["confidence"],
        bins=[-np.inf, 60, 80, 100],
        labels=["Low", "Medium", "High"]
    ).astype(str)
    df["signal_strength"] = df["final_score"].apply(
        lambda x: "🔥 Explosive" if x >= 90 else
                  "⚡ Strong" if x >= 80 else
                  "↑ Solid" if x >= 70 else
                  "→ Moderate" if x >= 60 else
                  "↓ Weak"
    )
    df["tag_reason"] = ""
    df["tag_date"] = tag_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    # Sort by tag order & score for UI
    tag_order = {"Buy": 2, "Watch": 1, "Avoid": 0}
    df["_tag_order"] = df["tag"].map(tag_order).fillna(0)
    df = df.sort_values(["_tag_order", sort_by], ascending=[False, ascending]).drop(columns=["_tag_order"])
    return df.reset_index(drop=True)

# ============================================================================
# ANOMALY DETECTOR
# ============================================================================

def run_anomaly_detector(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects price, volume, EPS, and technical anomalies. Flags and labels each.
    Adds columns: 'anomaly' (bool), 'anomaly_type' (str), 'anomaly_reason' (str), 'spike_score' (int).
    """
    df = df.copy()
    df["anomaly"] = False
    df["anomaly_type"] = ""
    df["anomaly_reason"] = ""
    df["spike_score"] = 0

    # Price spike: >10% 1d return
    if "ret_1d" in df:
        spike = df["ret_1d"].abs() > 10
        df.loc[spike, "anomaly"] = True
        df.loc[spike, "anomaly_type"] += "Price Spike; "
        df.loc[spike, "anomaly_reason"] += "1D move > 10%; "
        df.loc[spike, "spike_score"] += 1

    # Volume spike: RVOL > 3
    if "rvol" in df:
        volspike = df["rvol"] > 3
        df.loc[volspike, "anomaly"] = True
        df.loc[volspike, "anomaly_type"] += "Volume Spike; "
        df.loc[volspike, "anomaly_reason"] += "RVOL > 3; "
        df.loc[volspike, "spike_score"] += 1

    # EPS jump: >40% qtr growth
    if "eps_change_pct" in df:
        epsjump = df["eps_change_pct"].abs() > 40
        df.loc[epsjump, "anomaly"] = True
        df.loc[epsjump, "anomaly_type"] += "EPS Jump; "
        df.loc[epsjump, "anomaly_reason"] += "EPS % Change > 40; "
        df.loc[epsjump, "spike_score"] += 1

    # 52w breakout
    if "price" in df and "high_52w" in df:
        new_high = df["price"] >= df["high_52w"]
        df.loc[new_high, "anomaly"] = True
        df.loc[new_high, "anomaly_type"] += "New High; "
        df.loc[new_high, "anomaly_reason"] += "New 52W High; "
        df.loc[new_high, "spike_score"] += 1
    if "price" in df and "low_52w" in df:
        new_low = df["price"] <= df["low_52w"]
        df.loc[new_low, "anomaly"] = True
        df.loc[new_low, "anomaly_type"] += "New Low; "
        df.loc[new_low, "anomaly_reason"] += "New 52W Low; "
        df.loc[new_low, "spike_score"] += 1

    # Multiple anomaly bonus
    df["anomaly"] = df["spike_score"] > 0
    df["multi_anomaly"] = df["spike_score"] > 1
    return df

# ============================================================================
# EDGE FINDER
# ============================================================================

def compute_edge_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds key 'edge' flags and summary edge labels to each row.
    Edge types: momentum breakout, value outlier, sector leader, vol squeeze, new high/low.
    """
    df = df.copy()
    # 1. Momentum breakout (recent, near high)
    df["edge_momentum_breakout"] = (
        (df.get("ret_3d", 0) > 2) &
        (df.get("ret_7d", 0) > 5) &
        (df.get("price", 0) >= 0.98 * df.get("high_52w", 0))
    )
    # 2. Value outlier (cheap PE, strong EPS)
    df["edge_value_outlier"] = (
        (df.get("pe", 99) < 18) &
        (df.get("eps_score", 0) > 80) &
        (df.get("final_score", 0) > 70)
    )
    # 3. Sector leader
    df["edge_sector_leader"] = (
        (df.get("sector_score", 0) > 85) &
        (df.get("final_score", 0) > 80)
    )
    # 4. Volatility squeeze (low recent moves, high volume)
    df["edge_volatility_squeeze"] = (
        (df.get("ret_30d", 0).abs() < 2.5) &
        (df.get("ret_7d", 0).abs() < 1) &
        (df.get("vol_ratio_1d_90d", 0) > 2)
    )
    # 5. Fresh new high/low
    df["edge_new_high"] = df.get("price", 0) >= df.get("high_52w", 0)
    df["edge_new_low"]  = df.get("price", 0) <= df.get("low_52w", 0)
    # Label edges for UI or export
    edge_cols = [
        "edge_momentum_breakout", "edge_value_outlier", "edge_sector_leader",
        "edge_volatility_squeeze", "edge_new_high", "edge_new_low"
    ]
    pretty_labels = {
        "edge_momentum_breakout": "Momentum Breakout",
        "edge_value_outlier": "Value Outlier",
        "edge_sector_leader": "Sector Leader",
        "edge_volatility_squeeze": "Vol Squeeze",
        "edge_new_high": "New 52W High",
        "edge_new_low":  "New 52W Low"
    }
    df["edge_types"] = df.apply(
        lambda r: ", ".join([pretty_labels[col] for col in edge_cols if r.get(col, False)]),
        axis=1
    )
    df["has_edge"] = df[edge_cols].any(axis=1)
    df["edge_count"] = df[edge_cols].sum(axis=1)
    return df

def find_edges(df: pd.DataFrame, min_edges: int = 1) -> pd.DataFrame:
    """
    Return all stocks with >= min_edges (at least one 'edge' flag).
    """
    df = compute_edge_signals(df)
    return df[df["edge_count"] >= min_edges].copy()

# ============================================================================
# SECTOR MAPPER & ROTATION
# ============================================================================

def run_sector_mapper(sector_df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps sector rotation and strength for dashboard use.
    Returns: DataFrame with ['sector', 'sector_score', 'sector_rank', 'rotation_status', 'sector_count']
    """
    df = sector_df.copy()
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r"[^\w\s]", "", regex=True)
                  .str.replace(r"\s+", "_", regex=True)
    )
    if "sector" in df:
        df["sector"] = df["sector"].astype(str).str.strip().str.title()
    if "sector_count" in df:
        df["sector_count"] = pd.to_numeric(df["sector_count"], errors="coerce").fillna(0).astype(int)
    if "sector_avg_3m" in df:
        df["sector_avg_3m"] = pd.to_numeric(df["sector_avg_3m"], errors="coerce").fillna(0)
    # Calculate percentile-based sector score (0-100)
    df["sector_score"] = df["sector_avg_3m"].rank(pct=True, na_option="bottom") * 100
    df["sector_rank"] = df["sector_score"].rank(ascending=False, method="min").astype(int)
    # Rotation tag: Hot = top 25%, Moderate = next 25%, else Weak
    total = len(df)
    quartile = max(1, int(round(total * 0.25)))
    half = max(1, int(round(total * 0.5)))
    df["rotation_status"] = "Weak"
    if total >= 3:
        df.loc[df["sector_rank"] <= quartile, "rotation_status"] = "Hot"
        df.loc[(df["sector_rank"] > quartile) & (df["sector_rank"] <= half), "rotation_status"] = "Moderate"
    cols = [c for c in [
        "sector", "sector_score", "sector_rank", "rotation_status", "sector_count"
    ] if c in df]
    out = df[cols].sort_values("sector_score", ascending=False).reset_index(drop=True)
    return out

def compute_sector_rotation(sector_df: pd.DataFrame, metric: str = "sector_avg_3m") -> pd.DataFrame:
    """
    Computes sector rotation tags based on the specified return metric.
    Returns DataFrame with sector_score, rank, rotation_status, sector_count.
    """
    df = sector_df.copy()
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r"[^\w\s]", "", regex=True)
                  .str.replace(r"\s+", "_", regex=True)
    )
    if "sector" in df:
        df["sector"] = df["sector"].astype(str).str.strip().str.title()
    if "sector_count" in df:
        df["sector_count"] = pd.to_numeric(df["sector_count"], errors="coerce").fillna(0).astype(int)
    if metric in df:
        df[metric] = pd.to_numeric(df[metric], errors="coerce").fillna(0)
    else:
        raise ValueError(f"Column '{metric}' not found in sector_df.")
    df["sector_score"] = df[metric].rank(pct=True, na_option="bottom") * 100
    df["sector_rank"] = df["sector_score"].rank(ascending=False, method="min").astype(int)
    total = len(df)
    quart = max(1, int(round(total * 0.25)))
    half = max(1, int(round(total * 0.5)))
    df["rotation_status"] = "Weak"
    if total >= 3:
        df.loc[df["sector_rank"] <= quart, "rotation_status"] = "Hot"
        df.loc[(df["sector_rank"] > quart) & (df["sector_rank"] <= half), "rotation_status"] = "Moderate"
    cols = [c for c in [
        "sector", "sector_score", "sector_rank", "rotation_status", "sector_count"
    ] if c in df]
    out = df[cols].sort_values("sector_score", ascending=False).reset_index(drop=True)
    return out

def sector_rotation_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns a simple summary: counts of each rotation tag and top sectors.
    """
    counts = df["rotation_status"].value_counts().to_dict()
    top_sectors = list(df[df["rotation_status"] == "Hot"]["sector"].head(3))
    return {
        "total_sectors": len(df),
        "hot_count": counts.get("Hot", 0),
        "moderate_count": counts.get("Moderate", 0),
        "weak_count": counts.get("Weak", 0),
        "hot_sectors": top_sectors,
    }

# ============================================================================
# (OPTIONAL) WATCHLIST / SCENARIO HELPERS
# ============================================================================

def top_n(df: pd.DataFrame, n=20, by="final_score", tag=None):
    """Return top-N stocks by score (optionally filtered by tag)."""
    filt = df.copy()
    if tag: filt = filt[filt["tag"] == tag]
    return filt.sort_values(by=by, ascending=False).head(n).reset_index(drop=True)

def sector_leaders(df: pd.DataFrame, n_per_sector=2, by="final_score", tag=None):
    """Return top-N-per-sector stocks."""
    filt = df.copy()
    if tag: filt = filt[filt["tag"] == tag]
    return (filt.sort_values([ "sector", by], ascending=[True, False])
                .groupby("sector")
                .head(n_per_sector)
                .reset_index(drop=True))

def multi_spike_anomalies(df: pd.DataFrame, min_spike=3):
    """Return all stocks with spike_score >= min_spike."""
    return df[df.get("spike_score", 0) >= min_spike].sort_values("spike_score", ascending=False).reset_index(drop=True)

def laggard_reversal(df: pd.DataFrame):
    """Return laggard reversals: 1Y negative, 30D strong, Buy tag."""
    return df[(df.get("ret_1y", 0) < 0) & (df.get("ret_30d", 0) > 5) & (df.get("tag", "") == "Buy")].reset_index(drop=True)

def long_term_winners(df: pd.DataFrame, min_yrs=5, min_ret=15):
    """Return stocks with strong 5Y/3Y returns, not at high."""
    cols = ["ret_3y", "ret_5y", "from_high_pct"]
    for c in cols: df[c] = pd.to_numeric(df.get(c, 0), errors="coerce")
    f = (df.get("ret_5y", 0) > min_ret) & (df.get("ret_3y", 0) > min_ret/2) & (df.get("from_high_pct", 0) > 10)
    return df[f].sort_values("ret_5y", ascending=False).reset_index(drop=True)

def fresh_52w_high(df: pd.DataFrame):
    """Return stocks at new 52W high."""
    return df[df.get("price", 0) >= df.get("high_52w", 0)].sort_values("final_score", ascending=False).reset_index(drop=True)

def value_outliers(df: pd.DataFrame, pe_max=15, eps_min=75):
    """Return stocks with low PE, high EPS score."""
    f = (df.get("pe", 99) < pe_max) & (df.get("eps_score", 0) > eps_min)
    return df[f].sort_values("final_score", ascending=False).reset_index(drop=True)

# ============================================================================
# END OF FILE
# ============================================================================
