# config.py — FINAL (locked)  ❖  M.A.N.T.R.A. 2025-07-14
"""
Single source of truth for:
• Google-Sheet IDs / gids
• Required column lists (watchlist-only engine)
• Scoring weights, thresholds, colours
Nothing else in the codebase should redefine these values.
"""

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, List
import warnings


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DATACLASS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True, frozen=True)
class Config:
    # ─── DATA SOURCE ────────────────────────────────────────────────────────
    SHEET_ID: str = "1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"
    GIDS: Dict[str, str] = field(
        default_factory=lambda: {
            "watchlist": "2026492216",   # single-tab use-case
        }
    )

    def sheet_url(self, name: str) -> str:
        if name not in self.GIDS:
            raise ValueError(f"Unknown sheet: {name}")
        return f"https://docs.google.com/spreadsheets/d/{self.SHEET_ID}/export?format=csv&gid={self.GIDS[name]}"

    # ─── REQUIRED COLUMNS (exact headers after cleaning) ────────────────────
    REQUIRED_WATCHLIST: Set[str] = field(
        default_factory=lambda: {
            # core identity
            "ticker", "exchange", "company_name", "year",
            # fundamentals
            "market_cap", "category", "sector", "eps_tier", "pe",
            "eps_current", "eps_last_qtr", "eps_duplicate", "eps_change_pct",
            # price & range
            "price", "prev_close", "low_52w", "high_52w",
            "from_low_pct", "from_high_pct",
            # moving averages
            "sma_20d", "sma_50d", "sma_200d", "trading_under",
            # returns
            "ret_1d", "ret_3d", "ret_7d", "ret_30d", "ret_3m",
            "ret_6m", "ret_1y", "ret_3y", "ret_5y",
            # volume
            "volume_1d", "volume_7d", "volume_30d", "volume_3m",
            "vol_ratio_1d_90d", "vol_ratio_7d_90d", "vol_ratio_30d_90d",
            "rvol",
            # tiers
            "price_tier"
        }
    )

    # ─── CLEANING HINTS ─────────────────────────────────────────────────────
    PERCENT_COLS: List[str] = field(
        default_factory=lambda: [
            "from_low_pct", "from_high_pct", "eps_change_pct",
            "ret_1d", "ret_3d", "ret_7d", "ret_30d",
            "ret_3m", "ret_6m", "ret_1y", "ret_3y", "ret_5y"
        ]
    )
    TEXT_COLS: List[str] = field(
        default_factory=lambda: [
            "ticker", "exchange", "company_name", "sector",
            "category", "eps_tier", "price_tier", "trading_under"
        ]
    )

    # ─── SCORING WEIGHTS & FILTERS (adaptive_tag_model) ────────────────────
    FACTOR_WEIGHTS: Dict[str, float] = field(
        default_factory=lambda: {
            "mom_fast": 0.15,
            "mom_mid":  0.15,
            "mom_long": 0.10,
            "trend":    0.10,
            "value":    0.15,
            "eps":      0.15,
            "volume":   0.10,
            "risk":     0.05,
            "liquidity":0.05,
        }
    )
    TAG_THRESHOLDS: Dict[str, float] = field(
        default_factory=lambda: {"buy": 0.80, "watch": 0.60}
    )
    HARD_FILTERS: Dict[str, float] = field(
        default_factory=lambda: {
            "pe_max": 40,
            "eps_change_min": 15.0,
            "vol_ratio_30d_min": 1.2,
        }
    )

    # ─── UI SETTINGS ───────────────────────────────────────────────────────
    SIGNAL_COLORS: Dict[str, str] = field(
        default_factory=lambda: {
            "BUY":   "#28a745",
            "WATCH": "#ffd43b",
            "AVOID": "#fa5252",
        }
    )
    APP_NAME: str = "M.A.N.T.R.A."
    APP_ICON: str = "📈"
    APP_VERSION: str = "v1.0 (Locked)"

    # ─── PERFORMANCE ────────────────────────────────────────────────────────
    CACHE_TTL: int = 300          # seconds
    REQUEST_TIMEOUT: int = 30     # seconds
    MAX_RETRIES: int = 3

    # ─── POST-INIT VALIDATION ──────────────────────────────────────────────
    def __post_init__(self):
        # ensure factor weights sum to 1
        tot = sum(self.FACTOR_WEIGHTS.values())
        if abs(tot - 1.0) > 1e-3:
            raise ValueError(f"FACTOR_WEIGHTS must sum to 1.0 (got {tot})")
        if self.CACHE_TTL < 60:
            warnings.warn("CACHE_TTL <60 s — consider increasing for stability.")


# singleton export
CONFIG = Config()

# shorthand re-exports
__all__ = ["CONFIG", "Config"]
