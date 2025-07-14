# config.py - FINAL
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple
import warnings

@dataclass
class Config:
    BASE_URL: str = "https://docs.google.com/spreadsheets/d/1Wa4-4K7hyTTCrqJ0pUzS-NaLFiRQpBgI8KBdHx9obKk"
    SHEET_GIDS: Dict[str, str] = field(default_factory=lambda: {
        "watchlist": "2026492216",
        "returns": "100734077",
        "sector": "140104095"
    })
    REQUIRED_WATCHLIST: Set[str] = field(default_factory=lambda: {
        "ticker", "exchange", "company_name", "year", "market_cap", "category", "sector",
        "eps_tier", "price", "prev_close", "ret_1d", "low_52w", "high_52w",
        "from_low_pct", "from_high_pct", "sma_20d", "sma_50d", "sma_200d",
        "trading_under", "ret_3d", "ret_7d", "ret_30d", "ret_3m", "ret_6m",
        "ret_1y", "ret_3y", "ret_5y", "volume_1d", "volume_7d", "volume_30d",
        "volume_3m", "vol_ratio_1d_90d", "vol_ratio_7d_90d", "vol_ratio_30d_90d",
        "rvol", "price_tier", "pe", "eps_current", "eps_last_qtr", "eps_duplicate", 
        "eps_change_pct"
    })
    REQUIRED_RETURNS: Set[str] = field(default_factory=lambda: {
        "ticker", "company_name",
        "avg_ret_30d", "avg_ret_3m", "avg_ret_6m", "avg_ret_1y", "avg_ret_3y", "avg_ret_5y"
    })
    REQUIRED_SECTOR: Set[str] = field(default_factory=lambda: {
        "sector", "sector_ret_1d", "sector_ret_3d", "sector_ret_7d", "sector_ret_30d",
        "sector_ret_3m", "sector_ret_6m", "sector_ret_1y", "sector_ret_3y", "sector_ret_5y",
        "sector_avg_30d", "sector_avg_3m", "sector_avg_6m", "sector_avg_1y",
        "sector_avg_3y", "sector_avg_5y", "sector_count"
    })
    CACHE_TTL: int = 300
    REQUEST_TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    SCHEMA_VERSION: str = "2025.07.14"
    def get_sheet_url(self, name: str) -> str:
        if name not in self.SHEET_GIDS:
            raise ValueError(f"Unknown sheet: {name}")
        return f"{self.BASE_URL}/export?format=csv&gid={self.SHEET_GIDS[name]}"

CONFIG = Config()
