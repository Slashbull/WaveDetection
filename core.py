# core.py — FINAL (locked)  ❖  M.A.N.T.R.A. 2025-07-14
"""
Orchestration layer

Responsibilities
────────────────
1. Ingest Watchlist data from either:
   • Google Sheet (default — IDs in CONFIG)
   • Local CSV  (via --csv or run(csv=…))
2. Pass the cleaned DataFrame to adaptive_tag_model.tag_watchlist( )
3. Return or print the tagged DataFrame
4. Supply a tiny CLI for one-line testing:
      python core.py                    → pulls sheet, shows top 25
      python core.py --csv file.csv     → uses local CSV
"""

from __future__ import annotations
import pandas as pd, sys, argparse
from typing import Optional
from config import CONFIG
import data_loader as dl
from adaptive_tag_model import tag_watchlist


# ─────────────────────────────────────────────────────────────────────────────
# Public utility
# ─────────────────────────────────────────────────────────────────────────────
def run(
    sheet_id:  str | None = None,
    gid:       str | None = None,
    csv:       str | None = None,
    top:       int        = 25
) -> pd.DataFrame:
    """
    Fetch data ➜ tag ➜ return top N rows

    Parameters
    ----------
    sheet_id : override Google-Sheet ID (None = default)
    gid      : override worksheet gid   (None = default)
    csv      : path to local CSV (skips Google-Sheet download)
    top      : row count to return (after sorting by edge_prob desc)

    Returns
    -------
    pandas.DataFrame  with columns:
        ticker · category · sector · price_tier · tag · edge_prob
    """
    if csv:
        df_raw = dl.load_csv(csv)
    else:
        df_raw = dl.fetch_watchlist(sheet_id, gid)

    tagged = tag_watchlist(df_raw)
    return tagged.head(top)


# ─────────────────────────────────────────────────────────────────────────────
# CLI helper
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M.A.N.T.R.A. core orchestrator")
    p.add_argument("--csv",       help="Local CSV path (skip Google Sheet)")
    p.add_argument("--sheet-id",  help="Override Google-Sheet ID")
    p.add_argument("--gid",       help="Override worksheet gid")
    p.add_argument("--top", type=int, default=25, help="Rows to print")
    return p.parse_args()


def _cli() -> None:
    args = _parse_args()
    try:
        out = run(
            sheet_id=args.sheet_id,
            gid=args.gid,
            csv=args.csv,
            top=args.top
        )
        pd.set_option("display.float_format", lambda x: f"{x:0.3f}")
        print(out.to_string(index=False))
        print(f"\n✅  Rows shown: {len(out)}")
    except Exception as e:
        print(f"❌  Error: {e}", file=sys.stderr)
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _cli()
