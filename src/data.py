"""Loading the saved FRED snapshot."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# FRED series id -> maturity in years (Treasury constant-maturity par yields, in %)
CURVE_SERIES = {
    "DGS1MO": 1 / 12, "DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1.0, "DGS2": 2.0, "DGS3": 3.0,
    "DGS5": 5.0, "DGS7": 7.0, "DGS10": 10.0, "DGS20": 20.0, "DGS30": 30.0,
}
# ICE BofA US corporate option-adjusted spreads (in %)
CREDIT_SERIES = {"counterparty_BBB": "BAMLC0A4CBBB", "bank_A": "BAMLC0A3CA"}


def load_table() -> pd.DataFrame:
    return pd.read_csv(RAW / "fred_rates_spreads.csv", index_col="date", parse_dates=True)


def load_meta() -> dict:
    return json.loads((RAW / "snapshot_meta.json").read_text())


def curve_history() -> pd.DataFrame:
    """Par yields in decimals, one row per day where every maturity is quoted."""
    return load_table()[list(CURVE_SERIES)].dropna() / 100.0


def valuation_inputs(valuation_date: str | None = None) -> dict:
    """Par yields, SOFR and credit spreads on the valuation date (default: last full curve)."""
    table = load_table()
    curves = table[list(CURVE_SERIES)].dropna()
    day = pd.Timestamp(valuation_date) if valuation_date else curves.index.max()
    row = table.loc[day]
    return {
        "date": str(day.date()),
        "maturities": list(CURVE_SERIES.values()),
        "par_yields": [row[s] / 100.0 for s in CURVE_SERIES],
        "sofr": float(row["SOFR"]) / 100.0 if pd.notna(row["SOFR"]) else None,
        "spreads": {k: float(row[s]) / 100.0 for k, s in CREDIT_SERIES.items()},
    }
