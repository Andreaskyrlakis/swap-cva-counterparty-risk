"""Download the free market data used by the project (FRED, no account needed).

- US Treasury constant-maturity par yields, 1 month to 30 years (H.15, daily)
- SOFR overnight rate (for reference only)
- ICE BofA US corporate option-adjusted spreads: BBB (counterparty) and A (our bank)

Run:  python scripts/01_download_data.py
"""

import io
import json
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import CURVE_SERIES, CREDIT_SERIES, RAW  # noqa: E402

START = "2016-01-01"
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}"


def download_series(series: str) -> pd.Series:
    request = urllib.request.Request(URL.format(series=series, start=START),
                                     headers={"User-Agent": "curl/8.0"})
    for attempt in range(4):  # FRED is sometimes slow: retry a few times
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                text = response.read().decode()
            break
        except (TimeoutError, OSError) as error:
            print(f"  attempt {attempt + 1} failed ({error}), retrying", flush=True)
            time.sleep(5)
    else:
        raise RuntimeError(f"could not download {series} from FRED")
    table = pd.read_csv(io.StringIO(text))
    table.columns = ["date", series]
    table["date"] = pd.to_datetime(table["date"])
    values = pd.to_numeric(table[series], errors="coerce")  # FRED writes "." for missing days
    return pd.Series(values.values, index=table["date"], name=series)


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    all_series = list(CURVE_SERIES) + ["SOFR"] + list(CREDIT_SERIES.values())
    columns = []
    for name in all_series:
        print(f"downloading {name} ...", flush=True)
        columns.append(download_series(name))
    table = pd.concat(columns, axis=1).sort_index()
    table.index.name = "date"
    table.to_csv(RAW / "fred_rates_spreads.csv")

    complete_curve = table[list(CURVE_SERIES)].dropna()
    meta = {
        "downloaded_on": date.today().isoformat(),
        "source": "FRED, Federal Reserve Bank of St. Louis (fredgraph.csv endpoint)",
        "series": all_series,
        "first_date": str(table.index.min().date()),
        "last_complete_curve_date": str(complete_curve.index.max().date()),
    }
    (RAW / "snapshot_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
