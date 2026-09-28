"""Download the hourly es.wikipedia user pageviews used as the public-domain series.

Source: Wikimedia Analytics REST API, ``pageviews/aggregate`` endpoint
(project ``es.wikipedia``, access ``all-access``, agent ``user``, hourly).
Wikimedia releases its Analytics datasets under CC0.

The file is written with the same columns as the AWS exports so that
``tsa_final.data.load_raw`` reads it through ``SERIES_REGISTRY`` unchanged
(``load_balancer`` holds the project, ``target_group`` is empty).

Run from the repo root: uv run python tp-final/experiments/00_download_wikipedia_es.py
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT = "es.wikipedia"
START = "2026050100"
END = "2026092622"
METRIC = "Pageviews"
USER_AGENT = "tsa-tp-final/1.0 (carlosaularit@gmail.com)"

URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/aggregate/"
    f"{PROJECT}/all-access/user/hourly/{START}/{END}"
)
OUT = Path(__file__).resolve().parent.parent / "data" / "wikipedia-es-pageviews-hourly-since-2026-05-01.csv"


def main() -> None:
    request = urllib.request.Request(URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        items = json.load(response)["items"]

    start = pd.to_datetime([item["timestamp"] for item in items], format="%Y%m%d%H", utc=True)
    df = pd.DataFrame(
        {
            "load_balancer": PROJECT,
            "target_group": "",
            "metric_name": METRIC,
            "period_start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "period_end_utc": (start + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "value": [item["views"] for item in items],
        }
    )
    df.to_csv(OUT, index=False)
    print(f"{len(df)} rows, {df['period_start_utc'].iloc[0]} -> {df['period_start_utc'].iloc[-1]} -> {OUT}")


if __name__ == "__main__":
    main()
