from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data.loader import load_analysis_data
from data.quality import inspect_data_quality
from tools.analytics import (
    analyze_customer_behavior,
    analyze_delivery_performance,
    analyze_payment_behavior,
    analyze_product_performance,
    analyze_region_performance,
    analyze_sales_trend,
    compare_states,
    get_sales_overview,
)
from tools.drivers import analyze_recent_change_drivers


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    df = load_analysis_data(root / "analysis_table.csv")

    quality = inspect_data_quality(df)
    print("Dataset grain check:")
    print(json.dumps(quality, indent=2, ensure_ascii=False))
    print()

    assert quality["is_one_row_per_order"] is True
    assert quality["payment_value_requires_grain_review"] is False
    assert quality["complete_months"][-1] == "2018-08"
    assert {"2018-09", "2018-10"}.issubset(quality["partial_months"])

    overview = get_sales_overview(df)
    trend = analyze_sales_trend(df, months=6)
    drivers = analyze_recent_change_drivers(df, dimension="state", limit=5)
    assert abs(overview["gmv"] - float(df["payment_value"].sum())) < 0.01
    assert overview["delivery_kpi_orders"] == quality["delivery_kpi_orders"]
    assert trend["latest_month"] == "2018-08-01"
    assert trend["previous_month"] == "2018-07-01"
    assert abs(
        trend["gmv_change"]
        - trend["order_volume_contribution"]
        - trend["aov_contribution"]
    ) < 0.01
    assert drivers["current_complete_month"] == "2018-08-01"
    assert drivers["previous_complete_month"] == "2018-07-01"
    assert {"2018-09", "2018-10"}.issubset(drivers["excluded_partial_months"])

    duplicated = df.iloc[[0]].copy()
    duplicate_safe = get_sales_overview(pd.concat([df, duplicated], ignore_index=True))
    assert abs(duplicate_safe["gmv"] - overview["gmv"]) < 0.01

    checks = {
        "sales_overview": lambda: overview,
        "sales_trend": lambda: trend,
        "recent_change_state_drivers": lambda: drivers,
        "recent_change_category_drivers": lambda: analyze_recent_change_drivers(df, dimension="category", limit=5),
        "region": lambda: analyze_region_performance(df, metric="gmv", limit=5),
        "delivery": lambda: analyze_delivery_performance(df, limit=5),
        "customer": lambda: analyze_customer_behavior(df),
        "payment": lambda: analyze_payment_behavior(df),
        "product": lambda: analyze_product_performance(df, metric="gmv", limit=5),
        "state_comparison": lambda: compare_states(df, "SP", "RJ"),
    }

    failures: list[str] = []
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns.\n")

    for name, check in checks.items():
        try:
            result = check()
            if not isinstance(result, dict):
                raise TypeError(f"Expected dict, got {type(result).__name__}")
            print(f"[PASS] {name}")
            if name in {"product", "recent_change_category_drivers"} and result.get("available") is False:
                print("       Product category is not present; tool correctly reports it as unavailable.")
        except Exception as exc:
            failures.append(name)
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")

    print()
    if failures:
        raise SystemExit("Smoke test failed for: " + ", ".join(failures))

    print("All deterministic analytics tools passed the offline smoke test.")


if __name__ == "__main__":
    main()

