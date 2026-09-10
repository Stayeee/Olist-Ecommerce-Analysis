from __future__ import annotations

import json
from pathlib import Path

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


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    df = load_analysis_data(root / "analysis_table.csv")

    quality = inspect_data_quality(df)
    print("Dataset grain check:")
    print(json.dumps(quality, indent=2, ensure_ascii=False))
    print()

    if quality.get("payment_value_requires_grain_review"):
        print(
            "[WARN] The prepared table is not clearly one row per order. "
            "Review how payment_value was joined before treating row-level sums as final GMV."
        )
        print()

    checks = {
        "sales_overview": lambda: get_sales_overview(df),
        "sales_trend": lambda: analyze_sales_trend(df, months=6),
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
            if name == "product" and result.get("available") is False:
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
