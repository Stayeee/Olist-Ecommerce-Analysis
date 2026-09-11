from __future__ import annotations

from typing import Any

import pandas as pd

from data.metrics import complete_month_starts, order_facts


def _category_column(df: pd.DataFrame) -> str | None:
    for candidate in (
        "product_category_name_english",
        "product_category_name",
        "product_category",
    ):
        if candidate in df.columns:
            return candidate
    return None


def analyze_recent_change_drivers(
    df: pd.DataFrame,
    dimension: str = "state",
    limit: int = 8,
) -> dict[str, Any]:
    """Decompose GMV change across the two latest complete months.

    The newest calendar month in Olist is excluded by default because the public
    dataset can end mid-month. The tool compares the two months immediately
    before it and ranks segments by absolute GMV contribution to the change.
    """
    if "order_month" not in df.columns:
        return {"available": False, "reason": "order_month is unavailable."}

    facts = order_facts(df)
    observed_months = sorted(facts["order_month"].dropna().unique())
    months = complete_month_starts(facts)
    if len(months) < 2:
        return {
            "available": False,
            "reason": "At least two complete calendar months are required.",
        }

    current_month = pd.Timestamp(months[-1])
    previous_month = pd.Timestamp(months[-2])
    excluded_months = [
        pd.Timestamp(month).strftime("%Y-%m")
        for month in observed_months
        if pd.Timestamp(month) not in months
    ]

    if dimension == "state":
        dimension_col = "customer_state"
        output_name = "state"
    elif dimension == "category":
        dimension_col = _category_column(df)
        output_name = "category"
        if dimension_col is None:
            return {
                "available": False,
                "dimension": "category",
                "reason": "No product category column is available in analysis_table.csv.",
            }
    else:
        return {
            "available": False,
            "reason": "dimension must be either 'state' or 'category'.",
        }

    subset = facts[facts["order_month"].isin([previous_month, current_month])].copy()
    subset = subset.dropna(subset=[dimension_col])

    grouped = (
        subset.groupby(["order_month", dimension_col], as_index=False)
        .agg(
            gmv=("payment_value", "sum"),
            orders=("order_id", "nunique"),
        )
    )

    current = grouped[grouped["order_month"] == current_month].set_index(dimension_col)
    previous = grouped[grouped["order_month"] == previous_month].set_index(dimension_col)
    segments = current.index.union(previous.index)

    rows: list[dict[str, Any]] = []
    for segment in segments:
        current_gmv = float(current.loc[segment, "gmv"]) if segment in current.index else 0.0
        previous_gmv = float(previous.loc[segment, "gmv"]) if segment in previous.index else 0.0
        current_orders = int(current.loc[segment, "orders"]) if segment in current.index else 0
        previous_orders = int(previous.loc[segment, "orders"]) if segment in previous.index else 0
        gmv_change = current_gmv - previous_gmv
        orders_change = current_orders - previous_orders

        rows.append(
            {
                output_name: str(segment),
                "previous_gmv": previous_gmv,
                "current_gmv": current_gmv,
                "gmv_change": gmv_change,
                "gmv_change_pct": (
                    (gmv_change / previous_gmv) * 100 if previous_gmv else None
                ),
                "previous_orders": previous_orders,
                "current_orders": current_orders,
                "orders_change": orders_change,
            }
        )

    previous_total = float(previous["gmv"].sum())
    current_total = float(current["gmv"].sum())
    total_change = current_total - previous_total

    for row in rows:
        row["contribution_to_total_change_pct"] = (
            row["gmv_change"] / total_change * 100 if total_change else None
        )

    ranked = sorted(rows, key=lambda row: abs(row["gmv_change"]), reverse=True)[:limit]

    return {
        "available": True,
        "dimension": dimension,
        "excluded_partial_months": excluded_months,
        "previous_complete_month": str(previous_month.date()),
        "current_complete_month": str(current_month.date()),
        "previous_total_gmv": previous_total,
        "current_total_gmv": current_total,
        "total_gmv_change": total_change,
        "total_gmv_change_pct": (
            total_change / previous_total * 100 if previous_total else None
        ),
        "drivers": ranked,
        "gmv_definition": "Gross payment value at one row per order across all order statuses.",
    }

