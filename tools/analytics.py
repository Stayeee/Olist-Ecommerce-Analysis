from __future__ import annotations

from typing import Any

import pandas as pd


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def get_sales_overview(df: pd.DataFrame) -> dict[str, Any]:
    total_orders = int(df["order_id"].nunique())
    gmv = float(df["payment_value"].sum())
    unique_customers = int(df["customer_unique_id"].nunique())

    return {
        "gmv": gmv,
        "orders": total_orders,
        "aov": _safe_divide(gmv, total_orders),
        "unique_customers": unique_customers,
        "orders_per_customer": _safe_divide(total_orders, unique_customers),
        "late_rate": float(df["is_late"].mean() * 100),
        "avg_delivery_days": float(df["delivery_days"].mean()),
    }


def analyze_sales_trend(df: pd.DataFrame, months: int = 6) -> dict[str, Any]:
    monthly = (
        df.dropna(subset=["order_month"])
        .groupby("order_month", as_index=False)
        .agg(
            gmv=("payment_value", "sum"),
            orders=("order_id", "nunique"),
            customers=("customer_unique_id", "nunique"),
        )
        .sort_values("order_month")
    )
    monthly["aov"] = monthly["gmv"] / monthly["orders"].replace(0, pd.NA)
    monthly = monthly.tail(max(2, months)).copy()

    latest = monthly.iloc[-1]
    previous = monthly.iloc[-2]

    def pct_change(current: float, prior: float) -> float:
        return _safe_divide(current - prior, prior) * 100

    return {
        "latest_month": str(latest["order_month"].date()),
        "latest_gmv": float(latest["gmv"]),
        "latest_orders": int(latest["orders"]),
        "latest_aov": float(latest["aov"]),
        "gmv_mom_pct": pct_change(float(latest["gmv"]), float(previous["gmv"])),
        "orders_mom_pct": pct_change(float(latest["orders"]), float(previous["orders"])),
        "aov_mom_pct": pct_change(float(latest["aov"]), float(previous["aov"])),
        "series": [
            {
                "month": str(row.order_month.date()),
                "gmv": float(row.gmv),
                "orders": int(row.orders),
                "aov": float(row.aov),
            }
            for row in monthly.itertuples()
        ],
    }


def analyze_region_performance(
    df: pd.DataFrame,
    metric: str = "gmv",
    limit: int = 10,
    min_orders: int = 100,
) -> dict[str, Any]:
    region = (
        df.groupby("customer_state", as_index=False)
        .agg(
            gmv=("payment_value", "sum"),
            orders=("order_id", "nunique"),
            customers=("customer_unique_id", "nunique"),
            late_rate=("is_late", "mean"),
            avg_delivery_days=("delivery_days", "mean"),
        )
    )
    region["late_rate"] *= 100
    region["aov"] = region["gmv"] / region["orders"].replace(0, pd.NA)
    region = region[region["orders"] >= min_orders].copy()

    allowed = {"gmv", "orders", "aov", "late_rate", "avg_delivery_days"}
    if metric not in allowed:
        metric = "gmv"

    ranked = region.nlargest(limit, metric)
    return {
        "metric": metric,
        "min_orders": min_orders,
        "rows": [
            {
                "state": row.customer_state,
                "gmv": float(row.gmv),
                "orders": int(row.orders),
                "customers": int(row.customers),
                "aov": float(row.aov),
                "late_rate": float(row.late_rate),
                "avg_delivery_days": float(row.avg_delivery_days),
            }
            for row in ranked.itertuples()
        ],
    }


def analyze_delivery_performance(
    df: pd.DataFrame, limit: int = 5, min_orders: int = 100
) -> dict[str, Any]:
    regional = analyze_region_performance(
        df, metric="late_rate", limit=limit, min_orders=min_orders
    )
    return {
        "overall_late_rate": float(df["is_late"].mean() * 100),
        "avg_delivery_days": float(df["delivery_days"].mean()),
        "highest_risk_states": regional["rows"],
    }


def analyze_customer_behavior(df: pd.DataFrame) -> dict[str, Any]:
    customer_orders = (
        df.groupby("customer_unique_id")["order_id"].nunique().rename("orders")
    )
    total_customers = int(customer_orders.size)
    repeat_customers = int((customer_orders > 1).sum())

    return {
        "unique_customers": total_customers,
        "repeat_customers": repeat_customers,
        "repeat_customer_rate": _safe_divide(repeat_customers, total_customers) * 100,
        "avg_orders_per_customer": float(customer_orders.mean()),
        "max_orders_per_customer": int(customer_orders.max()),
    }


def analyze_payment_behavior(df: pd.DataFrame) -> dict[str, Any]:
    installments = df.get("payment_installments", pd.Series(index=df.index, dtype=float)).fillna(0)
    result: dict[str, Any] = {
        "avg_installments": float(installments.mean()),
        "multi_installment_share": float((installments > 1).mean() * 100),
    }

    if "payment_type" in df.columns:
        mix = (
            df.groupby("payment_type", as_index=False)["payment_value"]
            .sum()
            .sort_values("payment_value", ascending=False)
        )
        result["payment_mix"] = [
            {"payment_type": row.payment_type, "payment_value": float(row.payment_value)}
            for row in mix.itertuples()
        ]
    return result


def analyze_product_performance(
    df: pd.DataFrame, metric: str = "gmv", limit: int = 10
) -> dict[str, Any]:
    category_col = None
    for candidate in (
        "product_category_name_english",
        "product_category_name",
        "product_category",
    ):
        if candidate in df.columns:
            category_col = candidate
            break

    if category_col is None:
        return {
            "available": False,
            "reason": "No product category column is available in analysis_table.csv.",
        }

    grouped = (
        df.dropna(subset=[category_col])
        .groupby(category_col, as_index=False)
        .agg(
            gmv=("payment_value", "sum"),
            orders=("order_id", "nunique"),
            late_rate=("is_late", "mean"),
        )
    )
    grouped["late_rate"] *= 100
    grouped["aov"] = grouped["gmv"] / grouped["orders"].replace(0, pd.NA)

    if metric not in {"gmv", "orders", "aov", "late_rate"}:
        metric = "gmv"
    ranked = grouped.nlargest(limit, metric)

    return {
        "available": True,
        "metric": metric,
        "category_column": category_col,
        "rows": [
            {
                "category": getattr(row, category_col),
                "gmv": float(row.gmv),
                "orders": int(row.orders),
                "aov": float(row.aov),
                "late_rate": float(row.late_rate),
            }
            for row in ranked.itertuples(index=False)
        ],
    }


def compare_states(df: pd.DataFrame, state_a: str, state_b: str) -> dict[str, Any]:
    targets = [state_a.upper(), state_b.upper()]
    subset = df[df["customer_state"].isin(targets)]
    rows = analyze_region_performance(subset, metric="gmv", limit=2, min_orders=1)["rows"]
    by_state = {row["state"]: row for row in rows}

    return {
        "state_a": by_state.get(targets[0]),
        "state_b": by_state.get(targets[1]),
    }
