from __future__ import annotations

from typing import Any

import pandas as pd


def order_facts(df: pd.DataFrame) -> pd.DataFrame:
    """Return one trustworthy row per order for order-level KPI calculation.

    The prepared Olist table stores an order's total payment in ``payment_value``.
    Identical duplicate rows can arise if the table is later joined to items. They
    are collapsed here so payment totals, orders and delivery rates are not
    multiplied. Conflicting values fail loudly because summing or choosing one
    would silently change the metric definition.
    """
    if not df["order_id"].duplicated().any():
        return df.copy()

    protected = [
        column
        for column in ("payment_value", "customer_state", "order_status", "is_late")
        if column in df.columns
    ]
    conflicts = {
        column: int((df.groupby("order_id")[column].nunique(dropna=False) > 1).sum())
        for column in protected
    }
    conflicts = {column: count for column, count in conflicts.items() if count}
    if conflicts:
        raise ValueError(
            "The prepared table has conflicting order-level values: "
            + ", ".join(f"{column} ({count} orders)" for column, count in conflicts.items())
        )

    return df.drop_duplicates(subset=["order_id"], keep="first").copy()


def month_coverage(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Describe observed calendar coverage and classify complete purchase months."""
    facts = order_facts(df).dropna(subset=["order_purchase_timestamp"]).copy()
    timestamps = pd.to_datetime(facts["order_purchase_timestamp"], errors="coerce")
    facts = facts.assign(_purchase_date=timestamps.dt.normalize())
    facts = facts.dropna(subset=["_purchase_date"])
    facts["_month"] = facts["_purchase_date"].dt.to_period("M").dt.to_timestamp()

    coverage = (
        facts.groupby("_month", as_index=False)
        .agg(
            first_purchase=("_purchase_date", "min"),
            last_purchase=("_purchase_date", "max"),
            active_days=("_purchase_date", "nunique"),
            orders=("order_id", "nunique"),
        )
        .sort_values("_month")
    )

    rows: list[dict[str, Any]] = []
    for _, row in coverage.iterrows():
        is_complete = (
            row["first_purchase"].day <= 7
            and row["last_purchase"].day >= 25
            and row["active_days"] >= 20
        )
        rows.append(
            {
                "month": pd.Timestamp(row["_month"]).strftime("%Y-%m"),
                "first_purchase_date": pd.Timestamp(row["first_purchase"]).date().isoformat(),
                "last_purchase_date": pd.Timestamp(row["last_purchase"]).date().isoformat(),
                "active_days": int(row["active_days"]),
                "orders": int(row["orders"]),
                "is_complete": bool(is_complete),
            }
        )
    return rows


def complete_month_starts(df: pd.DataFrame) -> list[pd.Timestamp]:
    return [
        pd.Timestamp(row["month"] + "-01")
        for row in month_coverage(df)
        if row["is_complete"]
    ]


def delivered_order_facts(df: pd.DataFrame) -> pd.DataFrame:
    """Return orders eligible for delivery KPIs and their true denominator."""
    facts = order_facts(df)
    eligible = facts["delivery_days"].notna()
    if "order_status" in facts.columns:
        eligible &= facts["order_status"].eq("delivered")
    return facts.loc[eligible].copy()

