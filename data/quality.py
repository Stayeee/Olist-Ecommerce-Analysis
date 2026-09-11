from __future__ import annotations

from typing import Any

import pandas as pd

from data.metrics import delivered_order_facts, month_coverage


def inspect_data_quality(df: pd.DataFrame) -> dict[str, Any]:
    """Return lightweight checks that help detect unsafe metric aggregation.

    Olist tables can become one-to-many after joining orders, items and payments.
    This check does not guess how a prepared table was built; it makes the row
    grain visible so GMV and rate definitions can be reviewed before release.
    """
    rows = int(len(df))
    orders = int(df["order_id"].nunique())
    duplicated_order_rows = int(df.duplicated(subset=["order_id"], keep=False).sum())
    max_rows_per_order = int(df.groupby("order_id").size().max()) if orders else 0

    result: dict[str, Any] = {
        "rows": rows,
        "unique_orders": orders,
        "rows_per_order": float(rows / orders) if orders else 0.0,
        "duplicated_order_rows": duplicated_order_rows,
        "max_rows_per_order": max_rows_per_order,
        "is_one_row_per_order": rows == orders,
    }

    if "payment_value" in df.columns and orders:
        payment_nunique = df.groupby("order_id")["payment_value"].nunique(dropna=False)
        result["orders_with_multiple_payment_values"] = int((payment_nunique > 1).sum())
        result["payment_value_requires_grain_review"] = bool(
            rows != orders or (payment_nunique > 1).any()
        )

    if "is_late" in df.columns and orders:
        late_nunique = df.groupby("order_id")["is_late"].nunique(dropna=False)
        result["orders_with_conflicting_late_flags"] = int((late_nunique > 1).sum())

    coverage = month_coverage(df)
    result["complete_months"] = [row["month"] for row in coverage if row["is_complete"]]
    result["partial_months"] = [row["month"] for row in coverage if not row["is_complete"]]
    result["delivery_kpi_orders"] = int(len(delivered_order_facts(df)))
    result["gmv_definition"] = (
        "Sum of order-level payment_value across all orders in the prepared table; "
        "cancellations and refunds cannot be netted without refund data."
    )

    return result

