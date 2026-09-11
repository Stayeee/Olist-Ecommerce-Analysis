from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "order_id",
    "customer_unique_id",
    "order_purchase_timestamp",
    "payment_value",
    "customer_state",
    "customer_city",
    "delivery_days",
    "is_late",
}


def load_analysis_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the prepared Olist analysis table and standardize key types."""
    if path is None:
        path = Path(__file__).resolve().parents[1] / "analysis_table.csv"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Analysis data not found: {path}")

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            "analysis_table.csv is missing required columns: "
            + ", ".join(sorted(missing))
        )

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"], errors="coerce"
    )
    df["order_month"] = (
        df["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    )

    if "order_delivered_customer_date" in df.columns:
        df["order_delivered_customer_date"] = pd.to_datetime(
            df["order_delivered_customer_date"], errors="coerce"
        )

    return df
