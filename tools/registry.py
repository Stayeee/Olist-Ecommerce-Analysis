from __future__ import annotations

from typing import Any, Callable

import pandas as pd

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


ToolFn = Callable[..., dict[str, Any]]


def build_tool_registry(df: pd.DataFrame) -> dict[str, ToolFn]:
    return {
        "get_sales_overview": lambda **kwargs: get_sales_overview(df),
        "analyze_sales_trend": lambda **kwargs: analyze_sales_trend(df, **kwargs),
        "analyze_region_performance": lambda **kwargs: analyze_region_performance(df, **kwargs),
        "analyze_delivery_performance": lambda **kwargs: analyze_delivery_performance(df, **kwargs),
        "analyze_customer_behavior": lambda **kwargs: analyze_customer_behavior(df),
        "analyze_payment_behavior": lambda **kwargs: analyze_payment_behavior(df),
        "analyze_product_performance": lambda **kwargs: analyze_product_performance(df, **kwargs),
        "compare_states": lambda **kwargs: compare_states(df, **kwargs),
    }


TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "get_sales_overview",
        "description": "Return overall Olist business KPIs including GMV, orders, AOV, customers, late rate and delivery time.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "analyze_sales_trend",
        "description": "Analyze monthly GMV, orders and AOV trends. Use this for growth, decline, trend and recent-period diagnostic questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "months": {"type": "integer", "minimum": 2, "maximum": 24}
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "analyze_region_performance",
        "description": "Rank Brazilian states by GMV, orders, AOV, late rate or delivery time.",
        "parameters": {
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "enum": ["gmv", "orders", "aov", "late_rate", "avg_delivery_days"],
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 27},
                "min_orders": {"type": "integer", "minimum": 1, "maximum": 5000},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "analyze_delivery_performance",
        "description": "Analyze overall delivery quality and identify high-risk states using late-delivery rate.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                "min_orders": {"type": "integer", "minimum": 1, "maximum": 5000},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "analyze_customer_behavior",
        "description": "Analyze unique customers, repeat customers and purchase frequency.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "analyze_payment_behavior",
        "description": "Analyze installment behavior and payment mix when available.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "analyze_product_performance",
        "description": "Rank product categories by GMV, orders, AOV or late rate when category data is available.",
        "parameters": {
            "type": "object",
            "properties": {
                "metric": {"type": "string", "enum": ["gmv", "orders", "aov", "late_rate"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 30},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "compare_states",
        "description": "Compare two Brazilian state codes such as SP and RJ across GMV, orders, AOV and delivery KPIs.",
        "parameters": {
            "type": "object",
            "properties": {
                "state_a": {"type": "string"},
                "state_b": {"type": "string"},
            },
            "required": ["state_a", "state_b"],
            "additionalProperties": False,
        },
    },
]
