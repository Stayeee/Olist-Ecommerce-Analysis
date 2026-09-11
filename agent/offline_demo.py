from __future__ import annotations

import json
from time import perf_counter
from typing import Any, Callable

import pandas as pd

from agent.agent import AgentResult, AgentStep
from tools.analytics import (
    analyze_customer_behavior,
    analyze_delivery_performance,
    analyze_sales_trend,
    compare_states,
)
from tools.drivers import analyze_recent_change_drivers


GUIDED_DEMO_QUESTIONS = [
    "Why did sales decline recently?",
    "Which regions should operations prioritize and why?",
    "Compare SP and RJ business performance.",
    "What does customer repeat purchase behavior look like?",
]


def _step_pair(name: str, arguments: dict[str, Any], result: dict[str, Any]) -> list[AgentStep]:
    return [
        AgentStep("tool_call", name, json.dumps(arguments, ensure_ascii=False)),
        AgentStep(
            "observation",
            name,
            json.dumps(result, ensure_ascii=False, default=str)[:2500],
        ),
    ]


def _sales_decline(df: pd.DataFrame) -> tuple[str, list[AgentStep]]:
    trend = analyze_sales_trend(df, months=6)
    drivers = analyze_recent_change_drivers(df, dimension="state", limit=5)
    negative = sorted(drivers["drivers"], key=lambda row: row["gmv_change"])[:3]
    states = ", ".join(row["state"] for row in negative)
    answer = f"""### Finding
GMV fell **{abs(trend['gmv_mom_pct']):.1f}%** from July to August 2018 even though orders grew **{trend['orders_mom_pct']:.1f}%**. Lower AOV was the primary arithmetic driver.

### Evidence
Order-volume growth contributed approximately **R\\$ {trend['order_volume_contribution']:,.0f}**, while the AOV change contributed approximately **-R\\$ {abs(trend['aov_contribution']):,.0f}**. The largest negative state contributions were concentrated in **{states}**.

### Interpretation
The data establishes an AOV effect and state concentration. It does not contain traffic, conversion, pricing, inventory or campaign data, so it cannot prove the commercial cause.

### Recommended action
Review product mix, availability, conversion and promotion changes in {states} before changing regional spend."""
    steps = _step_pair("analyze_sales_trend", {"months": 6}, trend)
    steps += _step_pair(
        "analyze_recent_change_drivers",
        {"dimension": "state", "limit": 5},
        drivers,
    )
    return answer, steps


def _delivery_priority(df: pd.DataFrame) -> tuple[str, list[AgentStep]]:
    result = analyze_delivery_performance(df, limit=5, min_orders=100)
    risks = result["highest_risk_states"][:3]
    evidence = ", ".join(
        f"{row['state']} ({row['late_rate']:.1f}% late, {row['delivery_orders']:,} delivered orders)"
        for row in risks
    )
    answer = f"""### Finding
Delivery risk should be prioritized in the highest-volume states with elevated late rates.

### Evidence
The platform late rate is **{result['overall_late_rate']:.1f}%** across **{result['delivery_kpi_orders']:,}** delivered orders with recorded duration. The leading risk states are {evidence}.

### Interpretation
These states combine meaningful sample size with higher observed lateness. The dataset does not identify the responsible carrier or operational process.

### Recommended action
Compare carrier, seller and route-level performance inside these states before setting service-level interventions."""
    return answer, _step_pair(
        "analyze_delivery_performance", {"limit": 5, "min_orders": 100}, result
    )


def _state_comparison(df: pd.DataFrame) -> tuple[str, list[AgentStep]]:
    result = compare_states(df, "SP", "RJ")
    sp, rj = result["state_a"], result["state_b"]
    answer = f"""### Finding
SP is the much larger market, while the delivery-risk comparison should be judged using both rate and order volume.

### Evidence
SP has **{sp['orders']:,} orders**, **R\\$ {sp['gmv']:,.0f} GMV** and a **{sp['late_rate']:.1f}%** late rate. RJ has **{rj['orders']:,} orders**, **R\\$ {rj['gmv']:,.0f} GMV** and a **{rj['late_rate']:.1f}%** late rate.

### Interpretation
SP has greater commercial impact because of its scale. A higher late rate in either state indicates service risk but does not identify the underlying carrier, seller or route.

### Recommended action
Use absolute late-order volume and carrier-level data to size the operational opportunity before prioritizing one state."""
    return answer, _step_pair(
        "compare_states", {"state_a": "SP", "state_b": "RJ"}, result
    )


def _customer_repeat(df: pd.DataFrame) -> tuple[str, list[AgentStep]]:
    result = analyze_customer_behavior(df)
    answer = f"""### Finding
Repeat purchasing is limited in the observed Olist history.

### Evidence
There are **{result['unique_customers']:,} unique customers** and **{result['repeat_customers']:,} repeat customers**, a repeat-customer rate of **{result['repeat_customer_rate']:.1f}%**. Average orders per customer are **{result['avg_orders_per_customer']:.2f}**.

### Interpretation
The dataset suggests low observed repeat behavior, but its finite observation window means this is not a lifetime retention rate.

### Recommended action
Build cohort retention by first-purchase month and category before deciding whether lifecycle campaigns are underperforming."""
    return answer, _step_pair("analyze_customer_behavior", {}, result)


_HANDLERS: dict[str, Callable[[pd.DataFrame], tuple[str, list[AgentStep]]]] = {
    GUIDED_DEMO_QUESTIONS[0]: _sales_decline,
    GUIDED_DEMO_QUESTIONS[1]: _delivery_priority,
    GUIDED_DEMO_QUESTIONS[2]: _state_comparison,
    GUIDED_DEMO_QUESTIONS[3]: _customer_repeat,
}


def run_guided_demo(question: str, df: pd.DataFrame) -> AgentResult:
    """Run a curated, deterministic walkthrough when no API key is available."""
    started_at = perf_counter()
    handler = _HANDLERS.get(question)
    if handler is None:
        return AgentResult(
            answer="Choose one of the guided demo questions to run without an API key.",
            error="unsupported_guided_demo_question",
            model="guided-demo-no-llm",
            latency_ms=int((perf_counter() - started_at) * 1000),
        )

    answer, steps = handler(df)
    return AgentResult(
        answer=answer,
        steps=steps,
        model="guided-demo-no-llm",
        latency_ms=int((perf_counter() - started_at) * 1000),
        tool_rounds=sum(step.step_type == "tool_call" for step in steps),
    )
