from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# =========================
# 1. Page configuration
# =========================

st.set_page_config(
    page_title="Olist Commerce Intelligence",
    page_icon="🛒",
    layout="wide"
)


# =========================
# 2. Page style
# =========================

st.markdown(
    """
    <style>
    .stApp {
        background: #f6f8fc;
    }

    .block-container {
        max-width: 1240px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #152238;
        letter-spacing: -0.02em;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5eaf2;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 5px 18px rgba(26, 39, 64, 0.05);
    }

    [data-testid="stMetricLabel"] {
        color: #667085;
    }

    [data-testid="stMetricValue"] {
        color: #152238;
    }

    .hero {
        padding: 30px 34px;
        border-radius: 22px;
        background: linear-gradient(
            125deg,
            #152238 0%,
            #234a80 62%,
            #2f6fb3 100%
        );
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 14px 35px rgba(21, 34, 56, 0.16);
    }

    .hero h1 {
        color: white;
        margin: 0 0 8px;
        font-size: 2.25rem;
    }

    .hero p {
        color: #dbe8f7;
        margin: 0;
        font-size: 1.03rem;
    }

    .eyebrow {
        font-size: 0.76rem;
        letter-spacing: 0.13em;
        font-weight: 700;
        color: #8ec5ff;
    }

    .section-note {
        color: #667085;
        margin-top: -8px;
        margin-bottom: 18px;
    }

    .insight {
        padding: 20px 22px;
        background: #eef6ff;
        border: 1px solid #cfe4fb;
        border-left: 5px solid #2f6fb3;
        border-radius: 12px;
        color: #233852;
        line-height: 1.65;
    }

    .method {
        padding: 16px 18px;
        border-radius: 12px;
        background: #fff8e8;
        border: 1px solid #f2ddb0;
        color: #654f22;
    }

    div[data-testid="stButton"] > button {
        border-radius: 10px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# 3. Load data
# =========================

@st.cache_data(show_spinner=False)
def load_data():

    file_path = (
        Path(__file__).resolve().parent
        / "data"
        / "analysis_table.csv"
    )

    if not file_path.exists():
        st.error(
            "Data file not found. Please place "
            "analysis_table.csv inside the data folder."
        )
        st.stop()

    df = pd.read_csv(file_path)

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"],
        errors="coerce"
    )

    df["order_month"] = (
        df["order_purchase_timestamp"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    return df


df = load_data()


# =========================
# 4. Analysis functions
# =========================

def get_overview_metrics(data):

    total_orders = data["order_id"].nunique()
    gmv = data["payment_value"].sum()

    return {
        "total_orders": int(total_orders),
        "gmv": float(gmv),
        "aov": float(gmv / total_orders),
        "late_rate": float(
            data["is_late"].mean() * 100
        ),
        "avg_delivery_days": float(
            data["delivery_days"].mean()
        ),
        "unique_customers": int(
            data["customer_unique_id"].nunique()
        )
    }


def get_monthly_performance(data):

    return (
        data
        .dropna(subset=["order_month"])
        .groupby(
            "order_month",
            as_index=False
        )
        .agg(
            gmv=("payment_value", "sum"),
            orders=("order_id", "nunique")
        )
        .sort_values("order_month")
    )


def get_city_performance(data, limit=10):

    return (
        data
        .groupby(
            "customer_city",
            as_index=False
        )
        .agg(
            gmv=("payment_value", "sum"),
            orders=("order_id", "nunique")
        )
        .nlargest(limit, "gmv")
        .sort_values("gmv")
    )


def get_state_performance(
    data,
    min_orders=100
):

    state_data = (
        data
        .groupby(
            "customer_state",
            as_index=False
        )
        .agg(
            orders=("order_id", "nunique"),
            gmv=("payment_value", "sum"),
            late_rate=("is_late", "mean"),
            avg_delivery_days=(
                "delivery_days",
                "mean"
            )
        )
    )

    state_data["late_rate"] = (
        state_data["late_rate"] * 100
    )

    return state_data[
        state_data["orders"] >= min_orders
    ].copy()


# =========================
# 5. Agent intent detection
# =========================

INTENT_KEYWORDS = {

    "delay": (
        "delay",
        "late",
        "delivery",
        "logistics",
        "fulfilment",
        "fulfillment"
    ),

    "region": (
        "region",
        "state",
        "city",
        "location",
        "area"
    ),

    # Payment must come before customer.
    # Otherwise "How are customers paying?"
    # may be incorrectly classified as customer.
    "payment": (
        "payment",
        "pay",
        "paying",
        "installment",
        "instalment"
    ),

    "customer": (
        "customer",
        "user",
        "retention",
        "repeat",
        "buyer"
    ),

    "gmv": (
        "gmv",
        "overall",
        "sales",
        "performance",
        "revenue",
        "order"
    )
}


def detect_intent(question):

    question = question.lower().strip()

    for intent, keywords in (
        INTENT_KEYWORDS.items()
    ):

        if any(
            keyword in question
            for keyword in keywords
        ):
            return intent

    return "unknown"


def run_analysis(
    intent,
    data
):

    overview = get_overview_metrics(data)

    if intent == "gmv":

        return {
            "analysis_type": "gmv",
            **overview
        }

    if intent == "delay":

        state_data = get_state_performance(
            data,
            min_orders=100
        )

        top_state = (
            state_data
            .nlargest(1, "late_rate")
            .iloc[0]
        )

        return {
            "analysis_type": "delay",
            "late_rate":
                overview["late_rate"],
            "avg_delivery_days":
                overview["avg_delivery_days"],
            "highest_risk_state":
                top_state["customer_state"],
            "highest_state_late_rate":
                float(top_state["late_rate"]),
            "highest_state_orders":
                int(top_state["orders"])
        }

    if intent == "region":

        state_data = get_state_performance(
            data,
            min_orders=100
        )

        top_state = (
            state_data
            .nlargest(1, "gmv")
            .iloc[0]
        )

        return {
            "analysis_type": "region",
            "top_state":
                top_state["customer_state"],
            "top_state_gmv":
                float(top_state["gmv"]),
            "top_state_orders":
                int(top_state["orders"]),
            "states_covered":
                int(
                    data[
                        "customer_state"
                    ].nunique()
                )
        }

    if intent == "customer":

        unique_customers = (
            data[
                "customer_unique_id"
            ].nunique()
        )

        total_orders = (
            data[
                "order_id"
            ].nunique()
        )

        repeat_proxy = (
            1
            - unique_customers
            / max(total_orders, 1)
        )

        return {
            "analysis_type": "customer",
            "unique_customers":
                int(unique_customers),
            "repeat_order_proxy":
                float(
                    max(
                        repeat_proxy,
                        0
                    ) * 100
                ),
            "orders_per_customer":
                float(
                    total_orders
                    / max(
                        unique_customers,
                        1
                    )
                )
        }

    if intent == "payment":

        installments = (
            data[
                "payment_installments"
            ]
            .fillna(0)
        )

        return {
            "analysis_type": "payment",
            "avg_installments":
                float(
                    installments.mean()
                ),
            "installment_share":
                float(
                    (
                        installments > 1
                    ).mean() * 100
                ),
            "single_payment_share":
                float(
                    (
                        installments <= 1
                    ).mean() * 100
                )
        }

    return {
        "analysis_type": "unknown",
        "error":
            "I could not match this question "
            "to a supported analysis."
    }


def run_ai_agent(
    question,
    data
):

    intent = detect_intent(
        question
    )

    return run_analysis(
        intent,
        data
    )


# =========================
# 6. Business recommendations
# =========================

def generate_business_advice(result):

    analysis_type = result.get(
        "analysis_type"
    )

    if analysis_type == "gmv":

        return (
            f"The business generated "
            f"R$ {result['gmv']:,.0f} across "
            f"{result['total_orders']:,} orders, "
            f"with an AOV of "
            f"R$ {result['aov']:,.2f}. "
            f"The {result['late_rate']:.1f}% "
            f"late rate is the clearest "
            f"operational risk. Track GMV and "
            f"service quality together so growth "
            f"does not hide fulfilment problems."
        )

    if analysis_type == "delay":

        return (
            f"Overall, "
            f"{result['late_rate']:.1f}% of orders "
            f"were delivered late. "
            f"{result['highest_risk_state']} has "
            f"the highest late rate among states "
            f"with at least 100 orders "
            f"({result['highest_state_late_rate']:.1f}%). "
            f"Prioritise route and carrier "
            f"investigation in high-volume, "
            f"high-risk states."
        )

    if analysis_type == "region":

        return (
            f"{result['top_state']} is the leading "
            f"state by GMV, contributing "
            f"R$ {result['top_state_gmv']:,.0f} "
            f"from {result['top_state_orders']:,} "
            f"orders. Compare revenue concentration "
            f"with delivery performance to protect "
            f"the regions with the greatest impact."
        )

    if analysis_type == "customer":

        return (
            f"The dataset contains "
            f"{result['unique_customers']:,} unique "
            f"customers and "
            f"{result['orders_per_customer']:.2f} "
            f"orders per customer. The repeat-order "
            f"proxy is "
            f"{result['repeat_order_proxy']:.1f}%. "
            f"Cohort analysis would be the next step "
            f"before making a retention decision."
        )

    if analysis_type == "payment":

        return (
            f"{result['installment_share']:.1f}% "
            f"of records used more than one "
            f"instalment, with an average of "
            f"{result['avg_installments']:.1f}. "
            f"Analyse order value alongside "
            f"instalment count to assess whether "
            f"financing supports higher-value purchases."
        )

    return (
        "Try one of the supported topics: "
        "overall sales, delivery performance, "
        "regional performance, customers or payments."
    )


# =========================
# 7. Display functions
# =========================

def render_overview_metrics(metrics):

    columns = st.columns(5)

    values = [
        (
            "Total orders",
            f"{metrics['total_orders']:,}"
        ),
        (
            "GMV",
            f"R$ {metrics['gmv'] / 1_000_000:.2f}M"
        ),
        (
            "Average order value",
            f"R$ {metrics['aov']:,.2f}"
        ),
        (
            "Late delivery rate",
            f"{metrics['late_rate']:.1f}%"
        ),
        (
            "Avg. delivery time",
            f"{metrics['avg_delivery_days']:.1f} days"
        )
    ]

    for column, value in zip(
        columns,
        values
    ):

        column.metric(
            value[0],
            value[1]
        )


def render_agent_result(result):

    if (
        result.get("analysis_type")
        == "unknown"
    ):

        st.warning(
            result.get("error")
        )

        return

    labels = {
        "total_orders": "Total orders",
        "gmv": "GMV",
        "aov": "Average order value",
        "late_rate": "Late rate",
        "avg_delivery_days":
            "Avg. delivery days",
        "unique_customers":
            "Unique customers",
        "highest_risk_state":
            "Highest-risk state",
        "highest_state_late_rate":
            "State late rate",
        "highest_state_orders":
            "Orders in state",
        "top_state": "Top state",
        "top_state_gmv":
            "Top-state GMV",
        "top_state_orders":
            "Top-state orders",
        "states_covered":
            "States covered",
        "repeat_order_proxy":
            "Repeat-order proxy",
        "orders_per_customer":
            "Orders per customer",
        "avg_installments":
            "Avg. instalments",
        "installment_share":
            "Multi-instalment share",
        "single_payment_share":
            "Single-payment share"
    }

    currency_keys = {
        "gmv",
        "aov",
        "top_state_gmv"
    }

    percent_keys = {
        "late_rate",
        "highest_state_late_rate",
        "repeat_order_proxy",
        "installment_share",
        "single_payment_share"
    }

    items = [
        (key, value)
        for key, value
        in result.items()
        if key not in {
            "analysis_type",
            "error"
        }
    ]

    columns = st.columns(
        min(
            len(items),
            4
        )
    )

    for index, item in enumerate(items):

        key, value = item

        if key in currency_keys:
            display_value = (
                f"R$ {value:,.2f}"
            )

        elif key in percent_keys:
            display_value = (
                f"{value:.1f}%"
            )

        elif isinstance(value, int):
            display_value = (
                f"{value:,}"
            )

        elif isinstance(value, float):
            display_value = (
                f"{value:.2f}"
            )

        else:
            display_value = str(value)

        columns[
            index % len(columns)
        ].metric(
            labels.get(
                key,
                key.replace(
                    "_",
                    " "
                ).title()
            ),
            display_value
        )

    st.markdown(
        "#### Business insight"
    )

    advice = (
        generate_business_advice(
            result
        )
    )

    st.markdown(
        f"""
        <div class="insight">
            {advice}
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================
# 8. Main interface
# =========================

metrics = get_overview_metrics(
    df
)

st.markdown(
    '<div class="hero">'
    '<div class="eyebrow">OLIST · COMMERCE INTELLIGENCE</div>'
    '<h1>From business question to decision-ready insight</h1>'
    '<p>Explore sales, customers and delivery performance through an interactive analytical agent.</p>'
    '</div>',
    unsafe_allow_html=True
)


# =========================
# 9. Sidebar
# =========================

with st.sidebar:

    st.markdown(
        "### 🛒 Project controls"
    )

    st.caption(
        "Public Brazilian e-commerce dataset"
    )

    min_orders = st.slider(
        "Minimum state order volume",
        min_value=50,
        max_value=1000,
        value=100,
        step=50,
        help=(
            "Reduces noise when comparing "
            "state-level late rates."
        )
    )

    st.divider()

    st.markdown(
        "**Prototype capability**"
    )

    st.markdown(
        "Natural-language keyword routing, "
        "deterministic analysis skills, "
        "interactive visualisation and "
        "rule-based business recommendations."
    )

    st.caption(
        "No customer-confidential data "
        "or external model API is used."
    )


# =========================
# 10. Tabs
# =========================

overview_tab, agent_tab, delivery_tab, method_tab = (
    st.tabs(
        [
            "Executive overview",
            "Analysis agent",
            "Delivery risk",
            "Method & iteration"
        ]
    )
)


# =========================
# Tab 1: Overview
# =========================

with overview_tab:

    st.markdown(
        "## Business snapshot"
    )

    st.markdown(
        """
        <p class="section-note">
            A high-level view of commercial
            scale and fulfilment quality.
        </p>
        """,
        unsafe_allow_html=True
    )

    render_overview_metrics(
        metrics
    )

    monthly_data = (
        get_monthly_performance(
            df
        )
    )

    city_data = (
        get_city_performance(
            df
        )
    )

    left_column, right_column = (
        st.columns(
            [1.55, 1]
        )
    )

    with left_column:

        monthly_chart = px.area(
            monthly_data,
            x="order_month",
            y="gmv",
            title="Monthly GMV trend",
            labels={
                "order_month":
                    "Order month",
                "gmv":
                    "GMV (R$)"
            },
            color_discrete_sequence=[
                "#2f80ed"
            ]
        )

        monthly_chart.update_traces(
            line_width=2.4
        )

        monthly_chart.update_layout(
            margin=dict(
                l=10,
                r=10,
                t=55,
                b=10
            ),
            hovermode="x unified"
        )

        st.plotly_chart(
            monthly_chart,
            use_container_width=True
        )

    with right_column:

        city_chart = px.bar(
            city_data,
            x="gmv",
            y="customer_city",
            orientation="h",
            title="Top 10 cities by GMV",
            labels={
                "gmv":
                    "GMV (R$)",
                "customer_city":
                    "City"
            },
            color="gmv",
            color_continuous_scale=[
                "#dbeafe",
                "#2f80ed"
            ]
        )

        city_chart.update_layout(
            margin=dict(
                l=10,
                r=10,
                t=55,
                b=10
            ),
            coloraxis_showscale=False
        )

        st.plotly_chart(
            city_chart,
            use_container_width=True
        )


# =========================
# Tab 2: Agent
# =========================

with agent_tab:

    st.markdown(
        "## Ask the analysis agent"
    )

    st.markdown(
        """
        <p class="section-note">
            Choose an example or enter
            a supported business question
            in English.
        </p>
        """,
        unsafe_allow_html=True
    )

    examples = [
        "How is overall sales performance?",
        "Where is delivery risk highest?",
        "Which region generates the most GMV?",
        "What does customer retention look like?",
        "How are customers paying?"
    ]

    example_columns = (
        st.columns(5)
    )

    for index, example in enumerate(
        examples
    ):

        clicked = (
            example_columns[index]
            .button(
                example,
                key=f"example_{index}",
                use_container_width=True
            )
        )

        if clicked:
            st.session_state[
                "agent_question"
            ] = example

    question = st.text_input(
        "Business question",
        key="agent_question",
        placeholder=(
            "e.g. Which region has "
            "the highest delivery risk?"
        )
    )

    if st.button(
        "Analyse question",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "Enter a question or "
                "select an example first."
            )

        else:

            with st.spinner(
                "Matching intent and "
                "running analysis..."
            ):

                result = run_ai_agent(
                    question,
                    df
                )

            st.success(
                "Analysis complete · "
                f"matched skill: "
                f"{result.get('analysis_type')}"
            )

            render_agent_result(
                result
            )


# =========================
# Tab 3: Delivery
# =========================

with delivery_tab:

    st.markdown(
        "## Delivery risk diagnostics"
    )

    st.markdown(
        """
        <p class="section-note">
            State comparisons exclude
            small samples using the
            sidebar volume control.
        </p>
        """,
        unsafe_allow_html=True
    )

    state_data = (
        get_state_performance(
            df,
            min_orders=min_orders
        )
    )

    risk_data = (
        state_data
        .nlargest(
            12,
            "late_rate"
        )
        .sort_values(
            "late_rate"
        )
    )

    volume_data = (
        state_data
        .nlargest(
            12,
            "orders"
        )
        .sort_values(
            "orders"
        )
    )

    left_column, right_column = (
        st.columns(2)
    )

    with left_column:

        risk_chart = px.bar(
            risk_data,
            x="late_rate",
            y="customer_state",
            orientation="h",
            title=(
                "Highest late rates "
                "by state"
            ),
            labels={
                "late_rate":
                    "Late rate (%)",
                "customer_state":
                    "State"
            },
            color="late_rate",
            color_continuous_scale=[
                "#fde8e7",
                "#d64545"
            ]
        )

        risk_chart.update_layout(
            coloraxis_showscale=False
        )

        st.plotly_chart(
            risk_chart,
            use_container_width=True
        )

    with right_column:

        volume_chart = px.bar(
            volume_data,
            x="orders",
            y="customer_state",
            orientation="h",
            title=(
                "Largest states "
                "by order volume"
            ),
            labels={
                "orders":
                    "Orders",
                "customer_state":
                    "State"
            },
            color="orders",
            color_continuous_scale=[
                "#dbeafe",
                "#2f80ed"
            ]
        )

        volume_chart.update_layout(
            coloraxis_showscale=False
        )

        st.plotly_chart(
            volume_chart,
            use_container_width=True
        )

    scatter_chart = px.scatter(
        state_data,
        x="orders",
        y="late_rate",
        size="gmv",
        color="avg_delivery_days",
        hover_name="customer_state",
        title=(
            "Risk prioritisation: "
            "volume vs. late rate"
        ),
        labels={
            "orders":
                "Order volume",
            "late_rate":
                "Late rate (%)",
            "avg_delivery_days":
                "Avg. delivery days"
        },
        color_continuous_scale="Blues"
    )

    st.plotly_chart(
        scatter_chart,
        use_container_width=True
    )


# =========================
# Tab 4: Method
# =========================

with method_tab:

    st.markdown(
        "## Method, limits and V1 → V2 iteration"
    )

    st.markdown(
        """
This prototype was built from the public Olist
e-commerce dataset.

It uses a **rule-based analytical agent**:
the app detects intent from question keywords,
routes the question to a validated Pandas
analysis function, and returns structured
metrics plus deterministic recommendations.

It does **not** call an LLM or claim
generative reasoning.

### V1

V1 validated the basic functional loop:

**Question → Intent → Calculation → Recommendation**

Testing exposed three usability problems:

- Unsupported questions were unclear
- Raw field names were shown directly
- Isolated metrics lacked visual context

### V2

V2 adds:

- Example business questions
- Formatted KPI cards
- Commercial and delivery visualisations
- Clear capability boundaries
- Better error feedback
- Minimum-volume safeguards for regional risk
        """
    )

    st.markdown(
        """
        <div class="method">
            <strong>Evaluation principle</strong>
            <br><br>
            Metric outputs should be reconciled
            against the source table. Intent routing
            should be tested with supported,
            ambiguous and unsupported questions.
            Regional risk rankings should use a
            minimum sample threshold.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "### Tech stack"
    )

    st.markdown(
        "Python · Pandas · Streamlit · Plotly · "
        "rule-based intent routing · AI-assisted coding"
    )