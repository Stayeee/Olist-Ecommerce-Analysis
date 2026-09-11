import os

import pandas as pd
import plotly.express as px
import streamlit as st

from agent.agent import OlistBusinessAgent
from agent.offline_demo import GUIDED_DEMO_QUESTIONS, run_guided_demo
from data.loader import load_analysis_data
from tools.analytics import analyze_region_performance, analyze_sales_trend, get_sales_overview
from tools.drivers import analyze_recent_change_drivers


st.set_page_config(
    page_title="Olist AI Business Analyst",
    page_icon="🛒",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
    .hero {padding: 28px 30px; border-radius: 20px; background: #172033; color: white; margin-bottom: 22px;}
    .hero h1 {color: white; margin: 0 0 8px 0; font-size: 2.2rem;}
    .hero p {color: #d8deea; margin: 0;}
    .eyebrow {font-size: .75rem; letter-spacing: .12em; font-weight: 700; color: #a8c7fa; margin-bottom: 8px;}
    .note {padding: 14px 16px; border-radius: 12px; background: #f5f7fb; border: 1px solid #e5e9f0;}
    [data-testid="stMetric"] {border: 1px solid #e5e9f0; border-radius: 14px; padding: 14px; background: white;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_analysis_data()


df = get_data()
overview = get_sales_overview(df)

if "agent_history" not in st.session_state:
    st.session_state.agent_history = []

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">OLIST · AI BUSINESS ANALYST</div>
      <h1>Ask a business question. Let the agent decide how to investigate.</h1>
      <p>LLM tool calling + deterministic Pandas analytics + multi-step business diagnosis.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Agent settings")
    model = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "gpt-5-mini"))

    secret_key = ""
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        secret_key = ""

    api_key = os.getenv("OPENAI_API_KEY", "") or secret_key
    if not api_key:
        api_key = st.text_input("OpenAI API key (optional)", type="password")
        st.caption("Leave blank to use the curated no-key guided demo.")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.agent_history = []
        st.rerun()

    st.divider()
    st.markdown("**What makes this an Agent?**")
    st.caption(
        "The model chooses analytics tools, observes real computed results, and may call additional tools before answering."
    )
    st.markdown("**Guardrail**")
    st.caption("Business numbers must come from tool outputs rather than model memory.")

overview_tab, agent_tab, diagnostics_tab, method_tab = st.tabs(
    ["Business overview", "AI analyst", "Root-cause lab", "How it works"]
)

with overview_tab:
    st.subheader("Business snapshot")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Orders", f"{overview['orders']:,}")
    c2.metric("GMV", f"R$ {overview['gmv']/1_000_000:.2f}M")
    c3.metric("AOV", f"R$ {overview['aov']:.2f}")
    c4.metric("Late rate", f"{overview['late_rate']:.1f}%")
    c5.metric("Avg delivery", f"{overview['avg_delivery_days']:.1f} days")

    trend = analyze_sales_trend(df, months=12)
    trend_df = pd.DataFrame(trend["series"])
    trend_df["month"] = pd.to_datetime(trend_df["month"])

    left, right = st.columns([1.5, 1])
    with left:
        fig = px.line(trend_df, x="month", y="gmv", markers=True, title="GMV trend · complete months only")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        region = pd.DataFrame(
            analyze_region_performance(df, metric="gmv", limit=8, min_orders=100)["rows"]
        )
        fig = px.bar(
            region.sort_values("gmv"),
            x="gmv",
            y="state",
            orientation="h",
            title="Top states by GMV",
        )
        st.plotly_chart(fig, use_container_width=True)

with agent_tab:
    st.subheader("Ask the AI Business Analyst")
    st.caption("Try a business question instead of selecting a fixed dashboard filter.")

    examples = GUIDED_DEMO_QUESTIONS
    if not api_key:
        st.info(
            "Guided demo mode is active. It runs curated questions with real deterministic "
            "analytics tools and shows the resulting trace. It does not call an LLM."
        )
    selected = st.selectbox("Example question", ["Write my own question"] + examples)
    default_question = "" if selected == "Write my own question" else selected
    question = st.text_area("Business question", value=default_question, height=90)

    if st.button("Analyze", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Enter a business question first.")
        elif not api_key and question not in GUIDED_DEMO_QUESTIONS:
            st.warning("Choose one of the example questions when using the no-key guided demo.")
        else:
            try:
                if api_key:
                    with st.spinner("The agent is selecting and running analytics tools..."):
                        agent = OlistBusinessAgent(df=df, model=model, api_key=api_key)
                        result = agent.ask(question, history=st.session_state.agent_history)
                else:
                    with st.spinner("Running the guided analysis with deterministic tools..."):
                        result = run_guided_demo(question, df)
            except Exception as exc:
                st.error("The agent could not start. Check the API key, model name and local environment.")
                st.code(f"{type(exc).__name__}: {exc}")
            else:
                st.session_state.agent_history.extend(
                    [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": result.answer},
                    ]
                )

                if result.error:
                    st.warning("The agent returned an execution error instead of a completed analysis.")
                    with st.expander("Technical error"):
                        st.code(result.error)

                st.markdown("### Decision-ready answer")
                st.markdown(result.answer)
                st.caption(
                    f"Model: {result.model} · Tool rounds: {result.tool_rounds} · "
                    f"Tokens: {result.input_tokens:,} in / {result.output_tokens:,} out · "
                    f"Latency: {result.latency_ms / 1000:.1f}s"
                )
                if result.model == "guided-demo-no-llm":
                    st.caption(
                        "Guided demo: curated deterministic workflow for portfolio viewing; "
                        "dynamic LLM tool selection is available only when an API key is supplied."
                    )

                st.markdown("### Agent execution trace")
                if not result.steps:
                    st.info("No data tool was needed for this question.")
                else:
                    for idx, step in enumerate(result.steps, start=1):
                        if step.step_type == "tool_call":
                            st.markdown(f"**{idx}. Tool call · `{step.name}`**")
                            st.code(step.detail, language="json")
                        else:
                            with st.expander(f"{idx}. Observation · {step.name}"):
                                st.code(step.detail, language="json")

    if st.session_state.agent_history:
        with st.expander("Conversation context"):
            for message in st.session_state.agent_history[-6:]:
                st.markdown(f"**{message['role'].title()}:** {message['content']}")

with diagnostics_tab:
    st.subheader("Recent GMV root-cause analysis")
    st.markdown(
        """
        <div class="note">
        This view compares the latest two complete calendar months. It decomposes GMV into order-volume
        and AOV effects, then ranks states by their measured contribution to the change.
        </div>
        """,
        unsafe_allow_html=True,
    )

    trend = analyze_sales_trend(df, months=6)
    state_drivers = analyze_recent_change_drivers(df, dimension="state", limit=8)
    previous_label = pd.Timestamp(trend["previous_month"]).strftime("%b %Y")
    latest_label = pd.Timestamp(trend["latest_month"]).strftime("%b %Y")
    a, b, c = st.columns(3)
    a.metric(f"GMV · {latest_label} vs {previous_label}", f"{trend['gmv_mom_pct']:.1f}%")
    b.metric("Orders MoM", f"{trend['orders_mom_pct']:.1f}%")
    c.metric("AOV MoM", f"{trend['aov_mom_pct']:.1f}%")

    st.markdown("### Finding")
    if trend["gmv_change"] < 0 and trend["primary_arithmetic_driver"] == "aov":
        st.write(
            f"GMV fell **{abs(trend['gmv_mom_pct']):.1f}%**, even as order volume grew "
            f"**{trend['orders_mom_pct']:.1f}%**. The arithmetic decline was driven by lower AOV."
        )
    else:
        st.write(
            f"GMV changed **{trend['gmv_mom_pct']:.1f}%**; the larger arithmetic contribution came from "
            f"**{trend['primary_arithmetic_driver']}**."
        )

    st.markdown("### Evidence")
    contribution_df = pd.DataFrame(
        [
            {"driver": "Order volume", "gmv_contribution": trend["order_volume_contribution"]},
            {"driver": "AOV", "gmv_contribution": trend["aov_contribution"]},
        ]
    )
    driver_df = pd.DataFrame(state_drivers["drivers"])
    left, right = st.columns(2)
    with left:
        fig = px.bar(
            contribution_df,
            x="driver",
            y="gmv_contribution",
            color="gmv_contribution",
            color_continuous_scale=["#b42318", "#f2f4f7", "#027a48"],
            title="Orders vs AOV contribution to GMV change",
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.bar(
            driver_df.sort_values("gmv_change"),
            x="gmv_change",
            y="state",
            orientation="h",
            color="gmv_change",
            color_continuous_scale=["#b42318", "#f2f4f7", "#027a48"],
            title="State contribution to GMV change",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Interpretation")
    st.write(
        "Order count and AOV are mathematical drivers. State contributions show where the change was "
        "concentrated; they do not prove a causal explanation such as traffic, pricing, stock or competition."
    )

    st.markdown("### Recommended action")
    negative_states = driver_df.nsmallest(3, "gmv_change")["state"].tolist()
    st.write(
        "Prioritize the largest negative-contribution states "
        f"({', '.join(negative_states)}) and inspect traffic, conversion, product mix and availability before changing spend."
    )
    st.caption(
        "Excluded partial months: " + ", ".join(trend["excluded_partial_months"])
    )

with method_tab:
    st.subheader("Architecture")
    st.code(
        """User question
   ↓
LLM Agent (planning + tool selection)
   ↓
Analytics Tool Registry
   ↓
Pandas / Olist dataset
   ↓
Structured observation
   ↓
LLM decides: enough evidence?
   ├─ No → call another tool
   └─ Yes → finding + evidence + interpretation + action""",
        language="text",
    )

    st.markdown("### Why Tool Calling instead of RAG?")
    st.write(
        "GMV, AOV, order volume and delivery rates require exact calculation over structured data. "
        "RAG is better suited to retrieving unstructured knowledge, so it is intentionally not the core architecture here."
    )

    st.markdown("### Evaluation")
    st.write(
        "The repository includes an offline smoke test for deterministic tools plus 25 representative Agent questions covering overview, sales, root cause, region, delivery, customer, payment, product, comparisons, out-of-scope requests and guardrails."
    )
