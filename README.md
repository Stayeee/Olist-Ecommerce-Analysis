# Olist AI Business Analyst Agent

An AI-powered ecommerce analysis assistant built on the public Olist dataset.

The project started as a traditional Python + Streamlit analytics dashboard. It has been redesigned as a tool-calling business analysis agent: users ask business questions in natural language, the LLM decides which analytics tools to call, Pandas computes the numbers from the dataset, and the model can run multiple analysis steps before producing a decision-ready answer.

## Why this project exists

Traditional dashboards assume the user already knows which metric or chart to inspect. This prototype explores a different interaction model for operations and product teams:

**Business question → tool selection → structured observations → diagnosis → recommended action**

Example questions:

- Why did sales decline recently?
- Which regions should operations prioritize and why?
- Compare SP and RJ business performance.
- What does customer repeat purchase behavior look like?
- Where is delivery risk highest?

## What makes it an Agent

The original prototype used keyword routing and fixed answer templates. The upgraded version uses an LLM tool-calling loop.

```text
User question
   ↓
LLM Agent
(planning + tool selection)
   ↓
Analytics Tool Registry
   ↓
Pandas / Olist data
   ↓
Structured observation
   ↓
LLM decides whether more evidence is needed
   ├─ Yes → call another tool
   └─ No  → finding + evidence + interpretation + action
```

The model does not calculate or invent business metrics itself. Numeric claims are grounded in deterministic Python functions.

## Analytics tools

The current tool registry contains:

- `get_sales_overview` — GMV, orders, AOV, customers, delivery KPIs
- `analyze_sales_trend` — monthly GMV, orders, AOV and MoM changes
- `analyze_region_performance` — regional ranking by GMV, orders, AOV or delivery risk
- `analyze_delivery_performance` — platform delivery quality and high-risk states
- `analyze_customer_behavior` — repeat customers and purchase frequency
- `analyze_payment_behavior` — installment behavior and payment mix when available
- `analyze_product_performance` — category analysis when the prepared dataset contains product category fields
- `analyze_recent_change_drivers` — state/category contribution to the latest complete-month GMV change
- `compare_states` — direct comparison between two Brazilian states

## Root-cause analysis

A diagnostic question should not stop at one KPI.

For example, when a user asks **"Why did sales decline?"**, the agent can:

1. identify the latest two complete calendar months;
2. inspect GMV, order volume and AOV changes;
3. decompose the GMV change into exact order-volume and AOV contributions;
4. rank states or available product categories by contribution to the same period change;
5. separate arithmetic drivers and segment concentration from unmeasured causal hypotheses;
6. recommend the next operational investigation.

The dataset contains two incomplete tail months: September 2018 has 16 orders and October 2018 has 4. The previous implementation excluded only the last observed month, which still produced a false August-to-September collapse. The metric layer now detects month coverage from purchase dates and compares July 2018 with August 2018, the latest two complete months.

For that period, gross payment value fell 4.1% while orders grew 3.5% and AOV fell 7.4%. The exact arithmetic decomposition shows a positive order-volume contribution of about R$35.9k and a negative AOV contribution of about R$80.0k. This identifies the measured driver without claiming an unsupported cause such as traffic, price or inventory.

## Why Tool Calling instead of RAG

This project primarily analyzes structured ecommerce data. Metrics such as GMV, AOV, order volume and late-delivery rate require exact calculation, so Tool Calling + Pandas is a better core architecture than RAG.

RAG would become useful later if the product also needed to answer questions from unstructured operating manuals, metric definitions, policies or research documents.

## Data-quality guardrail

Olist source tables contain one-to-many relationships between orders, items and payments. After joins, a prepared analysis table may contain multiple rows per order. That matters because blindly summing a repeated `payment_value` can overstate GMV.

`data/quality.py` therefore checks:

- rows vs unique orders;
- rows per order;
- maximum rows per order;
- orders containing multiple payment values;
- conflicting late-delivery flags.

The checked-in table contains 99,441 rows and 99,441 unique orders, so `payment_value` is currently safe to sum once per order. The metric layer also collapses identical duplicate order rows if a future item join reintroduces them, and fails on conflicting order-level values instead of silently choosing an aggregation.

The dashboard's GMV label means **gross payment value across all order statuses**. The dataset has no refund table, so this should not be described as net revenue. Delivery KPIs use only delivered orders with a recorded delivery duration; canceled and still-open orders are not counted as on-time deliveries.

## Evaluation

The project uses two evaluation layers.

### 1. Offline analytics smoke test

This requires no API key and validates the prepared dataset plus deterministic analytics functions:

```bash
python -m evaluation.smoke_test
```

The same smoke test runs automatically in GitHub Actions for the upgrade branch and pull request.

CI also runs an offline fake-model test of the Agent loop:

```bash
python -m evaluation.agent_loop_test
```

It verifies that a root-cause request can execute a trend tool, feed the observation back into the loop, execute a driver tool, and then return a final answer without making a paid API request.

### 2. Agent evaluation

`evaluation/questions.json` contains 25 representative business questions covering business overview, sales trends, root-cause diagnosis, regions, delivery, customers, payments, products, comparisons, out-of-scope requests and hallucination guardrails.

Run with an API key:

```bash
python -m evaluation.evaluate
```

The `Live Agent Evaluation` GitHub Actions workflow can run the same suite using
an `OPENAI_API_KEY` repository Actions secret. It is limited to manual dispatch
or changes to its own workflow file, so ordinary branch pushes do not repeatedly
spend API tokens. Its detailed JSON result is retained as a workflow artifact.

The evaluation records required-tool coverage, multi-step completion, Finding/Evidence/Interpretation/Action structure, execution errors, latency, and input/output token usage. Detailed results are written to `evaluation/results.json` for review.

## Project structure

```text
.
├── app.py
├── analysis_table.csv
├── PROJECT_DESIGN.md
├── agent/
│   ├── __init__.py
│   ├── agent.py
│   └── prompts.py
├── data/
│   ├── __init__.py
│   ├── loader.py
│   └── quality.py
├── tools/
│   ├── __init__.py
│   ├── analytics.py
│   └── registry.py
├── evaluation/
│   ├── __init__.py
│   ├── questions.json
│   ├── smoke_test.py
│   └── evaluate.py
├── .github/workflows/smoke-test.yml
├── .env.example
└── requirements.txt
```

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the offline check first:

```bash
python -m evaluation.smoke_test
```

Then set your API key. You can use an environment variable or Streamlit secrets. Do not commit API keys to GitHub.

Optional model setting:

```text
OPENAI_MODEL=gpt-5-mini
```

Then run:

```bash
streamlit run app.py
```

You can also paste an API key into the Streamlit sidebar for a local demonstration.

## Tech stack

- Python
- Pandas
- Streamlit
- Plotly
- OpenAI Responses API
- Function / Tool Calling
- GitHub Actions

The default demo model is `gpt-5-mini`, a cost-sensitive API model that supports the Responses API and function calling. You can override it with `OPENAI_MODEL` when your API account has access to another compatible model.

## Product thinking behind the prototype

The target user is an ecommerce operations or product user who can describe a business problem but may not know SQL or Python.

The design prioritizes:

- **grounded answers** — metrics come from deterministic tools;
- **explainability** — the UI exposes Tool Calls and Observations rather than hidden reasoning;
- **multi-step diagnosis** — complex questions can trigger several analyses;
- **scope control** — unsupported questions should be rejected rather than answered with invented data;
- **business usefulness** — answers use a Finding → Evidence → Interpretation → Action structure;
- **metric reliability** — dataset grain is checked before assuming joined rows can be safely aggregated.

## Interview summary

A concise way to explain the project:

> I originally built Olist as a traditional ecommerce data-analysis dashboard. I later found that a dashboard still requires users to know which metric to inspect, so I redesigned it as an AI Business Analyst Agent. I encapsulated sales, customer, regional and delivery analysis as deterministic tools. The LLM is responsible for understanding the question and deciding which tools to call, while the actual numbers are calculated from the dataset. For diagnostic questions the agent can perform multiple tool calls before generating a business conclusion and recommended next action. I also added data-grain checks and evaluation coverage so the project tests both metric reliability and Agent behavior.

See `PROJECT_DESIGN.md` for the full product rationale and interview guide.

## Current limitations

- The agent depends on the prepared `analysis_table.csv` rather than querying a production database.
- GMV is gross payment value rather than net revenue because refunds are unavailable.
- Recent-change analysis identifies arithmetic and segment contributors; traffic, conversion, pricing, inventory and campaign data would be required for causal diagnosis.
- Product analysis is only available when a product-category field exists in the prepared table.
- Agent evaluation currently focuses on tool-selection behavior and should be extended with answer-quality and cost metrics.
- This is a portfolio prototype, not a production ecommerce analytics system.

