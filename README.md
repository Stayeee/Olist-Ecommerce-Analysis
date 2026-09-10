# Olist AI Business Analyst Agent

An AI-powered ecommerce analysis assistant built on the public Olist dataset.

The project started as a traditional Python + Streamlit analytics dashboard. It has been redesigned as a real tool-calling business analysis agent: users ask business questions in natural language, the LLM decides which analytics tools to call, Pandas computes the numbers from the dataset, and the model can run multiple analysis steps before producing a decision-ready answer.

## Why this project exists

Traditional dashboards assume the user already knows which metric or chart to inspect. This prototype explores a different interaction model for operations and product teams:

**Business question → analysis plan → tool calls → structured observations → diagnosis → recommended action**

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
   ├─ No → call another tool
   └─ Yes → finding + evidence + interpretation + action
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
- `compare_states` — direct comparison between two Brazilian states

## Root-cause analysis

A diagnostic question should not stop at one KPI.

For example, when a user asks **"Why did sales decline?"**, the agent can:

1. inspect recent GMV trends;
2. decompose the change into order volume and AOV;
3. call regional or product tools when more evidence is useful;
4. distinguish observed facts from possible business explanations;
5. recommend the next operational investigation.

The system prompt also tells the agent to treat Olist's latest month cautiously because a partial final month can look like a large decline even when it is only incomplete data.

## Why Tool Calling instead of RAG

This project primarily analyzes structured ecommerce data. Metrics such as GMV, AOV, order volume and late-delivery rate require exact calculation, so Tool Calling + Pandas is a better core architecture than RAG.

RAG would become useful later if the product also needed to answer questions from unstructured operating manuals, metric definitions, policies or research documents.

## Evaluation

`evaluation/questions.json` contains 25 representative business questions covering:

- business overview
- sales trends
- root-cause diagnosis
- regional performance
- delivery risk
- customer behavior
- payments
- products
- segment comparison
- out-of-scope requests
- hallucination guardrails

Run:

```bash
python evaluation/evaluate.py
```

The evaluation records which tools the model chose and checks whether expected tools were used. This is intentionally a lightweight first evaluation layer; future iterations can add grounded-answer scoring, latency and token cost.

## Project structure

```text
.
├── app.py
├── analysis_table.csv
├── agent/
│   ├── agent.py
│   └── prompts.py
├── data/
│   └── loader.py
├── tools/
│   ├── analytics.py
│   └── registry.py
├── evaluation/
│   ├── questions.json
│   └── evaluate.py
├── .env.example
└── requirements.txt
```

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your API key:

```bash
OPENAI_API_KEY=your_api_key_here
```

Optional model setting:

```bash
OPENAI_MODEL=gpt-5.6-luna
```

Then run:

```bash
streamlit run app.py
```

You can also paste an API key into the Streamlit sidebar for local demonstration. Do not commit API keys to GitHub.

## Tech stack

- Python
- Pandas
- Streamlit
- Plotly
- OpenAI Responses API
- Function / Tool Calling

## Product thinking behind the prototype

The target user is an ecommerce operations or product user who can describe a business problem but may not know SQL or Python.

The design prioritizes:

- **grounded answers** — metrics come from deterministic tools;
- **explainability** — the UI exposes Tool Calls and Observations rather than hidden reasoning;
- **multi-step diagnosis** — complex questions can trigger several analyses;
- **scope control** — unsupported questions should be rejected rather than answered with invented data;
- **business usefulness** — answers use a Finding → Evidence → Interpretation → Action structure.

## Interview summary

A concise way to explain the project:

> I originally built Olist as a traditional ecommerce data-analysis dashboard. I later found that a dashboard still requires users to know which metric to inspect, so I redesigned it as an AI Business Analyst Agent. I encapsulated sales, customer, regional and delivery analysis as deterministic tools. The LLM is responsible for understanding the question and deciding which tools to call, while the actual numbers are calculated from the dataset. For diagnostic questions the agent can perform multiple tool calls before generating a business conclusion and recommended next action. I also added an evaluation set to test tool selection and out-of-scope behavior.

## Current limitations

- The agent depends on the prepared `analysis_table.csv` rather than querying a production database.
- Product analysis is only available when a product-category field exists in the prepared table.
- Evaluation currently focuses on tool-selection behavior and should be extended with answer-quality and cost metrics.
- This is a portfolio prototype, not a production ecommerce analytics system.
