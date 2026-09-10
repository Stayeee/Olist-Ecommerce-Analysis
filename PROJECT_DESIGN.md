# Olist AI Business Analyst Agent — Project Design & Interview Guide

## 1. Business problem

Traditional dashboards require users to know which metric, filter or chart to inspect. An ecommerce operations user may instead start with a business question such as:

> Why did sales decline recently?

The product goal is to turn that natural-language question into a grounded, traceable analysis workflow.

## 2. Target user

Primary user: ecommerce operations / product / business analyst users who understand the business but may not write Python or SQL.

The prototype should help them move from a vague business question to:

1. a measurable finding;
2. supporting evidence;
3. a cautious interpretation;
4. a recommended next action.

## 3. Agent workflow

```text
User question
  ↓
LLM understands the task
  ↓
Select analytics tool(s)
  ↓
Pandas calculates metrics from Olist data
  ↓
Structured observation returned to LLM
  ↓
Enough evidence?
  ├─ No: call another relevant tool
  └─ Yes: produce finding + evidence + interpretation + action
```

The key architectural decision is that the model does not calculate business KPIs from memory. Deterministic Python functions own the metric calculation.

## 4. Why this is an Agent rather than a chatbot

The original prototype used keyword routing and fixed answer templates. The upgraded system gives the LLM control over which tool to call and allows multiple tool-call rounds before the final response.

The important Agent capabilities are:

- natural-language task understanding;
- dynamic tool selection;
- tool execution against real data;
- observation-driven follow-up analysis;
- conversational context;
- execution trace;
- scope and hallucination guardrails.

## 5. Why Tool Calling instead of RAG

The core data is structured. GMV, AOV, order count and late-delivery rate require precise calculations, so Tool Calling + Pandas is the correct first architecture.

RAG is useful when the system must retrieve unstructured knowledge such as operating manuals, policies, metric definitions or research documents. It is deliberately not added just to make the stack look more complex.

## 6. Current analytics tools

- `get_sales_overview`
- `analyze_sales_trend`
- `analyze_region_performance`
- `analyze_delivery_performance`
- `analyze_customer_behavior`
- `analyze_payment_behavior`
- `analyze_product_performance`
- `compare_states`

Each tool returns JSON-like structured evidence rather than prose.

## 7. Root-cause example

Question:

> Why did sales decline recently?

Expected investigation pattern:

```text
Sales Trend Tool
  ↓
Is GMV actually down?
  ↓
Decompose into Orders vs AOV
  ↓
If useful, inspect Region / Product contributors
  ↓
Separate measured facts from hypotheses
  ↓
Recommend the next operational investigation
```

Important guardrail: more than one tail month can be incomplete. The metric layer checks first/last purchase date and active-day coverage for every month. In this dataset it excludes both September and October 2018, then compares July with August 2018.

The latest valid comparison is:

- GMV: -4.1%
- Orders: +3.5%
- AOV: -7.4%
- Order-volume contribution: approximately +R$35.9k
- AOV contribution: approximately -R$80.0k

This supports the finding that lower AOV was the arithmetic driver. State-level contributions locate where the change concentrated, but do not prove why AOV or demand changed.

## 8. Data-quality design

Olist source tables contain one-to-many relationships among orders, items and payments. A prepared analysis table can therefore contain multiple rows per order after joins.

Before treating row-level `payment_value` sums as GMV, the project checks:

- number of rows vs unique orders;
- rows per order;
- maximum rows per order;
- whether one order contains multiple payment values;
- whether delivery flags conflict within an order.

The checked-in table passes the grain check: 99,441 rows and 99,441 unique orders. GMV is explicitly defined as gross order-level payment value across all statuses because refund data is not available. Delivery metrics use delivered orders with a recorded duration, so open or canceled orders do not dilute the late rate.

This check is intentionally visible because an intelligent Agent cannot compensate for incorrect metric definitions.

## 9. Evaluation strategy

### Layer A — deterministic smoke test

Run:

```bash
python -m evaluation.smoke_test
```

Purpose: verify the prepared dataset loads and every analytics function executes without an LLM or API cost.

CI also runs `python -m evaluation.agent_loop_test` with a fake model response sequence. This checks the multi-step orchestration path—tool call, observation, second tool call, final answer—without network access or API spend.

### Layer B — Agent evaluation

Run:

```bash
python evaluation/evaluate.py
```

The current question set covers 25 representative cases across sales, delivery, customer, region, product, payment, comparison, root-cause, out-of-scope and hallucination-guardrail scenarios.

Primary first metric: expected Tool selection coverage.

Future metrics:

- grounded-answer correctness;
- multi-step completion rate;
- unsupported-claim rate;
- latency;
- token/API cost.

## 10. What I would say in an interview

### 30-second version

I originally built Olist as a traditional ecommerce data-analysis dashboard. I found that dashboards still require users to know which metric to inspect, so I redesigned it as an AI Business Analyst Agent. I encapsulated sales, customer, regional and delivery analysis into deterministic Python tools. The LLM understands the business question and chooses which tools to call, while the actual metrics are calculated from the real dataset. For diagnostic questions it can perform multiple analysis steps before giving a conclusion and recommended action.

### What was the most important design decision?

I separated reasoning from calculation. The LLM decides what evidence it needs, but it does not invent or calculate business metrics itself. That reduces hallucination risk and makes the result traceable.

### Why did you not use RAG?

Because the main problem is calculation over structured ecommerce data, not document retrieval. RAG would be appropriate only if I later added business policies, metric definitions or other unstructured knowledge.

### How is this different from your first version?

The first version used keyword matching and fixed templates, so it was closer to a rule-based analytical assistant. The upgraded version uses real LLM Tool Calling, supports multiple tool rounds, keeps conversation context and exposes an execution trace.

## 11. Resume-ready positioning

**Olist AI Business Analyst Agent | Personal Project**

- Redesigned a Python/Streamlit ecommerce analytics dashboard into a tool-calling AI Business Analyst Agent over Olist order data, enabling natural-language analysis across sales, customers, regions and fulfilment.
- Encapsulated deterministic Pandas analytics as callable tools and implemented a multi-step agent loop for KPI decomposition and root-cause investigation while grounding numeric claims in tool outputs.
- Added conversational context, visible execution traces, dataset-grain checks and a 25-question evaluation set covering tool selection, diagnostic scenarios and hallucination guardrails.

## 12. Current boundary

This is a portfolio prototype, not a production analytics platform. It intentionally does not yet include a production database, role-based access control, enterprise observability, RAG, fine-tuning or multi-agent orchestration.

