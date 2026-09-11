from __future__ import annotations

import json
import os
from time import perf_counter
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from agent.prompts import SYSTEM_PROMPT
from tools.registry import TOOL_SCHEMAS, build_tool_registry


@dataclass
class AgentStep:
    step_type: str
    name: str
    detail: str


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    error: str | None = None
    model: str | None = None
    latency_ms: int = 0
    tool_rounds: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class OlistBusinessAgent:
    """Multi-step business analysis agent using OpenAI Responses API tool calling."""

    def __init__(
        self,
        df: pd.DataFrame,
        model: str | None = None,
        api_key: str | None = None,
        max_tool_rounds: int = 6,
        client: Any | None = None,
    ) -> None:
        self.df = df
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
        if client is not None:
            self.client = client
        else:
            resolved_key = api_key or os.getenv("OPENAI_API_KEY")
            if not resolved_key:
                raise ValueError("OPENAI_API_KEY is required to run the AI agent.")
            from openai import OpenAI

            self.client = OpenAI(api_key=resolved_key)
        self.registry = build_tool_registry(df)
        self.max_tool_rounds = max_tool_rounds

    def ask(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> AgentResult:
        started_at = perf_counter()
        input_tokens = 0
        output_tokens = 0
        tool_rounds = 0

        def build_result(answer: str, steps: list[AgentStep] | None = None, error: str | None = None) -> AgentResult:
            return AgentResult(
                answer=answer,
                steps=steps or [],
                error=error,
                model=self.model,
                latency_ms=int((perf_counter() - started_at) * 1000),
                tool_rounds=tool_rounds,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        if not question.strip():
            return build_result(answer="Please enter a business question.")

        input_items: list[Any] = []
        for message in (history or [])[-6:]:
            role = message.get("role")
            content = message.get("content")
            if role in {"user", "assistant"} and content:
                input_items.append({"role": role, "content": content})
        input_items.append({"role": "user", "content": question.strip()})

        steps: list[AgentStep] = []

        for _ in range(self.max_tool_rounds):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    instructions=SYSTEM_PROMPT,
                    input=input_items,
                    tools=TOOL_SCHEMAS,
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                return build_result(
                    answer=(
                        "The AI model request failed, so no business conclusion was generated. "
                        "Check the API key, model name, network connection and account access."
                    ),
                    steps=steps,
                    error=message,
                )

            usage = getattr(response, "usage", None)
            input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

            function_calls = [
                item for item in response.output if item.type == "function_call"
            ]

            if not function_calls:
                answer = response.output_text or "I could not generate a grounded answer."
                return build_result(answer=answer, steps=steps)

            input_items.extend(response.output)
            tool_rounds += 1

            for call in function_calls:
                tool_name = call.name
                try:
                    arguments = json.loads(call.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}

                steps.append(
                    AgentStep(
                        step_type="tool_call",
                        name=tool_name,
                        detail=json.dumps(arguments, ensure_ascii=False),
                    )
                )

                tool = self.registry.get(tool_name)
                if tool is None:
                    result = {"error": f"Unknown tool: {tool_name}"}
                else:
                    try:
                        result = tool(**arguments)
                    except Exception as exc:
                        result = {"error": f"{type(exc).__name__}: {exc}"}

                serialized_result = json.dumps(
                    result,
                    ensure_ascii=False,
                    default=str,
                )

                steps.append(
                    AgentStep(
                        step_type="observation",
                        name=tool_name,
                        detail=serialized_result[:2500],
                    )
                )

                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": serialized_result,
                    }
                )

        return build_result(
            answer=(
                "The analysis reached the tool-call limit before producing a final answer. "
                "Try a narrower business question."
            ),
            steps=steps,
            error="max_tool_rounds_reached",
        )
