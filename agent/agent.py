from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from openai import OpenAI

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


class OlistBusinessAgent:
    """Minimal multi-step business analysis agent using OpenAI tool calling."""

    def __init__(
        self,
        df: pd.DataFrame,
        model: str | None = None,
        api_key: str | None = None,
        max_tool_rounds: int = 6,
    ) -> None:
        self.df = df
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.registry = build_tool_registry(df)
        self.max_tool_rounds = max_tool_rounds

    def ask(self, question: str, history: list[dict[str, str]] | None = None) -> AgentResult:
        if not question.strip():
            return AgentResult(answer="Please enter a business question.")

        input_items: list[dict[str, Any]] = []
        for message in (history or [])[-6:]:
            role = message.get("role")
            content = message.get("content")
            if role in {"user", "assistant"} and content:
                input_items.append({"role": role, "content": content})
        input_items.append({"role": "user", "content": question})

        steps: list[AgentStep] = []

        for _ in range(self.max_tool_rounds):
            response = self.client.responses.create(
                model=self.model,
                instructions=SYSTEM_PROMPT,
                input=input_items,
                tools=TOOL_SCHEMAS,
            )

            function_calls = [
                item for item in response.output if item.type == "function_call"
            ]

            if not function_calls:
                answer = response.output_text or "I could not generate a grounded answer."
                return AgentResult(answer=answer, steps=steps)

            input_items.extend(response.output)

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
                    except Exception as exc:  # keep the agent loop recoverable
                        result = {"error": f"{type(exc).__name__}: {exc}"}

                steps.append(
                    AgentStep(
                        step_type="observation",
                        name=tool_name,
                        detail=json.dumps(result, ensure_ascii=False, default=str)[:2500],
                    )
                )

                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )

        return AgentResult(
            answer=(
                "The analysis reached the tool-call limit before producing a final answer. "
                "Try a narrower business question."
            ),
            steps=steps,
        )
