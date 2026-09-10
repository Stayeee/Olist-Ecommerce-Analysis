from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agent.agent import OlistBusinessAgent
from data.loader import load_analysis_data


class FakeResponses:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self._outputs = [
            SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="analyze_sales_trend",
                        arguments='{"months": 6}',
                        call_id="trend-call",
                    )
                ],
                output_text="",
                usage=SimpleNamespace(input_tokens=100, output_tokens=20),
            ),
            SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="analyze_recent_change_drivers",
                        arguments='{"dimension": "state", "limit": 5}',
                        call_id="driver-call",
                    )
                ],
                output_text="",
                usage=SimpleNamespace(input_tokens=120, output_tokens=30),
            ),
            SimpleNamespace(
                output=[],
                output_text=(
                    "Finding: GMV declined in the latest complete-month comparison. "
                    "Evidence: order and AOV contributions were calculated by tools."
                ),
                usage=SimpleNamespace(input_tokens=80, output_tokens=40),
            ),
        ]

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self._outputs.pop(0)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    df = load_analysis_data(root / "analysis_table.csv")
    agent = OlistBusinessAgent(df=df, api_key="offline-test-key")
    fake_responses = FakeResponses()
    agent.client = SimpleNamespace(responses=fake_responses)

    result = agent.ask("Why did sales decline recently?")
    called_tools = [step.name for step in result.steps if step.step_type == "tool_call"]

    assert result.error is None
    assert called_tools == ["analyze_sales_trend", "analyze_recent_change_drivers"]
    assert len(fake_responses.requests) == 3
    assert "Finding:" in result.answer
    assert result.tool_rounds == 2
    assert result.input_tokens == 300
    assert result.output_tokens == 90
    assert result.latency_ms >= 0

    second_round_input = fake_responses.requests[1]["input"]
    assert any(
        isinstance(item, dict)
        and item.get("type") == "function_call_output"
        and item.get("call_id") == "trend-call"
        for item in second_round_input
    )

    print("Offline multi-step agent loop passed.")


if __name__ == "__main__":
    main()

