from __future__ import annotations

from pathlib import Path

from agent.offline_demo import GUIDED_DEMO_QUESTIONS, run_guided_demo
from data.loader import load_analysis_data


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    df = load_analysis_data(root / "analysis_table.csv")

    for question in GUIDED_DEMO_QUESTIONS:
        result = run_guided_demo(question, df)
        assert result.error is None
        assert result.steps
        assert result.tool_rounds >= 1
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert all(
            heading in result.answer
            for heading in (
                "### Finding",
                "### Evidence",
                "### Interpretation",
                "### Recommended action",
            )
        )
        print(f"[PASS] {question}")

    unsupported = run_guided_demo("Forecast exact profit next year.", df)
    assert unsupported.error == "unsupported_guided_demo_question"
    print("All guided no-key demo cases passed.")


if __name__ == "__main__":
    main()
