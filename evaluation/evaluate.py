from __future__ import annotations

import json
from pathlib import Path

from agent.agent import OlistBusinessAgent
from data.loader import load_analysis_data


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / "evaluation" / "questions.json").read_text(encoding="utf-8"))
    df = load_analysis_data(root / "analysis_table.csv")
    agent = OlistBusinessAgent(df)

    rows = []
    for case in cases:
        result = agent.ask(case["question"])
        used_tools = [step.name for step in result.steps if step.step_type == "tool_call"]
        expected = case["expected_tools"]

        if expected:
            tool_match = all(tool in used_tools for tool in expected)
        else:
            tool_match = len(used_tools) == 0

        rows.append(
            {
                "category": case["category"],
                "question": case["question"],
                "expected_tools": expected,
                "used_tools": used_tools,
                "tool_match": tool_match,
                "answer": result.answer,
            }
        )
        print(f"[{ 'PASS' if tool_match else 'FAIL' }] {case['question']}")
        print(f"  expected={expected} used={used_tools}")

    output_path = root / "evaluation" / "results.json"
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    passed = sum(row["tool_match"] for row in rows)
    print(f"\nTool selection score: {passed}/{len(rows)} = {passed / len(rows):.1%}")
    print(f"Detailed results written to {output_path}")


if __name__ == "__main__":
    main()
