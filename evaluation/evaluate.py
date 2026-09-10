from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from agent.agent import OlistBusinessAgent
from data.loader import load_analysis_data


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cases = json.loads(
        (root / "evaluation" / "questions.json").read_text(encoding="utf-8")
    )
    df = load_analysis_data(root / "analysis_table.csv")
    agent = OlistBusinessAgent(df)

    rows = []
    category_scores: dict[str, list[bool]] = defaultdict(list)

    for case in cases:
        result = agent.ask(case["question"])
        used_tools = [
            step.name for step in result.steps if step.step_type == "tool_call"
        ]
        expected = case["expected_tools"]

        if expected:
            tool_match = all(tool in used_tools for tool in expected)
        else:
            tool_match = len(used_tools) == 0

        category_scores[case["category"]].append(tool_match)

        rows.append(
            {
                "category": case["category"],
                "question": case["question"],
                "expected_tools": expected,
                "used_tools": used_tools,
                "tool_call_count": len(used_tools),
                "multi_step": len(used_tools) > 1,
                "tool_match": tool_match,
                "execution_error": result.error,
                "answer": result.answer,
            }
        )

        status = "PASS" if tool_match and not result.error else "FAIL"
        print(f"[{status}] {case['question']}")
        print(f"  expected={expected} used={used_tools}")
        if result.error:
            print(f"  error={result.error}")

    output_path = root / "evaluation" / "results.json"
    output_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    passed = sum(row["tool_match"] and not row["execution_error"] for row in rows)
    errored = sum(bool(row["execution_error"]) for row in rows)
    multi_step = sum(row["multi_step"] for row in rows)

    print(f"\nOverall tool-selection score: {passed}/{len(rows)} = {passed / len(rows):.1%}")
    print(f"Execution errors: {errored}/{len(rows)}")
    print(f"Questions using >1 tool call: {multi_step}/{len(rows)}")

    print("\nCategory scores:")
    for category, scores in sorted(category_scores.items()):
        score = sum(scores) / len(scores)
        print(f"  {category}: {sum(scores)}/{len(scores)} = {score:.1%}")

    print(f"\nDetailed results written to {output_path}")


if __name__ == "__main__":
    main()
