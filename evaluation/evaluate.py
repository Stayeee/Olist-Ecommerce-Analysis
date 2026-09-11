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
        expected_multi_step = len(expected) > 1
        multi_step_match = not expected_multi_step or len(used_tools) > 1
        requires_business_structure = case["category"] not in {"out_of_scope", "guardrail"}
        normalized_answer = result.answer.lower()
        structure_match = not requires_business_structure or all(
            heading in normalized_answer
            for heading in ("finding", "evidence", "interpretation", "action")
        )

        rows.append(
            {
                "category": case["category"],
                "question": case["question"],
                "expected_tools": expected,
                "used_tools": used_tools,
                "tool_call_count": len(used_tools),
                "multi_step": len(used_tools) > 1,
                "multi_step_match": multi_step_match,
                "tool_match": tool_match,
                "structure_match": structure_match,
                "execution_error": result.error,
                "model": result.model,
                "latency_ms": result.latency_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "answer": result.answer,
            }
        )

        status = (
            "PASS"
            if tool_match and multi_step_match and structure_match and not result.error
            else "FAIL"
        )
        print(f"[{status}] {case['question']}")
        print(f"  expected={expected} used={used_tools}")
        if result.error:
            print(f"  error={result.error}")

    output_path = root / "evaluation" / "results.json"
    output_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    passed = sum(
        row["tool_match"]
        and row["multi_step_match"]
        and row["structure_match"]
        and not row["execution_error"]
        for row in rows
    )
    errored = sum(bool(row["execution_error"]) for row in rows)
    multi_step = sum(row["multi_step"] for row in rows)
    structured = sum(row["structure_match"] for row in rows)
    total_input_tokens = sum(row["input_tokens"] for row in rows)
    total_output_tokens = sum(row["output_tokens"] for row in rows)
    avg_latency_ms = sum(row["latency_ms"] for row in rows) / len(rows)

    print(f"\nOverall tool-selection score: {passed}/{len(rows)} = {passed / len(rows):.1%}")
    print(f"Execution errors: {errored}/{len(rows)}")
    print(f"Questions using >1 tool call: {multi_step}/{len(rows)}")
    print(f"Answer-structure score: {structured}/{len(rows)} = {structured / len(rows):.1%}")
    print(f"Average latency: {avg_latency_ms:.0f} ms")
    print(f"Total tokens: {total_input_tokens:,} input / {total_output_tokens:,} output")

    print("\nCategory scores:")
    for category, scores in sorted(category_scores.items()):
        score = sum(scores) / len(scores)
        print(f"  {category}: {sum(scores)}/{len(scores)} = {score:.1%}")

    print(f"\nDetailed results written to {output_path}")


if __name__ == "__main__":
    main()

