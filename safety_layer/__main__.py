"""Offline safety-layer evaluation and a reproducible task-4 demonstration."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .engine import DEFAULT_POLICY, AuditLogError, SafetyEngine
from .vignettes import VIGNETTES, load_vignettes, materialize_case


def run_demo(engine, output, vignettes=VIGNETTES):
    dataset = load_vignettes(vignettes)
    results = []
    for case in dataset["cases"]:
        decision = engine.evaluate(
            materialize_case(dataset, case), **case.get("kwargs", {})
        )
        fired = [hit.rule_id for hit in decision.triggered_rules]
        passed = decision.disposition == case["expected"] and set(fired) == set(
            case["rules"]
        )
        results.append(
            {
                "case_id": case["id"],
                "vignette": case["vignette"],
                "expected": case["expected"],
                "actual": decision.disposition,
                "expected_rules": case["rules"],
                "triggered_rules": fired,
                "passed": passed,
                "event_id": decision.event_id,
            }
        )
    hits = Counter(rule for result in results for rule in result["triggered_rules"])
    clinical = set(engine.clinical_rule_ids)
    report = {
        "policy_version": engine.version,
        "policy_sha256": engine.policy_sha256,
        "cases": len(results),
        "passed": sum(r["passed"] for r in results),
        "clinical_rules_exercised": len(clinical & set(hits)),
        "clinical_rules_total": len(clinical),
        "rule_hit_counts": dict(sorted(hits.items())),
        "limitation": "Agreement with software acceptance expectations, not clinical validation.",
        "results": results,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "vignette_results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Safety-layer vignette evaluation",
        "",
        f"Policy: {engine.version} (`{engine.policy_sha256}`).",
        "",
        f"Passed: **{report['passed']}/{report['cases']}** software acceptance vignettes.",
        f"Clinical rules exercised: **{report['clinical_rules_exercised']}/{len(clinical)}**.",
        "",
        report["limitation"],
        "",
        "| Case | Expected | Actual | Pass |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r['case_id']} | {r['expected']} | {r['actual']} | {'yes' if r['passed'] else 'NO'} |"
        for r in results
    ]
    (output / "evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Evaluate all fictional acceptance vignettes")
    demo.add_argument("--output-dir", type=Path, default=Path("safety_layer/output"))
    demo.add_argument("--vignettes", type=Path, default=VIGNETTES)
    single = sub.add_parser("evaluate", help="Evaluate a structured JSON profile")
    single.add_argument("input", type=Path)
    single.add_argument("--condition", default=None)
    single.add_argument("--audit-log", type=Path, required=True)
    for command in (demo, single):
        command.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    args = parser.parse_args()
    if args.command == "demo":
        engine = SafetyEngine(args.policy, audit_log=args.output_dir / "audit.jsonl")
        report = run_demo(engine, args.output_dir, args.vignettes)
        print(
            f"{report['passed']}/{report['cases']} vignettes passed; "
            f"{report['clinical_rules_exercised']}/{report['clinical_rules_total']} clinical rules exercised."
        )
        print(f"Results and append-only audit log: {args.output_dir.resolve()}")
        return 0 if report["passed"] == report["cases"] else 1
    profile = json.loads(args.input.read_text(encoding="utf-8"))
    decision = SafetyEngine(args.policy, args.audit_log).evaluate(
        profile, candidate_condition=args.condition
    )
    print(json.dumps(decision.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditLogError as exc:
        print(
            json.dumps(
                {"audit_error": str(exc), "safety": exc.decision.to_dict()}, indent=2
            )
        )
        print(
            "Safety advice is available, but audit storage must be repaired before continuing.",
            file=sys.stderr,
        )
        raise SystemExit(2)
