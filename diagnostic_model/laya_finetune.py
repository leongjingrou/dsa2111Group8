import argparse
import json
from pathlib import Path

from diagnostic_model.laya_model import (
    OUT_OF_SCOPE,
    EVIDENCE_MODES,
    compare_ebm,
    consultation_state,
    evaluate_laya,
    questions,
)
from diagnostic_model.scenario_data import (
    load_scenarios,
    regenerate_scenarios,
    scenario_split,
    scenario_targets,
)


def write_rows(data, folder, test_size=0.25, evidence_mode="conversation"):
    train_index, test_index = scenario_split(data, test_size=test_size)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, index in (("train", train_index), ("test", test_index)):
        path = folder / f"laya_{name}.jsonl"
        with path.open("w") as file:
            rows = data.loc[index]
            escalation = scenario_targets(rows)["escalation"]
            for (_, row), urgent in zip(rows.iterrows(), escalation):
                condition = row["condition_id"] if row["scope_status"] == "in_scope" else OUT_OF_SCOPE
                item = {
                    "state": consultation_state(row, evidence_mode=evidence_mode),
                    "questions": questions(),
                    "expected": {
                        "condition": condition,
                        "escalation": "urgent" if urgent else "routine",
                    },
                }
                file.write(json.dumps(item) + "\n")
        paths.append(path)
    return paths


if __name__ == "__main__":
    import laya
    from huggingface_hub import snapshot_download
    from laya.train import TrainConfig, finetune

    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostic_model/artifacts"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--test-size", type=float, default=0.25,
        help="Fraction reserved for stratified testing; use 0.5 for a 50/50 experiment.",
    )
    parser.add_argument("--train-encoder", action="store_true")
    parser.add_argument("--trace-conversation", action="store_true")
    parser.add_argument("--evidence-mode", choices=EVIDENCE_MODES, default="conversation")
    parser.add_argument(
        "--generate-scenarios",
        action="store_true",
        help="Regenerate the scenario CSV with data/generate.py before fine-tuning.",
    )
    args = parser.parse_args()
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")

    data = (
        regenerate_scenarios(args.data)
        if args.generate_scenarios
        else load_scenarios(args.data)
    )
    train_path, test_path = write_rows(
        data,
        args.output_dir,
        test_size=args.test_size,
        evidence_mode=args.evidence_mode,
    )
    base = Path(
        snapshot_download("convaiinnovations/laya", allow_patterns=["typed-decisions/*"])
    ) / "typed-decisions"
    config = TrainConfig(
        epochs=args.epochs,
        micro_batch=2 if args.train_encoder else 8,
        grad_accum=8 if args.train_encoder else 2,
        encoder_lr=1e-5,
        head_lr=5e-4,
        loss="soft-ce",
        freeze_encoder=not args.train_encoder,
        shuffle_options=("choice",),
        calib_frac=0.15,
        eval_data=str(test_path),
        seed=2111,
        gradient_checkpointing=args.train_encoder,
        log_every=5,
    )
    name = "laya_full" if args.train_encoder else "laya_head"
    summary = finetune(str(train_path), str(base), str(args.model_dir), config, device="cuda")
    (args.output_dir / f"{name}_finetune.json").write_text(json.dumps(summary, indent=2))

    model = laya.load(str(args.model_dir), device="cuda")
    report = evaluate_laya(
        data,
        args.output_dir,
        model,
        name=name,
        model_name=str(args.model_dir),
        trace_conversation=args.trace_conversation,
        test_size=args.test_size,
        evidence_mode=args.evidence_mode,
    )
    comparison = compare_ebm(
        data,
        args.output_dir,
        report,
        args.output_dir,
        name=f"{name}_comparison",
        test_size=args.test_size,
    )
    print(json.dumps(comparison, indent=2))
