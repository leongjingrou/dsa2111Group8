# V2 learned-stopping visualizations

- `ebm-split-overview.png` maps policy accuracy, urgent recall, question count,
  and stopping distribution across train, validation and test.
- `ebm-sample-maps.png` maps five live checkpoints for selected test cases. The
  green tile is the controller's chosen stop. The examples deliberately include
  an incorrect early-stop case.

Matching Laya PNGs are produced by the same renderer after the GPU fine-tune
writes `laya_sequential_metrics.json` and `laya_sequential_decisions.jsonl`.
They are not generated from the old final-answer-only Laya artifact because it
does not contain the intermediate confidence states required for a valid stop
policy audit.
