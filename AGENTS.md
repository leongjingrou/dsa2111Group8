# Project context

- Repository: https://github.com/leongjingrou/dsa2111Group8.
- The user is responsible for **item 4: Safety layer lead** in the team role breakdown.
- Their focus is **red-flag rules and escalation**.
- Listed deliverables: versioned rule config; scope limits covering age, pregnancy, out-of-scope conditions, and contradictions; rule tests against vignettes; and a log of which rule fired.
- The screenshot associates this item with **Role 2** in its rightmost column; that column's heading is cropped.
- See `references/team-role-breakdown.md` for the transcribed breakdown and `references/team-role-breakdown.png` for the source screenshot.

## Task 4 implementation

- Standalone work began on `codex/safety-layer`. The user then requested complete repository integration. Local branch `codex/safety-integration` is based on the complete `origin/main` at `9ee31c2`.
- `safety_layer/` is a standalone Python standard-library package. Read its `README.md` and `INTEGRATION.md` before changing its API.
- Entry points: `SafetyEngine.evaluate`, `guard_decision`, `guard_model_call`, and `python3 -m safety_layer demo`.
- Rules and source provenance: `safety_layer/config/rules.v1.json`. Age 18 to 64 and pregnancy exclusions are configurable project assumptions, not MOH eligibility rules. Clinical review is pending.
- Never turn missing or uncertain findings into negative answers, or let diagnostic output override a positive emergency rule.
- Run `python3 -m unittest discover -s safety_layer/tests` after safety changes. Saved fictional-vignette results and audit events are under `safety_layer/evaluation/v1.0.0/`.
- The safety layer is now wired into `LLMDrivenConsultation`, the conversation API and `/api/diagnose`, with SQLite session persistence and backend/API end-to-end tests. See `backend/README.md`.
- The user clarified that the frontend is unfinished; keep frontend work out of this task. Its source has not been modified.
- Deliver this implementation on `codex/safety-integration`. The user authorized a push after verification on 26 October 2026. Keep the commit message limited to the layer name, test-suite explanation and passing case counts.
- Verified tests use the clean Homebrew Python environment `.venv-e2e`, avoiding the system Anaconda pytest/readline crash.
- The user requires readable code, comments and structure: use names that describe the domain, focused functions, explicit control flow and comments explaining safety decisions. See `backend/REVIEW_GUIDE.md` for the review order.
