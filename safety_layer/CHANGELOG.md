# Safety policy changelog

## 1.1.0 — 27 October 2026

- Reduced the mandatory pre-model checklist to a compact set of immediately
  life-threatening warning signs. Other red flags remain active whenever they
  are reported and can be explored by the adaptive consultation flow.
- Grouped the mandatory warning signs into one patient-facing question that
  accepts either free text or an explicit "none" response.
- Kept grouped negative screening facts in the safety record without injecting
  them as simultaneous first-step diagnostic-model features.
- Model-only urgent labels now wait for the learned stopping policy; explicit
  emergency findings still stop immediately.
- Clinical review status remains **pending**; this is a conversational prototype
  policy change, not a validated clinical screening recommendation.

## Backend integration — 26 October 2026

- Integrated mandatory pre-model and post-model checks into both consultation
  APIs, with explicit screening, guarded explanation, persisted session state,
  retry handling and end-to-end tests.
- Separated result models, patient-fact normalization, policy validation and rule
  evaluation into focused modules. Clinical policy predicates remain at 1.0.0.
- Added regressions for malformed model actions during scope referral and for
  equivalent negative pregnancy values across Profile and EvidenceState. A
  positive or uncertain pregnancy finding still blocks routine assessment.
- See `../backend/REVIEW_GUIDE.md` and `../backend/evaluation/REPORT.md` for the
  review order and current software validation evidence.

## 1.0.0 — 26 October 2026

- Added deterministic red-flag and combination rules, independent of the LLM and
  diagnostic model. All matching rules are retained; the highest urgency wins.
- Added configurable adult age scope, pregnancy handling, six-condition scope,
  unknown-finding referral, missing-data handling and contradiction preservation.
- Added pre-model and post-model guards compatible with the inspected team
  Profile, EvidenceState and DecisionRecord interfaces.
- Added source URLs and access dates, semantic policy version and SHA-256
  fingerprint, fixed referral messages and JSONL audit events.
- Added 64 fictional vignette cases and additional guard, input, configuration
  and audit tests. Saved software acceptance evidence under `evaluation/v1.0.0/`.
- Clinical review status: **pending**. The age range and pregnancy exclusions are
  explicit project policies. Exact rule mappings and routing are not clinically
  validated.
