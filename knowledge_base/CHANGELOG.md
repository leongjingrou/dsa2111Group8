# Knowledge Base Changelog

## Version 1.1.0

### Changes
- Added "nose is itchy" to the Allergic Rhinitis keywords.
- Added "eyes are watery" to the Allergic Rhinitis keywords.
- Improved support for alternative symptom phrasing.
- No clinical guidance or escalation rules were changed.

### Evaluation
- Reused the same six synthetic retrieval queries from v1.0.0.
- Compared keyword retrieval and TF-IDF retrieval.
- Results are stored in `evaluation/baseline_v1_1.txt`.

### Limitations
- The evaluation dataset contains only six synthetic queries.
- Results measure retrieval performance, not diagnostic accuracy.
- Medical content is still pending source validation.