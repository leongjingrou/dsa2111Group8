# Knowledge Base Changelog

## Version 1.2.0

### Changes
- Added medical descriptions for all six conditions.
- Added source references from HealthHub Singapore and SingHealth.
- Replaced pediatric-specific Gastroenteritis and UTI references with more general patient-information articles.
- Preserved source IDs for compatibility with existing retrieval functions.
- Kept all medical content marked as pending review.

### Safety
- Red-flag information has not yet been curated or reviewed.
- Retrieval results must not be treated as diagnoses.
- Escalation decisions remain the responsibility of the independent safety layer.

### Evaluation
- Retained the six synthetic retrieval queries.
- No changes to symptom keywords or retrieval algorithms since v1.1.0.
- Retrieval performance is expected to remain unchanged.

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