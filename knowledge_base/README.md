# Knowledge Base — DSA2111 Group 8

## Overview

This component provides a version-controlled medical knowledge base for the AI-Assisted Telemedicine Routing and Escalation project.

It supports retrieval of relevant medical reference entries from structured JSON data using keyword matching and TF-IDF cosine similarity.

The component is designed to provide supporting information to other system components. It does not independently diagnose patients or make clinical escalation decisions.

## Knowledge Base Contents

Current version: **1.2.0**

The knowledge base contains six conditions:

1. Common Cold
2. Influenza
3. Gastroenteritis
4. Migraine
5. Allergic Rhinitis
6. Urinary Tract Infection

Each condition includes symptom keywords, a description, source references, review metadata, and a reserved field for red flags.

Medical references are sourced from HealthHub Singapore and SingHealth. All entries currently remain pending review.

## File Structure

```text
knowledge_base/
├── data/
│   ├── conditions.json
│   ├── sources.json
│   └── version.json
├── evaluation/
│   ├── queries.json
│   ├── evaluate.py
│   ├── baseline_v1.txt
│   └── baseline_v1_1.txt
├── __init__.py
├── retrieval.py
├── tfidf_retrieval.py
├── kb_service.py
├── test_retrieval.py
├── test_tfidf_retrieval.py
├── test_sources.py
├── test_kb_service.py
├── CHANGELOG.md
└── README.md
```

## Dependencies

Python 3.13 or another compatible Python 3 version.

Install dependencies from the repository root:

```powershell
python -m pip install scikit-learn pytest
```

## Retrieval Methods

### Keyword Retrieval

Performs case-insensitive exact-phrase matching against predefined symptom keywords.

### TF-IDF Retrieval

Uses TF-IDF vectors with unigrams and bigrams, followed by cosine similarity, to rank relevant condition entries.

Similarity scores measure textual relevance and must not be interpreted as probabilities of disease.

## Integration Interface

Import the service from the repository root:

```python
from knowledge_base.kb_service import retrieve_knowledge

response = retrieve_knowledge(
    "I have fever, cough and body aches",
    method="tfidf",
    top_k=3
)

print(response)
```

Parameters:

- `query`: Non-empty symptom description.
- `method`: `"tfidf"` (default) or `"keyword"`.
- `top_k`: Positive integer specifying the maximum number of returned entries.

The response contains:

- `kb_version`
- `method`
- `query`
- `results`
- `disclaimer`

Retrieved entries include condition identifiers, names, review statuses, source references, and version information. TF-IDF results additionally include similarity scores.

## Testing

Run all tests from the repository root:

```powershell
python -m pytest knowledge_base/ -v -p no:cacheprovider
```

Current result: **17 tests passed**.

## Evaluation

Run:

```powershell
python knowledge_base/evaluation/evaluate.py
```

Results:

| Metric | v1.0.0 | v1.1.0 |
|---|---|---|
| Keyword Top-1 Accuracy | 83.33% | 100% |
| Keyword Recall@3 | 83.33% | 100% |
| TF-IDF Top-1 Accuracy | 100% | 100% |
| TF-IDF Recall@3 | 100% | 100% |

The evaluation uses six synthetic queries. It measures retrieval of expected knowledge-base entries, not diagnostic accuracy or clinical safety.

The keyword method returns matches in dataset order rather than by relevance score.

## Version Control

The knowledge base is version-controlled using Git and version metadata.

- **v1.0.0:** Initial structured knowledge base and retrieval baseline.
- **v1.1.0:** Added alternative symptom phrases and demonstrated an update-and-retest improvement.
- **v1.2.0:** Added medical descriptions and source references.

Changes are documented in `CHANGELOG.md`.

## Safety and Limitations

- Retrieved conditions are informational matches, not diagnoses.
- All medical entries currently have pending review status.
- Empty `red_flags` fields do not indicate the absence of medical emergencies.
- The independent safety and escalation component must handle high-risk situations.
- TF-IDF relies on vocabulary overlap and may fail on unfamiliar descriptions or synonyms.
- The six-query evaluation is too small to establish clinical performance.
- The current prototype is not validated for real-world patient care.

## Integration Responsibilities

The knowledge base provides reference retrieval only.

The diagnostic component may use retrieved information as supporting context, subject to review and validation.

The safety and escalation component must independently assess urgent warning signs and must not rely solely on retrieved conditions or similarity scores.
