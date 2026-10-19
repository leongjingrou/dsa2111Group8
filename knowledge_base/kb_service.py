
from knowledge_base.retrieval import load_json, search_conditions
from knowledge_base.tfidf_retrieval import search_conditions_tfidf


def retrieve_knowledge(query, method="tfidf", top_k=3):
    """
    Retrieve relevant knowledge-base entries.

    Args:
        query: Patient symptom description.
        method: "tfidf" or "keyword".
        top_k: Maximum number of results.

    Returns:
        Dictionary containing KB metadata and results.

    Note:
        Results indicate textual relevance, not diagnosis.
        Clinical safety and escalation must be handled separately.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    if method not in ("tfidf", "keyword"):
        raise ValueError("method must be 'tfidf' or 'keyword'")

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    kb = load_json("conditions.json")

    if method == "tfidf":
        results = search_conditions_tfidf(
            query, top_k=top_k
        )
    else:
        results = search_conditions(query)[:top_k]

    return {
        "kb_version": kb["version"],
        "method": method,
        "query": query,
        "results": results,
        "disclaimer": (
            "Retrieval results are informational only. "
            "They are not diagnoses or clinical triage decisions."
        )
    }


if __name__ == "__main__":
    import json

    response = retrieve_knowledge(
        "I have fever, cough and body aches",
        method="tfidf",
        top_k=3
    )

    print(json.dumps(response, indent=2))
