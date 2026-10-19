from tfidf_retrieval import search_conditions_tfidf


def test_tfidf_returns_results():
    results = search_conditions_tfidf(
        "runny nose and sneezing"
    )

    assert len(results) > 0


def test_tfidf_top_k():
    results = search_conditions_tfidf(
        "fever and cough",
        top_k=2
    )

    assert len(results) <= 2


def test_tfidf_similarity_scores():
    results = search_conditions_tfidf(
        "painful urination"
    )

    for result in results:
        assert 0 < result["similarity_score"] <= 1


def test_tfidf_source_retrieval():
    results = search_conditions_tfidf(
        "common cold runny nose",
        top_k=6
    )

    common_cold = next(
        result
        for result in results
        if result["condition_id"] == "common_cold"
    )

    source_ids = [
        source["id"]
        for source in common_cold["sources"]
    ]

    assert "healthhub_common_cold" in source_ids