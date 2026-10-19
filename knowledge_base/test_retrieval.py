from retrieval import search_conditions


def test_matching_symptoms():
    results = search_conditions(
        "I have a runny nose and sore throat"
    )

    condition_ids = {
        result["condition_id"]
        for result in results
    }

    assert "common_cold" in condition_ids


def test_no_matching_symptoms():
    results = search_conditions(
        "This sentence contains no matching symptoms"
    )

    assert results == []


def test_source_retrieval():
    results = search_conditions("runny nose")

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


def test_kb_version():
    results = search_conditions("sore throat")

    assert results[0]["kb_version"] == "1.0.0"