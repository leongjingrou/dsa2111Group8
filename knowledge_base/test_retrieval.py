from retrieval import search_conditions


def test_matching_symptoms():
    results = search_conditions(
        "I have a runny nose and sore throat"
    )

    assert len(results) == 1
    assert results[0]["condition_id"] == "common_cold"


def test_no_matching_symptoms():
    results = search_conditions(
        "This sentence contains no matching symptoms"
    )

    assert results == []


def test_source_retrieval():
    results = search_conditions("runny nose")

    assert len(results) == 1

    source_ids = [
        source["id"]
        for source in results[0]["sources"]
    ]

    assert "healthhub_common_cold" in source_ids


def test_kb_version():
    results = search_conditions("sore throat")

    assert results[0]["kb_version"] == "1.0.0"