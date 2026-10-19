
import pytest

from knowledge_base.kb_service import retrieve_knowledge


def test_tfidf_service():
    response = retrieve_knowledge(
        "fever and body aches",
        method="tfidf"
    )

    assert response["kb_version"] == "1.2.0"
    assert response["method"] == "tfidf"
    assert len(response["results"]) > 0


def test_keyword_service():
    response = retrieve_knowledge(
        "painful urination",
        method="keyword"
    )

    assert response["method"] == "keyword"

    condition_ids = {
        result["condition_id"]
        for result in response["results"]
    }

    assert "urinary_tract_infection" in condition_ids


def test_invalid_method():
    with pytest.raises(ValueError):
        retrieve_knowledge(
            "headache",
            method="invalid"
        )


def test_empty_query():
    with pytest.raises(ValueError):
        retrieve_knowledge("")


def test_top_k_limit():
    response = retrieve_knowledge(
        "runny nose and sneezing",
        top_k=1
    )

    assert len(response["results"]) <= 1
