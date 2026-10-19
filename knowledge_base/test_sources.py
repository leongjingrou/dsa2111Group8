
from knowledge_base.retrieval import load_json


def test_all_source_ids_exist():
    kb = load_json("conditions.json")
    sources = load_json("sources.json")["sources"]

    valid_source_ids = {
        source["id"] for source in sources
    }

    for condition in kb["conditions"]:
        for source_id in condition["source_ids"]:
            assert source_id in valid_source_ids


def test_source_ids_are_unique():
    sources = load_json("sources.json")["sources"]

    ids = [source["id"] for source in sources]

    assert len(ids) == len(set(ids))


def test_source_review_status():
    sources = load_json("sources.json")["sources"]

    valid_statuses = {
        "pending",
        "reviewed"
    }

    for source in sources:
        assert source["review_status"] in valid_statuses


def test_all_conditions_have_source_records():
    kb = load_json("conditions.json")

    for condition in kb["conditions"]:
        assert len(condition["source_ids"]) >= 1
