import json
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


@lru_cache(maxsize=None)
def load_json(filename):
    """Load immutable knowledge-base documents once per serving process."""
    with open(DATA_DIR / filename, encoding="utf-8") as f:
        return json.load(f)


def search_conditions(query):
    kb = load_json("conditions.json")
    sources = load_json("sources.json")["sources"]

    source_lookup = {
        source["id"]: source
        for source in sources
    }

    query = query.lower().strip()
    results = []

    for condition in kb["conditions"]:
        matched = [
            keyword
            for keyword in condition["keywords"]
            if keyword.lower() in query
        ]

        if not matched:
            continue

        results.append({
            "condition_id": condition["id"],
            "name": condition["name"],
            "matched_keywords": matched,
            "description": condition["description"],
            "review_status": condition["review_status"],
            "sources": [
                source_lookup[source_id]
                for source_id in condition["source_ids"]
                if source_id in source_lookup
            ],
            "kb_version": kb["version"]
        })

    return results


if __name__ == "__main__":
    query = "I have a runny nose and sore throat"

    results = search_conditions(query)

    print(json.dumps(results, indent=2))
