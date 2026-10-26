import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from knowledge_base.retrieval import search_conditions
from knowledge_base.tfidf_retrieval import search_conditions_tfidf


def evaluate():
    queries_path = Path(__file__).parent / "queries.json"

    with open(queries_path, encoding="utf-8") as f:
        queries = json.load(f)

    methods = {
        "Keyword": search_conditions,
        "TF-IDF": search_conditions_tfidf
    }

    for method_name, search_function in methods.items():
        top1_correct = 0
        top3_correct = 0

        print(f"\n{method_name} Evaluation")
        print("-" * 50)

        for item in queries:
            results = search_function(item["query"])

            predicted_ids = [
                result["condition_id"]
                for result in results
            ]

            expected = item["expected_condition"]

            # Keyword retrieval has no explicit ranking.
            # Its current JSON order is used as the baseline.
            top1_hit = (
                len(predicted_ids) > 0
                and predicted_ids[0] == expected
            )

            top3_hit = expected in predicted_ids[:3]

            top1_correct += int(top1_hit)
            top3_correct += int(top3_hit)

            print(f"Query: {item['query']}")
            print(f"Expected: {expected}")
            print(f"Retrieved: {predicted_ids[:3]}")
            print()

        total = len(queries)

        print(f"Top-1 Accuracy: {top1_correct / total:.2%}")
        print(f"Recall@3: {top3_correct / total:.2%}")


if __name__ == "__main__":
    evaluate()