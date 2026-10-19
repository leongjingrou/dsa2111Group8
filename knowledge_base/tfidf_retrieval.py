import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from retrieval import load_json


def search_conditions_tfidf(query, top_k=3):
    kb = load_json("conditions.json")
    sources = load_json("sources.json")["sources"]

    source_lookup = {
        source["id"]: source
        for source in sources
    }

    conditions = kb["conditions"]

    documents = [
        " ".join([
            condition["name"],
            *condition["keywords"]
        ])
        for condition in conditions
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2)
    )

    document_vectors = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector, document_vectors
    )[0]

    ranked_indices = similarities.argsort()[::-1]

    results = []

    for index in ranked_indices[:top_k]:
        score = float(similarities[index])

        if score <= 0:
            continue

        condition = conditions[index]

        results.append({
            "condition_id": condition["id"],
            "name": condition["name"],
            "similarity_score": round(score, 4),
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
    query = "I have a runny nose and sneezing"

    results = search_conditions_tfidf(query)

    print(json.dumps(results, indent=2))