"""Build domain centroids from example queries per action for CentroidRouter.
Usage:
  Put examples in data/labels/query_examples.json:
    [{"query": "...", "action_id": 0}, {"query": "...", "action_id": 1}, ...]
  Run: python scripts/build_centroids.py
  Output: data/processed/centroids.json
Requires: pip install sentence-transformers
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_LABELS = ROOT / "data" / "labels"
DATA_PROCESSED = ROOT / "data" / "processed"


def get_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        return None


def main() -> None:
    examples_path = DATA_LABELS / "query_examples.json"
    if not examples_path.exists():
        print("Create data/labels/query_examples.json with [{\"query\": \"...\", \"action_id\": 0}, ...]")
        DATA_LABELS.mkdir(parents=True, exist_ok=True)
        sample = [
            {"query": "What is the refund policy?", "action_id": 0},
            {"query": "How does the API authenticate?", "action_id": 1},
            {"query": "SELECT * FROM users WHERE active = 1", "action_id": 2},
            {"query": "def get_user(id):", "action_id": 4},
        ]
        with open(examples_path, "w") as f:
            json.dump(sample, f, indent=2)
        print("Wrote example file. Add more examples per action_id then re-run.")
        return

    model = get_embedder()
    if model is None:
        print("Install sentence-transformers: pip install sentence-transformers")
        return

    with open(examples_path) as f:
        examples = json.load(f)

    by_action: dict[int, list[list[float]]] = defaultdict(list)
    for ex in examples:
        q = ex.get("query", "")
        aid = ex.get("action_id", 0)
        vec = model.encode(q, convert_to_numpy=True).tolist()
        by_action[aid].append(vec)

    dim = None
    centroids = []
    for action_id in sorted(by_action.keys()):
        vecs = by_action[action_id]
        if not vecs:
            continue
        dim = dim or len(vecs[0])
        centroid = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]
        centroids.append({"action_id": action_id, "vector": centroid})

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "centroids.json"
    with open(out_path, "w") as f:
        json.dump({"dim": dim or 384, "centroids": centroids}, f, indent=2)
    print(f"Wrote {out_path} with {len(centroids)} centroids (dim={dim}).")


if __name__ == "__main__":
    main()
