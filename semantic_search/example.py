#!/usr/bin/env python3
"""
Multilingual semantic search with geo-filtering (runnable example).

Demonstrates:
  1. Building a FAISS index over company description embeddings
  2. Parsing natural language queries (semantic part + location + radius)
  3. Applying geographic pre-filtering:
       - Province code:  bounding-box filter (covers entire region)
       - City name:      Haversine radius filter
  4. Scoring candidates with cosine similarity
  5. Returning ranked results

Usage:
    python example.py

The example runs several pre-defined queries so you can see all strategies
in action without any external API.
"""
import json
from pathlib import Path

import numpy as np
from indexer import build_index
from query_parser import parse_query
from searcher import search
from sentence_transformers import SentenceTransformer

DATA = Path(__file__).parent / "data"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEMO_QUERIES = [
    "CNC machining aluminum in Montréal within 30 km",
    "usinage CNC aluminium Montréal",           # French, no explicit location keyword
    "TIG welding aerospace in QC",              # Province-level search
    "powder coating anodizing",                 # No location — all candidates
    "laser cutting sheet metal near Laval",
    "additive manufacturing metal 3D printing",
]


def print_results(query: str, results: list[dict]) -> None:
    print(f"\n{'─'*60}")
    print(f"  Query: \"{query}\"")
    print(f"{'─'*60}")
    if not results:
        print("  No results found.")
        return
    for r in results:
        dist = f"  {r['distance_km']:.1f} km" if "distance_km" in r else ""
        print(
            f"  [{r['score']:.3f}]{dist}  {r['name']}  ({r['city']}, {r['province']})"
        )
        caps = ", ".join(r["capabilities"][:3])
        if caps:
            print(f"           Capabilities: {caps}")


def main() -> None:
    with open(DATA / "companies.json") as f:
        companies = json.load(f)

    # Build the index (in production this would be pre-computed and loaded from disk)
    index, metas, embs = build_index(companies, model_name=MODEL_NAME)
    embs_array = np.array(embs, dtype=np.float32) if not isinstance(embs, np.ndarray) else embs
    model = SentenceTransformer(MODEL_NAME)

    print("\n" + "="*60)
    print("  Semantic search demo")
    print("="*60)

    for raw_query in DEMO_QUERIES:
        semantic_q, location, radius_km = parse_query(raw_query)
        print(f"\n  Parsed: semantic='{semantic_q}'  location={location!r}  radius={radius_km} km")

        results = search(
            query=semantic_q,
            location=location,
            radius_km=radius_km,
            companies=metas,
            embeddings=embs_array,
            model=model,
            top_k=5,
        )
        print_results(raw_query, results)


if __name__ == "__main__":
    main()
