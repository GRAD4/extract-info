#!/usr/bin/env python3
"""
Neo4j Knowledge Graph Experiment (runnable example).

This script demonstrates:
  1. Loading company-capability data into Neo4j as a property graph
  2. Querying the graph with Cypher for structured retrieval
  3. Serialising graph results as LLM context
  4. Answering natural language questions grounded in graph data

Prerequisites:
  - A running Neo4j instance (see README for Docker instructions)
  - NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD set in .env
  - LLM_PROVIDER set to 'ollama' or 'bedrock'

Experimental note:
  This approach works well for explicit structural queries. For open-ended
  semantic search it requires an embedding layer on top, at which point a
  simpler relational DB + FAISS index achieves the same result.
  See the semantic_search/ module for that approach.
"""
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# knowledge_graph/ is self-contained; add its parent to path for shared llm module
sys.path.insert(0, str(Path(__file__).parent.parent / "rag_classification"))
from llm import LLMClient
from loader import load
from querier import GraphQuerier

DATA = Path(__file__).parent / "data"


def main() -> None:
    uri      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    user     = os.getenv("NEO4J_USER",     "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    with open(DATA / "companies.json") as f:
        companies = json.load(f)

    # ── 1. Load data into Neo4j ───────────────────────────────────────────────
    print("\n[1] Loading data into Neo4j…")
    load(companies, uri, user, password)

    querier = GraphQuerier(uri, user, password)

    # ── 2. Structured Cypher queries ──────────────────────────────────────────
    print("\n[2] Structured query: CNC Machining companies in QC")
    rows = querier.find_by_capability_and_province("CNC Machining", "QC")
    for r in rows:
        print(f"   {r['name']} ({r['city']}): {', '.join(r['capabilities'])}")

    print("\n[3] Companies certified with AS9100D")
    rows = querier.find_certified_by("AS9100D")
    for r in rows:
        print(f"   {r['name']} — {r['city']}, {r['province']}")

    print("\n[4] Capability co-occurrence (top pairs)")
    pairs = querier.capability_co_occurrence()
    for p in pairs[:5]:
        print(f"   '{p['cap_a']}' + '{p['cap_b']}': {p['co_count']} company/companies")

    # ── 3. Graph context → LLM ────────────────────────────────────────────────
    print("\n[5] Graph-as-context LLM demo")
    context = querier.build_context_for_query("CNC Machining", "QC")
    print("\nContext fed to LLM:")
    print(context)

    question = "Which company offers the most CNC machining subcategories and where is it located?"
    print(f"\nQuestion: {question}")

    try:
        provider = os.getenv("LLM_PROVIDER", "ollama")
        llm = LLMClient(provider=provider)
        answer = querier.answer_with_llm(question, context, llm.generate)
        print(f"\nLLM answer:\n{answer}")
    except requests.ConnectionError:
        print("\n[LLM skipped] Cannot connect to Ollama.")
        print("  → Start Ollama: ollama serve")
        print("  → Pull the model: ollama pull llama3.1")
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            print("\n[LLM skipped] Ollama returned 404 (model or endpoint not found).")
            print("  → Ensure Ollama is running and pull the model: ollama pull llama3.1")
        else:
            print(f"\n[LLM skipped] {e}")
    except Exception as e:
        print(f"\n[LLM skipped] {e}")
        print("  → Set LLM_PROVIDER and relevant credentials in .env to enable LLM answers.")

    querier.close()


if __name__ == "__main__":
    main()
