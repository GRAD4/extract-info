#!/usr/bin/env python3
"""
RAG-based company classification (runnable example).

Pipeline:
  1. Load a manufacturing capability taxonomy (categories.json)
  2. Read company website text (example_company.txt)
  3. Chunk the text and build a local Annoy vector index
  4. For each taxonomy subcategory, retrieve the most relevant snippet
  5. Assemble a prompt and call an LLM (Ollama by default)
  6. Parse the structured output into category / subcategory pairs

Also demonstrates certification extraction via normalized keyword matching.

Usage:
    python example.py
    LLM_PROVIDER=bedrock python example.py
"""
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from llm import LLMClient
from prompt import PromptBuilder
from rag import RAGIndex, chunk_text

load_dotenv()

DATA = Path(__file__).parent / "data"


def load_taxonomy() -> dict[str, list[str]]:
    with open(DATA / "categories.json") as f:
        return json.load(f)


def load_certifications() -> list[str]:
    with open(DATA / "certifications.json") as f:
        return json.load(f)


def extract_certifications(text: str, cert_list: list[str]) -> list[str]:
    """
    Keyword-based certification extraction.

    Normalizes the text by removing spaces and hyphens, then checks whether
    each certification acronym appears as a substring. This is simple but
    effective for well-known standard codes (ISO 9001, AS9100D, etc.).
    """
    norm = re.sub(r"[\s\-]+", "", text).upper()
    found = []
    for cert in cert_list:
        key = re.sub(r"[\s\-]+", "", cert).upper()
        if key in norm:
            found.append(cert)
    return found


def parse_specifications(llm_output: str, taxonomy: dict[str, list[str]]) -> list[tuple[str, str]]:
    """
    Extract valid category/subcategory pairs from LLM output lines.

    Only pairs that exist in the taxonomy are kept, which prevents hallucinated
    categories from leaking into the result.
    """
    valid = {
        (cat, sub)
        for cat, subs in taxonomy.items()
        for sub in subs
    }
    specs = []
    for line in re.findall(r"Specification\s*\d+\s*:\s*(.+)", llm_output, re.IGNORECASE):
        if " - " in line:
            cat, sub = line.split(" - ", 1)
            pair = (cat.strip(), sub.strip())
            if pair in valid:
                specs.append(pair)
    return specs


def main() -> None:
    taxonomy = load_taxonomy()
    certifications = load_certifications()

    company_name = "ACME Aerospace Parts Inc."
    text = (DATA / "example_company.txt").read_text()

    print(f"\n{'='*60}")
    print(f"  Classifying: {company_name}")
    print(f"{'='*60}\n")

    # ── 1. Build RAG index ────────────────────────────────────────────────────
    chunks = chunk_text(text, max_chars=400)
    labeled = [(f"chunk_{i}", c) for i, c in enumerate(chunks)]

    rag = RAGIndex()
    rag.build(labeled)
    print(f"[RAG] Index built over {len(labeled)} chunk(s).\n")

    # ── 2. Retrieve best snippet per subcategory ──────────────────────────────
    best: dict[tuple[str, str], tuple[str, str, float]] = {}
    for category, subcategories in taxonomy.items():
        for sub in subcategories:
            results = rag.query(sub, top_k=3)
            for label, snippet, sim in results:
                key = (category, sub)
                if key not in best or sim > best[key][2]:
                    best[key] = (label, snippet, sim)

    # Sort by similarity and keep top 10 to stay within token budget
    top_hits = sorted(best.values(), key=lambda x: -x[2])[:10]
    print(f"[RAG] Top {len(top_hits)} snippets selected.\n")

    # ── 3. Build prompt and call LLM ──────────────────────────────────────────
    pb = PromptBuilder(taxonomy, top_k=10)
    prompt = pb.build(company_name, top_hits)

    provider = os.getenv("LLM_PROVIDER", "ollama")
    llm = LLMClient(provider=provider)
    print(f"[LLM] Calling {provider} ({llm.model})…\n")
    response = llm.generate(prompt)

    print("=== LLM raw output ===")
    print(response)
    print()

    # ── 4. Parse and validate output ──────────────────────────────────────────
    specs = parse_specifications(response, taxonomy)
    certs = extract_certifications(text, certifications)

    print("=== Parsed specifications ===")
    for i, (cat, sub) in enumerate(specs, 1):
        print(f"  {i:2d}. {cat} — {sub}")

    print("\n=== Certifications detected ===")
    if certs:
        for cert in certs:
            print(f"  ✓ {cert}")
    else:
        print("  (none found)")


if __name__ == "__main__":
    main()
