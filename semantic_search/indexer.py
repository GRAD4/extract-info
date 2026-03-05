"""
FAISS index builder for semantic supplier search.

Each company is represented by a single embedding computed from its
description and capability list. Metadata (name, location, capabilities)
is stored separately in a parallel list so it can be returned alongside
search results without re-querying a database.
"""
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def build_company_text(company: dict) -> str:
    """
    Concatenate all text fields of a company into a single string for embedding.

    In a production system this would also include scraped website text.
    Here we combine the description with capability names.
    """
    caps = " ".join(company.get("capabilities", []))
    certs = " ".join(company.get("certifications", []))
    return " ".join(filter(None, [company.get("description", ""), caps, certs]))


def build_index(
    companies: list[dict],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    index_path: Path | None = None,
    meta_path: Path | None = None,
    embs_path: Path | None = None,
) -> tuple[faiss.Index, list[dict], np.ndarray]:
    """
    Encode companies and build a FAISS inner-product index.

    Args:
        companies:   List of company dicts from companies.json.
        model_name:  HuggingFace sentence-transformers model.
        index_path:  If provided, the FAISS index is saved here.
        meta_path:   If provided, company metadata is pickled here.
        embs_path:   If provided, the embedding matrix is saved here (.npy).

    Returns:
        (index, metadata, embeddings)
    """
    model = SentenceTransformer(model_name)

    texts = [build_company_text(c) for c in companies]
    metas = [
        {
            "id":           c["id"],
            "name":         c["name"],
            "city":         c["city"],
            "province":     c["province"],
            "lat":          c["lat"],
            "lon":          c["lon"],
            "capabilities": c.get("capabilities", []),
            "certifications": c.get("certifications", []),
            "summary":      c.get("description", "")[:120],
        }
        for c in companies
    ]

    print(f"Encoding {len(texts)} companies with '{model_name}'…")
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embs = np.array(embs, dtype=np.float32)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on L2-normalised vecs = cosine sim
    index.add(embs)
    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

    if index_path:
        faiss.write_index(index, str(index_path))
    if meta_path:
        meta_path.write_bytes(pickle.dumps(metas))
    if embs_path:
        np.save(str(embs_path), embs)

    return index, metas, embs
