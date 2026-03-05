"""
Local RAG index built on Annoy (Approximate Nearest Neighbours Oh Yeah).

The index stores text chunks as vector embeddings and supports querying
by keyword or phrase to retrieve the most relevant snippets.
"""
import re

from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer


class RAGIndex:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dim: int = 384,
    ):
        self.model = SentenceTransformer(model_name)
        self.dim = dim
        self._index = AnnoyIndex(dim, "angular")
        self._meta: dict[int, tuple[str, str]] = {}  # id -> (label, snippet)

    def build(self, chunks: list[tuple[str, str]], n_trees: int = 10) -> None:
        """
        Build the index from (label, text) pairs.

        Args:
            chunks:  List of (label, text) tuples. Label is an identifier
                     (e.g. page URL or chunk ID); text is the raw passage.
            n_trees: More trees → better recall, slower build.
        """
        texts = [text for _, text in chunks]
        vecs = self.model.encode(texts, convert_to_numpy=True, batch_size=32)
        for i, ((label, text), vec) in enumerate(zip(chunks, vecs)):
            self._index.add_item(i, vec)
            # Store a short snippet for display
            first_line = text.strip().split("\n")[0][:200]
            self._meta[i] = (label, first_line)
        self._index.build(n_trees)

    def query(self, query: str, top_k: int = 5) -> list[tuple[str, str, float]]:
        """
        Retrieve the top-k most relevant chunks for a query string.

        Returns:
            List of (label, snippet, similarity) sorted by descending similarity.
            Similarity is in [0, 1]; angular distance d maps to 1 - d/2.
        """
        vec = self.model.encode(query, convert_to_numpy=True)
        ids, distances = self._index.get_nns_by_vector(
            vec, top_k, include_distances=True
        )
        results = []
        for idx, dist in zip(ids, distances):
            label, snippet = self._meta[idx]
            similarity = max(0.0, 1.0 - dist / 2.0)
            results.append((label, snippet, similarity))
        return results


def chunk_text(text: str, max_chars: int = 400) -> list[str]:
    """
    Split text into chunks that respect sentence boundaries.

    Sentences are accumulated into a chunk until max_chars is reached,
    then a new chunk starts. This keeps semantic units together.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if current and len(current) + len(sent) > max_chars:
            chunks.append(current.strip())
            current = sent
        else:
            current = (current + " " + sent).strip()
    if current:
        chunks.append(current)
    return chunks
