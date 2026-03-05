"""
Prompt builder for the RAG-based classification step.

The prompt follows a simple structure:
  1. System instructions + full taxonomy listing
  2. Company name
  3. Retrieved RAG snippets as additional context
  4. Output format specification

If the assembled prompt exceeds the token budget, context is trimmed
progressively: first the snippets are reduced, then dropped entirely.
"""


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~0.75 tokens per word."""
    return int(len(text.split()) / 0.75)


SYSTEM = """\
You are classifying a manufacturing company based on text retrieved from its website.

Your task: select the subcategories from the taxonomy below that best describe
the company's manufacturing capabilities. Look for concrete evidence in the text —
specific processes, equipment, materials, and certifications mentioned.

Taxonomy:
{taxonomy}

Rules:
- Only output subcategories from the exact list above.
- Do not invent new categories or translate them.
- Respond in this format, one line per match:
    Specification 1: <Category> - <Subcategory>
    Specification 2: <Category> - <Subcategory>
    ...
- Output at most {top_k} specifications.
"""

USER = """\
Company: {company}

Retrieved context:
{context}
"""


class PromptBuilder:
    def __init__(self, taxonomy: dict[str, list[str]], top_k: int = 10, token_limit: int = 3000):
        self.taxonomy = taxonomy
        self.top_k = top_k
        self.token_limit = token_limit

        lines = []
        for category, subcategories in taxonomy.items():
            lines.append(f"{category}:")
            for sub in subcategories:
                lines.append(f"  - {sub}")
        self._taxonomy_str = "\n".join(lines)

    def build(
        self,
        company: str,
        snippets: list[tuple[str, str, float]],  # (label, text, similarity)
    ) -> str:
        system = SYSTEM.format(taxonomy=self._taxonomy_str, top_k=self.top_k)

        def _make_context(items: list[tuple[str, str, float]]) -> str:
            parts = []
            for label, text, sim in items:
                parts.append(f"[{label}] (relevance {sim:.2f})\n{text}")
            return "\n\n".join(parts)

        context = _make_context(snippets)
        user = USER.format(company=company, context=context)
        prompt = system + "\n\n" + user

        # Trim snippets if over budget
        if _estimate_tokens(prompt) > self.token_limit and len(snippets) > 3:
            context = _make_context(snippets[:3])
            user = USER.format(company=company, context=context)
            prompt = system + "\n\n" + user

        # Drop context entirely as last resort
        if _estimate_tokens(prompt) > self.token_limit:
            user = USER.format(company=company, context="(none retrieved)")
            prompt = system + "\n\n" + user

        return prompt
