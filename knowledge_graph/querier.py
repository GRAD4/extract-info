"""
Neo4j graph querier and LLM context builder.

This module demonstrates two approaches to answering manufacturing queries:

  1. Structured Cypher query: precise, but requires the user to know the
     exact capability and location names.

  2. Graph-as-context for an LLM: the graph result is serialised to text
     and fed as context to a language model, allowing natural language answers.

Findings from the experiment:
  - The graph excels at multi-hop structural queries (e.g. "find suppliers of
    my tier-1 suppliers that are AS9100D certified").
  - For free-text semantic queries ("companies that do precision aerospace
    parts"), the graph needs an embedding layer on top anyway — at which point
    a relational DB + FAISS index achieves the same result with less
    operational overhead.
  - We concluded that a hybrid approach (graph for explicit relationships,
    embeddings for semantic search) would be ideal, but adds complexity that
    was not justified for the current use case.
"""
import json

from neo4j import GraphDatabase


class GraphQuerier:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    # ── Structured queries ────────────────────────────────────────────────────

    def find_by_capability_and_province(
        self, category: str, province: str
    ) -> list[dict]:
        """
        Find all companies in a province that have at least one capability
        in the given category.

        This is a simple two-hop traversal: Company → Capability.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Company {province: $province})-[:HAS_CAPABILITY]->(cap:Capability {category: $category})
                RETURN c.name AS name, c.city AS city,
                       collect(DISTINCT cap.name) AS capabilities
                ORDER BY c.name
                """,
                province=province,
                category=category,
            )
            return [dict(r) for r in result]

    def find_certified_by(self, cert_name: str) -> list[dict]:
        """Find all companies holding a specific certification."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Company)-[:CERTIFIED_WITH]->(cert:Certification {name: $name})
                RETURN c.name AS name, c.city AS city, c.province AS province
                ORDER BY c.province, c.city
                """,
                name=cert_name,
            )
            return [dict(r) for r in result]

    def capability_co_occurrence(self) -> list[dict]:
        """
        Find pairs of capabilities that frequently appear together
        (in the same company). Useful for building a capability affinity graph.

        This type of query is where graph databases genuinely shine — it is
        awkward to express in SQL and natural in Cypher.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Company)-[:HAS_CAPABILITY]->(a:Capability),
                      (c)-[:HAS_CAPABILITY]->(b:Capability)
                WHERE a.name < b.name
                RETURN a.name AS cap_a, b.name AS cap_b, count(c) AS co_count
                ORDER BY co_count DESC
                LIMIT 10
                """
            )
            return [dict(r) for r in result]

    # ── Graph → LLM context ───────────────────────────────────────────────────

    def build_context_for_query(self, category: str, province: str) -> str:
        """
        Serialise graph results into a text block suitable as LLM context.

        The idea: run a structured graph query, then describe the results in
        plain text so an LLM can answer follow-up questions about them.
        """
        rows = self.find_by_capability_and_province(category, province)
        if not rows:
            return f"No companies found in {province} with '{category}' capabilities."

        lines = [
            f"Companies in {province} with {category} capabilities:\n"
        ]
        for r in rows:
            caps = ", ".join(r["capabilities"])
            lines.append(f"  - {r['name']} ({r['city']}): {caps}")

        return "\n".join(lines)

    def answer_with_llm(
        self,
        question: str,
        context: str,
        llm_generate,  # callable: (prompt: str) -> str
    ) -> str:
        """
        Use the graph context to ground an LLM answer.

        Args:
            question:     Natural language question from the user.
            context:      Text block produced by build_context_for_query.
            llm_generate: Any callable that accepts a prompt string and returns text.

        Returns:
            LLM-generated answer string.
        """
        prompt = (
            f"Use the following data about manufacturing companies to answer the question.\n\n"
            f"Data:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer concisely based only on the data provided."
        )
        return llm_generate(prompt)
