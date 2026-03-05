# Knowledge graph experiment (Neo4j)

An exploratory experiment using Neo4j to represent manufacturing company–capability
relationships as a property graph, and to evaluate whether graph-based retrieval
can ground LLM answers about the supplier ecosystem.

## 1. Graph schema

```
(Company {id, name, city, province, lat, lon})
    -[:HAS_CAPABILITY]->
(Capability {name, category})

(Company)
    -[:CERTIFIED_WITH]->
(Certification {name})
```

## 2. Quick start

### 2.1. Start Neo4j with Docker

```bash
docker run -d \
  --name neo4j-extract-info \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5
```

The browser UI is available at http://localhost:7474.

### 2.2. Install dependencies and run

```bash
pip install -r requirements.txt
cp .env.example .env   # edit if your Neo4j credentials differ
python example.py
```

## 3. What the example demonstrates

| Step | Description |
|------|-------------|
| Data loading | `loader.py` upserts companies, capabilities, and certifications as graph nodes |
| Structured retrieval | Cypher queries for capability-by-province, certified companies, capability co-occurrence |
| Graph-to-LLM context | Graph results serialised as text, passed as context to an LLM |
| Co-occurrence query | Which capability pairs most often appear together? (hard to express in SQL, natural in Cypher) |

## 4. Findings and limitations

Graph databases work well for explicit multi-hop structural queries, for example:
- "Find all tier-2 suppliers of companies certified AS9100D in QC"
- "Which capabilities always appear together?"

For semantic queries ("find companies that do precision aerospace parts"),
a graph database alone appears insufficient and an embedding layer is still needed.
Once embeddings are required, a simpler setup (relational DB + FAISS index)
achieves equivalent results with considerably less operational overhead.

The conclusion is that Neo4j adds value for explainable, relationship-centric queries
over a known schema. For open-ended semantic manufacturing search, the
embedding-first approach (see `semantic_search/`) proved more pragmatic.
