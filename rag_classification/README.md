# RAG-based сompany сlassification

Classifies a manufacturing company into capability categories using a
Retrieval-Augmented Generation (RAG) pipeline:

1. Company website text is chunked into passages.
2. Passages are embedded and stored in an **Annoy** approximate-nearest-neighbour index.
3. For each subcategory in the taxonomy, the top-k most relevant passages are retrieved.
4. An LLM (Ollama or AWS Bedrock) receives the retrieved context and outputs structured `Category - Subcategory` pairs.
5. Certifications are detected with normalized keyword matching.

## 1. Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure LLM provider
cp .env.example .env
# Edit .env, set LLM_PROVIDER=ollama (default) or bedrock

# 3. Run
python example.py
```

The example classifies `data/example_company.txt` against `data/categories.json`.
Replace those files with your own data to classify a different company.

## 2. Files

| File | Description |
|------|-------------|
| `rag.py` | Annoy-based vector index with `build()` and `query()` |
| `llm.py` | LLM client supporting Ollama and AWS Bedrock |
| `prompt.py` | Prompt builder with token-budget trimming |
| `example.py` | End-to-end pipeline script |
| `data/categories.json` | Example manufacturing capability taxonomy |
| `data/certifications.json` | Certification acronyms for keyword extraction |
| `data/example_company.txt` | Synthetic company text used as input |

## 3. Using Ollama (recommended for local runs)

```bash
# Install Ollama: https://ollama.com
ollama pull llama3.1
# Then run the example — it will use Ollama by default
python example.py
```

## 4. Using AWS Bedrock

Set `LLM_PROVIDER=bedrock` and fill in your AWS credentials in `.env`.
Any Bedrock model works; the client auto-detects Claude vs Llama format.
