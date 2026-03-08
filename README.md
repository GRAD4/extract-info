# Extract Info project methodology examples

*[Français ci-dessous](#extract-info--exemples-de-méthodologie)*

---

## Extract Info - methodology examples

Illustrative code examples from the **Extract Info** project, a pre-competitive research initiative carried out by [Axya](https://axya.co) and [Plumfind](https://plumfind.com), funded by the [Ministère de l'Économie, de l'Innovation et de l'Énergie du Québec (MEIE)](https://www.economie.gouv.qc.ca/) through the [Confiance IA](https://confiance.ai/) consortium, hosted by [CRIM](https://www.crim.ca/).

The project explores how AI can help structure data from manufacturing industry websites and e-commerce. Specifically, how to automatically classify companies into capability taxonomies, extract certifications, and enable multilingual semantic search over supplier directories.

This repository contains minimal, runnable examples demonstrating the core methodologies.

---

### 1. What's inside

| Directory | Description |
|-----------|-------------|
| `rag_classification/` | RAG-based company classification using Annoy vector index + LLM prompting (AWS Bedrock or Ollama) |
| `semantic_search/` | Multilingual semantic search with FAISS, geo-filtering strategies (city radius, province bounding box), and bilingual query parsing |
| `knowledge_graph/` | Exploratory Neo4j knowledge graph experiment: loading company–capability relationships and querying them as context for an LLM |

---

### 2. Requirements

- Python 3.10+
- Dependencies: see `requirements.txt` of each subdirectory
- For LLM calls: an [AWS Bedrock](https://aws.amazon.com/bedrock/) account or a local [Ollama](https://ollama.com/) installation

---

### 3. Quick start

```bash
git clone https://github.com/GRAD4/extract-info.git
cd extract-info

# Pick a module, install its dependencies, then follow its README
cd rag_classification
pip install -r requirements.txt
python example.py
```

Each subdirectory contains its own `README.md` with detailed instructions.

---

### 4. Acknowledgements

This work was supported by the Ministère de l'Économie, de l'Innovation et de l'Énergie du Québec (MEIE) through the Confiance IA consortium, administered by the Centre de recherche informatique de Montréal (CRIM).

The applied use cases driving this research were provided by Axya Inc. (manufacturing procurement platform) and Plumfind (fashion e-commerce search).

---

### 5. License

Apache 2.0 — see [LICENSE](LICENSE).

---
---

## Extract Info - exemples de méthodologie

Exemples de code illustratifs issus du projet **Extract Info**, une initiative de recherche précompétitive menée par [Axya](https://axya.co) et [Plumfind](https://plumfind.com), financée par le [Ministère de l'Économie, de l'Innovation et de l'Énergie du Québec (MEIE)](https://www.economie.gouv.qc.ca/) dans le cadre du consortium [Confiance IA](https://confiance.ia/), hébergé par le [CRIM](https://www.crim.ca/).

Le projet a exploré comment l'IA peut aider à structurer des données provenant de sites Web de l'industrie manufacturière et du commerce électronique - notamment comment classifier automatiquement des entreprises dans des taxonomies de capacités, extraire des certifications et permettre une recherche sémantique multilingue dans un répertoire de fournisseurs.

Ce dépôt contient des exemples minimaux et exécutables illustrant les méthodologies essentielles.

---

### 1. Contenu

| Répertoire | Description |
|------------|-------------|
| `rag_classification/` | Classification d'entreprises par RAG avec index vectoriel Annoy et inférence LLM (AWS Bedrock ou Ollama) |
| `semantic_search/` | Recherche sémantique multilingue avec FAISS, stratégies de géofiltrage (rayon pour les villes, boîte englobante pour les provinces) et analyse de requêtes bilingues |
| `knowledge_graph/` | Expérimentation exploratoire avec Neo4j : chargement de relations entreprise–capacité sous forme de graphe de connaissances et utilisation comme contexte pour un LLM |

---

### 2. Prérequis

- Python 3.10+
- Dépendances : voir le fichier `requirements.txt` de chaque sous-répertoire
- Pour les appels LLM : un compte [AWS Bedrock](https://aws.amazon.com/fr/bedrock/) ou une installation locale d'[Ollama](https://ollama.com/)

---

### 3. Démarrage rapide

```bash
git clone https://github.com/axya-inc/extract-info.git
cd extract-info

# Choisissez un module, installez ses dépendances, puis suivez son README
cd rag_classification
pip install -r requirements.txt
python example.py
```

Chaque sous-répertoire contient son propre fichier `README.md` avec des instructions détaillées.

---

### 4. Remerciements

Ces travaux ont été soutenus par le Ministère de l'Économie, de l'Innovation et de l'Énergie du Québec (MEIE) dans le cadre du consortium Confiance IA, administré par le Centre de recherche informatique de Montréal (CRIM).

Les cas d'utilisation appliqués à l'origine de cette recherche ont été fournis par Axya Inc. (plateforme d'approvisionnement manufacturier) et Plumfind (recherche pour le commerce électronique de mode).

---

### 5. Licence

Apache 2.0 — voir [LICENSE](LICENSE).
