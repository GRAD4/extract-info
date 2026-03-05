# Multilingual semantic search with geo-filtering

Demonstrates semantic search over a manufacturing supplier directory with
intelligent geographic pre-filtering:

| Location type | Strategy | Example |
|---------------|----------|---------|
| City / town | Haversine radius filter | `"laser cutting in Montréal within 30 km"` |
| Province / state | Bounding-box filter (covers entire region) | `"welding in QC"` |
| None | All candidates, ranked by similarity | `"powder coating anodizing"` |

## 1. Quick start

```bash
pip install -r requirements.txt
python example.py
```

No API keys required, runs entirely on local CPU with a small sentence-transformers model.

## 2. Files

| File | Description |
|------|-------------|
| `indexer.py` | Builds a FAISS `IndexFlatIP` over L2-normalised company embeddings |
| `geo_utils.py` | `haversine()`, `filter_by_radius()`, `filter_by_province()` with pre-defined bounding boxes |
| `query_parser.py` | Extracts semantic query, location, and radius from natural language (EN + FR) |
| `searcher.py` | Combines geo-filter + cosine similarity scoring |
| `example.py` | Runs several demo queries showing all three geo strategies |
| `data/companies.json` | 10 synthetic manufacturing companies with coordinates and capabilities |

## 3. Geographic strategies

**City search** (e.g. `in Montréal within 30 km`):
- Geocode the city name to coordinates
- Keep only companies within the Haversine radius
- Score remaining candidates by cosine similarity

**Province search** (e.g. `in QC`):
- Filter companies by province field
- Report the bounding-box centre and covering radius in the response
- Score remaining candidates by cosine similarity

The province strategy exists because a fixed radius (e.g. 50 km) around a province
capital covers only a fraction of that province. Users searching "in Québec"
expect results from across the entire province.

## 4. Multilingual support

`query_parser.py` normalises French prepositions to English equivalents before
location extraction, enabling queries like:
- `"usinage CNC dans la région de Québec"`. Inferred location: `"Québec"`
- `"soudure TIG à Montréal"`. Inferred location: `"Montréal"`
