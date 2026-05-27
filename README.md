# Nexus GraphRAG 🧠🔗

A **Hybrid Graph + Vector RAG** pipeline that ingests Apple and Microsoft SEC 10-K filings, extracts financial entities and relationships using Groq's LLaMA-3 70B, stores the knowledge graph in **Neo4j**, and stores semantic embeddings in **Qdrant**.

```
PDF (10-K) ──► LLaMA-3 70B (Groq) ──► Neo4j (Graph) + Qdrant (Vectors)
                                              │
                                      Neo4j Browser
                                  http://localhost:7474
```

---

## Architecture

| Component | Technology | Purpose |
|---|---|---|
| LLM | Groq — LLaMA-3 70B | Entity & relationship extraction |
| Embeddings | HuggingFace `bge-small-en-v1.5` | Semantic vector search |
| Graph DB | Neo4j 5.20 (APOC) | Knowledge graph storage |
| Vector DB | Qdrant | Dense vector search |
| Orchestration | LlamaIndex 0.10 | Pipeline glue |

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | `python --version` |
| Docker Desktop | Running before step 3 |
| Groq API Key | Free at [console.groq.com](https://console.groq.com) |

---

## Step-by-Step Setup

### Step 1 — Clone & install dependencies

```bash
git clone <repo-url>
cd nexus-graphRAG

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Step 2 — Add the 2025 10-K documents

Place the two PDF files inside the `data/` folder:

```
data/
├── apple_10k_fy2025.pdf
└── microsoft_10k_fy2025.pdf
```

### Step 3 — Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set your Groq API key:

```
GROQ_API_KEY=gsk_YOUR_KEY_HERE
```

Everything else defaults correctly for the local Docker setup.

### Step 4 — Start the databases

```bash
docker compose up -d
```

Verify both containers are healthy:

```bash
docker compose ps
```

You should see `nexus-neo4j` and `nexus-qdrant` both in **Up** state.

### Step 5 — Run the extraction engine

```bash
python src/pipeline/graph_extractor.py
```

> ⏳ **Be patient** — Groq is performing complex NLP: reading financial jargon, identifying companies, products, and financial metrics, and drawing relationship lines between them.  
> Expected time: **3–8 minutes** for the default 10-page slice.

When it finishes you will see:

```
============================================================
✅  SUCCESS: Hybrid Graph & Vector Indexing Complete!
============================================================
```

### Step 6 — Visualise the Knowledge Graph 📸

1. Open **http://localhost:7474** in your browser.
2. Login with `neo4j` / `password123`.
3. Paste this Cypher query into the top query bar and press **▶ Play**:

```cypher
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100
```

You will see a network of colourful floating nodes (companies, products, metrics) connected by labelled relationship arrows — your financial knowledge graph!

---

## Project Structure

```
nexus-graphRAG/
├── data/                          # 📄 Put your PDF 10-Ks here
│   ├── apple_10k_fy2025.pdf
│   └── microsoft_10k_fy2025.pdf
├── src/
│   ├── config.py                  # Centralised config (reads .env)
│   ├── db/
│   │   └── neo4j_setup.py         # Connection test utility
│   └── pipeline/
│       └── graph_extractor.py     # ⭐ Main extraction pipeline
├── docker-compose.yml             # Neo4j + Qdrant services
├── requirements.txt
├── .env.example                   # Copy → .env and fill in keys
└── README.md
```

---

## Tuning

| Env var | Default | Description |
|---|---|---|
| `MAX_PAGES` | `10` | Pages sent to the LLM. Raise for richer graphs, lower to avoid rate limits. |
| `GROQ_API_KEY` | — | Required. Free tier is sufficient. |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `GROQ_API_KEY is not set` | Create `.env` from `.env.example` and add your key. |
| `Failed to connect to Neo4j` | Run `docker compose up -d` and wait ~15 s for Neo4j to initialise. |
| `No PDF files found in data/` | Copy the two 10-K PDFs into the `data/` folder. |
| Groq rate-limit errors | Set `MAX_PAGES=5` in `.env` to reduce LLM calls. |
