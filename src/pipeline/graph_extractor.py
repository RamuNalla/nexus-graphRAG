"""
Nexus GraphRAG — Extraction Engine
====================================
Reads SEC 10-K PDFs from ./data/, uses Groq (LLaMA-3 70B) to extract
financial entities and relationships, and stores them in:
  • Neo4j  — property graph (nodes + edges)
  • Qdrant — vector store for semantic search

Run:
    python src/pipeline/graph_extractor.py
"""

import os
import sys
import json
import time
import textwrap
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — make `src` importable regardless of CWD
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config

# ---------------------------------------------------------------------------
# Validate secrets early
# ---------------------------------------------------------------------------
if not config.GROQ_API_KEY:
    sys.exit(
        "❌ GROQ_API_KEY is not set.\n"
        "   Copy .env.example → .env and paste your key from https://console.groq.com"
    )

# ---------------------------------------------------------------------------
# Third-party imports (all Python 3.14-compatible, no llama-index)
# ---------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("❌ PyMuPDF not installed. Run: pip install pymupdf")

try:
    from groq import Groq as GroqClient
except ImportError:
    sys.exit("❌ groq SDK not installed. Run: pip install groq")

try:
    from neo4j import GraphDatabase
except ImportError:
    sys.exit("❌ neo4j driver not installed. Run: pip install neo4j")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
except ImportError:
    sys.exit("❌ qdrant-client not installed. Run: pip install qdrant-client")

try:
    from fastembed import TextEmbedding
except ImportError:
    sys.exit("❌ fastembed not installed. Run: pip install fastembed")

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DATA_DIR      = PROJECT_ROOT / "data"
CHUNK_SIZE    = 1200          # characters per chunk sent to Groq
CHUNK_OVERLAP = 200
MAX_CHUNKS    = int(os.getenv("MAX_CHUNKS", "20"))   # free-tier guard
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"   # supported natively by fastembed
LLM_MODEL     = "llama-3.3-70b-versatile"  # llama3-70b-8192 was decommissioned
COLLECTION    = "sec_10k_filings"
EMBED_DIM     = 384

# Groq free tier: ~30 req/min — add a small sleep between calls
GROQ_SLEEP_S  = 2.5

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""\
    You are a financial knowledge-graph extractor.
    Given a passage from an SEC 10-K filing, extract every meaningful
    relationship as a list of JSON triples:

    [{"subject": "<entity>", "predicate": "<RELATION>", "object": "<entity>"}]

    Rules:
    - Entities: companies, products, people, metrics, geographies, dates.
    - Predicates: short UPPER_SNAKE_CASE verbs (e.g. REPORTED_REVENUE, ACQUIRED, OPERATES_IN).
    - Return ONLY the JSON array, no commentary, no markdown fences.
    - If nothing meaningful found, return [].
""")


# ===========================================================================
# Helpers
# ===========================================================================

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character chunks."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def extract_triples(groq: GroqClient, chunk: str) -> list[dict]:
    """Ask Groq to extract (subject, predicate, object) triples from a chunk."""
    response = groq.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": chunk},
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    raw = response.choices[0].message.content.strip()
    # Strip accidental markdown fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"```$", "", raw).strip()
    try:
        triples = json.loads(raw)
        if isinstance(triples, list):
            return triples
    except json.JSONDecodeError:
        print(f"\n   ⚠️  JSON parse failed for chunk, raw response: {raw[:120]}")
    return []


def upsert_graph(driver, triples: list[dict], source: str) -> int:
    """Merge nodes and relationships into Neo4j."""
    count = 0
    with driver.session() as session:
        for t in triples:
            subj = str(t.get("subject", "")).strip()
            pred = str(t.get("predicate", "")).strip().upper().replace(" ", "_")
            obj  = str(t.get("object",  "")).strip()
            if not (subj and pred and obj):
                continue
            # Sanitise relationship type — Neo4j only allows [A-Za-z0-9_]
            pred = re.sub(r"[^A-Z0-9_]", "_", pred)
            cypher = (
                f"MERGE (a:Entity {{name: $subj}}) "
                f"MERGE (b:Entity {{name: $obj}}) "
                f"MERGE (a)-[r:`{pred}` {{source: $source}}]->(b)"
            )
            session.run(cypher, subj=subj, obj=obj, source=source)
            count += 1
    return count


def ensure_qdrant_collection(qdrant: QdrantClient) -> None:
    """Create the Qdrant collection if it doesn't already exist."""
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )


def upsert_vectors(qdrant: QdrantClient, embedder: TextEmbedding,
                   chunks: list[str], source: str, offset: int) -> None:
    """Embed chunks and upsert into Qdrant."""
    vectors = list(embedder.embed(chunks))  # returns a generator of np arrays
    points  = [
        PointStruct(
            id=offset + i,
            vector=vec.tolist(),
            payload={"text": chunk, "source": source},
        )
        for i, (chunk, vec) in enumerate(zip(chunks, vectors))
    ]
    qdrant.upsert(collection_name=COLLECTION, points=points)


# ===========================================================================
# Main pipeline
# ===========================================================================

def build_hybrid_graph() -> None:

    # ------------------------------------------------------------------
    # 1. Validate data dir
    # ------------------------------------------------------------------
    if not DATA_DIR.is_dir():
        sys.exit(f"❌ data/ directory not found at {DATA_DIR}")

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        sys.exit(
            "❌ No PDF files found in data/.\n"
            "   Please copy apple_10k_fy2025.pdf and microsoft_10k_fy2025.pdf "
            "into the data/ folder."
        )

    print("\n" + "=" * 60)
    print(" Nexus GraphRAG — Extraction Engine")
    print("=" * 60)
    print(f"   PDFs found : {', '.join(p.name for p in pdf_files)}")
    print(f"   Max chunks : {MAX_CHUNKS} per document (set MAX_CHUNKS to change)")

    # ------------------------------------------------------------------
    # 2. Connect to services
    # ------------------------------------------------------------------
    print("\n🔧 Connecting to services…")

    groq_client = GroqClient(api_key=config.GROQ_API_KEY)
    print("   ✅ Groq SDK ready")

    neo4j_driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD),
    )
    neo4j_driver.verify_connectivity()
    print(f"   ✅ Neo4j connected  ({config.NEO4J_URI})")

    qdrant_client = QdrantClient(url=config.QDRANT_URL)
    ensure_qdrant_collection(qdrant_client)
    print(f"   ✅ Qdrant connected ({config.QDRANT_URL})  collection='{COLLECTION}'")

    print("\n🔧 Loading embedding model (BAAI/bge-small-en-v1.5 via fastembed)…")
    embedder = TextEmbedding(model_name=EMBED_MODEL)
    print("   ✅ Embedding model ready")

    # ------------------------------------------------------------------
    # 3. Process each PDF
    # ------------------------------------------------------------------
    total_triples = 0
    vector_offset = 0

    for pdf_path in pdf_files:
        print(f"\n📄 Processing: {pdf_path.name}")

        text   = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)[:MAX_CHUNKS]
        print(f"   Chunks to process: {len(chunks)} (of {len(chunk_text(text))} total)")

        # ── Vector embeddings ────────────────────────────────────────
        print("   Embedding chunks into Qdrant…", end=" ", flush=True)
        upsert_vectors(qdrant_client, embedder, chunks, pdf_path.name, vector_offset)
        vector_offset += len(chunks)
        print("done ✅")

        # ── Graph extraction via Groq ─────────────────────────────────
        print(f"   Extracting knowledge graph with LLaMA-3 70B…")
        file_triples = 0
        for i, chunk in enumerate(chunks, 1):
            triples = extract_triples(groq_client, chunk)
            if triples:
                written = upsert_graph(neo4j_driver, triples, pdf_path.name)
                file_triples   += written
                total_triples  += written
            pct = int(i / len(chunks) * 100)
            bar = ("█" * (pct // 5)).ljust(20)
            print(f"   [{bar}] {pct:3d}%  chunk {i}/{len(chunks)}  "
                  f"+{len(triples)} triples", end="\r", flush=True)
            time.sleep(GROQ_SLEEP_S)  # respect free-tier rate limit

        print(f"\n   ✅ {pdf_path.name}: {file_triples} graph relationships written")

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    neo4j_driver.close()

    print("\n" + "=" * 60)
    print("✅  SUCCESS: Hybrid Graph & Vector Indexing Complete!")
    print("=" * 60)
    print(f"   Total relationships written to Neo4j : {total_triples}")
    print(f"   Total chunks stored in Qdrant        : {vector_offset}")
    print("\n📊  Visualise your Knowledge Graph:")
    print("   1. Open  http://localhost:7474")
    print("   2. Login neo4j / password123")
    print("   3. Run Cypher:")
    print("\n      MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100\n")


if __name__ == "__main__":
    build_hybrid_graph()
