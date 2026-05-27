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
import nest_asyncio

# Allow nested event loops (needed for LlamaIndex async internals in scripts)
nest_asyncio.apply()

# ---------------------------------------------------------------------------
# Path setup — make `src` importable regardless of CWD
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from src import config

# ---------------------------------------------------------------------------
# Validate required secrets before importing heavy deps
# ---------------------------------------------------------------------------
if not config.GROQ_API_KEY:
    sys.exit(
        "❌ GROQ_API_KEY is not set.\n"
        "   Create a .env file (copy .env.example) and add your Groq API key.\n"
        "   Get a free key at https://console.groq.com"
    )

# ---------------------------------------------------------------------------
# Heavy imports (after validation so startup is fast on bad config)
# ---------------------------------------------------------------------------
from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex, Settings
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


# ---------------------------------------------------------------------------
# How many pages to process  (tweak to avoid Groq free-tier rate limits)
# ---------------------------------------------------------------------------
MAX_PAGES = int(os.getenv("MAX_PAGES", "10"))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def build_hybrid_graph() -> None:
    # ------------------------------------------------------------------
    # 1. Models
    # ------------------------------------------------------------------
    print("\n🔧 Step 1 — Initialising models…")
    llm = Groq(
        model="llama3-70b-8192",
        api_key=config.GROQ_API_KEY,
        temperature=0.0,
        # Stay well within the free-tier context window
        max_tokens=2048,
    )
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 64
    print("   ✅ Groq LLaMA-3 70B  |  HuggingFace bge-small-en-v1.5 ready.")

    # ------------------------------------------------------------------
    # 2. Neo4j
    # ------------------------------------------------------------------
    print("\n🔧 Step 2 — Connecting to Neo4j…")
    graph_store = Neo4jPropertyGraphStore(
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        url=config.NEO4J_URI,
    )
    print(f"   ✅ Connected to Neo4j at {config.NEO4J_URI}")

    # ------------------------------------------------------------------
    # 3. Qdrant
    # ------------------------------------------------------------------
    print("\n🔧 Step 3 — Connecting to Qdrant…")
    qdrant_client = QdrantClient(url=config.QDRANT_URL)
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name="sec_10k_filings",
    )
    print(f"   ✅ Connected to Qdrant at {config.QDRANT_URL}")

    # ------------------------------------------------------------------
    # 4. Load PDFs
    # ------------------------------------------------------------------
    print(f"\n🔧 Step 4 — Loading SEC 10-K PDFs from {DATA_DIR} …")
    if not os.path.isdir(DATA_DIR):
        sys.exit(f"❌ data/ directory not found at {DATA_DIR}")

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        sys.exit(
            "❌ No PDF files found in data/.\n"
            "   Please copy apple_10k_fy2025.pdf and microsoft_10k_fy2025.pdf "
            "into the data/ folder."
        )

    print(f"   Found PDFs: {', '.join(pdf_files)}")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    total_pages = len(documents)
    print(f"   Loaded {total_pages} total pages across all documents.")

    # ⚠️  Free-tier Groq has rate limits — slice to MAX_PAGES.
    #     The opening pages of a 10-K contain the richest relational content
    #     (Business Overview, Products, Risk Factors).
    docs_to_process = documents[:MAX_PAGES]
    print(
        f"   Processing first {len(docs_to_process)} pages "
        f"(set MAX_PAGES env var to change, current={MAX_PAGES})."
    )

    # ------------------------------------------------------------------
    # 5. Extract graph + vectors
    # ------------------------------------------------------------------
    print("\n🔧 Step 5 — Extracting entities & relationships with LLaMA-3 70B…")
    print("   ⏳  This will take several minutes.  Groq is reading financial")
    print("       jargon and drawing relationship lines between entities.\n")

    # SimpleLLMPathExtractor is the correct LlamaIndex 0.10.x API for
    # instructing the LLM to emit (subject, predicate, object) triples.
    kg_extractor = SimpleLLMPathExtractor(
        llm=llm,
        max_paths_per_chunk=10,
        num_workers=4,
    )

    index = PropertyGraphIndex.from_documents(
        docs_to_process,
        property_graph_store=graph_store,
        vector_store=vector_store,
        kg_extractors=[kg_extractor],
        show_progress=True,
    )

    # ------------------------------------------------------------------
    # 6. Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("✅  SUCCESS: Hybrid Graph & Vector Indexing Complete!")
    print("=" * 60)
    print("\n📊  Next steps:")
    print("  1. Open http://localhost:7474 in your browser.")
    print("  2. Login: neo4j / password123")
    print("  3. Run this Cypher query to visualise the graph:")
    print("\n     MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100\n")


if __name__ == "__main__":
    build_hybrid_graph()
