"""
Nexus GraphRAG — FastAPI Entry Point
======================================
Provides:
  POST /ask    — ask a financial question, routed via ReAct agent
  GET  /health — liveness check
  GET  /docs   — Swagger UI (auto-generated)
"""
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Make project root importable inside Docker and locally
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.chains.hybrid_synthesizer import GraphRAGSynthesizer

# ---------------------------------------------------------------------------
# Lifespan — replaces deprecated @app.on_event("startup"/"shutdown")
# ---------------------------------------------------------------------------
synthesizer: GraphRAGSynthesizer | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global synthesizer
    print("🚀 Starting Nexus-GraphRAG API...")
    synthesizer = GraphRAGSynthesizer()   # boots agent, connects to DBs
    yield
    # Shutdown
    if synthesizer:
        synthesizer.close()
    print("👋 Nexus-GraphRAG API shut down.")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Nexus-GraphRAG API",
    description=(
        "Enterprise Hybrid Graph-Vector RAG Engine over SEC 10-K Financial Filings.\n\n"
        "Uses **Neo4j** (knowledge graph) + **Qdrant** (vector search) + "
        "**Groq LLaMA-3.3 70B** (reasoning)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "question": "What is the primary business of Apple and what are its key entities?"
            }]
        }
    }

class QueryResponse(BaseModel):
    question: str
    answer: str

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/ask", response_model=QueryResponse, tags=["GraphRAG"])
def ask_question(request: QueryRequest):
    """
    Ask a financial question about Apple or Microsoft's 10-K filings.

    The agent automatically routes to:
    - **vector_search** for factual / numerical questions
    - **graph_search** for relationship / entity questions
    """
    if not synthesizer:
        raise HTTPException(status_code=503, detail="Synthesizer engine not ready yet.")
    answer = synthesizer.generate_answer(request.question)
    return QueryResponse(question=request.question, answer=answer)


@app.get("/health", tags=["System"])
def health_check():
    """Liveness check — confirms the API and agent are operational."""
    return {
        "status": "operational",
        "agent": "ready" if synthesizer else "initialising",
        "databases": ["Neo4j (graph)", "Qdrant (vector)"],
        "llm": "Groq llama-3.3-70b-versatile",
    }