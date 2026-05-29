import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.chains.hybrid_synthesizer import GraphRAGSynthesizer

# Initialize FastAPI App
app = FastAPI(
    title="Nexus-GraphRAG API",
    description="Enterprise Hybrid Graph-Vector RAG Engine over SEC Financial Filings.",
    version="1.0.0"
)

# Global instance of our synthesizer
synthesizer = None

@app.on_event("startup")
def startup_event():
    """Initializes the AI agent when the server starts."""
    global synthesizer
    synthesizer = GraphRAGSynthesizer()

# Define Request and Response Schemas
class QueryRequest(BaseModel):
    question: str
    
class QueryResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=QueryResponse, tags=["GraphRAG"])
def ask_question(request: QueryRequest):
    """
    Takes a user question, routes it to either Neo4j or Qdrant, and synthesizes an answer.
    """
    if not synthesizer:
        raise HTTPException(status_code=500, detail="Synthesizer engine not initialized.")
    
    answer = synthesizer.generate_answer(request.question)
    return QueryResponse(answer=answer)

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "operational", "databases": ["Neo4j", "Qdrant"]}