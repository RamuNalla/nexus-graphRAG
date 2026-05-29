"""
Nexus GraphRAG — Agentic Query Router
======================================
A ReAct-style agent that routes financial questions to the right tool:
  • vector_search  — semantic search over Qdrant (facts, figures, summaries)
  • graph_search   — Cypher queries over Neo4j  (relationships, multi-hop)

No llama-index dependency. Uses direct SDKs — fully Python 3.14 compatible.

Run:
    python3 src/agents/query_router.py
    python3 src/agents/query_router.py "What was Apple's revenue?"
"""

import os
import sys
import json
import re
import time
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config

if not config.GROQ_API_KEY:
    sys.exit("❌ GROQ_API_KEY not set. Copy .env.example → .env and add your key.")

from groq import Groq as GroqClient
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LLM_MODEL    = "llama-3.3-70b-versatile"
EMBED_MODEL  = "BAAI/bge-small-en-v1.5"
COLLECTION   = "sec_10k_filings"
MAX_REACT    = 6          # max ReAct iterations before giving up
GROQ_SLEEP   = 1.5        # seconds between LLM calls (free-tier rate limit)

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def vector_search(query: str, qdrant: QdrantClient, embedder: TextEmbedding,
                  top_k: int = 5) -> str:
    """Semantic similarity search over Qdrant. Returns concatenated text chunks."""
    from qdrant_client.models import NamedVector, QueryRequest as QdrantQueryRequest
    vector = list(embedder.embed([query]))[0].tolist()
    # qdrant-client 1.7+ uses query_points; fall back to search for older versions
    try:
        response = qdrant.query_points(
            collection_name=COLLECTION,
            query=vector,
            limit=top_k,
        )
        results = response.points
    except AttributeError:
        results = qdrant.search(
            collection_name=COLLECTION,
            query_vector=vector,
            limit=top_k,
        )
    if not results:
        return "No relevant passages found in the vector store."
    passages = []
    for r in results:
        source = r.payload.get("source", "unknown")
        text   = r.payload.get("text", "")
        passages.append(f"[{source}] {text.strip()}")
    return "\n\n---\n\n".join(passages)


def graph_search(query: str, neo4j_driver, groq: GroqClient) -> str:
    """
    Two-step: ask Groq to write a Cypher query, run it against Neo4j,
    return the results as a formatted string.
    """
    # Step 1 — text-to-Cypher
    cypher_prompt = textwrap.dedent(f"""\
        You are a Neo4j Cypher expert. The graph contains nodes with label :Entity
        and a `name` property. Relationships have descriptive UPPER_SNAKE_CASE types.

        Write a single Cypher query to answer the following question.
        Return ONLY the Cypher query, no explanation, no markdown fences.

        Question: {query}
    """)
    resp = groq.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": cypher_prompt}],
        temperature=0.0,
        max_tokens=256,
    )
    cypher = resp.choices[0].message.content.strip()
    cypher = re.sub(r"^```[a-z]*\n?", "", cypher)
    cypher = re.sub(r"```$", "", cypher).strip()

    # Step 2 — execute
    try:
        with neo4j_driver.session() as session:
            result = session.run(cypher)
            rows = [dict(record) for record in result]
        if not rows:
            return f"Cypher ran successfully but returned no results.\nQuery: {cypher}"
        # Serialise neo4j Node/Relationship objects to plain strings
        def _fmt(v):
            if hasattr(v, "_properties"):   # Node or Relationship
                return dict(v._properties)
            return v
        formatted = json.dumps([{k: _fmt(v) for k, v in row.items()} for row in rows],
                                indent=2, default=str)
        return f"Cypher: {cypher}\n\nResults:\n{formatted}"
    except Exception as e:
        return f"Cypher execution error: {e}\nQuery attempted: {cypher}"


# ---------------------------------------------------------------------------
# ReAct agent loop
# ---------------------------------------------------------------------------

REACT_SYSTEM = textwrap.dedent("""\
    You are an intelligent financial analysis agent with two tools:

    1. vector_search(query) — semantic search over 10-K filing text passages.
       Best for: revenue figures, business descriptions, risk factors, plain facts.

    2. graph_search(query) — graph database query over extracted entities & relationships.
       Best for: connections between entities, subsidiaries, acquisitions, multi-hop lookups.

    Follow this strict format for EVERY response until you have a final answer:

    Thought: <your reasoning about what to do next>
    Action: <tool name: either "vector_search" or "graph_search">
    Action Input: <the query string to pass to the tool>

    After you receive an Observation, continue with:

    Thought: <interpret the result>
    ... (repeat if needed) ...

    When you have enough information, write:

    Thought: I have enough information to answer.
    Final Answer: <your complete, well-structured answer>

    Rules:
    - Never make up facts. Only use what the tools return.
    - Always end with "Final Answer:".
""")


def run_react_agent(question: str, groq: GroqClient, qdrant: QdrantClient,
                    neo4j_driver, embedder: TextEmbedding) -> str:
    """Run a ReAct loop: Thought → Action → Observation → … → Final Answer."""
    messages = [
        {"role": "system", "content": REACT_SYSTEM},
        {"role": "user",   "content": question},
    ]

    for step in range(1, MAX_REACT + 1):
        time.sleep(GROQ_SLEEP)
        resp = groq.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
            stop=["Observation:"],   # stop before hallucinating the observation
        )
        assistant_msg = resp.choices[0].message.content.strip()
        print(f"\n--- Step {step} ---\n{assistant_msg}")
        messages.append({"role": "assistant", "content": assistant_msg})

        # Check for final answer
        if "Final Answer:" in assistant_msg:
            final = assistant_msg.split("Final Answer:", 1)[1].strip()
            return final

        # Parse Action / Action Input
        action_match = re.search(r"Action:\s*(.+)", assistant_msg)
        input_match  = re.search(r"Action Input:\s*(.+)", assistant_msg, re.DOTALL)

        if not action_match or not input_match:
            # Model deviated from format — nudge it
            messages.append({
                "role": "user",
                "content": "Please follow the format: Thought / Action / Action Input, or write Final Answer."
            })
            continue

        tool_name  = action_match.group(1).strip().lower().replace(" ", "_")
        tool_input = input_match.group(1).strip().strip('"').strip("'")

        # Execute tool
        print(f"\n🔧 Tool: {tool_name}")
        print(f"   Input: {tool_input}")
        if "graph" in tool_name:
            observation = graph_search(tool_input, neo4j_driver, groq)
        else:
            observation = vector_search(tool_input, qdrant, embedder)

        print(f"   Observation (first 300 chars): {observation[:300]}…")

        # Feed observation back
        messages.append({
            "role": "user",
            "content": f"Observation: {observation}\n\nContinue your reasoning."
        })

    return "⚠️  Agent reached maximum steps without a final answer."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class FinancialAgent:
    """Initialise once, then call .chat(question) repeatedly."""

    def __init__(self):
        print("🔧 Initialising connections…")
        self.groq    = GroqClient(api_key=config.GROQ_API_KEY)
        self.neo4j   = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD),
        )
        self.neo4j.verify_connectivity()
        self.qdrant  = QdrantClient(url=config.QDRANT_URL)
        self.embedder = TextEmbedding(model_name=EMBED_MODEL)
        print("   ✅ Groq | Neo4j | Qdrant | fastembed ready")

    def chat(self, question: str) -> str:
        print(f"\n{'='*56}")
        print(f"❓  {question}")
        print(f"{'='*56}")
        return run_react_agent(
            question, self.groq, self.qdrant, self.neo4j, self.embedder
        )

    def close(self):
        self.neo4j.close()


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent = FinancialAgent()

    # Accept a question from the command line, or run the two built-in tests
    if len(sys.argv) > 1:
        answer = agent.chat(" ".join(sys.argv[1:]))
        print(f"\n[FINAL ANSWER]: {answer}\n")
    else:
        print("\n" + "="*56)
        print("TEST 1: Simple Fact  →  should use vector_search")
        print("="*56)
        answer1 = agent.chat("What is the primary business of Apple?")
        print(f"\n[FINAL ANSWER]: {answer1}\n")

        print("Waiting 5 s to respect Groq free-tier rate limits…\n")
        time.sleep(5)

        print("="*56)
        print("TEST 2: Relationship  →  should use graph_search")
        print("="*56)
        answer2 = agent.chat("What entities or subsidiaries are related to Microsoft?")
        print(f"\n[FINAL ANSWER]: {answer2}\n")

    agent.close()
