"""
Nexus GraphRAG — Hybrid Synthesizer
=====================================
Thin wrapper around FinancialAgent that the FastAPI layer calls.
No llama-index. Python 3.14 compatible.
"""
import sys
import os
from pathlib import Path

# Make project root importable regardless of CWD (works both locally and in Docker)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.query_router import FinancialAgent


class GraphRAGSynthesizer:
    """
    Wraps FinancialAgent and exposes a simple generate_answer() interface
    for the FastAPI backend.
    """

    def __init__(self):
        print("🔧 Booting up Hybrid GraphRAG Synthesizer...")
        self.agent = FinancialAgent()
        print("   ✅ Synthesizer ready")

    def generate_answer(self, query: str) -> str:
        print(f"\n🧠 [AGENT TRIGGERED] Query: {query}")
        try:
            return self.agent.chat(query)
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"

    def close(self):
        self.agent.close()