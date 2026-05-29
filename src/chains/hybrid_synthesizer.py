import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.agents.query_router import initialize_agent

class GraphRAGSynthesizer:
    """
    A wrapper class that encapsulates the LlamaIndex ReAct Agent.
    This acts as the synthesizer engine for the FastAPI backend.
    """
    def __init__(self):
        print("🔧 Booting up Hybrid GraphRAG Synthesizer...")
        self.agent = initialize_agent()

    def generate_answer(self, query: str) -> str:
        print(f"\n🧠 [AGENT TRIGGERED] Query: {query}")
        try:
            response = self.agent.chat(query)
            return str(response)
        except Exception as e:
            return f"Error generating response: {str(e)}"