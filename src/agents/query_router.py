import os
import sys
import time
import nest_asyncio

# Prevent async event loop issues
nest_asyncio.apply()

from llama_index.core import Settings, PropertyGraphIndex
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from llama_index.core.indices.property_graph import VectorContextRetriever, TextToCypherRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent

# Ensure we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src import config

def initialize_agent():
    print("1. Initializing Models & Connections...")
    llm = Groq(model="llama3-70b-8192", api_key=config.GROQ_API_KEY, temperature=0.0)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = llm
    Settings.embed_model = embed_model

    graph_store = Neo4jPropertyGraphStore(
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        url=config.NEO4J_URI,
    )
    
    qdrant_client = QdrantClient(url=config.QDRANT_URL)
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name="sec_10k_filings")

    # Reconnect to our existing index from Phase 2
    index = PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        vector_store=vector_store,
    )

    print("2. Building Independent Retrieval Tools...")
    
    # TOOL 1: Vector Retriever (For simple text lookups)
    vector_retriever = VectorContextRetriever(
        graph_store=graph_store,
        vector_store=vector_store,
        embed_model=embed_model
    )
    vector_query_engine = RetrieverQueryEngine.from_args(vector_retriever, llm=llm)
    
    vector_tool = QueryEngineTool(
        query_engine=vector_query_engine,
        metadata=ToolMetadata(
            name="vector_search_tool",
            description="Use this tool to find direct facts, numerical figures, and standard text summaries from the financial documents (e.g., 'What was the revenue?', 'What are the risk factors?')."
        )
    )

    # TOOL 2: Graph/Cypher Retriever (For multi-hop & relationship lookups)
    cypher_retriever = TextToCypherRetriever(
        graph_store=graph_store,
        llm=llm
    )
    cypher_query_engine = RetrieverQueryEngine.from_args(cypher_retriever, llm=llm)
    
    graph_tool = QueryEngineTool(
        query_engine=cypher_query_engine,
        metadata=ToolMetadata(
            name="graph_search_tool",
            description="Use this tool for relational, structural, or multi-hop queries connecting entities (e.g., 'What companies did Microsoft acquire?', 'How is Entity X connected to Entity Y?')."
        )
    )

    print("3. Compiling the Agentic Router...")
    
    system_prompt = """
    You are an intelligent financial router agent. You have access to a Vector Database and a Graph Database.
    Think step-by-step. 
    First, analyze the user's question to determine its complexity.
    If it is a simple factual question, use the `vector_search_tool`.
    If it requires connecting entities, finding relationships, or mapping dependencies, use the `graph_search_tool`.
    Once you have the observation, summarize the final answer clearly.
    """

    agent = ReActAgent.from_tools(
        [vector_tool, graph_tool],
        llm=llm,
        verbose=True, # This makes the agent print its "Thoughts"
        system_prompt=system_prompt
    )
    
    return agent

if __name__ == "__main__":
    agent = initialize_agent()
    
    print("\n==================================================")
    print("TEST 1: Simple Fact Question (Should Route to Vector)")
    print("==================================================")
    q1 = "What is the primary business of Apple?"
    response1 = agent.chat(q1)
    print(f"\n[FINAL ANSWER]: {response1}\n")
    
    # Pause to respect Groq free-tier rate limits
    print("Waiting 5 seconds to avoid API rate limits...\n")
    time.sleep(5)
    
    print("==================================================")
    print("TEST 2: Relational/Multi-Hop Question (Should Route to Graph)")
    print("==================================================")
    # We ask about Microsoft's relationships/entities which the graph excels at
    q2 = "What entities or subsidiaries are related to Microsoft?"
    response2 = agent.chat(q2)
    print(f"\n[FINAL ANSWER]: {response2}\n")