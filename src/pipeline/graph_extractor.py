import os
import sys
import nest_asyncio

# Apply nest_asyncio to prevent event loop issues with LlamaIndex
nest_asyncio.apply()

from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src import config

def build_hybrid_graph():
    print("1. Initializing Models...")
    # Initialize Groq LLM (Used for extracting Entities and Edges)
    llm = Groq(model="llama3-70b-8192", api_key=config.GROQ_API_KEY, temperature=0.0)
    
    # Initialize HuggingFace Embeddings (Used for Vector Search)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # Set global settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512

    print("2. Connecting to Neo4j (Graph DB)...")
    graph_store = Neo4jPropertyGraphStore(
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        url=config.NEO4J_URI,
    )

    print("3. Connecting to Qdrant (Vector DB)...")
    qdrant_client = QdrantClient(url=config.QDRANT_URL)
    vector_store = QdrantVectorStore(
        client=qdrant_client, 
        collection_name="sec_10k_filings"
    )

    print("4. Loading SEC 10-K Documents...")
    # Load all PDFs from the data directory
    documents = SimpleDirectoryReader("./data").load_data()
    print(f"Loaded {len(documents)} total pages.")
    
    # ⚠️ CRITICAL FOR FREE TIER: Slice to the first 5 pages to avoid Rate Limits
    # The first pages of a 10-K typically contain the Business Summary (highly relational).
    docs_to_process = documents[:5] 
    print(f"Processing {len(docs_to_process)} pages to respect free-tier rate limits...")

    print("5. Extracting Graph Nodes/Edges & Generating Vector Embeddings...")
    print("This will take a few minutes as the LLM reads the text and maps relationships...")
    
    # This single command builds BOTH the Graph and the Vector representations!
    index = PropertyGraphIndex.from_documents(
        docs_to_process,
        property_graph_store=graph_store,
        vector_store=vector_store,
        show_progress=True,
        # We tell the LLM to extract financial entities
        kg_extractors=[
            {"type": "llm", "llm": llm, "max_paths_per_chunk": 10}
        ]
    )
    
    print("\n✅ SUCCESS: Hybrid Graph & Vector Indexing Complete!")
    print("Check your Neo4j Browser (http://localhost:7474) to see the Knowledge Graph!")

if __name__ == "__main__":
    build_hybrid_graph()