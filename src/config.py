import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Neo4j Configuration
# Use 127.0.0.1 explicitly — 'localhost' resolves to ::1 (IPv6) on macOS,
# which Docker does not expose, causing ConnectionRefusedError.
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# LLM Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")