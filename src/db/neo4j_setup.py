import sys
import os
from neo4j import GraphDatabase

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src import config

def test_neo4j_connection():
    print(f"Attempting to connect to Neo4j at {config.NEO4J_URI}...")
    
    try:
        # Initialize the Neo4j Driver
        driver = GraphDatabase.driver(
            config.NEO4J_URI, 
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
        )
        
        # Verify Connectivity
        driver.verify_connectivity()
        print("✅ SUCCESS: Successfully connected to Neo4j Graph Database!")
        
        # Run a quick test query to create and delete a node
        with driver.session() as session:
            session.run("CREATE (n:TestNode {message: 'Hello GraphRAG'})")
            print("✅ SUCCESS: Successfully wrote to Neo4j!")
            session.run("MATCH (n:TestNode) DELETE n")
            print("✅ SUCCESS: Successfully deleted test data. Database is ready for ingestion.")
            
        driver.close()
        
    except Exception as e:
        print(f"❌ ERROR: Failed to connect to Neo4j. Is the Docker container running?\nDetails: {e}")

if __name__ == "__main__":
    test_neo4j_connection()