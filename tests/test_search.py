import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from query import retrieve_vector
import lancedb
import pyarrow as pa
from langchain_ollama import OllamaEmbeddings

def run_test():
    # Setup dummy db
    db_dir = "/tmp/lancedb_test"
    db = lancedb.connect(db_dir)
    
    # Create notebooks table
    nb_schema = pa.schema([
        ("id", pa.string()),
        ("vector", pa.list_(pa.float32(), 768)), # Nomic embed has 768 dims
        ("text", pa.string()),
        ("notebook_id", pa.string()),
        ("source", pa.string()),
    ])
    
    embedder = OllamaEmbeddings(model="nomic-embed-text")
    vec1 = embedder.embed_query("pump manual")
    vec2 = embedder.embed_query("boiler parts")
    
    data = [
        {"id": "1", "vector": vec1, "text": "pump maintenance", "notebook_id": "nb_1", "source": "pump.pdf"},
        {"id": "2", "vector": vec2, "text": "boiler maintenance", "notebook_id": "nb_1", "source": "boiler.pdf"},
        {"id": "3", "vector": vec1, "text": "pump maintenance alt", "notebook_id": "nb_2", "source": "pump.pdf"},
    ]
    
    if "notebooks" in db.list_tables().tables:
        db.drop_table("notebooks")
    db.create_table("notebooks", data=data, schema=nb_schema)
    
    # Create core knowledge table
    core_schema = pa.schema([
        ("id", pa.string()),
        ("vector", pa.list_(pa.float32(), 768)),
        ("text", pa.string()),
        ("equipment_category", pa.string()),
    ])
    
    core_vec = embedder.embed_query("pump fundamentals core knowledge")
    core_data = [
        {"id": "c1", "vector": core_vec, "text": "core pump info", "equipment_category": "3_pumps_piping_fluid"},
        {"id": "c2", "vector": core_vec, "text": "core boiler info", "equipment_category": "5_boiler_steam"},
    ]
    
    if "core_knowledge" in db.list_tables().tables:
        db.drop_table("core_knowledge")
    db.create_table("core_knowledge", data=core_data, schema=core_schema)
    
    print("Testing Notebook Only...")
    res1 = retrieve_vector("pump", db_dir, "notebooks", 2, "nomic-embed-text", notebook_id="nb_1", selected_sources=["pump.pdf"], core_category="3_pumps_piping_fluid")
    print([r["text"] for r in res1])
    assert len(res1) == 1
    assert res1[0]["text"] == "pump maintenance"
    
    print("Testing Core Only...")
    res2 = retrieve_vector("pump", db_dir, "notebooks", 2, "nomic-embed-text", notebook_id="nb_1", selected_sources=["core"], core_category="3_pumps_piping_fluid")
    print([r["text"] for r in res2])
    assert len(res2) == 1
    assert res2[0]["text"] == "core pump info"
    
    print("Testing Both...")
    res3 = retrieve_vector("pump", db_dir, "notebooks", 4, "nomic-embed-text", notebook_id="nb_1", selected_sources=["core", "pump.pdf"], core_category="3_pumps_piping_fluid")
    print([r["text"] for r in res3])
    assert len(res3) == 2
    
    print("ALL TESTS PASSED")

if __name__ == "__main__":
    run_test()
