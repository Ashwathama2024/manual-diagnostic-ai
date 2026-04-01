"""
test_search.py — Integration tests for retrieve_vector.

These tests require a live Ollama instance and are skipped in CI
unless MANUALIQ_INTEGRATION=1 is set in the environment.
"""
import os
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

INTEGRATION = os.environ.get("MANUALIQ_INTEGRATION", "0") == "1"
skip_no_ollama = pytest.mark.skipif(
    not INTEGRATION,
    reason="Requires live Ollama — set MANUALIQ_INTEGRATION=1 to run"
)


@skip_no_ollama
def test_retrieve_vector_notebook_only(tmp_path):
    """retrieve_vector returns only chunks matching the requested notebook + source."""
    import lancedb
    import pyarrow as pa
    from langchain_ollama import OllamaEmbeddings
    from query import retrieve_vector

    db = lancedb.connect(str(tmp_path / "lancedb"))
    embedder = OllamaEmbeddings(model="nomic-embed-text")
    vec_pump   = embedder.embed_query("pump manual")
    vec_boiler = embedder.embed_query("boiler parts")

    schema = pa.schema([
        ("id",          pa.string()),
        ("vector",      pa.list_(pa.float32(), 768)),
        ("text",        pa.string()),
        ("notebook_id", pa.string()),
        ("source",      pa.string()),
    ])
    data = [
        {"id": "1", "vector": vec_pump,   "text": "pump maintenance",     "notebook_id": "nb_1", "source": "pump.pdf"},
        {"id": "2", "vector": vec_boiler, "text": "boiler maintenance",   "notebook_id": "nb_1", "source": "boiler.pdf"},
        {"id": "3", "vector": vec_pump,   "text": "pump maintenance alt", "notebook_id": "nb_2", "source": "pump.pdf"},
    ]
    db.create_table("manual_chunks", data=data, schema=schema)

    res = retrieve_vector(
        "pump", str(tmp_path / "lancedb"), "manual_chunks", 2,
        "nomic-embed-text", notebook_id="nb_1", selected_sources=["pump.pdf"]
    )
    assert len(res) == 1
    assert res[0]["text"] == "pump maintenance"


@skip_no_ollama
def test_retrieve_vector_core_only(tmp_path):
    """retrieve_vector returns only core knowledge chunks when 'core' is the source."""
    import lancedb
    import pyarrow as pa
    from langchain_ollama import OllamaEmbeddings
    from query import retrieve_vector

    db = lancedb.connect(str(tmp_path / "lancedb"))
    embedder = OllamaEmbeddings(model="nomic-embed-text")
    core_vec = embedder.embed_query("pump fundamentals core knowledge")

    core_schema = pa.schema([
        ("id",                 pa.string()),
        ("vector",             pa.list_(pa.float32(), 768)),
        ("text",               pa.string()),
        ("equipment_category", pa.string()),
    ])
    core_data = [
        {"id": "c1", "vector": core_vec, "text": "core pump info",   "equipment_category": "3_pumps_piping_fluid"},
        {"id": "c2", "vector": core_vec, "text": "core boiler info", "equipment_category": "5_boiler_steam"},
    ]
    db.create_table("core_knowledge", data=core_data, schema=core_schema)

    res = retrieve_vector(
        "pump", str(tmp_path / "lancedb"), "manual_chunks", 2,
        "nomic-embed-text", notebook_id="nb_1",
        selected_sources=["core"], core_category="3_pumps_piping_fluid"
    )
    assert len(res) == 1
    assert res[0]["text"] == "core pump info"
