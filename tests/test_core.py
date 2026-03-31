import pytest
import os
import sys
from pathlib import Path

# Add scripts directory to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from common import resolve_path, load_config
from index import split_markdown

def test_resolve_path():
    assert resolve_path("test.txt").name == "test.txt"
    assert resolve_path(Path("test.txt")).name == "test.txt"

def test_split_markdown():
    sample_md = """# Header 1
This is some text under header 1.
## Header 2
This is under header 2. Focus on fuel pump.
"""
    chunks = split_markdown(
        markdown=sample_md,
        min_chars=10,
        max_chars=500,
        overlap=50,
        source_name="test.md",
        notebook_id="test_nb"
    )
    
    assert len(chunks) == 2
    assert chunks[0].section_path == "Header 1"
    assert "fuel pump" in chunks[1].text
    assert chunks[1].section_path == "Header 1 > Header 2"
    assert chunks[0].source == "test.md"
