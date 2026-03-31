import logging
from pathlib import Path
import subprocess
import sys

# Make sure scripts directory is in path
sys.path.insert(0, str(Path(__file__).parent))
from common import resolve_path

log = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [preflight] %(message)s")
    
    core_dir = resolve_path("core_knowledge/fundamentals")
    if not core_dir.exists():
        log.info("core_knowledge/fundamentals not found. Skipping core preflight.")
        return
        
    log.info("Running core knowledge ingestion and indexing preflight...")
    
    cmd = [sys.executable, str(Path(__file__).parent / "ingest_core.py")]
    try:
        subprocess.run(cmd, check=True)
        log.info("Core knowledge preflight complete.")
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to process core knowledge: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
