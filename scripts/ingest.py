from __future__ import annotations

"""
ingest.py — PDF → Enhanced Markdown

Pipeline (with graceful fallback at each stage):
  1. Docling Python API  (OCR + table structure + image extraction)
     └─ For each extracted image → Ollama vision model (minicpm-v) → caption
  2. Fallback: Docling CLI  (plain markdown, no vision)

Figure captions are injected as a "## Figures and Diagrams" section so the
existing chunker picks them up as searchable text.
"""

import argparse
import base64
import io
import json
import logging
import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Callable

from common import ensure_command, load_config, require_path, resolve_path

log = logging.getLogger(__name__)

OLLAMA_BASE = "http://localhost:11434"

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _find_config() -> dict[str, Any]:
    """Load config.yaml from default locations."""
    here = Path(__file__).parent
    for p in [here.parent / "config.yaml", here / "config.yaml"]:
        if p.exists():
            try:
                return load_config(str(p))
            except Exception:
                pass
    return {}


def _vision_settings(cfg: dict | None) -> tuple[bool, str, int, int]:
    """Return (enabled, model_name, timeout_s, max_images_per_doc)."""
    c = cfg or {}
    vis = c.get("vision", {})
    mdl = c.get("models", {})
    return (
        bool(vis.get("enabled", True)),
        str(mdl.get("vision", "minicpm-v")),
        int(vis.get("timeout", 90)),
        int(vis.get("max_per_doc", 30)),
    )


# ---------------------------------------------------------------------------
# Vision captioning
# ---------------------------------------------------------------------------

def _call_vision(img_bytes: bytes, context: str, model: str, timeout: int) -> str:
    """
    Send a PNG image to an Ollama vision model and return its description.
    Uses raw urllib so no extra dependencies are required.
    Returns empty string on any failure.
    """
    b64 = base64.b64encode(img_bytes).decode()
    prompt = (
        f"Surrounding text context: {context[:400]}\n\n"
        "You are analyzing a page from a technical engineering manual. "
        "Describe this image thoroughly. Include: component names, labels, "
        "measurements, arrows and flow directions, connection points, warning symbols, "
        "any visible text, and what the diagram illustrates. "
        "Be specific and technical — this description will be used for maintenance "
        "and troubleshooting searches."
    )
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "images": [b64],
        "stream": False,
        "options": {"num_predict": 1000, "temperature": 0.1},
    }).encode()

    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip()
    except Exception as exc:
        log.warning("Vision caption failed (%s): %s", model, exc)
        return ""


# ---------------------------------------------------------------------------
# Inline figure placement helper
# ---------------------------------------------------------------------------

def _insert_figure_inline(md: str, caption_block: str, page_no: int | None, total_pages: int) -> str:
    """Insert a figure caption block at its approximate page position in the markdown.

    Uses the figure's page provenance to estimate a character offset, then snaps
    to the nearest paragraph boundary so the caption sits near the surrounding text
    rather than being dumped at the end of the document.

    Falls back to end-of-document if page provenance is unavailable.
    """
    if page_no is None or total_pages <= 1 or not md:
        return md + "\n\n" + caption_block

    # Fraction through the document (0-indexed page → 0.0–1.0)
    frac = max(0.0, (page_no - 1) / total_pages)
    target = int(frac * len(md))

    # Snap forward to the next paragraph break so we don't cut mid-sentence
    insert_at = md.find("\n\n", target)
    if insert_at == -1:
        insert_at = len(md)
    else:
        insert_at += 2  # place after the blank line

    return md[:insert_at] + caption_block + "\n\n" + md[insert_at:]


# ---------------------------------------------------------------------------
# Docling Python API path (OCR + vision)
# ---------------------------------------------------------------------------

def _pictures_from_doc(doc: Any) -> list[Any]:
    """Extract picture items from a DoclingDocument — tries multiple APIs."""
    # Docling 2.x: doc.pictures is a list of PictureItem
    pics = getattr(doc, "pictures", None)
    if pics is not None:
        return list(pics)

    # Fallback: iterate items looking for PictureItem
    try:
        from docling.datamodel.document import PictureItem
        return [item for item, _ in doc.iterate_items() if isinstance(item, PictureItem)]
    except Exception:
        pass

    try:
        from docling_core.types.doc import PictureItem
        return [item for item, _ in doc.iterate_items() if isinstance(item, PictureItem)]
    except Exception:
        pass

    return []


def _build_markdown_python_api(
    pdf_path: Path,
    vision_enabled: bool,
    vision_model: str,
    vision_timeout: int,
    max_images: int,
    images_budget: list[int] | None = None,
) -> str:
    """
    Convert PDF using Docling Python API.
    Returns enhanced markdown string (may include vision-captioned figures).
    Raises ImportError if Docling Python package is unavailable.

    images_budget: mutable single-element list [remaining_count] shared across
    batch calls so the per-document cap applies to the whole document, not per batch.
    When None a fresh budget of max_images is created (single-batch behaviour).
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat

    opts = PdfPipelineOptions()
    opts.do_ocr = True                  # ← critical for scanned / image-text PDFs
    opts.do_table_structure = True
    if vision_enabled:
        opts.generate_picture_images = True   # extract raster images for captioning
        opts.images_scale = 1.5               # higher resolution for better OCR/vision

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )

    log.info("[ingest] Docling Python API: converting %s (OCR=%s, vision=%s)",
             pdf_path.name, True, vision_enabled)
    result = converter.convert(str(pdf_path))
    doc = result.document

    # Stable export — handles text, headings, tables, code blocks
    md_content: str = doc.export_to_markdown()

    if not vision_enabled:
        return md_content

    # ------------------------------------------------------------------
    # Inject figure captions inline at approximate page position
    # ------------------------------------------------------------------
    pictures = _pictures_from_doc(doc)
    total_pages = max(len(getattr(doc, "pages", None) or {}), 1)
    log.info("[ingest] Found %d picture(s) in %s", len(pictures), pdf_path.name)

    budget = images_budget if images_budget is not None else [max_images]

    for idx, pic in enumerate(pictures, 1):
        if budget[0] <= 0:
            log.info("[ingest] Image budget exhausted — skipping remaining figures")
            break
        try:
            pil_img = pic.get_image(doc)
            if pil_img is None:
                continue

            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            # Pass leading markdown as context so minicpm-v knows what section it's in
            fig_global = max_images - budget[0] + 1
            log.info("[ingest]   Captioning figure %d (budget remaining: %d)…",
                     fig_global, budget[0])
            caption = _call_vision(img_bytes, md_content[:800], vision_model, vision_timeout)

            if caption:
                alt = getattr(pic, "caption", None) or getattr(pic, "text", None)
                alt_text = str(alt).strip() if alt else ""
                label = f"Figure {fig_global}" + (f": {alt_text}" if alt_text else "")

                caption_block = (
                    f"<!-- FIGURE {fig_global} -->\n"
                    f"**[{label}]**\n\n"
                    f"*Vision analysis:* {caption}"
                )

                # Get page provenance for inline placement
                page_no: int | None = None
                if hasattr(pic, "prov") and pic.prov:
                    page_no = getattr(pic.prov[0], "page_no", None)

                md_content = _insert_figure_inline(
                    md_content, caption_block, page_no, total_pages
                )
                budget[0] -= 1
                log.info("[ingest]   Figure %d: %d chars (page %s)",
                         fig_global, len(caption), page_no)
            else:
                log.warning("[ingest]   Figure %d: vision returned empty", idx)

        except Exception as exc:
            log.warning("[ingest]   Figure %d error: %s", idx, exc)

    return md_content


# ---------------------------------------------------------------------------
# CLI fallback (original approach)
# ---------------------------------------------------------------------------

def _ingest_via_cli(pdf_path: Path, processed_dir: Path, docling_exe: str) -> Path:
    """Original subprocess-based ingestion (no vision)."""
    tmp_dir = processed_dir / "_docling_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    temp_root = processed_dir / "_docling_runtime_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TMP"] = str(temp_root)
    env["TEMP"] = str(temp_root)

    cmd = [docling_exe, str(pdf_path), "--output", str(tmp_dir), "--to", "md"]
    log.info("[ingest] CLI fallback: %s", " ".join(cmd))
    sub = subprocess.run(cmd, check=False, capture_output=True, text=True,
                         env=env, timeout=300)

    if sub.returncode != 0:
        raise RuntimeError(
            f"Docling CLI failed.\nCommand: {' '.join(cmd)}\n"
            f"stdout:\n{sub.stdout}\nstderr:\n{sub.stderr}"
        )

    direct = tmp_dir / f"{pdf_path.stem}.md"
    if direct.exists():
        md_src = direct
    else:
        mds = sorted(tmp_dir.glob("*.md"))
        if not mds:
            raise FileNotFoundError(f"No Markdown output found in {tmp_dir}")
        md_src = mds[0]

    out_md = processed_dir / f"{pdf_path.stem}.md"
    out_md.write_text(md_src.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    return out_md


# ---------------------------------------------------------------------------
# PDF page-batch splitter  (RAM-safe large-file ingestion)
# ---------------------------------------------------------------------------

def split_pdf_pages(
    pdf_path: Path,
    pages_per_batch: int,
    overlap_pages: int = 2,
) -> tuple[list[Path], int]:
    """Split a large PDF into smaller temp PDFs for RAM-safe Docling processing.

    Each batch overlaps the previous one by `overlap_pages` pages so that
    diagrams or text that straddle a batch boundary are seen in full context
    by both Docling and the vision model.

    Returns (batch_paths, total_pages).
    If total pages <= pages_per_batch, returns ([pdf_path], total_pages) — no temp files.
    Temp batch files are written alongside pdf_path and named <stem>__batch_<NNNN>.pdf.
    The caller is responsible for deleting them after use.
    """
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        log.warning("[ingest] pypdf not installed — processing whole PDF (may be RAM-heavy)")
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            return [pdf_path], len(reader.pages)
        except Exception:
            return [pdf_path], 0

    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)

    if total <= pages_per_batch:
        return [pdf_path], total

    overlap = max(0, min(overlap_pages, pages_per_batch - 1))
    step = pages_per_batch - overlap  # advance by this many pages each batch

    log.info(
        "[ingest] PDF has %d pages — splitting into batches of %d (overlap=%d)",
        total, pages_per_batch, overlap,
    )
    batches: list[Path] = []
    start = 0
    while start < total:
        end = min(start + pages_per_batch, total)
        writer = PdfWriter()
        for p in range(start, end):
            writer.add_page(reader.pages[p])
        out = pdf_path.parent / f"{pdf_path.stem}__batch_{start:04d}.pdf"
        with open(out, "wb") as fh:
            writer.write(fh)
        batches.append(out)
        if end == total:
            break
        start += step

    return batches, total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_pdf(
    pdf_path: Path,
    processed_dir: Path,
    docling_exe: str = "docling",
    cfg: dict | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> Path:
    """
    Convert a PDF to enhanced Markdown and write to processed_dir.

    Tries Docling Python API (with OCR + vision captioning) first.
    Falls back to Docling CLI on any import or conversion error.

    Large PDFs are automatically split into page batches (pdf_pages_per_batch from cfg)
    so that Docling/RapidOCR never tries to allocate RAM for the full document at once.
    Batch temp files are cleaned up after merging.

    progress_cb — optional callable(msg: str) for live status updates (used by SSE upload).

    Returns the path to the output .md file.
    Called by server.py on upload and by main() for batch ingestion.
    """
    _cb = progress_cb or (lambda _: None)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_md = processed_dir / f"{pdf_path.stem}.md"

    # Load vision settings from cfg or default config
    if cfg is None:
        cfg = _find_config()

    vision_enabled, vision_model, vision_timeout, max_images = _vision_settings(cfg)
    pages_per_batch = int(cfg.get("indexing", {}).get("pdf_pages_per_batch", 15))
    overlap_pages = int(cfg.get("indexing", {}).get("batch_overlap_pages", 2))

    # ── Attempt 1: Python API (with page-batch splitting) ────────────────
    try:
        batches, total_pages = split_pdf_pages(pdf_path, pages_per_batch, overlap_pages)
        is_batched = len(batches) > 1

        # Shared budget so the max_images cap applies across all batches, not per-batch
        images_budget: list[int] = [max_images]

        md_parts: list[str] = []
        for idx, batch_path in enumerate(batches):
            if is_batched:
                start_pg = idx * (pages_per_batch - overlap_pages) + 1
                end_pg = min(start_pg + pages_per_batch - 1, total_pages)
                msg = f"Converting pages {start_pg}–{end_pg} of {total_pages}…"
                _cb(msg)
                log.info("[ingest] %s", msg)
            else:
                _cb(f"Converting {pdf_path.name}…")

            part = _build_markdown_python_api(
                batch_path, vision_enabled, vision_model, vision_timeout,
                max_images, images_budget
            )
            md_parts.append(part)

            # Clean up temp batch PDF (but NOT the original)
            if is_batched and batch_path != pdf_path:
                try:
                    batch_path.unlink()
                except Exception:
                    pass

        md_content = "\n\n---\n\n".join(p for p in md_parts if p.strip())
        if not md_content.strip():
            raise RuntimeError("Python API produced empty markdown")

        out_md.write_text(md_content, encoding="utf-8")
        log.info("[ingest] ✓ %s → %d chars (Python API, %d batch(es))",
                 pdf_path.name, len(md_content), len(batches))
        return out_md

    except ImportError:
        log.info("[ingest] Docling Python package not available; using CLI")
        # Clean up any orphaned batch files on ImportError
        try:
            for f in pdf_path.parent.glob(f"{pdf_path.stem}__batch_*.pdf"):
                f.unlink()
        except Exception:
            pass
    except Exception as exc:
        log.warning("[ingest] Python API failed (%s); falling back to CLI", exc)
        try:
            for f in pdf_path.parent.glob(f"{pdf_path.stem}__batch_*.pdf"):
                f.unlink()
        except Exception:
            pass

    # ── Attempt 2: CLI fallback ──────────────────────────────────────────
    _cb(f"Converting {pdf_path.name} via CLI…")
    result = _ingest_via_cli(pdf_path, processed_dir, docling_exe)
    log.info("[ingest] ✓ %s → CLI (no vision)", pdf_path.name)
    return result


# ---------------------------------------------------------------------------
# DOCX ingestion
# ---------------------------------------------------------------------------

def ingest_docx(
    docx_path: Path,
    processed_dir: Path,
    cfg: dict | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> Path:
    """
    Convert a DOCX / DOC file to Markdown using the Docling Python API.
    DOCX is already structured text so no OCR or page-batching is needed.
    Falls back to the Docling CLI on ImportError or conversion failure.
    """
    _cb = progress_cb or (lambda _: None)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_md = processed_dir / f"{docx_path.stem}.md"

    _cb(f"Converting {docx_path.name}…")
    log.info("[ingest] Converting DOCX: %s", docx_path.name)

    # Attempt 1: Docling Python API
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(docx_path))
        md_content = result.document.export_to_markdown()

        if not md_content.strip():
            raise RuntimeError("Docling returned empty markdown for DOCX")

        out_md.write_text(md_content, encoding="utf-8")
        log.info("[ingest] ✓ %s → %d chars (DOCX Python API)", docx_path.name, len(md_content))
        return out_md

    except ImportError:
        log.info("[ingest] Docling Python package not available; using CLI for DOCX")
    except Exception as exc:
        log.warning("[ingest] DOCX Python API failed (%s); falling back to CLI", exc)

    # Attempt 2: CLI fallback
    _cb(f"Converting {docx_path.name} via CLI…")
    result = _ingest_via_cli(docx_path, processed_dir, "docling")
    log.info("[ingest] ✓ %s → CLI (DOCX)", docx_path.name)
    return result


# ---------------------------------------------------------------------------
# Universal entry point — dispatches by extension
# ---------------------------------------------------------------------------

def ingest_file(
    file_path: Path,
    processed_dir: Path,
    docling_exe: str = "docling",
    cfg: dict | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> Path:
    """
    Convert any supported document (PDF, DOCX, DOC) to Markdown.
    Routes to ingest_pdf() for PDFs (with page-batching) or ingest_docx() for Word files.
    """
    ext = file_path.suffix.lower()
    if ext in {".docx", ".doc"}:
        return ingest_docx(file_path, processed_dir, cfg, progress_cb)
    else:
        return ingest_pdf(file_path, processed_dir, docling_exe, cfg, progress_cb)


# ---------------------------------------------------------------------------
# CLI entry point  (unchanged behaviour)
# ---------------------------------------------------------------------------

def run_docling(pdf_path: Path, out_dir: Path, docling_exe: str, temp_root: Path) -> None:
    """Direct Docling CLI call (kept for main() compatibility)."""
    temp_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TMP"] = str(temp_root)
    env["TEMP"] = str(temp_root)

    cmd = [docling_exe, str(pdf_path), "--output", str(out_dir), "--to", "md"]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            "Docling conversion failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def select_markdown(out_dir: Path, expected_stem: str) -> Path:
    direct = out_dir / f"{expected_stem}.md"
    if direct.exists():
        return direct
    md_files = sorted(out_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No Markdown output found in {out_dir}")
    return md_files[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PDF manual(s) to Markdown using Docling (with optional vision)."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input-pdf", default=None, help="Ingest a single PDF")
    parser.add_argument("--all", action="store_true", help="Ingest all PDFs in raw_dir")
    parser.add_argument("--output-md", default=None, help="Override output path (single-file only)")
    parser.add_argument("--no-vision", action="store_true", help="Disable vision captioning")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    if args.no_vision:
        cfg.setdefault("vision", {})["enabled"] = False

    docling_exe = ensure_command("docling", "Install with `pip install docling`.")

    paths = cfg["paths"]
    if "raw_dir" in paths:
        raw_dir = resolve_path(paths["raw_dir"])
        processed_dir = resolve_path(paths["processed_dir"])
    else:
        input_pdf_path = resolve_path(paths["input_pdf"])
        processed_dir = resolve_path(paths["markdown_output"]).parent
        raw_dir = input_pdf_path.parent

    processed_dir.mkdir(parents=True, exist_ok=True)

    if args.input_pdf:
        pdf_path = require_path(resolve_path(args.input_pdf), "Input PDF")
        if args.output_md:
            output_md = resolve_path(args.output_md)
            output_md.parent.mkdir(parents=True, exist_ok=True)
            # Route through ingest_pdf() so vision + batching are applied
            tmp_out = ingest_pdf(pdf_path, output_md.parent, docling_exe, cfg)
            if tmp_out.resolve() != output_md.resolve():
                tmp_out.rename(output_md)
            print(f"Markdown written: {output_md}")
        else:
            out_md = ingest_pdf(pdf_path, processed_dir, docling_exe, cfg)
            print(f"Markdown written: {out_md}")

    elif args.all:
        pdf_files = sorted(raw_dir.glob("*.pdf"))
        if not pdf_files:
            raise RuntimeError(f"No PDF files found in {raw_dir}")
        for pdf_path in pdf_files:
            print(f"Ingesting: {pdf_path.name} …")
            out_md = ingest_pdf(pdf_path, processed_dir, docling_exe, cfg)
            print(f"  Written: {out_md}")
        print(f"\nIngested {len(pdf_files)} PDF(s).")

    else:
        if "raw_dir" in paths:
            raise RuntimeError("Use --all or --input-pdf <path>.")
        legacy_pdf = require_path(resolve_path(paths["input_pdf"]), "Input PDF")
        out_md = ingest_pdf(legacy_pdf, processed_dir, docling_exe, cfg)
        print(f"Markdown written: {out_md}")


if __name__ == "__main__":
    main()
