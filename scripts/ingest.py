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
from typing import Any

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
        "options": {"num_predict": 500, "temperature": 0.1},
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
) -> str:
    """
    Convert PDF using Docling Python API.
    Returns enhanced markdown string (may include vision-captioned figures).
    Raises ImportError if Docling Python package is unavailable.
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
    # Append figure captions as a dedicated section
    # ------------------------------------------------------------------
    pictures = _pictures_from_doc(doc)
    log.info("[ingest] Found %d picture(s) in %s", len(pictures), pdf_path.name)

    figure_blocks: list[str] = []
    for idx, pic in enumerate(pictures[:max_images], 1):
        try:
            pil_img = pic.get_image(doc)
            if pil_img is None:
                continue

            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            log.info("[ingest]   Captioning figure %d/%d …",
                     idx, min(len(pictures), max_images))
            caption = _call_vision(img_bytes, "", vision_model, vision_timeout)

            if caption:
                # Alt text from docling (caption attribute on PictureItem)
                alt = getattr(pic, "caption", None) or getattr(pic, "text", None)
                alt_text = str(alt).strip() if alt else ""
                label = f"Figure {idx}" + (f": {alt_text}" if alt_text else "")

                figure_blocks.append(
                    f"<!-- FIGURE {idx} -->\n"
                    f"**[{label}]**\n\n"
                    f"*Vision analysis:* {caption}"
                )
                log.info("[ingest]   Figure %d: %d chars", idx, len(caption))
            else:
                log.warning("[ingest]   Figure %d: vision returned empty", idx)

        except Exception as exc:
            log.warning("[ingest]   Figure %d error: %s", idx, exc)

    if figure_blocks:
        md_content += (
            "\n\n## Figures and Diagrams\n\n"
            + "\n\n---\n\n".join(figure_blocks)
        )

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
# Public API
# ---------------------------------------------------------------------------

def ingest_pdf(
    pdf_path: Path,
    processed_dir: Path,
    docling_exe: str = "docling",
    cfg: dict | None = None,
) -> Path:
    """
    Convert a PDF to enhanced Markdown and write to processed_dir.

    Tries Docling Python API (with OCR + vision captioning) first.
    Falls back to Docling CLI on any import or conversion error.

    Returns the path to the output .md file.
    Called by server.py on upload and by main() for batch ingestion.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_md = processed_dir / f"{pdf_path.stem}.md"

    # Load vision settings from cfg or default config
    if cfg is None:
        cfg = _find_config()

    vision_enabled, vision_model, vision_timeout, max_images = _vision_settings(cfg)

    # ── Attempt 1: Python API ────────────────────────────────────────────
    try:
        md_content = _build_markdown_python_api(
            pdf_path, vision_enabled, vision_model, vision_timeout, max_images
        )
        if not md_content.strip():
            raise RuntimeError("Python API produced empty markdown")
        out_md.write_text(md_content, encoding="utf-8")
        log.info("[ingest] ✓ %s → %d chars (Python API)", pdf_path.name, len(md_content))
        return out_md
    except ImportError:
        log.info("[ingest] Docling Python package not available; using CLI")
    except Exception as exc:
        log.warning("[ingest] Python API failed (%s); falling back to CLI", exc)

    # ── Attempt 2: CLI fallback ──────────────────────────────────────────
    result = _ingest_via_cli(pdf_path, processed_dir, docling_exe)
    log.info("[ingest] ✓ %s → CLI (no vision)", pdf_path.name)
    return result


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
            tmp_dir = output_md.parent / "_docling_tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            temp_root = output_md.parent / "_docling_runtime_tmp"
            run_docling(pdf_path, tmp_dir, docling_exe, temp_root)
            produced_md = select_markdown(tmp_dir, pdf_path.stem)
            output_md.write_text(
                produced_md.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8"
            )
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
