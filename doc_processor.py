"""
Document Processing Pipeline — Semantic Edition
=================================================
Handles PDF manuals containing text, tables, diagrams, and images.
Extracts all content with SECTION-AWARE semantic chunking that preserves
the document structure (chapters, sections, subsections) so every chunk
carries its full context hierarchy.

Pipeline:
  PDF → PyMuPDF (text + structure) → pdfplumber (tables) → OCR (images)
      → Section detection → Semantic chunking → Chunks with full references

Every chunk knows:
  - Which manual it came from
  - Which chapter / section / subsection
  - Which page
  - What type of content (text, table, diagram)
"""

import os
import io
import re
import hashlib
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """A single chunk of extracted content from a manual, with full context."""
    text: str
    source_file: str
    page_number: int
    chunk_type: str  # "text", "table", "image_ocr"
    equipment_id: str
    section_title: str = ""       # e.g., "3.2.1 Fuel Injection Timing"
    section_hierarchy: str = ""   # e.g., "Chapter 3 > Fuel System > Injection Timing"
    chapter: str = ""             # e.g., "Chapter 3: Fuel System"
    chunk_id: str = ""
    image_path: str = ""           # path to saved image file (for image/diagram chunks)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.chunk_id:
            content_hash = hashlib.md5(
                f"{self.source_file}:{self.page_number}:{self.text[:100]}".encode()
            ).hexdigest()[:12]
            self.chunk_id = f"{self.equipment_id}_{content_hash}"

    @property
    def reference(self) -> str:
        """Human-readable reference string for citations."""
        parts = []
        if self.source_file:
            parts.append(self.source_file)
        if self.section_hierarchy:
            parts.append(self.section_hierarchy)
        elif self.section_title:
            parts.append(self.section_title)
        parts.append(f"Page {self.page_number}")
        return " > ".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Section / Heading detection
# ---------------------------------------------------------------------------

# Patterns that indicate section headings in technical manuals
HEADING_PATTERNS = [
    # "CHAPTER 3", "Chapter 3: Fuel System"
    re.compile(r'^(CHAPTER|Chapter)\s+(\d+)[\s:.–—-]*(.*)', re.MULTILINE),
    # "3.2.1 Fuel Injection Timing", "3.2.1. Fuel Injection"
    re.compile(r'^(\d+(?:\.\d+)+)\.?\s+([A-Z].*)', re.MULTILINE),
    # "SECTION 3.2", "Section 3.2 — Fuel System"
    re.compile(r'^(SECTION|Section)\s+([\d.]+)[\s:.–—-]*(.*)', re.MULTILINE),
    # "3. FUEL SYSTEM", "3 FUEL SYSTEM" (single-level numbered, ALL CAPS title)
    re.compile(r'^(\d+)\.?\s+([A-Z][A-Z\s]{3,})', re.MULTILINE),
    # "PART 3", "PART III"
    re.compile(r'^(PART)\s+(\w+)[\s:.–—-]*(.*)', re.MULTILINE),
    # ALL CAPS lines that are likely headers (min 4 chars, max 80)
    re.compile(r'^([A-Z][A-Z\s\-/&]{3,78})$', re.MULTILINE),
]


def detect_sections(text: str) -> list[dict]:
    """
    Detect section headings in text and split into sections.
    Returns list of {title, level, start_pos, text}.

    Level logic:
      - "CHAPTER X" → level 1
      - "X.Y Title" → level = count of dots + 1
      - "SECTION X" → level 2
      - ALL CAPS line → level 2
    """
    headings = []

    for pattern in HEADING_PATTERNS:
        for match in pattern.finditer(text):
            start = match.start()
            full_match = match.group(0).strip()

            # Determine level
            groups = match.groups()
            first = groups[0] if groups else ""

            if first.upper() in ("CHAPTER", "PART"):
                level = 1
                title = full_match
            elif first.upper() == "SECTION":
                level = 2
                title = full_match
            elif re.match(r'^\d+(?:\.\d+)+', first):
                # Numbered section: count dots for level
                level = first.count('.') + 1
                title = full_match
            elif re.match(r'^\d+$', first):
                level = 1
                title = full_match
            elif first == first.upper() and len(first) > 3:
                # ALL CAPS line
                level = 2
                title = full_match
            else:
                level = 2
                title = full_match

            # Skip very short "headings" that are probably not headings
            if len(title.strip()) < 3:
                continue

            headings.append({
                "title": title.strip(),
                "level": level,
                "start_pos": start,
            })

    # Remove duplicates (same position)
    seen_positions = set()
    unique_headings = []
    for h in sorted(headings, key=lambda x: x["start_pos"]):
        if h["start_pos"] not in seen_positions:
            seen_positions.add(h["start_pos"])
            unique_headings.append(h)

    return unique_headings


def build_section_hierarchy(headings: list[dict], current_idx: int) -> str:
    """
    Build a breadcrumb hierarchy string for the heading at current_idx.
    E.g., "Chapter 3: Fuel System > 3.2 Injection > 3.2.1 Timing Adjustment"
    """
    if not headings or current_idx < 0:
        return ""

    current = headings[current_idx]
    chain = [current["title"]]

    # Walk backwards to find parent headings (lower level numbers)
    target_level = current["level"] - 1
    for i in range(current_idx - 1, -1, -1):
        if headings[i]["level"] <= target_level:
            chain.insert(0, headings[i]["title"])
            target_level = headings[i]["level"] - 1
            if target_level < 1:
                break

    return " > ".join(chain)


def get_chapter_for_position(headings: list[dict], position: int) -> str:
    """Find the chapter (level-1 heading) that contains the given text position."""
    chapter = ""
    for h in headings:
        if h["start_pos"] > position:
            break
        if h["level"] == 1:
            chapter = h["title"]
    return chapter


def get_section_for_position(headings: list[dict], position: int) -> tuple[str, str]:
    """
    Find the nearest section heading and hierarchy for a text position.
    Returns (section_title, hierarchy_string).
    """
    if not headings:
        return "", ""

    # Find the last heading before this position
    best_idx = -1
    for i, h in enumerate(headings):
        if h["start_pos"] <= position:
            best_idx = i
        else:
            break

    if best_idx < 0:
        return "", ""

    section_title = headings[best_idx]["title"]
    hierarchy = build_section_hierarchy(headings, best_idx)
    return section_title, hierarchy


# ---------------------------------------------------------------------------
# OCR helper
# ---------------------------------------------------------------------------

def ocr_image(image: Image.Image) -> str:
    """Extract text from an image using Tesseract OCR."""
    try:
        import pytesseract
        text = pytesseract.image_to_string(image, config="--psm 6")
        return text.strip()
    except ImportError:
        logger.warning("pytesseract not installed — skipping OCR")
        return ""
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------

def save_image_to_disk(
    image_bytes: bytes,
    equipment_id: str,
    source_file: str,
    page_number: int,
    image_index: int,
    base_dir: str = "./images",
) -> str:
    """
    Save an extracted image to disk in an organized directory structure.

    Path: {base_dir}/{equipment_id}/{source_stem}_page{N}_img{M}.png

    Returns the relative path to the saved image, or "" on failure.
    """
    try:
        # Sanitize source filename for use in path
        stem = os.path.splitext(os.path.basename(source_file))[0]
        stem = re.sub(r'[^\w\-.]', '_', stem)

        # Build directory
        equip_dir = os.path.join(base_dir, equipment_id)
        os.makedirs(equip_dir, exist_ok=True)

        # Build filename
        filename = f"{stem}_page{page_number}_img{image_index}.png"
        filepath = os.path.join(equip_dir, filename)

        # Convert to PNG via PIL (handles JPEG, JPEG2000, etc.)
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode in ("CMYK", "P"):
            image = image.convert("RGB")
        image.save(filepath, format="PNG")

        logger.info(f"Saved image: {filepath}")
        return filepath
    except Exception as e:
        logger.warning(f"Failed to save image to disk: {e}")
        return ""


# ---------------------------------------------------------------------------
# Surrounding text extraction (spatial proximity)
# ---------------------------------------------------------------------------

def extract_surrounding_text(
    page_dict_blocks: list[dict],
    image_bbox: tuple,
    proximity_px: int = 80,
) -> dict:
    """
    Find text blocks spatially adjacent to an image on a PDF page.

    Uses bounding-box geometry to find text above, below, and the likely
    caption for a diagram.

    Args:
        page_dict_blocks: blocks from page.get_text("dict")["blocks"]
        image_bbox: (x0, y0, x1, y1) of the image on the page
        proximity_px: max pixel distance to consider text "near" the image

    Returns:
        {"above": str, "below": str, "caption": str}
    """
    if not page_dict_blocks or not image_bbox:
        return {"above": "", "below": "", "caption": ""}

    img_x0, img_y0, img_x1, img_y1 = image_bbox

    above_blocks = []
    below_blocks = []

    for block in page_dict_blocks:
        # Only consider text blocks (type 0), skip image blocks (type 1)
        if block.get("type", 0) != 0:
            continue

        bx0, by0, bx1, by1 = block["bbox"]

        # Check horizontal overlap (block and image share some x-range)
        h_overlap = (bx0 < img_x1) and (bx1 > img_x0)
        if not h_overlap:
            continue

        # Extract text from block's lines → spans
        block_text = ""
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                block_text += span.get("text", "") + " "
        block_text = block_text.strip()
        if not block_text:
            continue

        # Text ABOVE the image: block's bottom edge (by1) is above image top (img_y0)
        # and within proximity
        if by1 <= img_y0 and (img_y0 - by1) <= proximity_px:
            above_blocks.append({
                "text": block_text,
                "distance": img_y0 - by1,
            })

        # Text BELOW the image: block's top edge (by0) is below image bottom (img_y1)
        # and within proximity
        if by0 >= img_y1 and (by0 - img_y1) <= proximity_px:
            below_blocks.append({
                "text": block_text,
                "distance": by0 - img_y1,
            })

    # Sort by distance (closest first)
    above_blocks.sort(key=lambda x: x["distance"])
    below_blocks.sort(key=lambda x: x["distance"])

    above_text = " ".join(b["text"] for b in above_blocks[:3])
    below_text = " ".join(b["text"] for b in below_blocks[:3])

    # Caption = the single closest text below (likely "Fig. X.Y — ...")
    caption = below_blocks[0]["text"] if below_blocks else ""

    return {
        "above": above_text.strip(),
        "below": below_text.strip(),
        "caption": caption.strip(),
    }


# ---------------------------------------------------------------------------
# Vision model image description
# ---------------------------------------------------------------------------

# Domain-specific prompt for describing engineering diagrams
VISION_PROMPT_ENGINEERING = """You are analyzing a technical diagram from an industrial/marine equipment manual.

Describe this image in detail for a diagnostic knowledge base. Include:
1. WHAT it shows: cross-section, schematic, P&ID, wiring diagram, assembly drawing, exploded view, flow diagram, or photograph?
2. COMPONENTS: List every labeled component, part number, or callout visible.
3. CONNECTIONS: How components connect — flow paths, piping, wiring, mechanical linkages.
4. MEASUREMENTS: Any dimensions, tolerances, clearances, pressures, or temperatures shown.
5. OPERATING CONDITIONS: Normal values, alarm limits, or set points if visible.
6. CONTEXT: What system or subsystem this relates to (fuel, cooling, lubrication, hydraulic, electrical, etc.).

Be specific and technical. Use the exact labels and numbers shown in the diagram."""


def describe_image_with_vision(
    image_bytes: bytes,
    surrounding_context: str = "",
    vision_model: str = "minicpm-v",
    ollama_base_url: str = "http://localhost:11434",
) -> tuple:
    """
    Use an Ollama vision model to describe a technical diagram.
    Falls back to Tesseract OCR if vision model is unavailable.

    Args:
        image_bytes: Raw image bytes (PNG/JPEG)
        surrounding_context: Text found near the image for prompt context
        vision_model: Ollama vision model name
        ollama_base_url: Ollama server URL

    Returns:
        (description: str, method: str) where method is "vision", "ocr", or "none"
    """
    import base64

    # --- Try vision model first ---
    try:
        import ollama as ollama_lib

        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Build prompt with surrounding context if available
        prompt = VISION_PROMPT_ENGINEERING
        if surrounding_context:
            context_snippet = surrounding_context[:500]
            prompt = (
                f"This diagram appears in a section about: {context_snippet}\n\n"
                f"{prompt}"
            )

        client = ollama_lib.Client(host=ollama_base_url)
        response = client.chat(
            model=vision_model,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [b64_image],
            }],
        )

        # Extract response text (handle both dict and object formats)
        description = ""
        if isinstance(response, dict):
            description = response.get("message", {}).get("content", "")
        elif hasattr(response, "message"):
            description = response.message.content or ""

        if description and len(description) > 20:
            logger.info(f"Vision model described image ({len(description)} chars)")
            return description.strip(), "vision"
        else:
            logger.warning("Vision model returned empty/short response, falling back to OCR")

    except ImportError:
        logger.warning("ollama package not installed — falling back to OCR")
    except Exception as e:
        error_str = str(e).lower()
        if "connect" in error_str or "refused" in error_str:
            logger.warning("Ollama not running — falling back to OCR")
        elif "not found" in error_str:
            logger.warning(f"Vision model '{vision_model}' not found — falling back to OCR. "
                           f"Install with: ollama pull {vision_model}")
        else:
            logger.warning(f"Vision model failed ({e}) — falling back to OCR")

    # --- Fallback: Tesseract OCR ---
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        ocr_text = ocr_image(pil_image)
        if ocr_text and len(ocr_text) > 10:
            logger.info(f"OCR fallback produced {len(ocr_text)} chars")
            return ocr_text, "ocr"
    except Exception as e:
        logger.warning(f"OCR fallback also failed: {e}")

    return "", "none"


# ---------------------------------------------------------------------------
# PDF Text Extraction with structure (PyMuPDF)
# ---------------------------------------------------------------------------

def extract_text_with_structure(pdf_path: str) -> list[dict]:
    """
    Extract text page-by-page using PyMuPDF, preserving structure.
    Returns list of {page: int, text: str, headings: list}.
    """
    pages = []
    all_text = ""
    page_offsets = []  # (page_num, start_offset_in_all_text)

    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                page_offsets.append((page_num + 1, len(all_text)))
                all_text += text + "\n\n"
                pages.append({"page": page_num + 1, "text": text.strip()})
        doc.close()
    except Exception as e:
        logger.error(f"PyMuPDF extraction failed for {pdf_path}: {e}")
        return [], []

    # Detect sections across the entire document
    headings = detect_sections(all_text)

    # Map headings back to pages
    for heading in headings:
        pos = heading["start_pos"]
        heading["page"] = 1
        for page_num, offset in page_offsets:
            if offset <= pos:
                heading["page"] = page_num
            else:
                break

    return pages, headings, all_text, page_offsets


# ---------------------------------------------------------------------------
# PDF Image Extraction — Vision-Aware Pipeline
# ---------------------------------------------------------------------------

def extract_images_with_vision(
    pdf_path: str,
    equipment_id: str,
    headings: list = None,
    page_offsets: list = None,
    min_size: int = 100,
    vision_model: str = "minicpm-v",
    ollama_base_url: str = "http://localhost:11434",
    image_base_dir: str = "./images",
    use_vision: bool = True,
    progress_callback=None,
) -> list[dict]:
    """
    Extract images from PDF, save to disk, describe with vision model.

    For each image:
      1. Extract raw bytes from PDF (PyMuPDF)
      2. Get bounding box position on page
      3. Find surrounding text blocks (spatial proximity)
      4. Save image to disk as PNG
      5. Describe with Ollama vision model (or OCR fallback)
      6. Attach section metadata from heading detection

    Args:
        pdf_path: Path to the PDF
        equipment_id: Equipment ID for directory structure
        headings: Detected section headings (from extract_text_with_structure)
        page_offsets: (page_num, text_offset) tuples
        min_size: Minimum image dimension in pixels to process
        vision_model: Ollama vision model name
        ollama_base_url: Ollama server URL
        image_base_dir: Root directory for saved images
        use_vision: If False, skip vision model and use OCR only
        progress_callback: Optional callable(stage, progress_float)

    Returns:
        List of dicts with keys: page, text, image_index, image_path,
        surrounding_text, section_title, section_hierarchy, chapter, method
    """
    results = []
    source_file = os.path.basename(pdf_path)
    seen_hashes = set()  # Dedup by image content hash

    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        for page_num in range(total_pages):
            page = doc[page_num]
            images = page.get_images(full=True)

            if not images:
                continue

            # Get all text blocks on this page (for surrounding text extraction)
            page_dict = page.get_text("dict")
            text_blocks = page_dict.get("blocks", [])

            for img_idx, img_info in enumerate(images):
                xref = img_info[0]

                try:
                    # --- Extract raw image bytes ---
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    # --- Size check ---
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    if pil_image.width < min_size or pil_image.height < min_size:
                        continue

                    # --- Dedup by content hash ---
                    img_hash = hashlib.md5(image_bytes).hexdigest()
                    if img_hash in seen_hashes:
                        continue
                    seen_hashes.add(img_hash)

                    # --- Get image bounding box on page ---
                    image_bbox = None
                    try:
                        rects = page.get_image_rects(xref)
                        if rects:
                            r = rects[0]  # Use first occurrence
                            image_bbox = (r.x0, r.y0, r.x1, r.y1)
                    except Exception:
                        pass  # Some PDFs don't support this

                    # --- Extract surrounding text ---
                    surrounding = {"above": "", "below": "", "caption": ""}
                    if image_bbox and text_blocks:
                        surrounding = extract_surrounding_text(
                            text_blocks, image_bbox, proximity_px=80
                        )

                    surrounding_combined = ""
                    if surrounding["caption"]:
                        surrounding_combined = surrounding["caption"]
                    if surrounding["above"]:
                        surrounding_combined = (
                            surrounding["above"] + " " + surrounding_combined
                        ).strip()

                    # --- Save image to disk ---
                    image_path = save_image_to_disk(
                        image_bytes=image_bytes,
                        equipment_id=equipment_id,
                        source_file=source_file,
                        page_number=page_num + 1,
                        image_index=img_idx,
                        base_dir=image_base_dir,
                    )

                    # --- Describe image (vision or OCR) ---
                    if use_vision:
                        description, method = describe_image_with_vision(
                            image_bytes=image_bytes,
                            surrounding_context=surrounding_combined,
                            vision_model=vision_model,
                            ollama_base_url=ollama_base_url,
                        )
                    else:
                        ocr_text = ocr_image(pil_image)
                        description = ocr_text if ocr_text and len(ocr_text) > 10 else ""
                        method = "ocr" if description else "none"

                    if not description:
                        continue  # Nothing useful extracted

                    # --- Section metadata ---
                    section_title = ""
                    section_hierarchy = ""
                    chapter = ""
                    if headings and page_offsets:
                        # Find text offset for this page
                        img_offset = 0
                        for pnum, poff in page_offsets:
                            if pnum == page_num + 1:
                                img_offset = poff
                                break
                        section_title, section_hierarchy = get_section_for_position(
                            headings, img_offset
                        )
                        chapter = get_chapter_for_position(headings, img_offset)

                    results.append({
                        "page": page_num + 1,
                        "text": description,
                        "image_index": img_idx,
                        "image_path": image_path,
                        "surrounding_text": surrounding_combined,
                        "section_title": section_title,
                        "section_hierarchy": section_hierarchy,
                        "chapter": chapter,
                        "method": method,
                    })

                    if progress_callback:
                        pct = (page_num + 1) / total_pages
                        progress_callback(
                            f"Image {len(results)}: page {page_num+1} ({method})", pct
                        )

                except Exception as e:
                    logger.warning(
                        f"Image extraction failed page {page_num+1} img {img_idx}: {e}"
                    )

        doc.close()
    except Exception as e:
        logger.error(f"Image extraction failed for {pdf_path}: {e}")

    return results


def _extract_images_legacy(pdf_path: str, min_size: int = 100) -> list[dict]:
    """
    Legacy image extraction — Tesseract OCR only.
    Kept as internal fallback reference. Use extract_images_with_vision() instead.
    """
    results = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            images = page.get_images(full=True)
            for img_idx, img_info in enumerate(images):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    if image.width < min_size or image.height < min_size:
                        continue
                    ocr_text = ocr_image(image)
                    if ocr_text and len(ocr_text) > 10:
                        results.append({
                            "page": page_num + 1,
                            "text": ocr_text,
                            "image_index": img_idx,
                        })
                except Exception as e:
                    logger.warning(f"Image extraction failed page {page_num+1} img {img_idx}: {e}")
        doc.close()
    except Exception as e:
        logger.error(f"PyMuPDF image extraction failed for {pdf_path}: {e}")
    return results


# ---------------------------------------------------------------------------
# Table Extraction (pdfplumber)
# ---------------------------------------------------------------------------

def extract_tables_pdfplumber(pdf_path: str) -> list[dict]:
    """
    Extract tables from PDF using pdfplumber.
    Returns list of {page: int, text: str, table_index: int}.
    """
    results = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for tbl_idx, table in enumerate(tables):
                    if not table or len(table) < 2:
                        continue
                    md_lines = []
                    for row_idx, row in enumerate(table):
                        clean_row = [str(cell).strip() if cell else "" for cell in row]
                        md_lines.append("| " + " | ".join(clean_row) + " |")
                        if row_idx == 0:
                            md_lines.append("|" + "|".join(["---"] * len(clean_row)) + "|")
                    table_text = "\n".join(md_lines)
                    if table_text.strip():
                        results.append({
                            "page": page_num + 1,
                            "text": table_text,
                            "table_index": tbl_idx,
                        })
    except Exception as e:
        logger.error(f"pdfplumber table extraction failed for {pdf_path}: {e}")
    return results


# ---------------------------------------------------------------------------
# Semantic Text Chunking
# ---------------------------------------------------------------------------

def semantic_chunk_text(
    text: str,
    headings: list[dict],
    page_offsets: list[tuple],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict]:
    """
    Semantically-aware chunking that respects section boundaries.

    Instead of blindly cutting at character count, this:
    1. Tries to break at section boundaries first
    2. Falls back to paragraph boundaries
    3. Falls back to sentence boundaries
    4. Adds section context to every chunk

    Returns list of {text, page, section_title, section_hierarchy, chapter}.
    """
    if not text or not text.strip():
        return []

    chunks = []

    # If we have headings, try section-based chunking first
    if headings:
        # Create section ranges
        sections = []
        for i, heading in enumerate(headings):
            start = heading["start_pos"]
            end = headings[i + 1]["start_pos"] if i + 1 < len(headings) else len(text)
            section_text = text[start:end].strip()
            if section_text:
                hierarchy = build_section_hierarchy(headings, i)
                chapter = get_chapter_for_position(headings, start)
                page = heading.get("page", 1)
                sections.append({
                    "text": section_text,
                    "title": heading["title"],
                    "hierarchy": hierarchy,
                    "chapter": chapter,
                    "page": page,
                    "start_pos": start,
                })

        # Handle text before first heading
        if headings and headings[0]["start_pos"] > 0:
            pre_text = text[:headings[0]["start_pos"]].strip()
            if pre_text:
                page = _find_page_for_offset(page_offsets, 0)
                sections.insert(0, {
                    "text": pre_text,
                    "title": "",
                    "hierarchy": "",
                    "chapter": "",
                    "page": page,
                    "start_pos": 0,
                })

        # Now chunk each section
        for section in sections:
            section_chunks = _split_section(
                section["text"], chunk_size, chunk_overlap
            )
            for chunk_text in section_chunks:
                # Find the page for this chunk
                chunk_offset = text.find(chunk_text[:50])
                page = section["page"]
                if chunk_offset >= 0:
                    page = _find_page_for_offset(page_offsets, chunk_offset)

                chunks.append({
                    "text": chunk_text,
                    "page": page,
                    "section_title": section["title"],
                    "section_hierarchy": section["hierarchy"],
                    "chapter": section["chapter"],
                })
    else:
        # No headings detected — fall back to basic semantic chunking
        basic_chunks = _split_section(text, chunk_size, chunk_overlap)
        for chunk_text in basic_chunks:
            chunk_offset = text.find(chunk_text[:50])
            page = _find_page_for_offset(page_offsets, chunk_offset) if chunk_offset >= 0 else 1
            chunks.append({
                "text": chunk_text,
                "page": page,
                "section_title": "",
                "section_hierarchy": "",
                "chapter": "",
            })

    return chunks


def _find_page_for_offset(page_offsets: list[tuple], offset: int) -> int:
    """Find which page a text offset belongs to."""
    page = 1
    for page_num, page_start in page_offsets:
        if page_start <= offset:
            page = page_num
        else:
            break
    return page


def _split_section(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split a section of text into chunks, preferring paragraph and sentence breaks.
    """
    if not text or not text.strip():
        return []

    if len(text) <= chunk_size:
        return [text.strip()]

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        )
        return splitter.split_text(text)
    except ImportError:
        # Fallback: paragraph-aware splitting
        chunks = []
        paragraphs = text.split("\n\n")
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk += ("\n\n" if current_chunk else "") + para
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # If single paragraph is too big, split by sentences
                if len(para) > chunk_size:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    current_chunk = ""
                    for sent in sentences:
                        if len(current_chunk) + len(sent) + 1 <= chunk_size:
                            current_chunk += (" " if current_chunk else "") + sent
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sent
                else:
                    current_chunk = para

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Main Processing Pipeline
# ---------------------------------------------------------------------------

def process_pdf(
    pdf_path: str,
    equipment_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    progress_callback=None,
    use_vision: bool = True,
    vision_model: str = None,
    image_base_dir: str = None,
    source_filename: str = "",
) -> list[DocumentChunk]:
    """
    Full processing pipeline for a single PDF manual.

    Steps:
      1. Extract text with structure detection (PyMuPDF)
      2. Detect section headings and build hierarchy
      3. Extract tables (pdfplumber)
      4. Extract images → Vision model description (or OCR fallback)
      5. Semantic chunking with section context
      6. Return DocumentChunk list with full references

    Args:
        pdf_path: Path to the PDF file
        equipment_id: Unique identifier for the equipment
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        progress_callback: Optional callable(stage: str, progress: float)
        use_vision: If True, use Ollama vision model for diagram description
        vision_model: Ollama vision model name (default: env VISION_MODEL or "minicpm-v")
        image_base_dir: Root dir for saved images (default: env IMAGE_STORE_DIR or "./images")
        source_filename: Original filename of the uploaded PDF (for citations).
                         If empty, falls back to os.path.basename(pdf_path).

    Returns:
        List of DocumentChunk objects with section-aware metadata
    """
    filename = source_filename or os.path.basename(pdf_path)
    all_chunks: list[DocumentChunk] = []

    def _progress(stage, pct):
        if progress_callback:
            progress_callback(stage, min(pct, 0.99))

    # --- Stage 1: Text + structure extraction ---
    _progress("Extracting text & structure...", 0.0)
    result = extract_text_with_structure(pdf_path)
    if not result or len(result) < 4:
        logger.error(f"Failed to extract text from {pdf_path}")
        return all_chunks

    pages, headings, full_text, page_offsets = result
    logger.info(f"Extracted {len(pages)} pages, {len(headings)} section headings")

    # --- Stage 2: Semantic chunking ---
    _progress("Semantic chunking...", 0.20)
    text_chunks = semantic_chunk_text(
        full_text, headings, page_offsets, chunk_size, chunk_overlap
    )
    logger.info(f"Created {len(text_chunks)} semantic text chunks")

    for chunk_data in text_chunks:
        # Prefix chunk with section context for better retrieval
        section_prefix = ""
        if chunk_data["section_hierarchy"]:
            section_prefix = f"[{chunk_data['section_hierarchy']}]\n\n"
        elif chunk_data["section_title"]:
            section_prefix = f"[{chunk_data['section_title']}]\n\n"

        all_chunks.append(DocumentChunk(
            text=section_prefix + chunk_data["text"],
            source_file=filename,
            page_number=chunk_data["page"],
            chunk_type="text",
            equipment_id=equipment_id,
            section_title=chunk_data["section_title"],
            section_hierarchy=chunk_data["section_hierarchy"],
            chapter=chunk_data["chapter"],
        ))

    # --- Stage 3: Table extraction ---
    _progress("Extracting tables...", 0.50)
    tables = extract_tables_pdfplumber(pdf_path)
    logger.info(f"Extracted {len(tables)} tables")

    for table_data in tables:
        # Find section context for this table's page
        # Estimate offset from page
        table_offset = 0
        for pnum, poff in page_offsets:
            if pnum == table_data["page"]:
                table_offset = poff
                break
        section_title, hierarchy = get_section_for_position(headings, table_offset)
        chapter = get_chapter_for_position(headings, table_offset)

        table_prefix = ""
        if hierarchy:
            table_prefix = f"[TABLE in {hierarchy}]\n\n"
        elif section_title:
            table_prefix = f"[TABLE in {section_title}]\n\n"

        # Tables: chunk if too big, otherwise keep whole
        table_text_chunks = _split_section(table_data["text"], chunk_size, chunk_overlap)
        for chunk_text_str in table_text_chunks:
            all_chunks.append(DocumentChunk(
                text=table_prefix + chunk_text_str,
                source_file=filename,
                page_number=table_data["page"],
                chunk_type="table",
                equipment_id=equipment_id,
                section_title=section_title,
                section_hierarchy=hierarchy,
                chapter=chapter,
                metadata={"table_index": table_data["table_index"]},
            ))

    # --- Stage 4: Image Vision Processing ---
    _progress("Processing images (vision)...", 0.70)

    # Resolve vision config from params or environment
    _vision_model = vision_model or os.environ.get("VISION_MODEL", "minicpm-v")
    _image_dir = image_base_dir or os.environ.get("IMAGE_STORE_DIR", "./images")
    _ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    image_results = extract_images_with_vision(
        pdf_path=pdf_path,
        equipment_id=equipment_id,
        headings=headings,
        page_offsets=page_offsets,
        vision_model=_vision_model,
        ollama_base_url=_ollama_url,
        image_base_dir=_image_dir,
        use_vision=use_vision,
        progress_callback=lambda stage, pct: _progress(stage, min(0.70 + pct * 0.25, 0.95)),
    )

    vision_count = sum(1 for r in image_results if r["method"] == "vision")
    ocr_count = sum(1 for r in image_results if r["method"] == "ocr")
    logger.info(f"Processed {len(image_results)} images "
                f"({vision_count} vision, {ocr_count} OCR fallback)")

    for img_data in image_results:
        # Build the chunk text with surrounding context + description
        section_title = img_data.get("section_title", "")
        hierarchy = img_data.get("section_hierarchy", "")
        chapter = img_data.get("chapter", "")
        method = img_data.get("method", "ocr")

        # Section prefix (same pattern as text/table chunks)
        diagram_prefix = ""
        if hierarchy:
            diagram_prefix = f"[DIAGRAM in {hierarchy}]\n\n"
        elif section_title:
            diagram_prefix = f"[DIAGRAM in {section_title}]\n\n"

        # Combine surrounding context + description for rich embedding
        combined_text = ""
        surrounding = img_data.get("surrounding_text", "")
        if surrounding:
            combined_text += f"[SURROUNDING CONTEXT]: {surrounding}\n\n"

        description = img_data.get("text", "")
        if description:
            label = "DIAGRAM DESCRIPTION" if method == "vision" else "DIAGRAM OCR TEXT"
            combined_text += f"[{label}]: {description}"

        if not combined_text.strip():
            continue

        full_chunk_text = diagram_prefix + combined_text

        # Determine chunk type based on method used
        chunk_type = "image_vision" if method == "vision" else "image_ocr"

        # Split if too large
        img_chunks = _split_section(full_chunk_text, chunk_size, chunk_overlap)
        for chunk_text_str in img_chunks:
            all_chunks.append(DocumentChunk(
                text=chunk_text_str,
                source_file=filename,
                page_number=img_data["page"],
                chunk_type=chunk_type,
                equipment_id=equipment_id,
                section_title=section_title,
                section_hierarchy=hierarchy,
                chapter=chapter,
                image_path=img_data.get("image_path", ""),
                metadata={
                    "image_index": img_data["image_index"],
                    "vision_model": _vision_model if method == "vision" else "",
                    "has_surrounding_context": bool(surrounding),
                    "description_method": method,
                },
            ))

    _progress("Done", 1.0)
    logger.info(f"Total chunks from {filename}: {len(all_chunks)} "
                f"({len(text_chunks)} text, {len(tables)} tables, "
                f"{len(image_results)} images)")
    return all_chunks


def process_directory(
    dir_path: str,
    equipment_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    progress_callback=None,
) -> list[DocumentChunk]:
    """Process all PDFs in a directory for a given equipment."""
    all_chunks = []
    pdf_files = sorted(Path(dir_path).glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {dir_path}")
        return all_chunks

    for i, pdf_file in enumerate(pdf_files):
        def _progress(stage, pct):
            if progress_callback:
                overall = (i + pct) / len(pdf_files)
                progress_callback(f"{pdf_file.name}: {stage}", overall)

        chunks = process_pdf(
            str(pdf_file), equipment_id, chunk_size, chunk_overlap,
            progress_callback=_progress,
        )
        all_chunks.extend(chunks)

    return all_chunks


# ---------------------------------------------------------------------------
# Processing stats
# ---------------------------------------------------------------------------

def get_processing_stats(chunks: list[DocumentChunk]) -> dict:
    """Return summary statistics for processed chunks."""
    if not chunks:
        return {"total_chunks": 0}

    type_counts = {}
    page_set = set()
    file_set = set()
    section_set = set()
    total_chars = 0

    for chunk in chunks:
        type_counts[chunk.chunk_type] = type_counts.get(chunk.chunk_type, 0) + 1
        page_set.add((chunk.source_file, chunk.page_number))
        file_set.add(chunk.source_file)
        if chunk.section_title:
            section_set.add(chunk.section_title)
        total_chars += len(chunk.text)

    return {
        "total_chunks": len(chunks),
        "total_characters": total_chars,
        "avg_chunk_size": total_chars // len(chunks) if chunks else 0,
        "files_processed": len(file_set),
        "pages_covered": len(page_set),
        "sections_detected": len(section_set),
        "chunks_by_type": type_counts,
        "files": sorted(file_set),
        "sections": sorted(section_set)[:20],  # Top 20 for display
    }
