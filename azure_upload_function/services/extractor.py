"""
extractor.py — Unified file ingestion + processing pipeline

Supported types:
  .csv   → structured table (pandas)
  .xlsx  → structured table, multi-sheet (pandas + openpyxl)
  .pdf   → text via Azure Document Intelligence (OCR)
  .jpg / .jpeg / .png / .svg → image metadata + OCR (Document Intelligence)
  .docx  → paragraph text (python-docx)
  .txt   → raw UTF-8 text

Public API:
  process_file(file_bytes, filename)  → unified dict response
  extract_text(file_bytes, filename)  → plain text string (for RAG pipeline)
  extract_with_structured(...)        → (text, structured_data | None)
"""

import io
import os
import logging
from typing import Any

# ---------------------------------------------------------------------------
# Azure Document Intelligence client (PDF + images)
# ---------------------------------------------------------------------------
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials  import AzureKeyCredential

_di_client: DocumentAnalysisClient | None = None


def _get_di_client() -> DocumentAnalysisClient:
    global _di_client
    if _di_client is None:
        _di_client = DocumentAnalysisClient(
            endpoint   = os.environ["DOC_INTELLIGENCE_ENDPOINT"],
            credential = AzureKeyCredential(os.environ["DOC_INTELLIGENCE_KEY"]),
        )
        logging.info("DocumentAnalysisClient initialised.")
    return _di_client


# ---------------------------------------------------------------------------
# Unified public entry point — returns structured JSON response
# ---------------------------------------------------------------------------

def process_file(file_bytes: bytes, filename: str) -> dict[str, Any]:
    """
    Main entry point. Detects file type, routes to the correct processor,
    and returns a unified JSON-serialisable response dict.

    Response shape:
    {
        "type":     "table" | "image" | "document",
        "filename": str,
        "preview":  str,          # first 500 chars of text / first 5 rows as string
        "content":  str,          # full extracted text
        "metadata": dict,         # file-specific metadata
        "structured": {           # only for CSV/Excel
            "columns": [...],
            "rows":    [...],     # all rows
            "preview_rows": [...] # first 5 rows only
        } | None,
        "chart_ready": {          # bonus: chart-ready output for CSV/Excel
            "labels": [...],
            "values": [...]
        } | None
    }
    """
    if not file_bytes:
        raise ValueError(f"Empty file: '{filename}'")

    name = filename.lower()
    logging.info("process_file: '%s' (%d bytes)", filename, len(file_bytes))

    # ── Route by extension ────────────────────────────────────────────────
    if name.endswith(".csv"):
        return _process_csv(file_bytes, filename)

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return _process_excel(file_bytes, filename)

    if name.endswith(".pdf"):
        return _process_pdf(file_bytes, filename)

    if name.endswith((".jpg", ".jpeg", ".png", ".svg")):
        return _process_image(file_bytes, filename)

    if name.endswith(".docx"):
        return _process_docx(file_bytes, filename)

    if name.endswith(".txt"):
        return _process_txt(file_bytes, filename)

    raise RuntimeError(
        f"Unsupported file type: '{filename}'. "
        "Supported: csv, xlsx, xls, pdf, jpg, jpeg, png, svg, docx, txt"
    )


# ---------------------------------------------------------------------------
# Existing RAG-pipeline helpers (unchanged API)
# ---------------------------------------------------------------------------

def extract_text(file_bytes: bytes, filename: str) -> str:
    """Plain text extraction for the RAG pipeline."""
    text, _ = extract_with_structured(file_bytes, filename)
    return text


def extract_with_structured(file_bytes: bytes, filename: str) -> tuple[str, dict | None]:
    """
    Returns (text, structured_data).
    structured_data = {"columns": [...], "rows": [...]} for CSV/Excel, else None.
    Raises RuntimeError if text < 10 chars.
    """
    name = filename.lower()
    structured_data = None

    if name.endswith(".pdf") or name.endswith((".jpg", ".jpeg", ".png", ".svg")):
        text = _ocr(file_bytes, name)
    elif name.endswith(".csv"):
        text, structured_data = _csv_to_text_and_struct(file_bytes)
    elif name.endswith((".xlsx", ".xls")):
        text, structured_data = _excel_to_text_and_struct(file_bytes)
    elif name.endswith(".docx"):
        text = _docx_to_text(file_bytes)
    elif name.endswith(".txt"):
        text = _txt_to_text(file_bytes)
    else:
        raise RuntimeError(f"Unsupported file type: '{filename}'")

    text = text.strip()
    logging.info("extractor: '%s' → %d chars | structured=%s",
                 filename, len(text), structured_data is not None)

    # Only enforce minimum text for non-image file types.
    # Images may legitimately have no OCR text (e.g. photos, diagrams).
    name_lower = filename.lower()
    is_image   = name_lower.endswith((".jpg", ".jpeg", ".png", ".svg", ".gif", ".bmp", ".webp"))
    if len(text) < 10 and not is_image:
        raise RuntimeError(
            f"Extraction produced too little text ({len(text)} chars) for '{filename}'."
        )
    return text, structured_data


# ---------------------------------------------------------------------------
# A. CSV processor
# ---------------------------------------------------------------------------

def _process_csv(file_bytes: bytes, filename: str) -> dict:
    from services.cleaner import read_csv_clean
    try:
        df = read_csv_clean(file_bytes, source_label=filename)
    except Exception as exc:
        raise RuntimeError(f"CSV parse failed: {exc}") from exc

    columns   = list(df.columns)
    all_rows  = df.to_dict(orient="records")
    stats     = _basic_stats(df)
    text      = df.to_string(index=False)

    logging.info("CSV: %d rows × %d cols | columns=%s", len(df), len(columns), columns)

    return {
        "type":     "table",
        "filename": filename,
        "preview":  df.head(5).to_string(index=False),
        "content":  text,
        "metadata": {
            "rows":         len(df),
            "columns":      len(columns),
            "column_names": columns,
            "stats":        stats,
        },
        "structured": {
            "columns":      columns,
            "rows":         all_rows,
            "preview_rows": all_rows[:5],
        },
        "chart_ready": _chart_ready(df),
    }


# ---------------------------------------------------------------------------
# B. Excel processor
# ---------------------------------------------------------------------------

def _process_excel(file_bytes: bytes, filename: str) -> dict:
    import pandas as pd
    from services.cleaner import read_excel_clean
    try:
        xf = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
    except Exception as exc:
        raise RuntimeError(f"Excel parse failed: {exc}") from exc

    parts, all_rows, columns = [], [], []
    sheets: dict = {}

    for sheet in xf.sheet_names:
        try:
            df = read_excel_clean(file_bytes, sheet_name=sheet, source_label=filename)
        except Exception as exc:
            logging.warning("Excel sheet '%s' skipped: %s", sheet, exc)
            continue

        parts.append(f"Sheet: {sheet}\n{df.to_string(index=False)}")
        logging.info("Excel: sheet '%s' — %d rows × %d cols | columns=%s",
                     sheet, len(df), len(df.columns), df.columns.tolist())

        sheet_cols = list(df.columns)
        sheets[sheet] = {
            "columns": sheet_cols,
            "rows":    df.to_dict(orient="records"),
        }
        if not columns:
            columns = sheet_cols
        for record in df.to_dict(orient="records"):
            row = {str(k): v for k, v in record.items()}
            row["_sheet"] = sheet
            all_rows.append(row)

    if not sheets:
        raise RuntimeError(f"No valid sheets found in '{filename}'.")

    text     = "\n\n".join(parts)
    first_df = read_excel_clean(file_bytes, sheet_name=xf.sheet_names[0], source_label=filename)

    return {
        "type":     "table",
        "filename": filename,
        "preview":  first_df.head(5).to_string(index=False),
        "content":  text,
        "metadata": {
            "sheets":     xf.sheet_names,
            "total_rows": len(all_rows),
            "columns":    columns,
        },
        "structured": {
            "columns":      columns,
            "rows":         all_rows,
            "sheets":       sheets,
            "preview_rows": all_rows[:5],
        },
        "chart_ready": _chart_ready(first_df),
    }


# ---------------------------------------------------------------------------
# C. PDF processor
# ---------------------------------------------------------------------------

def _process_pdf(file_bytes: bytes, filename: str) -> dict:
    text = _ocr(file_bytes, filename)
    return {
        "type":     "document",
        "filename": filename,
        "preview":  text[:500],
        "content":  text,
        "metadata": {
            "char_count": len(text),
            "word_count": len(text.split()),
        },
        "structured":  None,
        "chart_ready": None,
    }


# ---------------------------------------------------------------------------
# D. Image processor (.jpg, .jpeg, .png, .svg)
# ---------------------------------------------------------------------------

def _process_image(file_bytes: bytes, filename: str) -> dict:
    name = filename.lower()
    meta: dict = {"filename": filename}

    # Extract image metadata using Pillow (skip for SVG — it's XML)
    if not name.endswith(".svg"):
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(file_bytes))
            meta.update({
                "format": img.format,
                "mode":   img.mode,
                "width":  img.width,
                "height": img.height,
                "size_bytes": len(file_bytes),
            })
            logging.info("Image: %s %dx%d", img.format, img.width, img.height)
        except Exception as exc:
            logging.warning("Pillow metadata extraction failed: %s", exc)
            meta["size_bytes"] = len(file_bytes)
    else:
        # SVG is XML — extract text content directly
        try:
            svg_text = file_bytes.decode("utf-8", errors="replace")
            meta["size_bytes"] = len(file_bytes)
            meta["format"] = "SVG"
            return {
                "type":     "image",
                "filename": filename,
                "preview":  svg_text[:500],
                "content":  svg_text,
                "metadata": meta,
                "structured":  None,
                "chart_ready": None,
            }
        except Exception:
            pass

    # OCR the image via Document Intelligence
    ocr_text = ""
    try:
        ocr_text = _ocr(file_bytes, filename)
        meta["ocr_char_count"] = len(ocr_text)
    except Exception as exc:
        logging.warning("Image OCR failed (non-fatal): %s", exc)
        ocr_text = ""

    return {
        "type":     "image",
        "filename": filename,
        "preview":  ocr_text[:500] if ocr_text else "(no text detected)",
        "content":  ocr_text,
        "metadata": meta,
        "structured":  None,
        "chart_ready": None,
    }


# ---------------------------------------------------------------------------
# E. Word (.docx) processor
# ---------------------------------------------------------------------------

def _process_docx(file_bytes: bytes, filename: str) -> dict:
    text = _docx_to_text(file_bytes)
    return {
        "type":     "document",
        "filename": filename,
        "preview":  text[:500],
        "content":  text,
        "metadata": {
            "char_count": len(text),
            "word_count": len(text.split()),
            "paragraph_count": text.count("\n") + 1,
        },
        "structured":  None,
        "chart_ready": None,
    }


# ---------------------------------------------------------------------------
# F. Plain text (.txt) processor
# ---------------------------------------------------------------------------

def _process_txt(file_bytes: bytes, filename: str) -> dict:
    text = _txt_to_text(file_bytes)
    return {
        "type":     "document",
        "filename": filename,
        "preview":  text[:500],
        "content":  text,
        "metadata": {
            "char_count": len(text),
            "word_count": len(text.split()),
            "line_count": text.count("\n") + 1,
        },
        "structured":  None,
        "chart_ready": None,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ocr(file_bytes: bytes, filename: str) -> str:
    """Azure Document Intelligence OCR for PDF and images."""
    if not file_bytes:
        raise RuntimeError("Empty file bytes.")
    logging.info("OCR: sending %d bytes for '%s'", len(file_bytes), filename)
    client = _get_di_client()
    poller = client.begin_analyze_document(
        model_id = "prebuilt-read",
        document = io.BytesIO(file_bytes),
    )
    result = poller.result()
    lines  = [line.content for page in result.pages for line in page.lines]
    text   = "\n".join(lines)
    logging.info("OCR: extracted %d chars from %d pages", len(text), len(result.pages))
    return text


def _csv_to_text_and_struct(file_bytes: bytes) -> tuple[str, dict]:
    from services.cleaner import read_csv_clean
    df = read_csv_clean(file_bytes, source_label="csv")
    return df.to_string(index=False), {
        "columns": list(df.columns),
        "rows":    df.to_dict(orient="records"),
    }


def _excel_to_text_and_struct(file_bytes: bytes) -> tuple[str, dict]:
    import pandas as pd
    from services.cleaner import read_excel_clean
    xf = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
    parts, all_rows, columns = [], [], []
    sheets: dict = {}

    for sheet in xf.sheet_names:
        try:
            df = read_excel_clean(file_bytes, sheet_name=sheet, source_label=sheet)
        except Exception as exc:
            logging.warning("_excel_to_text_and_struct: sheet '%s' skipped: %s", sheet, exc)
            continue
        parts.append(f"Sheet: {sheet}\n{df.to_string(index=False)}")
        sheet_cols = list(df.columns)
        sheets[sheet] = {
            "columns": sheet_cols,
            "rows":    df.to_dict(orient="records"),
        }
        if not columns:
            columns = sheet_cols
        for record in df.to_dict(orient="records"):
            row = {str(k): v for k, v in record.items()}
            row["_sheet"] = sheet
            all_rows.append(row)

    return "\n\n".join(parts), {
        "columns": columns,
        "rows":    all_rows,
        "sheets":  sheets,
    }


def _docx_to_text(file_bytes: bytes) -> str:
    from docx import Document
    doc   = Document(io.BytesIO(file_bytes))
    lines = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(lines)


def _txt_to_text(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="replace")


def _basic_stats(df) -> dict:
    """Return basic numeric stats for a DataFrame."""
    try:
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            return {}
        return {
            col: {
                "min":  round(float(numeric[col].min()), 4),
                "max":  round(float(numeric[col].max()), 4),
                "mean": round(float(numeric[col].mean()), 4),
            }
            for col in numeric.columns
        }
    except Exception:
        return {}


def _chart_ready(df) -> dict | None:
    """
    Bonus: produce chart-ready {labels, values} from the first two columns
    of a DataFrame (first column = labels, second numeric column = values).
    Returns None if not applicable.
    """
    try:
        if len(df.columns) < 2:
            return None
        label_col = df.columns[0]
        # Find first numeric column after the label column
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            return None
        value_col = numeric_cols[0]
        return {
            "labels": df[label_col].astype(str).tolist()[:50],
            "values": df[value_col].fillna(0).tolist()[:50],
            "label_key":  str(label_col),
            "value_key":  str(value_col),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Legacy alias
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Backward-compatible alias."""
    return _ocr(file_bytes, "file.pdf")
