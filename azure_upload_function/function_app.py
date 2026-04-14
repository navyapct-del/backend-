import azure.functions as func
import logging
import json
import time
import os
import uuid

try:
    from services.config import log_config_status, require_env
    from services.blob_service    import BlobService
    from services.table_service   import TableService
    from services.extractor       import extract_text, extract_with_structured
    from services.search_service  import ensure_index, index_document, vector_search, delete_index
    from services.openai_service  import (
        generate_summary, generate_tags, generate_embedding,
        generate_rag_answer, extract_structured_data, generate_explanation,
        smart_chart_from_structured,
    )
    from services.query_engine import generate_plan, execute_plan, structured_to_df, get_series_from_data, detect_dual_axis_from_rows
    from services.delete_service       import delete_document
except Exception as _import_exc:
    logging.error("STARTUP IMPORT ERROR: %s", _import_exc)
    raise

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# ---------------------------------------------------------------------------
# Custom JSON encoder — handles NaN, Infinity, numpy types from pandas
# ---------------------------------------------------------------------------

import math

class _SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return None if math.isnan(obj) else float(obj)
            if isinstance(obj, (np.ndarray,)):  return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)

    def encode(self, obj):
        # Replace NaN/Infinity in the final string — catches all edge cases
        result = super().encode(obj)
        return result

def _safe_json(obj) -> str:
    """Serialize to JSON, replacing NaN/Infinity with null."""
    raw = json.dumps(obj, cls=_SafeEncoder)
    # Replace bare NaN and Infinity tokens (not valid JSON)
    import re as _re
    raw = _re.sub(r'\bNaN\b',       'null', raw)
    raw = _re.sub(r'\bInfinity\b',  'null', raw)
    raw = _re.sub(r'\b-Infinity\b', 'null', raw)
    return raw

# ---------------------------------------------------------------------------
# GET /health — validate all required env vars, return status
# ---------------------------------------------------------------------------

@app.route(route="health", methods=["GET"])
def health(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("GET /health")
    required = [
        "AZURE_STORAGE_CONNECTION_STRING",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_KEY",
        "DOC_INTELLIGENCE_ENDPOINT",
        "DOC_INTELLIGENCE_KEY",
    ]
    status = {}
    all_ok = True
    for k in required:
        present = bool(os.environ.get(k, "").strip())
        status[k] = "OK" if present else "MISSING"
        if not present:
            all_ok = False

    return func.HttpResponse(
        json.dumps({"healthy": all_ok, "config": status}, indent=2),
        status_code=200 if all_ok else 503,
        mimetype="application/json",
    )


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

_CHART_KW = {"plot", "graph", "chart", "trend", "visualize", "growth", "pie", "bar chart", "line chart", "pie chart", "show as chart", "show as graph"}
_TABLE_KW = {"compare", "comparison", "difference", "versus", " vs ", "year-wise",
             "yearwise", "state-wise", "breakdown", "statewise"}

# Keywords that signal aggregation chart intent even without explicit "plot/chart"
_AGG_CHART_KW = {"average", "avg", "sum", "total", "count", "by", "per",
                 "distribution", "breakdown"}

def _detect_type(query: str) -> str:
    q = " " + query.lower() + " "
    if any(f" {k} " in q for k in _CHART_KW): return "chart"
    if any(k in q for k in _TABLE_KW): return "table"
    return "text"

def _is_analytical(query: str) -> bool:
    return _detect_type(query) in ("chart", "table")

def _is_chart_intent(query: str) -> bool:
    """True when query explicitly asks for a chart OR implies aggregation visualisation."""
    if _detect_type(query) == "chart":
        return True
    q = query.lower()
    return sum(1 for k in _AGG_CHART_KW if k in q) >= 2   # at least 2 agg signals


def _chart_type_from_query(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ("trend", "over time", "growth", "line")):
        return "line"
    if any(k in q for k in ("distribution", "share", "proportion", "pie")):
        return "pie"
    return "bar"


def _promote_to_chart(result: dict, query: str) -> dict:
    """
    Convert a table/text engine result to a chart response when:
    - The result has exactly 2 columns (label + numeric value), OR
    - The result came from a groupby aggregation (has numeric columns)

    First column → label (xKey)
    Remaining numeric columns → series values

    Returns the result unchanged if it cannot be meaningfully charted.
    """
    rows    = result.get("rows", [])
    columns = result.get("columns", [])

    if not rows or len(columns) < 2:
        return result

    # Find label col (first non-numeric) and value cols (numeric)
    sample      = rows[0]
    label_col   = columns[0]
    value_cols  = [
        c for c in columns[1:]
        if any(isinstance(r.get(c), (int, float)) and not isinstance(r.get(c), bool)
               for r in rows[:5])
    ]

    if not value_cols:
        return result   # no numeric columns — keep as table

    # Clean: remove rows where label is null/empty
    clean_rows = [
        r for r in rows
        if r.get(label_col) is not None and str(r.get(label_col, "")).strip() not in ("", "nan", "None")
    ]
    if not clean_rows:
        return result

    # Convert values to float safely
    def _safe_float(v):
        try:
            return round(float(v), 4) if v is not None else None
        except (TypeError, ValueError):
            return None

    chart_rows = []
    for r in clean_rows:
        row = {label_col: str(r[label_col])}   # preserve original column name as xKey
        for vc in value_cols:
            row[vc] = _safe_float(r.get(vc))
        chart_rows.append(row)

    chart_type = _chart_type_from_query(query)
    x_key      = label_col   # use real column name, not "label"
    series     = value_cols

    logging.info("_promote_to_chart: %d rows, xKey=%s, series=%s, type=%s",
                 len(chart_rows), x_key, series, chart_type)

    return {
        "type":    "chart",
        "answer":  f"Chart generated from {len(chart_rows)} data points.",
        "data":    chart_rows,
        "columns": columns,
        "rows":    clean_rows,
        "chart_config": {
            "type":     chart_type,
            "xKey":     x_key,
            "series":   series,
            "dualAxis": False,
        },
        "script":  result.get("script", ""),
    }


def _run_query_engine(user_query: str, structured: dict) -> dict | None:
    """
    Run the LLM query planner + pandas execution engine against stored structured data.
    Promotes groupby aggregation results to chart format when query intent warrants it.
    """
    import pandas as pd
    import json as _json
    try:
        df = structured_to_df(structured)
        if df.empty:
            logging.warning("[QUERY] Structured data produced empty DataFrame")
            return None

        df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
        columns = list(df.columns)

        logging.info("[QUERY] User query: %s", user_query)
        logging.info("[QUERY] Schema columns: %s", columns)

        plan   = generate_plan(user_query, columns)
        logging.info("[QUERY] Generated plan: op=%s select=%s group_by=%s aggs=%d filters=%d",
                     plan.get("operation"), plan.get("select"),
                     plan.get("group_by"), len(plan.get("aggregations", [])),
                     len(plan.get("filters", [])))

        result = execute_plan(df, plan)
        resp_type = result.get("type")
        row_count = len(result.get("rows", []))

        if resp_type == "error":
            logging.warning("[QUERY ERROR] Invalid columns | answer=%s", result.get("answer"))
        elif not result.get("rows") and resp_type != "text":
            logging.warning("[QUERY] Empty result set | type=%s op=%s", resp_type, plan.get("operation"))
        else:
            logging.info("[QUERY] Result: type=%s rows=%d", resp_type, row_count)

        # Promote to chart if:
        # 1. Query has chart/aggregation intent, AND
        # 2. Result is a table (not already a chart), AND
        # 3. Result has numeric columns suitable for charting
        if (result.get("type") != "chart"
                and result.get("type") != "error"
                and _is_chart_intent(user_query)
                and plan.get("group_by")):
            result = _promote_to_chart(result, user_query)

        return result

    except ValueError as exc:
        # _validate_plan raises ValueError with JSON payload when all requested
        # columns are invalid — surface as a structured error result.
        try:
            detail = _json.loads(str(exc))
            invalid_cols = detail.get("invalid_columns", [])
            available    = detail.get("available_columns", columns if 'columns' in dir() else [])
            suggestions  = detail.get("suggestions", [])
            hint = (f"Did you mean: {', '.join(suggestions)}?" if suggestions
                    else f"Available columns: {', '.join(available)}")
            logging.warning("_run_query_engine: invalid columns requested=%s suggestions=%s",
                            invalid_cols, suggestions)
            return {
                "type":               "error",
                "answer":             (
                    f"The dataset does not contain "
                    f"{'column' if len(invalid_cols) == 1 else 'columns'} "
                    f"{', '.join(repr(c) for c in invalid_cols)}. {hint}"
                ),
                "invalid_columns":    invalid_cols,
                "available_columns":  available,
                "suggestions":        suggestions,
                "columns":            [],
                "rows":               [],
                "chart_config":       None,
                "script":             "",
            }
        except (_json.JSONDecodeError, Exception):
            logging.warning("_run_query_engine schema error: %s", exc)
            return None

    except Exception as exc:
        logging.warning("_run_query_engine failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# POST /upload
# Upload → Blob → OCR → Embedding → Table Storage + AI Search
# ---------------------------------------------------------------------------

@app.route(route="upload", methods=["POST"])
def upload(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("POST /upload")
    try:
        file = req.files.get("file")
        if not file:
            return func.HttpResponse(json.dumps({"error": "No file provided."}),
                                     status_code=400, mimetype="application/json")

        filename    = req.form.get("filename") or file.filename
        description = req.form.get("description", "")
        tags_input  = req.form.get("tags", "")
        temp_flag   = req.form.get("temp", "false").lower() == "true"
        session_id  = req.form.get("session_id", "")

        if not filename:
            return func.HttpResponse(json.dumps({"error": "filename is required."}),
                                     status_code=400, mimetype="application/json")

        # ── File type validation ───────────────────────────────────────────
        ALLOWED_MIME_TYPES = {
            # Images
            "image/jpeg", "image/png", "image/gif", "image/svg+xml", "image/webp",
            # PDF
            "application/pdf",
            # CSV — browsers report different MIME types
            "text/csv", "application/csv", "text/plain",
            # Excel
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            # Word
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            # Generic binary (some browsers send this for xlsx/docx)
            "application/octet-stream",
            "application/zip",
        }
        ALLOWED_EXTENSIONS = {
            "jpg", "jpeg", "png", "gif", "svg", "webp",
            "pdf",
            "csv",
            "xls", "xlsx",
            "doc", "docx",
            "txt",
        }

        content_type = (file.content_type or "").lower().split(";")[0].strip()
        file_ext     = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        # Validate by extension first (more reliable than MIME type)
        if file_ext not in ALLOWED_EXTENSIONS and content_type not in ALLOWED_MIME_TYPES:
            return func.HttpResponse(
                json.dumps({"error": f"Unsupported file type '.{file_ext}'. Allowed: jpg, jpeg, png, pdf, csv, xls, xlsx, doc, docx, txt."}),
                status_code=400, mimetype="application/json")

        # ── Temp upload validation ─────────────────────────────────────────
        if temp_flag and not session_id:
            return func.HttpResponse(
                json.dumps({"error": "session_id is required for temporary uploads."}),
                status_code=400, mimetype="application/json")

        file_bytes = file.read()

        # Guard against oversized uploads (default cap: 50 MB)
        MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "50")) * 1024 * 1024
        if len(file_bytes) > MAX_UPLOAD_BYTES:
            return func.HttpResponse(
                json.dumps({"error": f"File too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024*1024)} MB."}),
                status_code=413, mimetype="application/json")

        # ── 1. Blob Storage ───────────────────────────────────────────────
        # ── 1. Blob Storage ───────────────────────────────────────────────
        # Check for duplicate filename (non-temp uploads only)
        if not temp_flag:
            table_svc_check = TableService()
            existing = table_svc_check.find_by_filename(filename)
            if existing:
                logging.info("Duplicate upload rejected: '%s' already exists (id=%s)", filename, existing.get("id"))
                return func.HttpResponse(
                    json.dumps({
                        "error": f"A file named '{filename}' already exists. Please delete it first or rename your file.",
                        "duplicate": True,
                        "existing_id": existing.get("id", ""),
                    }),
                    status_code=409, mimetype="application/json")

        blob_svc = BlobService()
        # Use temp prefix when temp flag is set
        if temp_flag:
            custom_blob_name = f"temp/{session_id}/{uuid.uuid4().hex}_{filename}"
        else:
            custom_blob_name = ""  # BlobService will generate UUID prefix
        blob_url = blob_svc.upload(filename, file_bytes,
                                   file.content_type or "application/octet-stream",
                                   blob_name=custom_blob_name)
        logging.info("Blob uploaded: %s", blob_url)

        # ── 2. Table Storage placeholder (status=processing) ──────────────
        table_svc = TableService()
        record_id = table_svc.insert_entity(filename, blob_url, description, tags_input,
                                            temp=temp_flag, session_id=session_id)

        # ── 3. Text extraction (universal — PDF, CSV, Excel, Word, TXT, Image) ──
        t0   = time.time()
        try:
            text, structured_data = extract_with_structured(file_bytes, filename)
        except RuntimeError as extraction_err:
            # Images with no detectable text raise RuntimeError("too little text").
            # Fall back to filename-based text so the upload can still complete.
            logging.warning("Extraction warning for '%s': %s — using filename fallback", filename, extraction_err)
            text            = filename
            structured_data = None

        logging.info("Text extracted: %d chars in %.2fs from '%s' | structured=%s",
                     len(text), time.time() - t0, filename, structured_data is not None)

        # For images/files with no extractable text, use filename as minimal content
        # so the record is still searchable by filename.
        name_lower = filename.lower()
        is_image   = name_lower.endswith((".jpg", ".jpeg", ".png", ".svg", ".gif", ".bmp", ".webp"))
        if len(text.strip()) < 10:
            if is_image:
                logging.info("Image '%s' produced little/no OCR text — using filename as content", filename)
                text = f"Image file: {filename}"
            else:
                return func.HttpResponse(
                    json.dumps({"error": f"Extraction returned too little text from '{filename}'."}),
                    status_code=422, mimetype="application/json")

        # ── 4. OpenAI: summary + tags + embedding ─────────────────────────
        t1 = time.time()

        # generate_summary with error handling (Task 9.3)
        try:
            summary = generate_summary(text)
            if not summary:
                logging.warning("generate_summary returned empty for '%s'", filename)
                summary = ""
        except Exception as sum_exc:
            logging.warning("generate_summary failed for '%s': %s — storing empty summary", filename, sum_exc)
            summary = ""

        # generate_tags with error handling (Task 9.3)
        # Always generate AI tags; merge with any user-provided tags
        try:
            ai_tags_str = generate_tags(text)
            if not ai_tags_str:
                logging.warning("generate_tags returned empty for '%s'", filename)
                ai_tags_str = ""
        except Exception as tag_exc:
            logging.warning("generate_tags failed for '%s': %s — storing empty tags", filename, tag_exc)
            ai_tags_str = ""

        # Merge user-provided tags with AI-generated tags (deduplicated)
        user_tags = [t.strip() for t in tags_input.split(",") if t.strip()]
        ai_tags   = [t.strip() for t in ai_tags_str.split(",") if t.strip()]
        merged    = list(dict.fromkeys(user_tags + ai_tags))  # preserve order, deduplicate
        tags_str  = ", ".join(merged)

        tags_list = [t.strip() for t in tags_str.split(",") if t.strip()]

        embedding = generate_embedding(text)
        logging.info("Summary+tags+embedding in %.2fs | Embedding size: %d",
                     time.time() - t1, len(embedding) if embedding else 0)

        if not embedding:
            return func.HttpResponse(
                json.dumps({"error": "Embedding generation failed. Check AZURE_OPENAI_API_KEY."}),
                status_code=502, mimetype="application/json")

        # ── 5. Upload text + structured_data to Blob, store URLs in Table ──
        t_blob   = time.time()
        text_url = ""
        sd_url   = ""
        try:
            text_url = blob_svc.upload_text(record_id, text)
            logging.info("Text blob uploaded in %.2fs: %s", time.time() - t_blob, text_url)
        except Exception as exc:
            logging.warning("Text blob upload failed (will use inline fallback): %s", exc)

        if structured_data:
            try:
                sd_url = blob_svc.upload_structured_data(record_id, structured_data)
                logging.info("Structured data blob uploaded: %s", sd_url)
            except Exception as exc:
                logging.warning("Structured data blob upload failed: %s", exc)

        table_svc.update_ai_fields(
            filename, text, summary, tags_str,
            structured_data     = structured_data if not sd_url else None,  # inline only if no URL
            text_url            = text_url,
            structured_data_url = sd_url,
        )
        logging.info("Table Storage updated: %s → completed", filename)

        # ── 6. Azure AI Search — vector index ─────────────────────────────
        t_search = time.time()
        ensure_index()
        index_document(
            doc_id    = record_id,
            filename  = filename,
            content   = text,
            summary   = summary,
            tags      = tags_list,
            blob_url  = blob_url,
            embedding = embedding,
        )
        logging.info("AI Search indexed in %.2fs: id=%s", time.time() - t_search, record_id)

        return func.HttpResponse(
            json.dumps({"id": record_id, "filename": filename,
                        "blob_url": blob_url, "message": "Upload successful."}),
            status_code=201, mimetype="application/json")

    except Exception as exc:
        logging.exception("Upload error.")
        return func.HttpResponse(json.dumps({"error": "Internal server error.", "detail": str(exc)}),
                                 status_code=500, mimetype="application/json")


# ---------------------------------------------------------------------------
# POST /reset-index — delete and recreate AI Search index with correct schema
# ---------------------------------------------------------------------------

@app.route(route="reset-index", methods=["POST"])
def reset_index(req: func.HttpRequest) -> func.HttpResponse:
    """
    Deletes the existing AI Search index and recreates it with the correct schema.
    Call this ONCE if the index was created with a broken schema.
    After calling this, re-upload all documents.
    """
    logging.info("POST /reset-index")
    try:
        deleted = delete_index()
        if not deleted:
            return func.HttpResponse(
                json.dumps({"error": "Failed to delete index."}),
                status_code=500, mimetype="application/json")
        ensure_index()
        return func.HttpResponse(
            json.dumps({"message": "Index reset successfully. Re-upload your documents."}),
            status_code=200, mimetype="application/json")
    except Exception as exc:
        logging.exception("reset-index error.")
        return func.HttpResponse(json.dumps({"error": str(exc)}),
                                 status_code=500, mimetype="application/json")


# ---------------------------------------------------------------------------
# POST /reprocess — auto-reprocess stale documents (schema version upgrade)
# ---------------------------------------------------------------------------

@app.route(route="reprocess", methods=["POST"])
def reprocess(req: func.HttpRequest) -> func.HttpResponse:
    """
    Finds all documents with outdated schema_version and reprocesses them
    by re-downloading from Blob Storage and re-extracting structured data.

    Call this after any backend logic change that affects structured_data format.
    No manual re-upload required.
    """
    logging.info("POST /reprocess")
    try:
        from services.table_service import SCHEMA_VERSION
        from services.extractor     import extract_with_structured
        from azure.storage.blob     import BlobServiceClient

        table_svc = TableService()
        stale     = table_svc.get_stale_documents()

        if not stale:
            return func.HttpResponse(
                json.dumps({"message": f"All documents are up to date (v{SCHEMA_VERSION}).",
                            "updated": 0}),
                status_code=200, mimetype="application/json")

        conn_str   = require_env("AZURE_STORAGE_CONNECTION_STRING")
        blob_svc_c = BlobServiceClient.from_connection_string(conn_str)

        updated = 0
        failed  = []

        for doc in stale:
            filename = doc["filename"]
            blob_url = doc["blob_url"]
            try:
                # Download file bytes from Blob Storage using the SDK's URL parser
                # (avoids fragile split("/") which breaks on blob names containing "/")
                from azure.storage.blob import BlobClient
                blob_client = BlobClient.from_blob_url(
                    blob_url   = blob_url,
                    credential = blob_svc_c.credential,
                )
                file_bytes  = blob_client.download_blob().readall()

                # Re-extract with current logic
                text, structured_data = extract_with_structured(file_bytes, filename)

                # Update Table Storage with new schema
                table_svc.update_ai_fields(
                    filename, text,
                    summary         = "",   # keep existing summary (don't re-call OpenAI)
                    tags            = "",
                    structured_data = structured_data,
                )
                logging.info("Reprocessed: %s → v%d", filename, SCHEMA_VERSION)
                updated += 1

            except Exception as exc:
                logging.exception("Reprocess failed for %s", filename)
                failed.append({"filename": filename, "error": str(exc)})

        return func.HttpResponse(
            json.dumps({
                "message":        f"Reprocess complete. Schema v{SCHEMA_VERSION}.",
                "updated":        updated,
                "failed":         len(failed),
                "failed_details": failed,
            }),
            status_code=200, mimetype="application/json")

    except Exception as exc:
        logging.exception("Reprocess endpoint error.")
        return func.HttpResponse(json.dumps({"error": str(exc)}),
                                 status_code=500, mimetype="application/json")


# ---------------------------------------------------------------------------
# GET /documents — list from Table Storage (lightweight, for UI polling)
# ---------------------------------------------------------------------------

@app.route(route="documents", methods=["GET"])
def documents(req: func.HttpRequest) -> func.HttpResponse:
    t0 = time.time()
    try:
        docs = TableService().list_documents()
        logging.info("/documents: %d docs in %.3fs", len(docs), time.time() - t0)
        return func.HttpResponse(json.dumps(docs), status_code=200, mimetype="application/json")
    except Exception as exc:
        logging.exception("/documents error.")
        return func.HttpResponse(json.dumps({"error": str(exc)}),
                                 status_code=500, mimetype="application/json")


# ---------------------------------------------------------------------------
# GET /diagnose — raw Table Storage state for debugging
# ---------------------------------------------------------------------------

@app.route(route="diagnose", methods=["GET"])
def diagnose(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("GET /diagnose")
    try:
        table_svc = TableService()
        docs      = table_svc.list_documents()
        # Augment with text_chars + has_summary by fetching full entities via search
        from azure.data.tables import TableServiceClient
        from services.config import require_env as _re
        conn_str = _re("AZURE_STORAGE_CONNECTION_STRING")
        raw_client = TableServiceClient.from_connection_string(conn_str).get_table_client("documentsmetadata")
        entities   = list(raw_client.query_entities(query_filter="PartitionKey eq 'documents'"))
        entity_map = {e.get("RowKey", ""): e for e in entities}
        report = []
        for d in docs:
            e = entity_map.get(d["id"], {})
            report.append({
                "filename":    d["filename"],
                "status":      d["status"],
                "text_chars":  len(e.get("text", "")),
                "has_summary": bool(d["summary"]),
                "RowKey":      d["id"],
            })
        return func.HttpResponse(json.dumps(report, indent=2),
                                 status_code=200, mimetype="application/json")
    except Exception as exc:
        logging.exception("diagnose error.")
        return func.HttpResponse(json.dumps({"error": str(exc)}),
                                 status_code=500, mimetype="application/json")


# ---------------------------------------------------------------------------
# GET|POST /query — vector search → RAG / table / chart
# ---------------------------------------------------------------------------

@app.route(route="query", methods=["GET", "POST"])
def query(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("POST /query")

    user_query      = req.params.get("q", "")
    filename_filter = req.params.get("filename", "")
    if not user_query:
        try:
            body            = req.get_json()
            user_query      = body.get("q", "")
            filename_filter = body.get("filename", filename_filter)
        except Exception:
            pass

    if not user_query:
        return func.HttpResponse(json.dumps({"error": "'q' is required."}),
                                 status_code=400, mimetype="application/json")

    try:
        analytical = _is_analytical(user_query)
        top_k      = 5 if analytical else 3   # text queries: max 3 docs

        # ── Vector search via Azure AI Search ─────────────────────────────
        t0              = time.time()
        query_embedding = generate_embedding(user_query)

        if not query_embedding:
            return func.HttpResponse(
                json.dumps({"type": "text", "answer": "Failed to generate query embedding.", "sources": []}),
                status_code=502, mimetype="application/json")

        docs = vector_search(query_embedding, user_query, top=top_k,
                             filename_filter=filename_filter)
        logging.info("vector_search: %d docs in %.3fs (analytical=%s)",
                     len(docs), time.time() - t0, analytical)

        if not docs:
            return func.HttpResponse(
                json.dumps({"type": "text", "answer": "No relevant data found in documents.", "sources": []}),
                status_code=200, mimetype="application/json")

        # Cap sources to top 3 most relevant only
        sources = [
            {"filename": d["filename"], "summary": d["summary"], "blob_url": d["blob_url"]}
            for d in docs[:3]
        ]

        # ── Try query engine on structured data first (works for ALL query types) ──
        # Only use query engine if the structured data columns are relevant to the query
        table_svc = TableService()
        stored_sd = None
        stored_sd_filename = None

        def _columns_relevant_to_query(query: str, columns: list) -> bool:
            """Check if any column name is semantically related to the query."""
            if not columns:
                return False
            q_lower = query.lower()
            # Extract meaningful words from query (3+ chars)
            q_words = set(w.strip(".,!?;:()") for w in q_lower.split() if len(w) >= 3)

            # Common aggregation/math keywords — if present, try the query engine
            # regardless of column match (user is asking for computation)
            agg_keywords = {"sum", "total", "average", "avg", "count", "max", "min",
                            "mean", "how many", "number of", "add up", "calculate"}
            if any(kw in q_lower for kw in agg_keywords):
                # Still need at least one column word to match to avoid wrong doc
                # Use a looser check for aggregation queries
                for col in columns:
                    col_lower = col.lower()
                    col_words = set(w.strip(".,!?;:()") for w in col_lower.split() if len(w) >= 3)
                    # Check if any query word is a substring of any column word or vice versa
                    for qw in q_words:
                        for cw in col_words:
                            if qw in cw or cw in qw:
                                return True

            # Standard check: direct word overlap
            for col in columns:
                col_lower = col.lower()
                col_words = set(w.strip(".,!?;:()") for w in col_lower.split() if len(w) >= 3)
                if q_words & col_words:
                    return True
                # Substring check
                for qw in q_words:
                    if qw in col_lower:
                        return True
            return False

        for doc in docs:
            sd = table_svc.get_structured_data(doc["filename"])
            if sd:
                # Get columns from structured data
                sd_columns = sd.get("columns", [])
                if not sd_columns and sd.get("sheets"):
                    for sheet_data in sd["sheets"].values():
                        sd_columns = sheet_data.get("columns", [])
                        if sd_columns:
                            break

                # Only use this structured data if columns are relevant to the query
                if _columns_relevant_to_query(user_query, sd_columns):
                    stored_sd = sd
                    stored_sd_filename = doc["filename"]
                    logging.info("Found relevant structured data for '%s' (columns: %s)",
                                 doc["filename"], sd_columns[:5])
                    break
                else:
                    logging.info("Skipping structured data for '%s' — columns not relevant to query (columns: %s)",
                                 doc["filename"], sd_columns[:5])
            else:
                # Stale schema — inline reprocess from Blob
                logging.info("Stale schema for '%s' — attempting inline reprocess", doc["filename"])
                try:
                    from services.extractor import extract_with_structured
                    from azure.storage.blob import BlobClient, BlobServiceClient
                    blob_url   = doc["blob_url"]
                    bc         = BlobClient.from_blob_url(
                        blob_url   = blob_url,
                        credential = BlobServiceClient.from_connection_string(
                            require_env("AZURE_STORAGE_CONNECTION_STRING")
                        ).credential,
                    )
                    file_bytes = bc.download_blob().readall()
                    _, sd_r    = extract_with_structured(file_bytes, doc["filename"])
                    if sd_r:
                        sd_columns = sd_r.get("columns", [])
                        if _columns_relevant_to_query(user_query, sd_columns):
                            table_svc.update_ai_fields(doc["filename"], "", "", "", structured_data=sd_r)
                            stored_sd = sd_r
                            stored_sd_filename = doc["filename"]
                            logging.info("Inline reprocess succeeded for '%s'", doc["filename"])
                            break
                except Exception as rp_exc:
                    logging.warning("Inline reprocess failed for '%s': %s", doc["filename"], rp_exc)

        if stored_sd:
            engine_result = _run_query_engine(user_query, stored_sd)
            if engine_result:
                resp_type    = engine_result.get("type", "text")
                chart_config = engine_result.get("chart_config")
                rows         = engine_result.get("rows", [])
                columns      = engine_result.get("columns", [])

                # error — invalid / unknown columns → fall through to RAG
                if resp_type == "error":
                    logging.info("Query engine returned error — falling through to RAG: %s",
                                 engine_result.get("answer", ""))
                    # Don't return error — fall through to RAG which knows the document content
                    stored_sd = None  # force RAG fallback

                # empty result → fall through to RAG
                elif not rows and resp_type != "text":
                    logging.info("Query engine returned 0 rows — falling through to RAG")
                    stored_sd = None  # force RAG fallback

        if stored_sd and engine_result:
            resp_type    = engine_result.get("type", "text")
            chart_config = engine_result.get("chart_config")
            rows         = engine_result.get("rows", [])
            columns      = engine_result.get("columns", [])

            # chart
            if resp_type == "chart" and chart_config:
                x_key      = chart_config.get("xKey", "")
                axis_info  = detect_dual_axis_from_rows(rows, x_key)
                chart_config["series"]   = axis_info["series"]
                chart_config["dualAxis"] = axis_info["dual_axis"]
                if axis_info["dual_axis"]:
                    chart_config["type"] = axis_info.get("chart_type", "composed")
                return func.HttpResponse(
                    _safe_json({
                        "type":         "chart",
                        "answer":       engine_result.get("answer", ""),
                        "data":         rows,
                        "chart_config": chart_config,
                        "script":       engine_result.get("script", ""),
                        "query":        user_query,
                        "sources":      sources,
                    }),
                    status_code=200, mimetype="application/json")

            # table with actual rows
            if resp_type == "table" and rows:
                return func.HttpResponse(
                    _safe_json({
                        "type":    "table",
                        "answer":  engine_result.get("answer", ""),
                        "columns": columns,
                        "rows":    rows,
                        "script":  engine_result.get("script", ""),
                        "query":   user_query,
                        "sources": sources,
                    }),
                    status_code=200, mimetype="application/json")

            # text / scalar with actual content
            if resp_type == "text" and engine_result.get("answer"):
                return func.HttpResponse(
                    json.dumps({
                        "type":    "text",
                        "answer":  engine_result.get("answer", ""),
                        "query":   user_query,
                        "sources": sources,
                    }),
                    status_code=200, mimetype="application/json")

        # ── Fallback: RAG answer from document text ────────────────────────
        result    = generate_rag_answer(user_query, docs)
        resp_type = result.get("type", "text")

        if resp_type == "table":
            return func.HttpResponse(
                json.dumps({
                    "type":    "table",
                    "columns": result.get("columns", []),
                    "rows":    result.get("rows", []),
                    "answer":  result.get("answer", ""),
                    "query":   user_query,
                    "sources": sources,
                }),
                status_code=200, mimetype="application/json")

        if resp_type == "chart":
            # Support both shapes:
            # New: { labels: [...], values: [...] }
            # Legacy: { data: [{xKey: label, yKey: value}, ...] }
            labels = result.get("labels", [])
            values = result.get("values", [])
            raw_data = result.get("data")

            if labels and values:
                # New flat shape — pass directly to frontend
                return func.HttpResponse(
                    json.dumps({
                        "type":       "chart",
                        "chart_type": result.get("chart_type", "bar"),
                        "labels":     labels,
                        "values":     values,
                        "answer":     result.get("answer", ""),
                        "query":      user_query,
                        "sources":    sources,
                    }),
                    status_code=200, mimetype="application/json")

            if not raw_data:
                if labels and values:
                    raw_data = [{"label": l, "value": v} for l, v in zip(labels, values)]

            if raw_data and len(raw_data) > 0:
                keys    = [k for k in raw_data[0].keys()]
                x_key   = keys[0]
                y_keys  = keys[1:] if len(keys) > 1 else keys[:1]
                series  = y_keys if y_keys else [keys[-1]]
            else:
                raw_data = []
                x_key    = "label"
                series   = ["value"]

            return func.HttpResponse(
                json.dumps({
                    "type":   "chart",
                    "answer": result.get("answer", ""),
                    "data":   raw_data,
                    "chart_config": {
                        "type":     result.get("chart_type", "bar"),
                        "xKey":     x_key,
                        "series":   series,
                        "dualAxis": False,
                    },
                    "query":   user_query,
                    "sources": sources,
                }),
                status_code=200, mimetype="application/json")

        answer = result.get("answer", "No relevant data found in documents.") or "No relevant data found in documents."
        return func.HttpResponse(
            json.dumps({"type": "text", "answer": answer,
                        "query": user_query, "sources": sources}),
            status_code=200, mimetype="application/json")

    except Exception as exc:
        logging.exception("Query error.")
        return func.HttpResponse(json.dumps({"error": "Internal server error.", "detail": str(exc)}),
                                 status_code=500, mimetype="application/json")


# ---------------------------------------------------------------------------
# DELETE /api/document/{id} — synchronous cascade delete
# ---------------------------------------------------------------------------

@app.route(route="document/{id}", methods=["DELETE"])
def delete_document_endpoint(req: func.HttpRequest) -> func.HttpResponse:
    """
    Synchronous cascade delete.

    Deletes all associated resources (Blob, Search, Table) in-request.
    Queue-based async deletion is disabled — set ENABLE_QUEUE=true in
    local.settings.json to re-enable it and restore the queue worker.
    """
    record_id = req.route_params.get("id", "").strip().strip("{}")
    logging.info("[DELETE API] Request received | id=%s", record_id)

    if not record_id:
        logging.warning("[DELETE API] Missing document ID in request")
        return func.HttpResponse(
            json.dumps({"error": "Document ID is required."}),
            status_code=400,
            mimetype="application/json",
        )

    try:
        logging.info("[DELETE] Processing document | id=%s", record_id)
        result = delete_document(record_id)

        if not result.found:
            logging.warning("[DELETE] Document not found | id=%s", record_id)
            return func.HttpResponse(
                json.dumps({"error": "Document not found", "id": record_id}),
                status_code=404,
                mimetype="application/json",
            )

        if result.success:
            logging.info("[DELETE SUCCESS] id=%s cid=%s", record_id, result.correlation_id[:8])
        else:
            logging.error("[DELETE PARTIAL] id=%s errors=%s", record_id, result.errors)

        return func.HttpResponse(
            json.dumps(result.to_dict()),
            status_code=200,
            mimetype="application/json",
        )

    except Exception as exc:
        logging.error("[DELETE ERROR] id=%s error=%s", record_id, exc)
        logging.exception("DELETE /api/document/%s — unexpected error", record_id)
        return func.HttpResponse(
            json.dumps({"error": "Internal server error.", "detail": str(exc)}),
            status_code=500,
            mimetype="application/json",
        )




# ---------------------------------------------------------------------------
# POST /cleanup-session — delete all temp blobs for a session
# Called by frontend "Clear Chat" to remove all temp-uploaded files
# ---------------------------------------------------------------------------

@app.route(route="cleanup-session", methods=["POST"])
def cleanup_session(req: func.HttpRequest) -> func.HttpResponse:
    """
    Delete all temporary blobs and Table Storage entities for a given session_id.
    Called when the user clicks "Clear Chat" in the Files Knowledge Bot.
    """
    logging.info("POST /cleanup-session")
    try:
        body       = req.get_json(silent=True) or {}
        session_id = body.get("session_id", "").strip()

        if not session_id:
            return func.HttpResponse(
                json.dumps({"error": "session_id is required."}),
                status_code=400, mimetype="application/json")

        from azure.storage.blob import BlobServiceClient, ContainerClient
        from services.config import require_env as _re

        conn_str   = _re("AZURE_STORAGE_CONNECTION_STRING")
        blob_svc_c = BlobServiceClient.from_connection_string(conn_str)

        deleted_blobs    = 0
        deleted_entities = 0
        errors           = []

        # Delete all blobs under temp/{session_id}/ prefix in both containers
        for container_name in ("documents", "images"):
            try:
                container: ContainerClient = blob_svc_c.get_container_client(container_name)
                prefix = f"temp/{session_id}/"
                blobs  = list(container.list_blobs(name_starts_with=prefix))
                for blob in blobs:
                    try:
                        container.delete_blob(blob.name)
                        deleted_blobs += 1
                        logging.info("Deleted temp blob: %s/%s", container_name, blob.name)
                    except Exception as be:
                        errors.append(f"blob:{container_name}/{blob.name}: {be}")
            except Exception as ce:
                logging.warning("cleanup-session: container '%s' error: %s", container_name, ce)

        # Delete Table Storage entities with matching session_id
        try:
            table_svc = TableService()
            docs      = table_svc.list_documents()
            for doc in docs:
                if doc.get("session_id") == session_id:
                    try:
                        from services.delete_service import delete_document as _del_doc
                        _del_doc(doc["id"])
                        deleted_entities += 1
                        logging.info("Deleted temp entity: %s", doc["id"])
                    except Exception as de:
                        errors.append(f"entity:{doc['id']}: {de}")
        except Exception as te:
            logging.warning("cleanup-session: table cleanup error: %s", te)

        logging.info("cleanup-session: session=%s blobs=%d entities=%d errors=%d",
                     session_id, deleted_blobs, deleted_entities, len(errors))

        return func.HttpResponse(
            json.dumps({
                "message":          f"Session cleanup complete.",
                "deleted_blobs":    deleted_blobs,
                "deleted_entities": deleted_entities,
                "errors":           errors,
            }),
            status_code=200, mimetype="application/json")

    except Exception as exc:
        logging.exception("cleanup-session error.")
        return func.HttpResponse(
            json.dumps({"error": "Internal server error.", "detail": str(exc)}),
            status_code=500, mimetype="application/json")
