import os
import re
import json
import math
import logging
from openai import AzureOpenAI
from services.config import require_env, get_env

# Shared client — instantiated once per worker lifetime.
# Endpoint: standard Azure OpenAI (openai.azure.com)
# API version 2024-05-01-preview supports gpt-4o and text-embedding-3-small.
_client: AzureOpenAI | None = None

def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        endpoint = require_env("AZURE_OPENAI_ENDPOINT").rstrip("/")
        # Standard Azure OpenAI endpoint (openai.azure.com) — 2024-05-01-preview
        # supports gpt-4o and text-embedding-3-small.
        # Override via AZURE_OPENAI_API_VERSION env var if needed.
        api_version = get_env("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
        _client = AzureOpenAI(
            api_key        = require_env("AZURE_OPENAI_API_KEY"),
            api_version    = api_version,
            azure_endpoint = endpoint,
        )
        logging.info("AzureOpenAI client initialised | endpoint=%s | api_version=%s",
                     endpoint, api_version)
    return _client

def _deployment() -> str:
    name = get_env("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    logging.info("Using chat deployment: %s", name)
    return name

_EMBED_MODEL = get_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def generate_embedding(text: str) -> list[float]:
    """
    Generate an embedding vector for the given text.
    Input is truncated to 2000 chars to keep cost low.
    Returns [] on failure so callers can fall back to keyword search.
    """
    if not text:
        return []
    try:
        resp = _get_client().embeddings.create(
            model = _EMBED_MODEL,
            input = text[:2000],
        )
        return resp.data[0].embedding
    except Exception as exc:
        logging.exception("generate_embedding failed.")
        return []


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def generate_summary(text: str) -> str:
    """Generate a 3-4 line summary of the document."""
    if not text:
        return ""
    prompt = f"Summarize the following document in 3-4 concise lines:\n\n{text[:2000]}"
    try:
        resp = _get_client().chat.completions.create(
            model       = _deployment(),
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.2,
            max_tokens  = 150,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        logging.exception("generate_summary failed.")
        return text[:300].strip()


# ---------------------------------------------------------------------------
# Tags / key phrases
# ---------------------------------------------------------------------------

def generate_tags(text: str) -> str:
    """Extract up to 10 key phrases as a comma-separated string."""
    if not text:
        return ""
    prompt = (
        "Extract up to 10 key topics or phrases from the text below. "
        "Return ONLY a comma-separated list, nothing else.\n\n"
        f"{text[:2000]}"
    )
    try:
        resp = _get_client().chat.completions.create(
            model       = _deployment(),
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.0,
            max_tokens  = 100,
        )
        return resp.choices[0].message.content.strip()[:500]
    except Exception:
        logging.exception("generate_tags failed.")
        return ""


# ---------------------------------------------------------------------------
# RAG answer
# ---------------------------------------------------------------------------

def generate_rag_answer(query: str, docs: list[dict]) -> dict:
    """
    Generate a structured response from retrieved document chunks.
    Returns a dict with keys: type, answer (and optionally columns/rows for table,
    labels/values for chart).
    Always returns valid JSON-serialisable dict — never raises.
    """
    if not query or not query.strip():
        return {"type": "text", "answer": "No question provided."}

    if not docs:
        return {"type": "text", "answer": "No relevant documents found."}

    context_parts = []
    for i, doc in enumerate(docs, 1):
        # Fetch full text from Blob URL if available; fall back to inline text
        text     = ""
        text_url = doc.get("text_url", "")
        if text_url:
            try:
                from services.blob_service import BlobService
                text = BlobService().download_text(text_url)
                logging.info("RAG: fetched full text from blob for '%s' (%d chars)",
                             doc.get("filename", ""), len(text))
            except Exception as exc:
                logging.warning("RAG: blob text fetch failed for '%s': %s", doc.get("filename"), exc)
                text = (doc.get("content") or doc.get("text") or "").strip()
        else:
            text = (doc.get("content") or doc.get("text") or doc.get("extracted_text") or "").strip()

        text = text[:6000]   # increased — gives LLM enough content to count/extract data for charts
        if text:
            context_parts.append(f"[Doc {i}: {doc.get('filename', f'Doc {i}')}]\n{text}")

    if not context_parts:
        return {"type": "text", "answer": "No relevant documents found."}

    context = ("\n\n".join(context_parts))[:10000]   # increased to give LLM full document context for counting/extraction

    safe_prompt = f"""You are a precise assistant. Answer the question using ONLY the information in the documents below.

RULES:
1. Respond with ONLY a single valid JSON object — no markdown, no text outside the JSON.
2. Start your response with {{ and end with }}.
3. NEVER fabricate, invent, or hallucinate data. Only use what is explicitly in the documents.
4. If the answer is not in the documents, return: {{"type":"text","answer":"The documents do not contain specific information about this."}}

RESPONSE FORMATS:
- Text answer:  {{"type":"text","answer":"your answer here"}}
- Table answer: {{"type":"table","columns":["Col1","Col2"],"rows":[{{"Col1":"val","Col2":"val"}}],"answer":"optional summary"}}
- Chart answer: {{"type":"chart","chart_type":"bar","labels":["A","B"],"values":[1,2],"answer":"optional summary"}}

WHEN TO USE EACH FORMAT:
- Use "table" when the question asks to list, enumerate, or show multiple items with attributes (e.g. "list all guidelines", "show all regulations")
- Use "chart" ONLY when the question explicitly asks for a graph, chart, plot, or visualization AND you have actual numeric data from the document
- Use "text" for all other questions

FOR TABLE RESPONSES:
- Include ALL items found in the document — do not summarize or truncate
- Use the exact names/values from the document text

Documents:
{context}

Question: {query}"""

    deployment = _deployment()
    logging.info("Using model: %s | Prompt length: %d", deployment, len(safe_prompt))

    try:
        resp = _get_client().chat.completions.create(
            model    = deployment,
            messages = [{"role": "user", "content": safe_prompt.strip()}],
            temperature = 0.1,
            max_tokens  = 4000,   # increased significantly — table responses with many rows need more tokens
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if model wraps in ```json
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()

        # Try to find JSON anywhere in the response (model sometimes adds preamble text)
        def _extract_json(s: str) -> dict | None:
            # Try direct parse first
            try:
                p = json.loads(s)
                if isinstance(p, dict) and "type" in p:
                    return p
            except Exception:
                pass
            # Try finding first { ... } block
            match = re.search(r'\{[\s\S]*\}', s)
            if match:
                try:
                    p = json.loads(match.group())
                    if isinstance(p, dict) and "type" in p:
                        return p
                except Exception:
                    pass
            return None

        parsed = _extract_json(raw)
        if parsed:
            logging.info("Structured response type: %s", parsed.get("type"))
            # If answer field is itself a JSON string, unwrap it
            if isinstance(parsed.get("answer"), str):
                inner = _extract_json(parsed["answer"])
                if inner:
                    return inner
            return parsed

        # Fallback: return as plain text answer
        return {"type": "text", "answer": raw}

    except Exception as exc:
        logging.error("generate_rag_answer failed | deployment=%s | error=%s", deployment, str(exc))
        return {"type": "text", "answer": "Failed to generate answer."}


# ---------------------------------------------------------------------------
# Smart chart builder — query-aware filtering + pivoting of stored data
# ---------------------------------------------------------------------------

def smart_chart_from_structured(query: str, structured: dict) -> dict | None:
    """
    Given stored structured_data {"columns": [...], "rows": [...]} from an
    Excel/CSV upload, intelligently:
      1. Detects entities mentioned in the query (e.g. Bihar, Maharashtra)
      2. Finds the column that contains those entities (e.g. "State")
      3. Filters rows to only those entities
      4. Detects the x-axis column (Year, Date, Month, etc.)
      5. Detects the value column (numeric)
      6. Pivots: rows of (x, entity, value) → columns per entity
      7. Returns chart-ready data + chart_config

    Returns None if the data cannot be meaningfully charted.
    """
    try:
        import pandas as pd

        sheets = structured.get("sheets", {})   # per-sheet data if available

        # ── 1. Build working DataFrame for entity detection ───────────────
        # Prefer flat rows; fall back to merging all sheet rows
        all_rows = structured.get("rows", [])
        if not all_rows and sheets:
            for sname, sd in sheets.items():
                for r in sd.get("rows", []):
                    row = dict(r)
                    row["_sheet"] = sname
                    all_rows.append(row)

        if not all_rows:
            return None

        df_all   = pd.DataFrame(all_rows)
        q_lower  = query.lower()
        entities = _extract_entities_from_query(q_lower, df_all)
        logging.info("smart_chart: entities detected = %s", entities)

        # ── 2. Select the best sheet (if per-sheet data available) ────────
        if sheets:
            best_sheet = _select_best_sheet(q_lower, entities, sheets)
            logging.info("smart_chart: selected sheet = '%s'", best_sheet)
            if best_sheet:
                sd   = sheets[best_sheet]
                cols = sd["columns"]
                rows = sd["rows"]
                df   = pd.DataFrame(rows, columns=cols)
            else:
                # No clear winner — use all rows but skip _sheet column
                df = df_all.drop(columns=["_sheet"], errors="ignore")
        else:
            df = pd.DataFrame(
                structured.get("rows", []),
                columns=structured.get("columns") or None,
            )

        if df.empty:
            return None

        # ── 2. Find entity column (categorical column whose values match entities) ──
        entity_col = _find_entity_column(df, entities)
        logging.info("smart_chart: entity_col = %s", entity_col)

        # ── 3. Filter rows to only the requested entities ─────────────────
        if entity_col and entities:
            mask = df[entity_col].astype(str).str.upper().isin(
                [e.upper() for e in entities]
            )
            df = df[mask]
            if df.empty:
                logging.warning("smart_chart: filter produced empty DataFrame")
                return None

        # ── 4. Find x-axis column (Year, Date, Month, Quarter, Category) ──
        x_col = _find_column(df, ["year","date","month","quarter","period","category","name"])
        if not x_col:
            # fallback: first non-entity, non-numeric column
            for c in df.columns:
                if c != entity_col and df[c].dtype == object:
                    x_col = c
                    break
        if not x_col:
            x_col = df.columns[0]

        # ── 5. Find value column (first numeric column that isn't x or entity) ──
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        value_col    = next(
            (c for c in numeric_cols if c not in (x_col, entity_col)), None
        )
        if not value_col and numeric_cols:
            value_col = numeric_cols[0]
        if not value_col:
            logging.warning("smart_chart: no numeric column found")
            return None

        logging.info("smart_chart: x=%s entity=%s value=%s", x_col, entity_col, value_col)

        # ── 6. Pivot: (x, entity) → wide format ───────────────────────────
        if entity_col and entity_col != x_col:
            pivot = df.pivot_table(
                index   = x_col,
                columns = entity_col,
                values  = value_col,
                aggfunc = "sum",
            ).reset_index()
            pivot.columns = [str(c) for c in pivot.columns]
            series = [c for c in pivot.columns if c != x_col]
            data   = pivot.to_dict(orient="records")
        else:
            # No entity column — simple x vs value
            pivot  = df[[x_col, value_col]].copy()
            pivot.columns = [x_col, value_col]
            series = [value_col]
            data   = pivot.to_dict(orient="records")

        # ── 7. Determine chart type from query ────────────────────────────
        chart_type = "line" if any(k in q_lower for k in ["trend","growth","over time","line"]) else "bar"

        return {
            "data":         data,
            "chart_config": {
                "type":   chart_type,
                "xKey":   x_col,
                "series": series,
            },
            "script": (
                f"SELECT {x_col}, {entity_col}, {value_col} FROM data "
                f"WHERE {entity_col} IN ({', '.join(repr(e) for e in entities)}) "
                f"ORDER BY {x_col};"
            ) if entity_col else f"SELECT {x_col}, {value_col} FROM data ORDER BY {x_col};",
        }

    except Exception:
        logging.exception("smart_chart_from_structured failed.")
        return None


def _extract_entities_from_query(q_lower: str, df) -> list[str]:
    """
    Find words/phrases in the query that match actual values in any categorical column.
    Case-insensitive matching — returns original-cased values from the DataFrame.
    """
    import re

    # Build a lookup: lowercase_value → original_cased_value
    value_map: dict[str, str] = {}
    for col in df.select_dtypes(include="object").columns:
        if col.lower() in {"_sheet", "_file", "_source"}:
            continue
        for val in df[col].dropna().astype(str).unique():
            value_map[val.lower()] = val

    if not value_map:
        return []

    # Try multi-word matches first (e.g. "Tamil Nadu"), then single words
    matched = []
    seen    = set()

    # Multi-word: check every 2-3 word window
    words = re.findall(r"[a-zA-Z]+", q_lower)
    for size in (3, 2, 1):
        for i in range(len(words) - size + 1):
            phrase = " ".join(words[i:i + size])
            if phrase in value_map and value_map[phrase] not in seen:
                matched.append(value_map[phrase])
                seen.add(value_map[phrase])

    return matched


def _find_entity_column(df, entities: list[str]) -> str | None:
    """Find the column whose values contain the queried entities. Skips internal columns."""
    if not entities:
        return None
    entity_lower = {e.lower() for e in entities}
    # Skip internal/metadata columns
    skip = {"_sheet", "_file", "_source"}
    for col in df.select_dtypes(include="object").columns:
        if col.lower() in skip:
            continue
        col_vals = set(df[col].dropna().astype(str).str.lower().unique())
        if entity_lower & col_vals:   # intersection
            return col
    return None


def _find_column(df, candidates: list[str]) -> str | None:
    """Return the first column whose name (lowercased) matches any candidate."""
    lc = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lc:
            return lc[cand]
    return None


def _select_best_sheet(q_lower: str, entities: list[str], sheets: dict) -> str | None:
    """
    Pick the most relevant sheet for the query.

    Strategy (in order):
    1. If any entity value appears in a sheet's rows → prefer that sheet
    2. If any query word appears in the sheet name → prefer that sheet
    3. Return the first sheet as fallback
    """
    import pandas as pd

    entity_lower = {e.lower() for e in entities}

    # Score each sheet
    scores: dict[str, int] = {}
    for sheet_name, sd in sheets.items():
        score = 0
        # Check if sheet name contains query words
        sheet_lower = sheet_name.lower()
        for word in q_lower.split():
            if len(word) > 3 and word in sheet_lower:
                score += 2

        # Check if entity values exist in this sheet's data
        if entity_lower and sd.get("rows"):
            df = pd.DataFrame(sd["rows"])
            for col in df.select_dtypes(include="object").columns:
                if col.lower() in {"_sheet", "_file"}:
                    continue
                col_vals = set(df[col].dropna().astype(str).str.lower().unique())
                matches  = entity_lower & col_vals
                score   += len(matches) * 5   # entity match is high signal

        scores[sheet_name] = score
        logging.info("smart_chart: sheet '%s' score=%d", sheet_name, score)

    if not scores:
        return None

    best       = max(scores, key=lambda k: scores[k])
    best_score = scores[best]

    # Only use best sheet if it has a meaningful score
    if best_score > 0:
        return best

    # Fallback: first sheet
    return next(iter(sheets))


# ---------------------------------------------------------------------------
# Structured data extraction (for table/chart responses)
# ---------------------------------------------------------------------------

def extract_structured_data(query: str, docs: list[dict]) -> list[dict]:
    """
    Ask OpenAI to extract structured rows from document text.
    Returns a list of flat dicts, or [] on failure.
    """
    if not docs:
        return []

    context_parts = []
    for i, doc in enumerate(docs, 1):
        text = (doc.get("content") or doc.get("text") or doc.get("extracted_text") or "")[:2000]
        context_parts.append(f"[Document {i}: {doc.get('filename','')}]\n{text}")
    context = "\n\n".join(context_parts)

    prompt = (
        "You are a data extraction assistant.\n"
        "Extract ALL numerical/tabular data relevant to the question from the context.\n"
        "Return ONLY a valid JSON array of flat objects with consistent keys.\n"
        "Example: [{\"year\": 2018, \"state\": \"Bihar\", \"value\": 6239}]\n"
        "If no structured data exists, return: []\n"
        "Do NOT include any explanation — only the JSON array.\n\n"
        f"Question: {query}\n\nContext:\n{context}\n\nJSON:"
    )
    try:
        resp = _get_client().chat.completions.create(
            model       = _deployment(),
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.0,
            max_tokens  = 1500,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception:
        logging.exception("extract_structured_data failed.")
        return []


# ---------------------------------------------------------------------------
# Short explanation of structured data
# ---------------------------------------------------------------------------

def generate_explanation(query: str, data: list[dict]) -> str:
    if not data:
        return "No structured data could be extracted."
    sample = json.dumps(data[:6], indent=2)
    prompt = f"Summarize this data in 2-3 sentences relevant to: '{query}'\n\nData:\n{sample}\n\nSummary:"
    try:
        resp = _get_client().chat.completions.create(
            model       = _deployment(),
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.2,
            max_tokens  = 150,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        logging.exception("generate_explanation failed.")
        return ""
