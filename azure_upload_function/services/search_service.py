import os
import logging
import requests
from time import sleep
from services.config import require_env

SEARCH_INDEX = "documents-index"
_API_VERSION = "2023-11-01"

_headers: dict | None = None

def _get_headers() -> dict:
    global _headers
    if _headers is None:
        _headers = {
            "Content-Type": "application/json",
            "api-key":      require_env("AZURE_SEARCH_KEY"),
        }
    return _headers

def _endpoint() -> str:
    return require_env("AZURE_SEARCH_ENDPOINT").rstrip("/")


# ---------------------------------------------------------------------------
# Index management — keyword search only (compatible with all tiers)
# ---------------------------------------------------------------------------

def delete_index() -> bool:
    url  = f"{_endpoint()}/indexes/{SEARCH_INDEX}?api-version={_API_VERSION}"
    resp = requests.delete(url, headers=_get_headers(), timeout=10)
    if resp.status_code in (200, 204, 404):
        logging.info("AI Search index '%s' deleted (or did not exist).", SEARCH_INDEX)
        return True
    logging.error("Failed to delete index: %s — %s", resp.status_code, resp.text)
    return False


def ensure_index() -> None:
    """
    Create a keyword-search index compatible with ALL Azure AI Search tiers
    including Free. No vector fields — uses full-text search only.
    """
    url   = f"{_endpoint()}/indexes/{SEARCH_INDEX}?api-version={_API_VERSION}"
    check = requests.get(url, headers=_get_headers(), timeout=10)
    if check.status_code == 200:
        logging.info("AI Search index '%s' already exists.", SEARCH_INDEX)
        return

    schema = {
        "name": SEARCH_INDEX,
        "fields": [
            {"name": "id",       "type": "Edm.String",             "key": True,  "searchable": False, "filterable": True,  "retrievable": True},
            {"name": "filename", "type": "Edm.String",             "key": False, "searchable": True,  "filterable": True,  "retrievable": True},
            {"name": "content",  "type": "Edm.String",             "key": False, "searchable": True,  "filterable": False, "retrievable": True},
            {"name": "summary",  "type": "Edm.String",             "key": False, "searchable": True,  "filterable": False, "retrievable": True},
            {"name": "tags",     "type": "Collection(Edm.String)", "key": False, "searchable": True,  "filterable": True,  "retrievable": True},
            {"name": "blob_url", "type": "Edm.String",             "key": False, "searchable": False, "filterable": False, "retrievable": True},
        ],
    }

    resp = requests.put(url, headers=_get_headers(), json=schema, timeout=20)
    logging.info("Index creation status: %s", resp.status_code)
    if resp.status_code not in (200, 201):
        logging.error("Index creation failed: %s", resp.text)
        resp.raise_for_status()
    else:
        logging.info("AI Search index '%s' created successfully.", SEARCH_INDEX)


# ---------------------------------------------------------------------------
# Index a document — no embedding field
# ---------------------------------------------------------------------------

def index_document(
    doc_id:   str,
    filename: str,
    content:  str,
    summary:  str,
    tags:     list[str],
    blob_url: str,
    embedding: list[float] = None,   # accepted but not stored (future use)
    retries:  int = 3,
) -> None:
    """Push a document into AI Search using keyword fields only."""
    url  = f"{_endpoint()}/indexes/{SEARCH_INDEX}/docs/index?api-version={_API_VERSION}"
    body = {
        "value": [{
            "@search.action": "upload",
            "id":       doc_id,
            "filename": filename,
            "content":  content[:32000],
            "summary":  summary,
            "tags":     tags,
            "blob_url": blob_url,
        }]
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, headers=_get_headers(), json=body, timeout=20)
            if resp.status_code in (200, 207):
                logging.info("AI Search indexed: id=%s filename=%s", doc_id, filename)
                return
            logging.warning("index_document attempt %d/%d: %s — %s",
                            attempt, retries, resp.status_code, resp.text[:200])
        except requests.RequestException as exc:
            logging.warning("index_document attempt %d/%d exception: %s", attempt, retries, exc)
        if attempt < retries:
            sleep(2 ** attempt)

    raise RuntimeError(f"AI Search indexing failed for id={doc_id} after {retries} attempts")


# ---------------------------------------------------------------------------
# Keyword search (full-text, works on all tiers)
# ---------------------------------------------------------------------------

def vector_search(
    query_embedding: list[float],
    query_text:      str,
    top:             int = 3,          # default reduced to 3
    filename_filter: str = "",
    min_score:       float = 0.01,     # AI Search BM25 scores vary; filter near-zero
) -> list[dict]:
    """
    Full-text keyword search with relevance filtering and top-K capping.
    Returns at most `top` results with score >= min_score.
    """
    url  = f"{_endpoint()}/indexes/{SEARCH_INDEX}/docs/search?api-version={_API_VERSION}"
    body: dict = {
        "search":       query_text,
        "searchFields": "content,summary,tags,filename",
        "select":       "id,filename,summary,content,blob_url,tags",
        "top":          top,
        "queryType":    "simple",
    }
    if filename_filter:
        # Use search.ismatch for partial/case-insensitive filename filtering
        body["filter"] = f"search.ismatch('{filename_filter}', 'filename')"

    try:
        resp = requests.post(url, headers=_get_headers(), json=body, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("value", [])

        # Filter by minimum relevance score and cap at top-K
        filtered = [
            {
                "id":       r.get("id", ""),
                "filename": r.get("filename", ""),
                "blob_url": r.get("blob_url", ""),
                "summary":  r.get("summary", ""),
                "tags":     r.get("tags", []),
                "text":     r.get("content", ""),
                "score":    round(r.get("@search.score", 0), 4),
            }
            for r in results
            if r.get("@search.score", 0) >= min_score
        ][:top]   # hard cap

        logging.info("keyword_search: %d/%d results (score>=%.3f) for query='%s'",
                     len(filtered), len(results), min_score, query_text[:60])
        return filtered

    except requests.HTTPError as exc:
        logging.error("keyword_search HTTP error: %s — %s",
                      exc.response.status_code, exc.response.text[:300])
        return []
    except Exception:
        logging.exception("keyword_search failed.")
        return []
