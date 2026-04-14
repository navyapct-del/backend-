import os
import uuid
import time
import json
import logging
from datetime import datetime, timezone
from azure.data.tables import TableServiceClient, TableClient, UpdateMode
from services.config import require_env

TABLE_NAME    = "documentsmetadata"
PARTITION_KEY = "documents"

# Increment this when the structured_data schema changes.
# Any stored record with a lower version will be auto-reprocessed.
SCHEMA_VERSION = 3   # v3 = dynamic header detection + clean column names

_table_client: TableClient | None = None


def _get_client() -> TableClient:
    global _table_client
    if _table_client is not None:
        return _table_client

    conn_str = require_env("AZURE_STORAGE_CONNECTION_STRING")
    svc = TableServiceClient.from_connection_string(
        conn_str, connection_timeout=5, read_timeout=8,
    )
    try:
        svc.create_table_if_not_exists(TABLE_NAME)
        logging.info("Table '%s' ready.", TABLE_NAME)
    except Exception as exc:
        logging.warning("Table init warning (non-fatal): %s", exc)

    _table_client = svc.get_table_client(TABLE_NAME)
    return _table_client


class TableService:
    def __init__(self):
        self._client = _get_client()

    # ------------------------------------------------------------------
    # INSERT — status starts as "processing"
    # ------------------------------------------------------------------

    def insert_entity(self, filename: str, blob_url: str, description: str, tags: str,
                      temp: bool = False, session_id: str = "") -> str:
        row_key = str(uuid.uuid4())
        entity  = {
            "PartitionKey":   PARTITION_KEY,
            "RowKey":         row_key,
            "filename":       filename,
            "blob_url":       blob_url,          # original file URL
            "description":    description[:500], # hard cap
            "tags":           tags[:500],
            "summary":        "",                # max 2KB — stored inline
            "text_url":       "",                # → Blob: metadata/{id}/text.txt
            "structured_data_url": "",           # → Blob: metadata/{id}/structured_data.json
            "status":         "processing",
            "schema_version": SCHEMA_VERSION,
            "created_at":     datetime.now(timezone.utc).isoformat(),
            "temp":           temp,
            "session_id":     session_id[:100] if session_id else "",
        }
        try:
            self._client.create_entity(entity=entity)
            logging.info("Table insert: RowKey=%s filename=%s temp=%s", row_key, filename, temp)
        except Exception:
            logging.exception("Table insert failed for filename=%s", filename)
            raise
        return row_key

    # ------------------------------------------------------------------
    # UPDATE — write text + summary + tags, mark completed
    # ------------------------------------------------------------------

    def update_ai_fields(self, filename: str, text: str, summary: str, tags: str,
                         structured_data: dict | None = None,
                         text_url: str = "",
                         structured_data_url: str = "") -> bool:
        """
        Store metadata in Table Storage (lightweight fields only).
        Large content (text, structured_data) is stored in Blob Storage —
        pass their URLs here. Falls back to inline truncated text if no URL.
        """
        # Summary: max 2KB inline (safe for Table Storage)
        summary_safe = summary[:2000] if summary else ""

        # Tags: max 500 chars inline
        tags_safe = tags[:500] if tags else ""

        # Text: only store inline if no blob URL (fallback, truncated to 8KB)
        text_inline = ""
        if not text_url and text:
            text_inline = text.encode("utf-8")[:8000].decode("utf-8", errors="ignore")
            logging.warning("update_ai_fields: no text_url — storing truncated text inline (%d chars)",
                            len(text_inline))

        try:
            entities = list(self._client.query_entities(
                query_filter=f"PartitionKey eq '{PARTITION_KEY}' and filename eq '{filename}'"
            ))
            if not entities:
                logging.warning("update_ai_fields: no entity for filename=%s", filename)
                return False

            e     = entities[0]
            # Structured data: only store inline if no blob URL (small payloads only)
            sd_inline = ""
            if not structured_data_url and structured_data:
                try:
                    sd_json = json.dumps(structured_data, ensure_ascii=False)
                    # Table Storage property limit is 64 KB; stay well under it
                    if len(sd_json.encode("utf-8")) <= 32000:
                        sd_inline = sd_json
                    else:
                        logging.warning(
                            "update_ai_fields: structured_data too large for inline storage "
                            "(%d bytes) — skipping inline for filename=%s",
                            len(sd_json.encode("utf-8")), filename,
                        )
                except Exception as exc:
                    logging.warning("update_ai_fields: could not serialise structured_data: %s", exc)

            patch = {
                "PartitionKey":          e["PartitionKey"],
                "RowKey":                e["RowKey"],
                "summary":               summary_safe,
                "tags":                  tags_safe,
                "status":                "completed",
                "schema_version":        SCHEMA_VERSION,
                "processed_at":          datetime.now(timezone.utc).isoformat(),
                "text_url":              text_url,
                "structured_data_url":   structured_data_url,
                "text":                  text_inline,        # fallback only
                "structured_data":       sd_inline,          # fallback only (small files)
            }

            self._client.update_entity(entity=patch, mode=UpdateMode.MERGE)
            logging.info(
                "Table update: %s → completed v%d | text_url=%s | sd_url=%s",
                filename, SCHEMA_VERSION, bool(text_url), bool(structured_data_url),
            )
            return True
        except Exception:
            logging.exception("Table update failed for filename=%s", filename)
            raise

    def get_structured_data(self, filename: str) -> dict | None:
        """
        Retrieve structured data — downloads from Blob Storage if URL exists,
        falls back to inline field. Returns None if stale or missing.
        """
        try:
            entities = list(self._client.query_entities(
                query_filter=f"PartitionKey eq '{PARTITION_KEY}' and filename eq '{filename}'"
            ))
            if not entities:
                return None

            e  = entities[0]
            sv = int(e.get("schema_version", 0))
            if sv < SCHEMA_VERSION:
                logging.warning("get_structured_data: '%s' stale v%d < v%d", filename, sv, SCHEMA_VERSION)
                return None

            # Prefer Blob URL
            sd_url = e.get("structured_data_url", "")
            if sd_url:
                from services.blob_service import BlobService
                sd = BlobService().download_structured_data(sd_url)
                if sd.get("sheets") or sd.get("rows"):
                    return sd
                return None

            # Fallback: inline field (legacy / small files)
            raw = e.get("structured_data", "")
            if not raw:
                return None
            sd = json.loads(raw)
            if int(sd.get("_version", 0)) < SCHEMA_VERSION:
                return None
            return sd if (sd.get("sheets") or sd.get("rows")) else None

        except Exception:
            logging.exception("get_structured_data failed for filename=%s", filename)
            return None

    def get_text(self, filename: str) -> str:
        """
        Retrieve extracted text — downloads from Blob Storage if URL exists,
        falls back to inline truncated text.
        """
        try:
            entities = list(self._client.query_entities(
                query_filter=f"PartitionKey eq '{PARTITION_KEY}' and filename eq '{filename}'"
            ))
            if not entities:
                return ""
            e        = entities[0]
            text_url = e.get("text_url", "")
            if text_url:
                from services.blob_service import BlobService
                return BlobService().download_text(text_url)
            return e.get("text", "")   # inline fallback
        except Exception:
            logging.exception("get_text failed for filename=%s", filename)
            return ""

    def get_stale_documents(self) -> list[dict]:
        """
        Return all completed documents whose schema_version < SCHEMA_VERSION.
        Used by /reprocess endpoint to auto-update stale records.
        """
        try:
            entities = list(self._client.query_entities(
                query_filter=f"PartitionKey eq '{PARTITION_KEY}' and status eq 'completed'"
            ))
            stale = []
            for e in entities:
                stored_v = int(e.get("schema_version", 0))
                if stored_v < SCHEMA_VERSION:
                    stale.append({
                        "PartitionKey": e["PartitionKey"],
                        "RowKey":       e["RowKey"],
                        "filename":     e.get("filename", ""),
                        "blob_url":     e.get("blob_url", ""),
                        "schema_version": stored_v,
                    })
            logging.info("get_stale_documents: %d stale (v<%d)", len(stale), SCHEMA_VERSION)
            return stale
        except Exception:
            logging.exception("get_stale_documents failed.")
            return []

    # ------------------------------------------------------------------
    # UPDATE EMBEDDING — called after update_ai_fields
    # ------------------------------------------------------------------

    def update_embedding(self, filename: str, embedding: list[float]) -> bool:
        """Store the embedding vector as a JSON string on the entity."""
        try:
            entities = list(self._client.query_entities(
                query_filter=f"PartitionKey eq '{PARTITION_KEY}' and filename eq '{filename}'"
            ))
            if not entities:
                return False
            e = entities[0]
            self._client.update_entity(entity={
                "PartitionKey": e["PartitionKey"],
                "RowKey":       e["RowKey"],
                "embedding":    json.dumps(embedding),
            }, mode=UpdateMode.MERGE)
            logging.info("Embedding stored for filename=%s (%d dims)", filename, len(embedding))
            return True
        except Exception:
            logging.exception("update_embedding failed for filename=%s", filename)
            return False

    # ------------------------------------------------------------------
    # SEMANTIC SEARCH — embedding cosine similarity, falls back to keyword
    # ------------------------------------------------------------------

    def semantic_search(
        self,
        query_embedding: list[float],
        query: str,
        top: int = 5,
        filename_filter: str = "",
    ) -> list[dict]:
        """
        Rank all completed documents by cosine similarity to the query embedding.
        Falls back to keyword search if no embeddings are stored yet.
        """
        from services.openai_service import cosine_similarity
        t0 = time.time()
        try:
            filter_str = f"PartitionKey eq '{PARTITION_KEY}' and status eq 'completed'"
            entities   = list(self._client.query_entities(query_filter=filter_str))

            if filename_filter:
                entities = [e for e in entities
                            if filename_filter.lower() in e.get("filename", "").lower()]

            scored = []
            no_embedding_count = 0

            for e in entities:
                raw_emb = e.get("embedding", "")
                logging.info("Embedding exists for '%s': %s", e.get("filename"), bool(raw_emb))
                if not raw_emb:
                    no_embedding_count += 1
                    continue
                try:
                    doc_emb = json.loads(raw_emb)
                    score   = cosine_similarity(query_embedding, doc_emb)
                    scored.append((score, e))
                except Exception:
                    logging.exception("Failed to parse embedding for '%s'", e.get("filename"))
                    no_embedding_count += 1

            # Fall back to keyword search if no embeddings exist
            if not scored:
                logging.info("semantic_search: no embeddings found, falling back to keyword search")
                return self.search_documents(query, top=top, filename_filter=filename_filter)

            scored.sort(key=lambda x: x[0], reverse=True)
            results = [{
                "id":       e.get("RowKey", ""),
                "filename": e.get("filename", ""),
                "blob_url": e.get("blob_url", ""),
                "summary":  e.get("summary", ""),
                "tags":     e.get("tags", ""),
                "text":     e.get("text", ""),
                "text_url": e.get("text_url", ""),
                "score":    round(sc, 4),
            } for sc, e in scored[:top]]

            logging.info(
                "semantic_search: top=%d best_score=%.3f in %.3fs (skipped %d without embeddings)",
                len(results), results[0]["score"] if results else 0,
                time.time() - t0, no_embedding_count,
            )
            return results

        except Exception as exc:
            logging.error("semantic_search failed: %s", exc)
            return self.search_documents(query, top=top, filename_filter=filename_filter)

    def search_documents(self, query: str, top: int = 5, filename_filter: str = "") -> list[dict]:
        """Keyword search. Text is fetched from Blob URL when available."""
        t0 = time.time()
        try:
            filter_str = f"PartitionKey eq '{PARTITION_KEY}' and status eq 'completed'"
            entities   = list(self._client.query_entities(query_filter=filter_str))
            q_lower    = query.lower()
            results    = []

            for e in entities:
                if filename_filter and filename_filter.lower() not in e.get("filename", "").lower():
                    continue

                # Use inline text for search (fast) — full text fetched on demand
                text = e.get("text", "")
                summary  = e.get("summary", "")
                tags     = e.get("tags", "")
                filename = e.get("filename", "")

                if (q_lower in text.lower() or q_lower in tags.lower()
                        or q_lower in summary.lower() or q_lower in filename.lower()):
                    results.append({
                        "id":        e.get("RowKey", ""),
                        "filename":  filename,
                        "blob_url":  e.get("blob_url", ""),
                        "summary":   summary,
                        "tags":      tags,
                        "text":      text,          # inline (may be truncated)
                        "text_url":  e.get("text_url", ""),   # full text URL
                    })

            if not results and not filename_filter:
                results = [{
                    "id":       e.get("RowKey", ""),
                    "filename": e.get("filename", ""),
                    "blob_url": e.get("blob_url", ""),
                    "summary":  e.get("summary", ""),
                    "tags":     e.get("tags", ""),
                    "text":     e.get("text", ""),
                    "text_url": e.get("text_url", ""),
                } for e in entities]

            logging.info("search_documents: %d results in %.3fs", len(results[:top]), time.time() - t0)
            return results[:top]
        except Exception as exc:
            logging.error("search_documents failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # REINDEX — fetch all completed docs that are missing embeddings
    # ------------------------------------------------------------------

    def get_docs_missing_embeddings(self) -> list[dict]:
        """Return all completed entities that have text but no embedding."""
        try:
            entities = list(self._client.query_entities(
                query_filter=f"PartitionKey eq '{PARTITION_KEY}' and status eq 'completed'"
            ))
            missing = []
            for e in entities:
                has_embedding = bool(e.get("embedding", ""))
                has_text      = bool(e.get("text", ""))
                logging.info("Doc '%s': embedding=%s text=%s", e.get("filename"), has_embedding, has_text)
                if has_text and not has_embedding:
                    missing.append({
                        "PartitionKey": e["PartitionKey"],
                        "RowKey":       e["RowKey"],
                        "filename":     e.get("filename", ""),
                        "text":         e.get("text", ""),
                    })
            logging.info("get_docs_missing_embeddings: %d docs need reindex", len(missing))
            return missing
        except Exception:
            logging.exception("get_docs_missing_embeddings failed.")
            return []

    def find_by_filename(self, filename: str) -> dict | None:
        """Return the first entity matching the given filename, or None."""
        try:
            entities = list(self._client.query_entities(
                query_filter=f"PartitionKey eq '{PARTITION_KEY}' and filename eq '{filename}'",
                select=["RowKey", "filename", "status", "temp"],
            ))
            for e in entities:
                if not e.get("temp", False):
                    return {"id": e.get("RowKey", ""), "filename": e.get("filename", ""), "status": e.get("status", "")}
            return None
        except Exception as exc:
            logging.error("find_by_filename failed for '%s': %s", filename, exc)
            return None

    def delete_session_documents(self, session_id: str) -> int:
        """Delete all temp entities for a given session_id. Returns count deleted."""
        try:
            entities = list(self._client.query_entities(
                query_filter=f"PartitionKey eq '{PARTITION_KEY}' and session_id eq '{session_id}'"
            ))
            count = 0
            for e in entities:
                self._client.delete_entity(partition_key=e["PartitionKey"], row_key=e["RowKey"])
                count += 1
            logging.info("delete_session_documents: deleted %d entities for session=%s", count, session_id)
            return count
        except Exception as exc:
            logging.error("delete_session_documents failed: %s", exc)
            return 0
        """Patch a single entity with its embedding by RowKey."""
        self._client.update_entity(entity={
            "PartitionKey": partition_key,
            "RowKey":       row_key,
            "embedding":    json.dumps(embedding),
        }, mode=UpdateMode.MERGE)

    # ------------------------------------------------------------------
    # LIST — lightweight, for frontend polling (no text field)
    # ------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        t0 = time.time()
        try:
            entities = list(self._client.query_entities(
                query_filter=f"PartitionKey eq '{PARTITION_KEY}'",
                select=["RowKey", "filename", "summary", "tags", "blob_url", "status", "created_at", "temp"],
            ))
            docs = [{
                "id":         e.get("RowKey", ""),
                "filename":   e.get("filename", ""),
                "summary":    e.get("summary", ""),
                "tags":       e.get("tags", ""),
                "blob_url":   e.get("blob_url", ""),
                "status":     e.get("status", "processing"),
                "created_at": e.get("created_at", ""),
            } for e in entities
              if not e.get("temp", False)]   # exclude temp uploads from Files Knowledge Bot
            docs.sort(key=lambda d: d["created_at"], reverse=True)
            logging.info("list_documents: %d docs in %.3fs", len(docs), time.time() - t0)
            return docs
        except Exception as exc:
            logging.error("list_documents failed: %s", exc)
            return []
