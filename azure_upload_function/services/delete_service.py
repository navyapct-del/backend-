"""
services/delete_service.py
==========================
Cascade delete service — removes a document and ALL its associated artifacts
across Azure Blob Storage, Azure Table Storage, and Azure AI Search.

Design principles:
  - Idempotent: repeated calls on the same ID are safe (missing = already deleted)
  - Ordered:    Blob → Search → Table  (Table is always last)
  - Fault-tolerant: each step is isolated; failures are logged but do not
    prevent remaining cleanup steps from running
  - Observable: structured logging at every step with correlation via record_id
  - No HTTP logic: pure service layer, callable from any context
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

import requests
from azure.core.exceptions import ResourceNotFoundError
from azure.data.tables import TableServiceClient
from azure.storage.blob import BlobClient, BlobServiceClient

from services.config import require_env

# ---------------------------------------------------------------------------
# Constants — single source of truth, no hardcoding in callers
# ---------------------------------------------------------------------------
_DOCS_CONTAINER     = "documents"
_METADATA_CONTAINER = "metadata"
_TABLE_NAME         = "documentsmetadata"
_PARTITION_KEY      = "documents"
_SEARCH_INDEX       = "documents-index"
_SEARCH_API_VERSION = "2023-11-01"


# ---------------------------------------------------------------------------
# Result dataclass — structured return value, no exceptions leak to callers
# ---------------------------------------------------------------------------

@dataclass
class DeletionResult:
    record_id:          str
    found:              bool                  = True
    success:            bool                  = False
    blob_deleted:       bool                  = False
    text_blob_deleted:  bool                  = False
    sd_blob_deleted:    bool                  = False
    search_deleted:     bool                  = False
    table_deleted:      bool                  = False
    errors:             list[str]             = field(default_factory=list)
    correlation_id:     str                   = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> dict:
        if not self.found:
            return {
                "id":    self.record_id,
                "error": "Document not found",
            }
        return {
            "id":      self.record_id,
            "status":  "deleted" if self.success else "partial",
            "message": (
                "Document and all associated resources deleted successfully"
                if self.success
                else "Document deleted with some cleanup warnings — see errors"
            ),
            "details": {
                "blob_deleted":      self.blob_deleted,
                "text_blob_deleted": self.text_blob_deleted,
                "sd_blob_deleted":   self.sd_blob_deleted,
                "search_deleted":    self.search_deleted,
                "table_deleted":     self.table_deleted,
            },
            "errors":          self.errors,
            "correlation_id":  self.correlation_id,
        }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def delete_document(record_id: str) -> DeletionResult:
    """
    Orchestrate full cascade deletion for a document identified by record_id.

    Execution order (CRITICAL — Table is always last):
      1. Read metadata from Table Storage
      2. Delete raw file blob from documents container
      3. Delete metadata blobs (text.txt, structured_data.json)
      4. Delete document from Azure AI Search index
      5. Delete entity from Table Storage  ← FINAL STEP

    Returns a DeletionResult. Never raises — all exceptions are captured.
    """
    result = DeletionResult(record_id=record_id)
    log    = _logger(record_id, result.correlation_id)

    log.info("CASCADE DELETE started")

    # ── STEP 1: Read metadata ─────────────────────────────────────────────
    entity = _fetch_entity(record_id, result, log)
    if entity is None:
        # _fetch_entity sets result.found = False on not-found
        return result

    blob_url  = entity.get("blob_url", "")
    text_url  = entity.get("text_url", "")
    sd_url    = entity.get("structured_data_url", "")
    filename  = entity.get("filename", "")

    log.info("Metadata read | filename=%s | blob_url=%s | text_url=%s | sd_url=%s",
             filename, bool(blob_url), bool(text_url), bool(sd_url))

    # ── STEP 2: Delete raw file blob ──────────────────────────────────────
    if blob_url:
        result.blob_deleted = _delete_blob_by_url(blob_url, "raw file", result, log)
    else:
        log.warning("blob_url missing — skipping raw blob deletion")
        result.blob_deleted = True   # treat as already gone (idempotent)

    # ── STEP 3a: Delete text metadata blob ───────────────────────────────
    if text_url:
        result.text_blob_deleted = _delete_blob_by_url(text_url, "text metadata", result, log)
    else:
        # Fall back to well-known path pattern
        result.text_blob_deleted = _delete_blob_by_path(
            _METADATA_CONTAINER, f"{record_id}/text.txt", "text metadata (path)", result, log
        )

    # ── STEP 3b: Delete structured_data metadata blob ────────────────────
    if sd_url:
        result.sd_blob_deleted = _delete_blob_by_url(sd_url, "structured_data metadata", result, log)
    else:
        # Fall back to well-known path pattern (may not exist — that's fine)
        result.sd_blob_deleted = _delete_blob_by_path(
            _METADATA_CONTAINER, f"{record_id}/structured_data.json",
            "structured_data metadata (path)", result, log
        )

    # ── STEP 4: Delete from Azure AI Search ──────────────────────────────
    result.search_deleted = _delete_from_search(record_id, result, log)

    # ── STEP 5: Delete Table entity (ALWAYS LAST) ─────────────────────────
    result.table_deleted = _delete_table_entity(
        entity["PartitionKey"], entity["RowKey"], result, log
    )

    # ── Determine overall success ─────────────────────────────────────────
    # Success = Table record gone (primary source of truth).
    # Blob/Search failures are non-fatal warnings.
    result.success = result.table_deleted

    if result.success:
        log.info("CASCADE DELETE completed successfully")
    else:
        log.error("CASCADE DELETE completed with failures | errors=%s", result.errors)

    return result


# ---------------------------------------------------------------------------
# Step implementations — each isolated, each idempotent
# ---------------------------------------------------------------------------

def _fetch_entity(record_id: str, result: DeletionResult, log) -> Optional[dict]:
    """
    Fetch the Table Storage entity by RowKey.
    Returns the entity dict, or None if not found (sets result.found = False).
    """
    try:
        conn_str   = require_env("AZURE_STORAGE_CONNECTION_STRING")
        svc        = TableServiceClient.from_connection_string(conn_str)
        table      = svc.get_table_client(_TABLE_NAME)
        entity     = table.get_entity(partition_key=_PARTITION_KEY, row_key=record_id)
        log.info("Entity found in Table Storage")
        return dict(entity)
    except ResourceNotFoundError:
        log.info("Entity not found in Table Storage — treating as already deleted")
        result.found = False
        return None
    except Exception as exc:
        # Unexpected error reading metadata — abort to avoid partial deletes
        # without knowing what to clean up
        msg = f"Failed to read Table entity: {exc}"
        log.error(msg)
        result.errors.append(msg)
        result.found = False   # conservative: don't proceed blind
        return None


def _delete_blob_by_url(url: str, label: str, result: DeletionResult, log) -> bool:
    """
    Delete a blob identified by its full URL.
    Returns True if deleted or already gone (idempotent).
    """
    try:
        conn_str    = require_env("AZURE_STORAGE_CONNECTION_STRING")
        blob_svc    = BlobServiceClient.from_connection_string(conn_str)
        blob_client = BlobClient.from_blob_url(
            blob_url   = url,
            credential = blob_svc.credential,
        )
        blob_client.delete_blob(delete_snapshots="include")
        log.info("Deleted %s blob | url=%s", label, url[:80])
        return True
    except ResourceNotFoundError:
        log.info("%s blob already gone (idempotent) | url=%s", label, url[:80])
        return True   # already deleted — that's fine
    except Exception as exc:
        msg = f"Failed to delete {label} blob (url={url[:80]}): {exc}"
        log.error(msg)
        result.errors.append(msg)
        return False


def _delete_blob_by_path(
    container: str, blob_name: str, label: str,
    result: DeletionResult, log
) -> bool:
    """
    Delete a blob identified by container + blob name.
    Returns True if deleted or already gone (idempotent).
    """
    try:
        conn_str    = require_env("AZURE_STORAGE_CONNECTION_STRING")
        blob_svc    = BlobServiceClient.from_connection_string(conn_str)
        blob_client = blob_svc.get_blob_client(container=container, blob=blob_name)
        blob_client.delete_blob(delete_snapshots="include")
        log.info("Deleted %s blob | container=%s blob=%s", label, container, blob_name)
        return True
    except ResourceNotFoundError:
        log.info("%s blob not found (idempotent) | container=%s blob=%s",
                 label, container, blob_name)
        return True   # already gone
    except Exception as exc:
        msg = f"Failed to delete {label} blob (container={container}, blob={blob_name}): {exc}"
        log.error(msg)
        result.errors.append(msg)
        return False


def _delete_from_search(record_id: str, result: DeletionResult, log) -> bool:
    """
    Remove the document from Azure AI Search index.
    Returns True if removed or not present (idempotent).
    """
    try:
        endpoint   = require_env("AZURE_SEARCH_ENDPOINT").rstrip("/")
        api_key    = require_env("AZURE_SEARCH_KEY")
        url        = f"{endpoint}/indexes/{_SEARCH_INDEX}/docs/index?api-version={_SEARCH_API_VERSION}"
        headers    = {"Content-Type": "application/json", "api-key": api_key}
        body       = {
            "value": [{
                "@search.action": "delete",
                "id":             record_id,
            }]
        }

        resp = requests.post(url, headers=headers, json=body, timeout=15)

        # 200/207 = success; 404 = index doesn't exist (treat as already gone)
        if resp.status_code in (200, 207):
            log.info("Deleted from AI Search index | id=%s", record_id)
            return True
        if resp.status_code == 404:
            log.info("AI Search index not found — treating as already deleted")
            return True

        msg = f"AI Search delete returned unexpected status {resp.status_code}: {resp.text[:200]}"
        log.error(msg)
        result.errors.append(msg)
        return False

    except Exception as exc:
        msg = f"Failed to delete from AI Search: {exc}"
        log.error(msg)
        result.errors.append(msg)
        return False


def _delete_table_entity(
    partition_key: str, row_key: str,
    result: DeletionResult, log
) -> bool:
    """
    Delete the entity from Table Storage.
    This is the FINAL step — only called after all blob/search cleanup.
    Returns True if deleted or already gone (idempotent).
    """
    try:
        conn_str = require_env("AZURE_STORAGE_CONNECTION_STRING")
        svc      = TableServiceClient.from_connection_string(conn_str)
        table    = svc.get_table_client(_TABLE_NAME)
        table.delete_entity(partition_key=partition_key, row_key=row_key)
        log.info("Deleted Table entity | PartitionKey=%s RowKey=%s", partition_key, row_key)
        return True
    except ResourceNotFoundError:
        log.info("Table entity already gone (idempotent) | RowKey=%s", row_key)
        return True   # already deleted
    except Exception as exc:
        msg = f"Failed to delete Table entity (RowKey={row_key}): {exc}"
        log.error(msg)
        result.errors.append(msg)
        return False


# ---------------------------------------------------------------------------
# Structured logger helper — injects record_id + correlation_id into every log
# ---------------------------------------------------------------------------

class _ContextLogger:
    """Thin wrapper that prefixes every log message with [DELETE id=... cid=...]."""

    def __init__(self, record_id: str, correlation_id: str):
        self._prefix = f"[DELETE id={record_id} cid={correlation_id[:8]}]"

    def info(self, msg: str, *args):
        logging.info(f"{self._prefix} {msg}", *args)

    def warning(self, msg: str, *args):
        logging.warning(f"{self._prefix} {msg}", *args)

    def error(self, msg: str, *args):
        logging.error(f"{self._prefix} {msg}", *args)


def _logger(record_id: str, correlation_id: str) -> _ContextLogger:
    return _ContextLogger(record_id, correlation_id)
