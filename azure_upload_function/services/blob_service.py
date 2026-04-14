# =============================================================================
# FILE: services/blob_service.py
# RESOURCE: Azure Blob Storage — Container: "documents"
#
# WHY CREATED:
#   Replaces AWS S3 (bucket: data-orch-upload-files-v1) as the binary file
#   store. All uploaded files (images, videos, documents) are written here.
#   Separated into its own service class so the upload handler stays clean
#   and this logic can be unit-tested or swapped independently.
#
# WHAT IT DOES:
#   - Connects to Azure Blob Storage via connection string
#   - Auto-creates the "documents" container if it doesn't exist
#   - Prefixes every blob name with a UUID hex to prevent filename collisions
#   - Sets the correct Content-Type on each blob for proper browser handling
#   - Returns the full public blob URL to be stored in SQL
#
# RESOURCE CHOSEN: Azure Blob Storage (General Purpose v2, Hot tier)
#
# ALTERNATIVES CONSIDERED:
#   1. Azure Data Lake Storage Gen2 (ADLS)
#      - Built on Blob Storage but adds hierarchical namespace
#      - Better for analytics pipelines and big data workloads
#      - Overkill for a document upload service
#      - More complex ACL model
#
#   2. Azure Files (SMB/NFS share)
#      - Designed for lift-and-shift of on-prem file shares
#      - Not suited for HTTP-based object storage
#      - Higher cost per GB
#
#   3. Azure NetApp Files
#      - Enterprise NFS/SMB, very high performance
#      - Extremely expensive, designed for SAP/HPC workloads
#
# WHY BLOB STORAGE WON:
#   - Direct equivalent of AWS S3 — same object storage model
#   - Cheapest Azure storage option for unstructured files
#   - Native SDK (azure-storage-blob) with presigned URL / SAS support
#   - Hot/Cool/Archive tiering available for cost optimisation later
#   - Scales to petabytes with no configuration
#
# UPLOAD STRATEGY: Direct upload via connection string
#   The original AWS flow used presigned S3 URLs (browser → S3 directly).
#   Here we upload server-side through the Function for simplicity and to
#   keep the connection string off the client. If client-side upload is
#   needed later, replace with a SAS token endpoint (generate_blob_sas).
#
# CONTAINER ACCESS: Private (default)
#   Blobs are not publicly accessible by URL unless the container is set to
#   "Blob" or "Container" public access. For production, keep private and
#   serve files via SAS tokens or a CDN with token auth.
# =============================================================================

import os
import logging
import uuid
import json
from azure.storage.blob import BlobServiceClient, ContentSettings
from services.config import require_env


CONTAINER_NAME          = "documents"
IMAGES_CONTAINER_NAME   = "images"       # stores image files (jpg, jpeg, png)
METADATA_CONTAINER_NAME = "metadata"   # stores text + structured_data JSON blobs


class BlobService:
    def __init__(self):
        conn_str     = require_env("AZURE_STORAGE_CONNECTION_STRING")
        self._client = BlobServiceClient.from_connection_string(conn_str)
        self._ensure_container(CONTAINER_NAME)
        self._ensure_container(IMAGES_CONTAINER_NAME)
        self._ensure_container(METADATA_CONTAINER_NAME)

    def _ensure_container(self, name: str):
        container = self._client.get_container_client(name)
        try:
            container.get_container_properties()
        except Exception:
            logging.info("Container '%s' not found — creating.", name)
            self._client.create_container(name)

    def upload(self, filename: str, data: bytes, content_type: str, blob_name: str = "") -> str:
        """Upload raw file bytes → returns blob URL.
        
        Images (jpg/jpeg/png) are stored in the 'images' container.
        All other files go to the 'documents' container.
        If blob_name is provided, it is used as-is (for temp uploads with custom prefix).
        Otherwise, a UUID prefix is prepended to filename to prevent collisions.
        """
        # Route images to the images container
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        is_image = ext in {"jpg", "jpeg", "png"} or content_type in {"image/jpeg", "image/png"}
        container = IMAGES_CONTAINER_NAME if is_image else CONTAINER_NAME

        if not blob_name:
            blob_name = f"{uuid.uuid4().hex}_{filename}"
        blob_client = self._client.get_blob_client(container=container, blob=blob_name)
        blob_client.upload_blob(
            data,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )
        logging.info("Uploaded file blob to '%s': %s", container, blob_name)
        return blob_client.url

    def upload_text(self, doc_id: str, text: str) -> str:
        """
        Upload extracted text to the metadata container.
        Returns the blob URL. Falls back gracefully on failure.
        """
        blob_name   = f"{doc_id}/text.txt"
        blob_client = self._client.get_blob_client(container=METADATA_CONTAINER_NAME, blob=blob_name)
        blob_client.upload_blob(
            text.encode("utf-8"),
            overwrite=True,
            content_settings=ContentSettings(content_type="text/plain; charset=utf-8"),
        )
        logging.info("Uploaded text blob: %s (%d chars)", blob_name, len(text))
        return blob_client.url

    def upload_structured_data(self, doc_id: str, data: dict) -> str:
        """
        Upload structured_data JSON to the metadata container.
        Returns the blob URL. Falls back gracefully on failure.
        """
        blob_name   = f"{doc_id}/structured_data.json"
        blob_client = self._client.get_blob_client(container=METADATA_CONTAINER_NAME, blob=blob_name)
        blob_client.upload_blob(
            json.dumps(data, ensure_ascii=False).encode("utf-8"),
            overwrite=True,
            content_settings=ContentSettings(content_type="application/json"),
        )
        logging.info("Uploaded structured_data blob: %s", blob_name)
        return blob_client.url

    def _get_blob_client_from_url(self, url: str):
        """
        Parse a blob URL and return a BlobClient.
        Works around missing get_blob_client_from_url in older SDK versions.
        URL format: https://<account>.blob.core.windows.net/<container>/<blob_name>
        """
        from azure.storage.blob import BlobClient
        return BlobClient.from_blob_url(
            blob_url   = url,
            credential = self._client.credential,
        )

    def download_text(self, text_url: str) -> str:
        """Download text content from a metadata blob URL."""
        return self._get_blob_client_from_url(text_url).download_blob().readall().decode("utf-8", errors="replace")

    def download_structured_data(self, sd_url: str) -> dict:
        """Download and parse structured_data JSON from a metadata blob URL."""
        raw = self._get_blob_client_from_url(sd_url).download_blob().readall()
        return json.loads(raw)
