import os
import logging
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from services.config import require_env


class DocIntelligenceService:
    """
    Wraps Azure Document Intelligence (Form Recognizer) to extract
    full plain text from PDF documents.
    """

    def __init__(self):
        endpoint = require_env("DOC_INTELLIGENCE_ENDPOINT")
        key      = require_env("DOC_INTELLIGENCE_KEY")
        self._client = DocumentAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key),
        )

    def extract_text(self, file_bytes: bytes) -> str:
        """
        Run the prebuilt 'read' model against raw PDF bytes.

        Args:
            file_bytes: Raw bytes of the uploaded document.

        Returns:
            Full extracted text as a single string, pages joined by newline.
        """
        try:
            poller = self._client.begin_analyze_document(
                model_id="prebuilt-read",
                document=file_bytes,
            )
            result = poller.result()

            lines = []
            for page in result.pages:
                for line in page.lines:
                    lines.append(line.content)

            extracted = "\n".join(lines)
            logging.info("Document Intelligence extracted %d lines.", len(lines))
            return extracted

        except Exception:
            logging.exception("DocIntelligenceService.extract_text failed.")
            raise
