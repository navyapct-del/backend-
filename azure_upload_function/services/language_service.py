import os
import logging
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# Truncate input to 2000 chars — single API call, no chunking loop needed.
# Key phrases from the opening text are the most representative anyway.
_AI_TEXT_LIMIT = 2000


class LanguageService:
    def __init__(self):
        endpoint     = os.environ["LANGUAGE_ENDPOINT"]
        key          = os.environ["LANGUAGE_KEY"]
        self._client = TextAnalyticsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key),
        )

    def extract_key_phrases(self, text: str) -> list[str]:
        """
        Extract key phrases from the first 2000 characters of text.
        Single API call — no chunking, no loop.

        Returns:
            Deduplicated list of key phrase strings.
        """
        if not text:
            return []

        truncated = text[:_AI_TEXT_LIMIT]

        try:
            response = self._client.extract_key_phrases(documents=[truncated])
            doc      = response[0]

            if doc.is_error:
                logging.warning(
                    "LanguageService error: %s — %s", doc.error.code, doc.error.message
                )
                return []

            # Deduplicate while preserving order
            seen, unique = set(), []
            for phrase in doc.key_phrases:
                lower = phrase.lower()
                if lower not in seen:
                    seen.add(lower)
                    unique.append(phrase)

            logging.info("LanguageService: extracted %d key phrases.", len(unique))
            return unique

        except Exception:
            logging.exception("LanguageService.extract_key_phrases failed.")
            raise
