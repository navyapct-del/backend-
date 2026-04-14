import os
import logging
from openai import AzureOpenAI

# Truncate input — enough context for a good summary, minimal tokens sent
_SUMMARY_TEXT_LIMIT = 2000


class SummaryService:
    """
    Generates a concise document summary using Azure OpenAI.
    Separate from RAGService so it can run in parallel with keyword extraction.
    Client is instantiated per-call to stay safe in threaded contexts.
    """

    def __init__(self):
        endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key    = os.getenv("AZURE_OPENAI_API_KEY")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

        if not endpoint or not api_key:
            raise EnvironmentError(
                "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set."
            )

        self._deployment = deployment
        self._client     = AzureOpenAI(
            api_key        = api_key,
            api_version    = "2024-02-01",
            azure_endpoint = endpoint,
        )

    def summarize(self, text: str) -> str:
        """
        Generate a 3–4 line summary of the document text.

        Args:
            text: Full extracted text (will be truncated internally).

        Returns:
            Summary string, or first 300 chars as fallback on failure.
        """
        if not text:
            return ""

        truncated = text[:_SUMMARY_TEXT_LIMIT]

        # Minimal prompt — fewer tokens = faster response
        prompt = f"Summarize the following document in 3–4 lines:\n\n{truncated}"

        try:
            response = self._client.chat.completions.create(
                model       = self._deployment,
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0.2,
                max_tokens  = 150,   # summary needs very few tokens
            )
            summary = response.choices[0].message.content.strip()
            logging.info("SummaryService: generated %d char summary.", len(summary))
            return summary

        except Exception:
            logging.exception("SummaryService.summarize failed — using text truncation fallback.")
            return text[:300].strip()
