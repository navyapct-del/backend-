import os
import logging
from openai import AzureOpenAI

_TEXT_LIMIT_PER_DOC = 1000


class RAGService:
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
            api_version    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
            azure_endpoint = endpoint,
        )

    def generate_answer(self, query: str, documents: list[dict]) -> str:
        """
        Build context from retrieved documents and generate a grounded answer.

        Args:
            query     : User question.
            documents : List of search result dicts (summary, extracted_text, blob_url).

        Returns:
            Answer string grounded strictly in the provided context.
        """
        if not documents:
            logging.warning("generate_answer: no documents provided — returning fallback.")
            return "Not enough information in the available documents to answer this question."

        # Build context: summary + first 1000 chars of extracted_text per doc
        context_parts = []
        for i, doc in enumerate(documents, start=1):
            summary  = doc.get("summary", "").strip()
            text     = doc.get("extracted_text", "")[:_TEXT_LIMIT_PER_DOC].strip()
            filename = doc.get("filename", f"Document {i}")
            context_parts.append(
                f"[Document {i}: {filename}]\n"
                f"Summary: {summary}\n"
                f"Text: {text}"
            )

        context = "\n\n".join(context_parts)

        prompt = (
            "You are an AI assistant. Answer the question using ONLY the context below.\n"
            "If the answer is not in the context, say: 'Not enough information.'\n\n"
            "FORMAT RULES:\n"
            "- Use numbered bullet points\n"
            "- Each point MUST be on its own line\n"
            "- Keep each point concise (1-2 lines max)\n"
            "- Do NOT combine multiple points on one line\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        try:
            logging.info("generate_answer: calling Azure OpenAI (deployment=%s).", self._deployment)
            response = self._client.chat.completions.create(
                model       = self._deployment,
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0.2,
                max_tokens  = 512,
            )
            answer = response.choices[0].message.content.strip()

            # Post-process: ensure numbered points each start on a new line
            # Handles cases where the model still returns "1. A 2. B 3. C"
            import re
            answer = re.sub(r'\s+(\d+\.)\s+', r'\n\1 ', answer).strip()

            logging.info("generate_answer: success (%d chars).", len(answer))
            return answer

        except Exception:
            logging.exception("generate_answer: Azure OpenAI call failed.")
            raise
