import os
import json
import re
import logging
from openai import AzureOpenAI

# How many docs to pull for analytical queries — more docs = richer cross-doc data
ANALYTICS_TOP_K = 8

# Intent keywords
_TABLE_KEYWORDS    = {"compare", "comparison", "difference", "versus", "vs", "breakdown",
                      "year-wise", "yearwise", "state-wise", "statewise", "category-wise"}
_CHART_KEYWORDS    = {"plot", "graph", "chart", "trend", "visualize", "bar", "line",
                      "histogram", "distribution", "growth"}
_TABULAR_KEYWORDS  = _TABLE_KEYWORDS | _CHART_KEYWORDS


def detect_intent(query: str) -> str:
    """
    Classify query into: 'chart' | 'table' | 'text'

    chart  → user wants a visual graph
    table  → user wants structured comparison data
    text   → plain Q&A
    """
    lower = query.lower()
    if any(k in lower for k in _CHART_KEYWORDS):
        return "chart"
    if any(k in lower for k in _TABLE_KEYWORDS):
        return "table"
    return "text"


def is_analytical(query: str) -> bool:
    """Returns True if the query needs structured data extraction."""
    lower = query.lower()
    return any(k in lower for k in _TABULAR_KEYWORDS)


class AnalyticsService:
    """
    Handles multi-document structured data extraction, table/chart generation.
    Keeps RAGService untouched — this is an additive layer.
    """

    def __init__(self):
        endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key    = os.getenv("AZURE_OPENAI_API_KEY")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

        if not endpoint or not api_key:
            raise EnvironmentError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set.")

        self._deployment = deployment
        self._client     = AzureOpenAI(
            api_key        = api_key,
            api_version    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
            azure_endpoint = endpoint,
        )

    # ------------------------------------------------------------------
    # STEP 1 — Extract structured JSON rows from multi-doc text chunks
    # ------------------------------------------------------------------

    def extract_structured_data(self, query: str, documents: list[dict]) -> list[dict]:
        """
        Ask the LLM to extract structured rows from all retrieved document chunks.
        Returns a list of flat dicts (rows), or [] on failure.
        """
        if not documents:
            return []

        # Build a compact context — use more text per doc for analytical queries
        context_parts = []
        for i, doc in enumerate(documents, start=1):
            text = doc.get("extracted_text", "")[:2000].strip()
            context_parts.append(f"[Document {i}: {doc.get('filename', '')}]\n{text}")
        context = "\n\n".join(context_parts)

        prompt = (
            "You are a data extraction assistant.\n"
            "Extract ALL numerical/tabular data relevant to the question from the context below.\n"
            "Return ONLY a valid JSON array of flat objects. Each object must have consistent keys.\n"
            "Example: [{\"year\": 2018, \"state\": \"Bihar\", \"value\": 6239}, ...]\n"
            "If no structured data exists, return an empty array: []\n"
            "Do NOT include any explanation — only the JSON array.\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            "JSON:"
        )

        try:
            response = self._client.chat.completions.create(
                model       = self._deployment,
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0.0,   # deterministic for data extraction
                max_tokens  = 1500,
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            data = json.loads(raw)
            if isinstance(data, list):
                logging.info("extract_structured_data: extracted %d rows.", len(data))
                return data
            return []

        except Exception:
            logging.exception("extract_structured_data failed.")
            return []

    # ------------------------------------------------------------------
    # STEP 2 — Pivot flat rows into chart-friendly format + config
    # ------------------------------------------------------------------

    def generate_chart_config(self, data: list[dict], query: str) -> dict:
        """
        Given flat rows like [{year, state, value}, ...], pivot into:
          [{year: 2018, Bihar: 6239, Maharashtra: 425389}, ...]
        and produce a chart_config for Recharts.

        Returns dict with keys: data (pivoted), chart_config, script
        Falls back to raw data if pivot is not possible.
        """
        if not data:
            return {"data": [], "chart_config": {}, "script": ""}

        keys = list(data[0].keys())

        # Heuristic: find x-axis key (year/date/category) and series key (state/name/label)
        x_key      = _find_key(keys, ["year", "date", "month", "quarter", "category", "period"])
        series_key = _find_key(keys, ["state", "name", "label", "region", "type", "category"])
        value_key  = _find_key(keys, ["value", "total", "amount", "count", "revenue", "tax"])

        # Determine chart type from query
        lower      = query.lower()
        chart_type = "line" if any(k in lower for k in ["trend", "growth", "over time"]) else "bar"

        if x_key and series_key and value_key:
            # Pivot: group by x_key, spread series_key values as columns
            pivoted    = {}
            series_set = set()
            for row in data:
                x   = row.get(x_key)
                s   = str(row.get(series_key, ""))
                v   = row.get(value_key)
                series_set.add(s)
                if x not in pivoted:
                    pivoted[x] = {x_key: x}
                pivoted[x][s] = v

            pivoted_list = sorted(pivoted.values(), key=lambda r: str(r.get(x_key, "")))
            series_list  = sorted(series_set)

            chart_config = {
                "type":   chart_type,
                "xKey":   x_key,
                "series": series_list,
            }
            script = (
                f"SELECT {x_key}, {series_key}, {value_key} "
                f"FROM documents "
                f"WHERE {series_key} IN ({', '.join(repr(s) for s in series_list)}) "
                f"ORDER BY {x_key}, {series_key};"
            )
            return {"data": pivoted_list, "chart_config": chart_config, "script": script}

        # Fallback: return raw data with best-guess config
        chart_config = {
            "type":   chart_type,
            "xKey":   keys[0] if keys else "x",
            "series": keys[1:] if len(keys) > 1 else [],
        }
        return {"data": data, "chart_config": chart_config, "script": ""}

    # ------------------------------------------------------------------
    # STEP 3 — Generate natural language explanation
    # ------------------------------------------------------------------

    def generate_explanation(self, query: str, data: list[dict]) -> str:
        """Short LLM-generated explanation of the structured data."""
        if not data:
            return "No structured data could be extracted from the documents."

        sample = json.dumps(data[:6], indent=2)
        prompt = (
            f"Summarize the following data in 2-3 concise sentences relevant to: '{query}'\n\n"
            f"Data:\n{sample}\n\nSummary:"
        )
        try:
            response = self._client.chat.completions.create(
                model       = self._deployment,
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0.2,
                max_tokens  = 200,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            logging.exception("generate_explanation failed.")
            return ""


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _find_key(keys: list[str], candidates: list[str]) -> str | None:
    """Return the first key from `keys` that matches any candidate (case-insensitive)."""
    lower_keys = {k.lower(): k for k in keys}
    for c in candidates:
        if c in lower_keys:
            return lower_keys[c]
    return None
