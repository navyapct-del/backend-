"""
Centralised environment variable access.
All services import from here — no direct os.environ calls scattered around.
"""
import os
import logging


def require_env(var: str) -> str:
    """Return env var value or raise a clear error if missing."""
    val = os.environ.get(var, "").strip()
    if not val:
        raise EnvironmentError(f"Missing required environment variable: {var}")
    return val


def get_env(var: str, default: str = "") -> str:
    """Return env var value with optional default (non-critical settings)."""
    return os.environ.get(var, default).strip()


def log_config_status() -> None:
    """
    Print current env var status to logs on startup.
    Shows PRESENT/MISSING — never logs actual key values.
    """
    keys = [
        "AZURE_STORAGE_CONNECTION_STRING",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT_NAME",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        "DOC_INTELLIGENCE_ENDPOINT",
        "DOC_INTELLIGENCE_KEY",
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_KEY",
    ]
    for k in keys:
        val = os.environ.get(k, "")
        status = "PRESENT" if val.strip() else "MISSING"
        logging.info("CONFIG  %-45s %s", k, status)
