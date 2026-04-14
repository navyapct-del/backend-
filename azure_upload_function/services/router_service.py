import logging

# Only route to SQL for explicit raw SQL patterns.
# Everything else — including "list", "show", "compare", natural language — goes to RAG.
_SQL_PREFIXES = ("select ", "show table", "show tables")


def route_query(query: str) -> str:
    """
    Route a user query to 'rag' or 'sql'.

    Default is 'rag'. SQL is only triggered for explicit SQL-syntax queries
    that start with SELECT or SHOW TABLE — patterns that cannot be document Q&A.

    Args:
        query: Raw user query string.

    Returns:
        'rag' (default) or 'sql'
    """
    lower = query.lower().strip()
    for prefix in _SQL_PREFIXES:
        if lower.startswith(prefix):
            logging.info("Router: '%s' → sql (matched prefix '%s')", query[:60], prefix)
            return "sql"

    logging.info("Router: '%s' → rag", query[:60])
    return "rag"
