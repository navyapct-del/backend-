"""
Dynamic Query Engine — fully LLM-driven, zero hardcoded column logic.

Flow:
  1. generate_plan(query, columns)  → LLM returns a structured JSON plan
  2. execute_plan(df, plan)         → pandas executes the plan dynamically
  3. build_chart_config(result, plan) → derives chart axes from actual result columns

Works for ANY dataset — no assumptions about column names or data shape.
"""

import re
import json
import logging
import pandas as pd


# ---------------------------------------------------------------------------
# Step 1 — LLM Query Planner
# ---------------------------------------------------------------------------

_PLAN_SCHEMA = """{
  "operation": "filter | count | groupby | aggregate | list | select",
  "select": ["col1", "col2"],
  "distinct": false,
  "filters": [
    { "column": "<exact col name>", "operator": "= | != | > | < | >= | <= | contains | isnull | notnull", "value": "<value or null>" }
  ],
  "group_by": ["<col>"],
  "aggregations": [
    { "type": "count | sum | avg | min | max | nunique", "column": "<col or *>" }
  ],
  "derived_columns": [
    { "name": "<new col name>", "source_column": "<existing col>", "conditions": [
        { "operator": "isnull | notnull | = | contains", "value": "<val or null>", "label": "<category label>" }
    ], "default": "<label when no condition matches>" }
  ],
  "pivot": {
    "index": "<column for rows>",
    "columns": "<column whose values become series>",
    "values": "<column to aggregate, or * for count>"
  },
  "order_by": { "column": "<col>", "ascending": true },
  "limit": null,
  "chart": null
}"""

_CHART_SCHEMA = """{
  "type": "bar | line | pie",
  "x_col": "<column for x-axis>",
  "y_cols": ["<column(s) for y-axis>"],
  "pivot_col": "<column to pivot into series, or null>"
}"""

# Semantic intent → operator mapping injected into the LLM prompt.
# Key = phrase that may appear in the query (lowercase).
# Value = (operator, value, hint shown to LLM)
_SEMANTIC_HINTS = [
    # NULL / missing payment
    (["not paid", "unpaid", "missing payment", "no payment", "haven't paid",
      "did not pay", "no fee", "missing fee", "fee not paid"],
     "isnull", None,
     "words like 'not paid', 'unpaid', 'missing' mean the value is NULL — use operator 'isnull'"),

    # NOT NULL / has paid
    (["has paid", "already paid", "completed payment", "fee paid", "payment done",
      "paid their fee", "paid the fee"],
     "notnull", None,
     "words like 'paid', 'completed payment' mean the value is NOT NULL — use operator 'notnull'"),

    # Explicit zero
    (["fee is 0", "fee = 0", "zero fee", "paid 0", "amount 0", "value 0"],
     "=", "0",
     "explicit zero value — use operator '=' with value 0"),
]


def _detect_semantic_hint(query: str) -> str:
    """
    Scan the query for semantic intent keywords and return an extra
    instruction line to inject into the LLM prompt.
    Returns empty string if no hint matches.
    """
    q = query.lower()
    for phrases, operator, value, hint in _SEMANTIC_HINTS:
        if any(p in q for p in phrases):
            val_str = f"value: {value}" if value is not None else "value: null"
            return (
                f"\nSEMANTIC RULE (MUST FOLLOW): {hint}. "
                f"Use operator \"{operator}\", {val_str} for the relevant column.\n"
            )
    return ""


def _detect_intent_hint(query: str, columns: list[str]) -> str:
    """
    Detect list/distinct/aggregation intent and inject a targeted hint.
    Maps natural language phrases to specific plan operations.
    Also resolves column name mentions in the query to actual column names.
    """
    q = query.lower()

    # Build a case-insensitive map: lowercase word → real column name
    col_map = {c.lower(): c for c in columns}

    def _find_col(words: list[str]) -> str | None:
        """Find the first word that matches a real column name."""
        for w in words:
            if w in col_map:
                return col_map[w]
        return None

    words = re.findall(r"[a-zA-Z]+", q)

    # ── LIST / DISTINCT intent ────────────────────────────────────────────
    list_triggers = ["list all", "show all", "all unique", "unique", "distinct",
                     "list unique", "what are the", "show unique"]
    if any(t in q for t in list_triggers):
        target_col = _find_col(words)
        if target_col:
            return (
                f'\nINTENT RULE (MUST FOLLOW): Query asks for unique/distinct values. '
                f'Use operation "select" with "distinct": true on column "{target_col}". '
                f'Set "select": ["{target_col}"] and "distinct": true. '
                f'Do NOT use SELECT * or groupby.\n'
            )
        return (
            '\nINTENT RULE (MUST FOLLOW): Query asks to list/show items. '
            'Use operation "select". Do NOT use SELECT *. '
            'Identify the relevant column from the dataset columns and set "select" to that column.\n'
        )

    # ── AVERAGE intent ────────────────────────────────────────────────────
    avg_triggers = ["average", "avg", "mean"]
    if any(t in q for t in avg_triggers):
        target_col = _find_col(words)
        group_col  = None
        if "by" in words:
            by_idx = words.index("by")
            group_col = _find_col(words[by_idx + 1:by_idx + 4])
        hint = '\nINTENT RULE (MUST FOLLOW): Query asks for an average. '
        hint += f'Use aggregation type "avg" on column "{target_col}". ' if target_col else 'Use aggregation type "avg". '
        if group_col:
            hint += f'Group by "{group_col}". Use operation "groupby".'
        else:
            hint += 'Use operation "aggregate".'
        return hint + "\n"

    # ── SUM intent ────────────────────────────────────────────────────────
    sum_triggers = ["total", "sum of", "sum"]
    if any(t in q for t in sum_triggers):
        target_col = _find_col(words)
        hint = '\nINTENT RULE (MUST FOLLOW): Query asks for a sum/total. '
        hint += f'Use aggregation type "sum" on column "{target_col}". ' if target_col else 'Use aggregation type "sum". '
        return hint + "\n"

    return ""


def generate_plan(query: str, columns: list[str]) -> dict:
    """
    Ask the LLM to produce a structured execution plan for the given query
    against a dataset with the provided column names.

    Returns a validated plan dict, or raises ValueError on failure.
    """
    from services.openai_service import _get_client, _deployment

    col_list      = ", ".join(f'"{c}"' for c in columns)
    semantic_hint = _detect_semantic_hint(query)
    intent_hint   = _detect_intent_hint(query, columns)

    prompt = f"""You are a data query planner. Given a user question and a list of dataset columns, produce a JSON execution plan.

Dataset columns: [{col_list}]

User question: "{query}"
{semantic_hint}{intent_hint}
Rules:
- Use ONLY column names from the list above — do NOT invent columns
- Match column names case-insensitively: "department" → use the real column name from the list
- "operation" must be one of: filter, count, groupby, aggregate, list, select
- "filters" operator must be one of: =, !=, >, <, >=, <=, contains, isnull, notnull
- For count/nunique of rows, use column "*"
- If the query asks for a chart/graph/plot, set "chart" with type, x_col, y_cols, pivot_col
- "select" lists columns to return; leave empty [] to return all
- Return ONLY valid JSON matching this schema — no explanation, no markdown

SELECT * PREVENTION (CRITICAL):
- NEVER generate "select": [] or SELECT * unless the query explicitly says "show everything" or "all columns"
- "list all X" → select only column X with distinct:true, NOT SELECT *
- "show all X" → select only column X, NOT SELECT *
- Always identify the specific column(s) the user is asking about

DISTINCT / LIST RULES:
- "list all departments", "unique courses", "what departments exist" → operation:"select", select:["<col>"], distinct:true
- Do NOT use groupby for list/distinct queries

AGGREGATION INTENT DETECTION:
- "count", "how many", "number of", "distribution", "breakdown" → COUNT(*)
- "total", "sum of", "sum" → SUM(column)
- "average", "avg", "mean" → type:"avg" aggregation on the relevant column
- "chart", "graph", "plot" + category column → groupby + COUNT(*) + chart
- NEVER use operator "=" with value null — always use "isnull" or "notnull"

CATEGORICAL COMPARISON (A vs B, split by category):
- Queries like "paid vs unpaid per course", "X vs Y by Z", "split by status" need derived_columns + pivot
- Use "derived_columns" to create a new categorical column from an existing one:
  Example for "paid vs unpaid":
  {{ "name": "payment_status", "source_column": "<fee column>",
     "conditions": [{{"operator": "notnull", "value": null, "label": "Paid"}},
                    {{"operator": "isnull",  "value": null, "label": "Unpaid"}}],
     "default": "Unknown" }}
- Then use "pivot" to spread that category into columns:
  {{ "index": "<group column e.g. Course>", "columns": "payment_status", "values": "*" }}
- Set chart with x_col = pivot index, y_cols = expected category labels (e.g. ["Paid","Unpaid"])
- "derived_columns" source_column MUST be a real column from the dataset columns list
- "pivot.index" and "pivot.columns" MUST be real or derived column names

Column selection for payment/fee queries:
- Identify the column whose name contains words like "fee", "amount", "payment", "paid", "cost"
- Apply the filter on THAT column — do not hardcode a column name

{_PLAN_SCHEMA}

If chart is needed, "chart" must match:
{_CHART_SCHEMA}

JSON plan:"""

    client     = _get_client()
    deployment = _deployment()

    try:
        resp = client.chat.completions.create(
            model       = deployment,
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.0,
            max_tokens  = 600,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$",           "", raw).strip()

        plan = json.loads(raw)
        _validate_plan(plan, columns)
        plan = _fix_groupby_plan(plan)
        plan = _enforce_null_semantics(query, plan, columns)

        logging.info("generate_plan: operation=%s filters=%d group_by=%s chart=%s",
                     plan.get("operation"), len(plan.get("filters", [])),
                     plan.get("group_by"), plan.get("chart") is not None)
        return plan

    except json.JSONDecodeError as exc:
        logging.error("generate_plan: LLM returned invalid JSON: %s", exc)
        raise ValueError(f"LLM returned invalid JSON: {exc}")
    except Exception as exc:
        logging.error("generate_plan failed: %s", exc)
        raise
def _fix_groupby_plan(plan: dict) -> dict:
    """
    Post-process the LLM plan to enforce correct GROUP BY structure.

    Fixes:
    1. GROUP BY with no aggregations → inject COUNT(*)
    2. SELECT contains '*' alongside GROUP BY → clear select (output comes from groupby+agg)
    3. chart.y_cols contains '*' or is empty → set to ["count"]
    4. GROUP BY with SELECT * → clear select
    """
    group_by = plan.get("group_by", [])
    if not group_by:
        return plan   # not a groupby query — nothing to fix

    # Fix 1: ensure at least one aggregation
    aggs = plan.get("aggregations", [])
    if not aggs:
        plan["aggregations"] = [{"type": "count", "column": "*"}]
        logging.info("_fix_groupby_plan: injected default COUNT(*) aggregation")

    # Fix 2: clear select when group_by is present — avoids SELECT Course, *
    select = plan.get("select", [])
    if "*" in select or set(select) - set(group_by):
        plan["select"] = []
        logging.info("_fix_groupby_plan: cleared select (was %s) — output from groupby+agg", select)

    # Fix 3: fix chart y_cols
    chart = plan.get("chart")
    if chart:
        y_cols = chart.get("y_cols", [])
        # Remove '*' and empty strings from y_cols
        clean_y = [c for c in y_cols if c and c != "*"]
        if not clean_y:
            agg_type = plan["aggregations"][0].get("type", "count")
            agg_col  = plan["aggregations"][0].get("column", "*")
            clean_y  = [_agg_alias(agg_type, agg_col)]
            logging.info("_fix_groupby_plan: set chart.y_cols to %s", clean_y)
        chart["y_cols"] = clean_y

        # Fix x_col if missing
        if not chart.get("x_col") and group_by:
            chart["x_col"] = group_by[0]
            logging.info("_fix_groupby_plan: set chart.x_col to '%s'", group_by[0])

    return plan


def _enforce_null_semantics(query: str, plan: dict, columns: list[str]) -> dict:
    """
    Post-process the LLM plan to enforce correct NULL semantics.
    Dynamically identifies payment/fee columns by name pattern.

    If the query means "not paid" but LLM generated operator "=" with value 0 or null,
    replace with "isnull". Vice versa for "paid" queries.
    """
    q = query.lower()

    # Detect which semantic intent applies
    is_null_intent   = any(p in q for p in
        ["not paid", "unpaid", "missing payment", "no payment",
         "haven't paid", "did not pay", "no fee", "missing fee", "fee not paid"])
    is_notnull_intent = any(p in q for p in
        ["has paid", "already paid", "completed payment",
         "paid their fee", "paid the fee", "payment done"])

    if not is_null_intent and not is_notnull_intent:
        return plan   # no semantic correction needed

    # Identify payment-related columns dynamically by name pattern
    payment_keywords = {"fee", "amount", "payment", "paid", "cost", "price", "charge"}
    col_lower_map    = {c.lower(): c for c in columns}
    payment_cols     = {
        original for lower, original in col_lower_map.items()
        if any(kw in lower for kw in payment_keywords)
    }

    if not payment_cols:
        return plan   # no payment column found — can't correct

    correct_op = "isnull" if is_null_intent else "notnull"

    for f in plan.get("filters", []):
        col = f.get("column", "")
        # Check if this filter targets a payment column
        if col in payment_cols or col.lower() in {c.lower() for c in payment_cols}:
            old_op = f.get("operator")
            # Fix: = 0, = null, = "0", = "" → correct operator
            if old_op in ("=", "!=", "==") or old_op != correct_op:
                logging.info(
                    "_enforce_null_semantics: corrected filter on '%s': "
                    "operator '%s' → '%s' (query intent: %s)",
                    col, old_op, correct_op,
                    "isnull" if is_null_intent else "notnull"
                )
                f["operator"] = correct_op
                f["value"]    = None

    return plan


def _validate_plan(plan: dict, columns: list[str]) -> None:
    """
    Remove any column references that don't exist in the actual dataset.
    Raises ValueError if the user's core intent (select/group_by) referenced
    only columns that don't exist — prevents silent SELECT * fallback.
    """
    import difflib

    col_set = {c.lower() for c in columns}

    def _valid(col: str) -> bool:
        return col == "*" or col.lower() in col_set

    # Capture original non-wildcard select before sanitization
    original_select = [c for c in plan.get("select", []) if c != "*"]

    # Sanitise filters
    plan["filters"] = [
        f for f in plan.get("filters", [])
        if _valid(f.get("column", ""))
    ]

    # Sanitise group_by — capture originals first
    original_group_by = list(plan.get("group_by", []))
    plan["group_by"] = [c for c in original_group_by if _valid(c)]

    # Sanitise aggregations
    plan["aggregations"] = [
        a for a in plan.get("aggregations", [])
        if _valid(a.get("column", ""))
    ]

    # Sanitise select — strip '*' when group_by is present (invalid SQL)
    if plan.get("group_by"):
        plan["select"] = [
            c for c in plan.get("select", [])
            if c != "*" and _valid(c)
        ]
    else:
        plan["select"] = [
            c for c in plan.get("select", [])
            if _valid(c)
        ]

    # Sanitise derived_columns — source_column must exist
    plan["derived_columns"] = [
        dc for dc in plan.get("derived_columns", [])
        if _valid(dc.get("source_column", ""))
    ]

    # Sanitise pivot columns
    pivot = plan.get("pivot")
    if pivot:
        if pivot.get("index") and not _valid(pivot["index"]):
            pivot["index"] = None
        # pivot.columns may be a derived column name — allow it through
        # pivot.values = "*" is always valid

    # Sanitise chart columns
    chart = plan.get("chart")
    if chart:
        if chart.get("x_col") and not _valid(chart["x_col"]):
            chart["x_col"] = None
        chart["y_cols"] = [c for c in chart.get("y_cols", []) if _valid(c)]
        if chart.get("pivot_col") and not _valid(chart["pivot_col"]):
            chart["pivot_col"] = None

    # ── Post-sanitization emptiness check ────────────────────────────────
    # If the user explicitly requested columns (select or group_by) but ALL
    # of them were stripped as invalid, raise — prevents silent SELECT * fallback.
    select_was_meaningful  = bool(original_select)
    select_now_empty       = not plan.get("select")
    groupby_was_meaningful = bool(original_group_by)
    groupby_now_empty      = not plan.get("group_by")
    aggs_empty             = not plan.get("aggregations")
    filters_empty          = not plan.get("filters")

    if (select_was_meaningful and select_now_empty
            and groupby_now_empty and aggs_empty and filters_empty):
        # All requested columns were invalid — build a helpful error
        invalid_cols = [c for c in original_select if not _valid(c)]
        suggestions  = difflib.get_close_matches(
            invalid_cols[0].lower() if invalid_cols else "",
            [c.lower() for c in columns],
            n=3, cutoff=0.4,
        )
        # Map suggestions back to original casing
        lower_to_orig = {c.lower(): c for c in columns}
        suggestions   = [lower_to_orig[s] for s in suggestions]
        raise ValueError(json.dumps({
            "invalid_columns": invalid_cols,
            "available_columns": columns,
            "suggestions": suggestions,
        }))

    if (groupby_was_meaningful and groupby_now_empty
            and select_now_empty and aggs_empty and filters_empty):
        invalid_cols = [c for c in original_group_by if not _valid(c)]
        suggestions  = difflib.get_close_matches(
            invalid_cols[0].lower() if invalid_cols else "",
            [c.lower() for c in columns],
            n=3, cutoff=0.4,
        )
        lower_to_orig = {c.lower(): c for c in columns}
        suggestions   = [lower_to_orig[s] for s in suggestions]
        raise ValueError(json.dumps({
            "invalid_columns": invalid_cols,
            "available_columns": columns,
            "suggestions": suggestions,
        }))


# ---------------------------------------------------------------------------
# Step 2 — Pandas Execution Engine
# ---------------------------------------------------------------------------

def execute_plan(df: pd.DataFrame, plan: dict) -> dict:
    """
    Execute a query plan against a DataFrame.
    Returns:
      {
        "type":         "text" | "table" | "chart",
        "answer":       str,
        "columns":      [...],
        "rows":         [...],
        "chart_config": {...} | None,
        "script":       str,
      }
    """
    try:
        operation = plan.get("operation", "select")
        result_df = df.copy()

        # ── 1. Apply filters ──────────────────────────────────────────────
        for f in plan.get("filters", []):
            col = _resolve_col(result_df, f["column"])
            if col is None:
                logging.warning("execute_plan: filter column '%s' not found — skipping", f["column"])
                continue
            op  = f.get("operator", "=")
            val = f.get("value")
            result_df = _apply_filter(result_df, col, op, val)

        # ── 1b. Apply derived columns (categorical splits) ────────────────
        derived = plan.get("derived_columns", [])
        if derived:
            result_df = _apply_derived_columns(result_df, derived)

        # ── 1c. Apply pivot (A vs B comparisons) ─────────────────────────
        pivot_spec = plan.get("pivot")
        if pivot_spec:
            result_df, pivot_series = _apply_pivot(result_df, pivot_spec)
            resp_type    = "chart" if plan.get("chart") else "table"
            chart_config = None
            chart_plan   = plan.get("chart")
            if chart_plan:
                # Use actual pivot series as y-axis — overrides LLM y_cols
                x_col = _resolve_col(result_df, pivot_spec.get("index", "")) or result_df.columns[0]
                chart_config = {
                    "type":     chart_plan.get("type", "bar"),
                    "xKey":     x_col,
                    "series":   pivot_series,
                    "pivotCol": None,
                }
            result_df = result_df.where(pd.notnull(result_df), None)
            script    = _build_script(plan)
            rows      = result_df.to_dict(orient="records")
            logging.info("execute_plan: pivot produced %d rows, series=%s", len(rows), pivot_series)
            return {
                "type":         resp_type,
                "answer":       f"Comparison across {len(rows)} groups.",
                "columns":      list(result_df.columns),
                "rows":         rows[:20],
                "chart_config": chart_config,
                "script":       script,
            }

        # ── 2. Apply group_by + aggregations ──────────────────────────────
        group_cols = [
            _resolve_col(result_df, c)
            for c in plan.get("group_by", [])
            if _resolve_col(result_df, c)
        ]

        aggs      = plan.get("aggregations", [])
        operation = plan.get("operation", "select")

        if group_cols and aggs:
            result_df = _apply_groupby(result_df, group_cols, aggs)
            resp_type = "table"

        elif group_cols and not aggs:
            # groupby with no explicit aggregation → count rows per group
            result_df = (
                result_df.groupby(group_cols, as_index=False)
                .size()
                .rename(columns={"size": "count"})
            )
            resp_type = "table"

        elif aggs and not group_cols:
            # Scalar aggregation — returns a single-row summary
            result_df = _apply_scalar_agg(result_df, aggs)
            resp_type = "text"

        else:
            # Guard: if plan is completely empty (no select, group_by, aggs, filters)
            # do NOT fall through to returning the full DataFrame.
            has_intent = (
                plan.get("select")
                or plan.get("group_by")
                or plan.get("aggregations")
                or plan.get("filters")
                or plan.get("pivot")
                or plan.get("derived_columns")
            )
            if not has_intent:
                return {
                    "type":    "error",
                    "answer":  "Query could not be mapped to any column in the dataset.",
                    "columns": [],
                    "rows":    [],
                    "chart_config": None,
                    "script":  "",
                }
            resp_type = "table" if operation in ("list", "select", "filter") else "text"

        # ── 3. Apply select ───────────────────────────────────────────────
        raw_select = plan.get("select", [])

        # Resolve real column names, skip "*" and any unresolvable names
        select_cols = [
            _resolve_col(result_df, c)
            for c in raw_select
            if c != "*" and _resolve_col(result_df, c) is not None
        ]

        logging.info("execute_plan: select_cols=%s", select_cols)

        if select_cols:
            # Only keep columns that actually exist in the current result_df
            valid = [c for c in select_cols if c in result_df.columns]
            if valid:
                result_df = result_df[valid]

        # ── 3b. Apply distinct ────────────────────────────────────────────
        if plan.get("distinct"):
            # Drop nulls from selected columns before deduplication
            cols_to_check = [c for c in result_df.columns if c in result_df.columns]
            result_df = result_df.dropna(subset=cols_to_check, how="all")
            result_df = result_df.drop_duplicates().reset_index(drop=True)
            logging.info("execute_plan: distinct applied → %d unique non-null rows", len(result_df))

        # ── 4. Apply order_by ─────────────────────────────────────────────
        order = plan.get("order_by")
        if order:
            order_col = _resolve_col(result_df, order.get("column", ""))
            if order_col:
                result_df = result_df.sort_values(
                    order_col, ascending=order.get("ascending", True)
                )

        # ── 5. Apply limit ────────────────────────────────────────────────
        limit = plan.get("limit")
        if limit:
            result_df = result_df.head(int(limit))

        # ── 6. Clean NaN / non-serialisable values ────────────────────────
        result_df = result_df.where(pd.notnull(result_df), None)

        columns = list(result_df.columns)
        rows    = result_df.to_dict(orient="records")

        # ── 7. Build answer text ──────────────────────────────────────────
        if resp_type == "text" and len(rows) == 1:
            # Single-row scalar result — format as sentence
            answer = "; ".join(f"{k}: {v}" for k, v in rows[0].items() if v is not None)
        elif resp_type == "text":
            answer = f"{len(rows)} result(s) found."
        else:
            answer = f"{len(rows)} row(s) returned."

        # ── 8. Chart config ───────────────────────────────────────────────
        chart_config = None
        chart_plan   = plan.get("chart")
        if chart_plan:
            resp_type    = "chart"
            chart_config = _build_chart_config(result_df, chart_plan)
            answer       = f"Chart generated from {len(rows)} data points."

        # ── 9. Build SQL-like script for transparency ─────────────────────
        script = _build_script(plan)

        return {
            "type":         resp_type,
            "answer":       answer,
            "columns":      columns,
            "rows":         rows[:20],   # hard cap for response size
            "chart_config": chart_config,
            "script":       script,
        }

    except Exception as exc:
        logging.exception("execute_plan failed.")
        return {
            "type":         "text",
            "answer":       f"Query execution failed: {exc}",
            "columns":      [],
            "rows":         [],
            "chart_config": None,
            "script":       "",
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_derived_columns(df: pd.DataFrame, derived_columns: list[dict]) -> pd.DataFrame:
    """
    Create new categorical columns from existing ones using condition lists.
    Each condition is evaluated in order — first match wins.
    No eval(), no exec() — uses only pandas operations.

    derived_column spec:
      {
        "name": "payment_status",
        "source_column": "Fee Paid",
        "conditions": [
          { "operator": "notnull", "value": null, "label": "Paid" },
          { "operator": "isnull",  "value": null, "label": "Unpaid" }
        ],
        "default": "Unknown"
      }
    """
    for dc in derived_columns:
        name       = dc.get("name", "derived")
        src        = _resolve_col(df, dc.get("source_column", ""))
        conditions = dc.get("conditions", [])
        default    = dc.get("default", "Other")

        if not src:
            logging.warning("_apply_derived_columns: source_column '%s' not found — skipping",
                            dc.get("source_column"))
            continue

        series  = df[src]
        result  = pd.Series([default] * len(df), index=df.index, dtype=str)

        # Apply conditions in reverse so first condition has highest priority
        for cond in reversed(conditions):
            op    = cond.get("operator", "=")
            val   = cond.get("value")
            label = cond.get("label", str(val))

            if op == "isnull":
                mask = series.isna() | (series.astype(str).str.strip().isin(["", "nan", "None"]))
            elif op == "notnull":
                mask = series.notna() & (~series.astype(str).str.strip().isin(["", "nan", "None"]))
            elif op == "contains":
                mask = series.astype(str).str.contains(str(val), case=False, na=False)
            else:
                try:
                    num_val = float(str(val))
                    ops = {"=": series == num_val, "!=": series != num_val,
                           ">": series > num_val,  "<": series < num_val}
                    mask = ops.get(op, pd.Series([False] * len(df), index=df.index))
                except (ValueError, TypeError):
                    mask = series.astype(str).str.lower() == str(val).lower()

            result[mask] = label

        df[name] = result
        logging.info("_apply_derived_columns: created '%s' from '%s' with %d conditions",
                     name, src, len(conditions))

    return df


def _apply_pivot(df: pd.DataFrame, pivot_spec: dict) -> tuple[pd.DataFrame, list[str]]:
    """
    Pivot a DataFrame using the plan's pivot spec.
    Returns (pivoted_df, series_columns).

    pivot_spec:
      { "index": "Course", "columns": "payment_status", "values": "*" }

    values="*" means count rows (size), otherwise sum the values column.
    """
    index_col   = _resolve_col(df, pivot_spec.get("index", ""))
    columns_col = _resolve_col(df, pivot_spec.get("columns", ""))
    values_col  = pivot_spec.get("values", "*")

    if not index_col or not columns_col:
        logging.warning("_apply_pivot: index='%s' or columns='%s' not found",
                        pivot_spec.get("index"), pivot_spec.get("columns"))
        return df, []

    if values_col == "*":
        # Count rows per (index, columns) combination
        pivoted = (
            df.groupby([index_col, columns_col])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
    else:
        val_col = _resolve_col(df, values_col)
        if not val_col:
            logging.warning("_apply_pivot: values column '%s' not found — falling back to count", values_col)
            pivoted = (
                df.groupby([index_col, columns_col])
                .size()
                .unstack(fill_value=0)
                .reset_index()
            )
        else:
            pivoted = (
                df.groupby([index_col, columns_col])[val_col]
                .sum()
                .unstack(fill_value=0)
                .reset_index()
            )

    # Flatten column names (MultiIndex after unstack)
    pivoted.columns = [str(c) for c in pivoted.columns]
    series_cols = [c for c in pivoted.columns if c != index_col]

    logging.info("_apply_pivot: index='%s' columns='%s' → series=%s",
                 index_col, columns_col, series_cols)
    return pivoted, series_cols


def _resolve_col(df: pd.DataFrame, name: str) -> str | None:
    """Case-insensitive column name resolution."""
    if name == "*":
        return "*"
    lc = {c.lower(): c for c in df.columns}
    return lc.get(name.lower())


def _apply_filter(df: pd.DataFrame, col: str, op: str, val) -> pd.DataFrame:
    try:
        series = df[col]
        if op == "isnull":
            return df[series.isna() | (series.astype(str).str.strip() == "")]
        if op == "notnull":
            return df[series.notna() & (series.astype(str).str.strip() != "")]
        if op == "contains":
            return df[series.astype(str).str.contains(str(val), case=False, na=False)]
        # Try numeric comparison first, fall back to string
        try:
            num_val = float(val)
            ops = {"=": series == num_val, "!=": series != num_val,
                   ">": series > num_val,  "<": series < num_val,
                   ">=": series >= num_val, "<=": series <= num_val}
        except (TypeError, ValueError):
            ops = {"=": series.astype(str).str.lower() == str(val).lower(),
                   "!=": series.astype(str).str.lower() != str(val).lower()}
        mask = ops.get(op)
        return df[mask] if mask is not None else df
    except Exception as exc:
        logging.warning("_apply_filter failed col=%s op=%s: %s", col, op, exc)
        return df


def _agg_alias(atype: str, col: str) -> str:
    """
    Generate the column alias for an aggregation — matches _build_script exactly.

    COUNT(*) → "count"
    SUM("Fee Paid") → "sum_fee_paid"
    AVG(Score) → "avg_score"
    NUNIQUE(Name) → "nunique_name"
    """
    if col == "*":
        return atype.lower()   # count, nunique, etc.
    safe_col = col.lower().replace(" ", "_").replace('"', "")
    return f"{atype.lower()}_{safe_col}"


# Map LLM agg type names → valid pandas method/agg names.
# pandas has no .avg() — must use .mean()
_PANDAS_AGG_MAP: dict[str, str] = {
    "avg":     "mean",
    "average": "mean",
    "mean":    "mean",
    "count":   "count",
    "sum":     "sum",
    "min":     "min",
    "max":     "max",
    "nunique": "nunique",
}


def _pandas_agg(atype: str) -> str:
    """Translate LLM agg type to a valid pandas aggregation name."""
    return _PANDAS_AGG_MAP.get(atype.lower(), atype.lower())


def _apply_groupby(df: pd.DataFrame, group_cols: list[str], aggs: list[dict]) -> pd.DataFrame:
    """
    Apply group_by + aggregations to a DataFrame.
    Output column names match SQL aliases produced by _build_script exactly.
    """
    base       = df.groupby(group_cols, as_index=False)
    count_star = any(a.get("column", "*") == "*" for a in aggs)
    col_aggs   = [a for a in aggs if a.get("column", "*") != "*"]

    if count_star and not col_aggs:
        # COUNT(*) only — simplest path
        alias  = _agg_alias("count", "*")
        result = base.size().rename(columns={"size": alias})
        logging.info("_apply_groupby: COUNT(*) AS %s", alias)
        return result

    if col_aggs:
        # Build pandas agg dict: {real_col: [pandas_agg_name, ...]}
        agg_dict: dict[str, list] = {}
        for a in col_aggs:
            col   = _resolve_col(df, a.get("column", "")) or df.columns[0]
            atype = _pandas_agg(a.get("type", "count"))   # avg → mean
            agg_dict.setdefault(col, []).append(atype)

        result = base.agg(agg_dict)

        # Flatten MultiIndex columns pandas produces after .agg()
        if isinstance(result.columns, pd.MultiIndex):
            new_cols = []
            for col, agg in result.columns:
                if col in group_cols:
                    new_cols.append(col)
                else:
                    # Find the original LLM agg type (not the pandas name) for alias
                    orig_atype = next(
                        (a.get("type", agg) for a in col_aggs
                         if _resolve_col(df, a.get("column", "")) == col
                         and _pandas_agg(a.get("type", "count")) == agg),
                        agg
                    )
                    new_cols.append(_agg_alias(orig_atype, col))
            result.columns = new_cols
        else:
            # Single-agg case — rename each non-group column to its alias
            new_cols = []
            for c in result.columns:
                if c in group_cols:
                    new_cols.append(c)
                else:
                    # Find the atype for this column
                    atype = next(
                        (a.get("type", "count") for a in col_aggs
                         if _resolve_col(df, a.get("column", "")) == c),
                        "agg"
                    )
                    new_cols.append(_agg_alias(atype, c))
            result.columns = new_cols

        if count_star:
            alias  = _agg_alias("count", "*")
            counts = base.size().rename(columns={"size": alias})
            result = result.merge(counts[group_cols + [alias]], on=group_cols, how="left")

        logging.info("_apply_groupby: output columns=%s", list(result.columns))
        return result

    # Fallback: COUNT(*)
    alias = _agg_alias("count", "*")
    return base.size().rename(columns={"size": alias})


def _apply_scalar_agg(df: pd.DataFrame, aggs: list[dict]) -> pd.DataFrame:
    """Scalar aggregation — returns a single-row summary with aliased column names."""
    row = {}
    for a in aggs:
        col        = a.get("column", "*")
        atype      = a.get("type", "count")
        alias      = _agg_alias(atype, col)
        pandas_fn  = _pandas_agg(atype)   # avg → mean
        try:
            if col == "*" or atype == "count":
                row[alias] = len(df)
            elif atype == "nunique":
                target     = _resolve_col(df, col)
                row[alias] = df[target].nunique() if target else len(df)
            else:
                target = _resolve_col(df, col)
                if target:
                    fn         = getattr(df[target], pandas_fn, None)
                    row[alias] = fn() if fn else None
        except Exception as exc:
            logging.warning("scalar agg failed %s(%s): %s", atype, col, exc)
            row[alias] = None
    logging.info("_apply_scalar_agg: output columns=%s", list(row.keys()))
    return pd.DataFrame([row])


def _detect_scale_groups(df: pd.DataFrame, series: list[str]) -> dict:
    """
    Group series by scale magnitude.
    If the ratio between the largest and smallest max-value exceeds 10,
    split into left/right axis assignments.

    Returns:
      {
        "dual_axis": bool,
        "left":  ["col1", ...],   # smaller-scale series
        "right": ["col2", ...],   # larger-scale series
      }
    """
    if len(series) < 2:
        return {"dual_axis": False, "left": series, "right": []}

    maxes: dict[str, float] = {}
    for col in series:
        try:
            m = df[col].dropna().abs().max()
            maxes[col] = float(m) if m == m else 0.0   # NaN guard
        except Exception:
            maxes[col] = 0.0

    if not maxes or max(maxes.values()) == 0:
        return {"dual_axis": False, "left": series, "right": []}

    overall_max = max(maxes.values())
    overall_min = min(v for v in maxes.values() if v > 0) if any(v > 0 for v in maxes.values()) else 1

    ratio = overall_max / overall_min
    if ratio <= 10:
        return {"dual_axis": False, "left": series, "right": []}

    # Split: series whose max is within 10x of the smallest go left, rest go right
    threshold = overall_min * 10
    left  = [c for c in series if maxes[c] <= threshold]
    right = [c for c in series if maxes[c] >  threshold]

    # Ensure both sides are non-empty
    if not left:
        left  = [series[0]]
        right = series[1:]
    if not right:
        right = [series[-1]]
        left  = series[:-1]

    logging.info("_detect_scale_groups: ratio=%.1f → dual_axis left=%s right=%s",
                 ratio, left, right)
    return {"dual_axis": True, "left": left, "right": right}


def _build_chart_config(df: pd.DataFrame, chart_plan: dict) -> dict:
    """
    Build chart_config from the actual result DataFrame.
    Series are derived dynamically from all numeric columns except xKey.
    Applies dual-axis detection when series have incompatible scales (ratio > 10).
    """
    chart_type = chart_plan.get("type", "bar")
    x_col      = _resolve_col(df, chart_plan.get("x_col") or "")
    pivot_col  = _resolve_col(df, chart_plan.get("pivot_col") or "")

    # Auto-detect x: prefer first non-numeric column
    if not x_col:
        non_numeric = [c for c in df.columns
                       if df[c].dtype == object or str(df[c].dtype) == "category"]
        x_col = non_numeric[0] if non_numeric else df.columns[0]

    # Derive all numeric columns from the actual DataFrame, excluding xKey
    all_numeric = df.select_dtypes(include="number").columns.tolist()
    auto_series = [c for c in all_numeric if c != x_col]

    # If no numeric columns found, fall back to all non-x columns
    # (handles edge cases where aggregation aliases aren't typed as numeric)
    if not auto_series:
        auto_series = [c for c in df.columns if c != x_col]

    # Validate LLM-specified y_cols against real columns
    requested       = [_resolve_col(df, c) for c in chart_plan.get("y_cols", [])
                       if _resolve_col(df, c)]
    valid_requested = [c for c in requested if c != x_col and c in df.columns]
    series          = valid_requested if valid_requested else auto_series

    # Final guard — never return empty series
    if not series and df.columns.tolist():
        series = [c for c in df.columns if c != x_col] or [df.columns[-1]]

    logging.info("_build_chart_config: x=%s series=%s (from %d numeric cols)",
                 x_col, series, len(all_numeric))

    # ── Scale detection ───────────────────────────────────────────────────
    scale = _detect_scale_groups(df, series)

    if scale["dual_axis"]:
        # Dual-axis: annotate each series with its axis assignment
        series_config = (
            [{"key": c, "axis": "left"}  for c in scale["left"]] +
            [{"key": c, "axis": "right"} for c in scale["right"]]
        )
        return {
            "type":      chart_type,
            "xKey":      x_col,
            "series":    series_config,   # list of {key, axis} objects
            "dualAxis":  True,
            "pivotCol":  pivot_col,
        }

    # Same scale — flat series list (existing behaviour)
    return {
        "type":     chart_type,
        "xKey":     x_col,
        "series":   series,
        "dualAxis": False,
        "pivotCol": pivot_col,
    }


def _quote_col(col: str) -> str:
    """Wrap column name in double quotes if it contains spaces or special chars."""
    if " " in col or any(c in col for c in "()[],-/\\"):
        return f'"{col}"'
    return col


def _filter_to_sql(f: dict) -> str:
    """
    Convert a single filter dict to a valid SQL WHERE clause fragment.

    Operator mapping:
      isnull   → col IS NULL
      notnull  → col IS NOT NULL
      contains → col LIKE '%value%'
      =, !=, >, <, >=, <=  → col op value  (quoted if string, unquoted if numeric)
    """
    col = _quote_col(f.get("column", "col"))
    op  = (f.get("operator") or "=").strip().lower()
    val = f.get("value")

    if op == "isnull":
        return f"{col} IS NULL"

    if op == "notnull":
        return f"{col} IS NOT NULL"

    if op == "contains":
        return f"{col} LIKE '%{val}%'"

    # Numeric or string value
    if val is None:
        # Treat None value with = / != as IS NULL / IS NOT NULL
        return f"{col} IS NULL" if op == "=" else f"{col} IS NOT NULL"

    try:
        float(str(val))
        return f"{col} {op.upper()} {val}"
    except (ValueError, TypeError):
        return f"{col} {op.upper()} '{val}'"


def normalize_sql(sql: str) -> str:
    """
    Sanitise a SQL string to fix common LLM output mistakes.
    Applied as a final pass — catches anything _build_script misses.
    """
    import re

    # isnull / notnull operator words
    sql = re.sub(r"\bisnull\b",   "IS NULL",     sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bnotnull\b",  "IS NOT NULL", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bis_null\b",  "IS NULL",     sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bis_notnull\b", "IS NOT NULL", sql, flags=re.IGNORECASE)

    # = null / = 'None' / = "None" / = '' patterns
    sql = re.sub(r"=\s*['\"]?None['\"]?", "IS NULL",     sql, flags=re.IGNORECASE)
    sql = re.sub(r"=\s*null\b",           "IS NULL",     sql, flags=re.IGNORECASE)
    sql = re.sub(r"!=\s*['\"]?None['\"]?","IS NOT NULL", sql, flags=re.IGNORECASE)
    sql = re.sub(r"!=\s*null\b",          "IS NOT NULL", sql, flags=re.IGNORECASE)

    # Quote unquoted column names that contain spaces
    # Pattern: word boundary, two+ words with spaces before SQL keywords
    def _quote_unquoted(m):
        col = m.group(1)
        if " " in col and not col.startswith('"'):
            return f'"{col}"'
        return col
    sql = re.sub(r'(?<!["\w])([A-Za-z][A-Za-z0-9 ]+[A-Za-z0-9])(?=\s+(?:IS|=|!=|>|<|LIKE|IN)\b)',
                 lambda m: f'"{m.group(1)}"' if " " in m.group(1) and not m.group(0).startswith('"') else m.group(0),
                 sql)

    return sql


def _build_case_expr(dc: dict) -> str:
    """
    Build a SQL CASE expression from a derived_column spec.

    Binary optimisation:
      If there are exactly 2 conditions that are pure opposites
      (one isnull + one notnull, or any pair where second is the logical inverse),
      collapse to:  CASE WHEN <cond> THEN 'A' ELSE 'B' END
      — no redundant ELSE 'Unknown' branch.

    Otherwise emit full WHEN ... WHEN ... ELSE form.
    """
    src        = _quote_col(dc.get("source_column", "col"))
    alias      = dc.get("name", "derived")
    conditions = dc.get("conditions", [])
    default    = dc.get("default", "Other")

    def _when_predicate(op: str, val, label: str) -> tuple[str, str]:
        """Returns (predicate_sql, label)."""
        op = (op or "=").lower()
        if op == "isnull":
            return f"{src} IS NULL", label
        if op == "notnull":
            return f"{src} IS NOT NULL", label
        if op == "contains":
            return f"{src} LIKE '%{val}%'", label
        try:
            float(str(val))
            return f"{src} {op.upper()} {val}", label
        except (ValueError, TypeError):
            return f"{src} {op.upper()} '{val}'", label

    # ── Binary optimisation ───────────────────────────────────────────────
    _BINARY_PAIRS = {("isnull", "notnull"), ("notnull", "isnull")}
    if len(conditions) == 2:
        op0 = (conditions[0].get("operator") or "").lower()
        op1 = (conditions[1].get("operator") or "").lower()
        if (op0, op1) in _BINARY_PAIRS:
            pred, lbl   = _when_predicate(op0, conditions[0].get("value"), conditions[0]["label"])
            else_label  = conditions[1]["label"]
            return f"CASE WHEN {pred} THEN '{lbl}' ELSE '{else_label}' END AS {alias}"

    # ── General form ──────────────────────────────────────────────────────
    when_clauses = []
    for cond in conditions:
        pred, lbl = _when_predicate(
            cond.get("operator", "="), cond.get("value"), cond.get("label", "")
        )
        when_clauses.append(f"WHEN {pred} THEN '{lbl}'")

    when_str = " ".join(when_clauses)
    return f"CASE {when_str} ELSE '{default}' END AS {alias}"


def _build_script(plan: dict) -> str:
    """
    Generate a valid, human-readable SQL script that accurately reflects
    the actual execution plan — including derived columns and pivot.
    """
    frm = "data"

    # ── CASE expressions for derived columns ─────────────────────────────
    case_parts: list[str] = [
        _build_case_expr(dc) for dc in plan.get("derived_columns", [])
    ]

    # ── SELECT clause ─────────────────────────────────────────────────────
    pivot_spec  = plan.get("pivot")
    group_by    = plan.get("group_by", [])
    aggs        = plan.get("aggregations", [])
    select_cols = plan.get("select", [])

    if pivot_spec:
        # Pivot query: SELECT index, derived_col, COUNT(*)
        idx_col  = _quote_col(pivot_spec.get("index", ""))
        cat_col  = pivot_spec.get("columns", "")
        val_col  = pivot_spec.get("values", "*")
        agg_expr = "COUNT(*) AS count" if val_col == "*" else f"SUM({_quote_col(val_col)}) AS total"
        select_parts = [p for p in [idx_col] + case_parts if p]
        if cat_col and cat_col not in [dc.get("name") for dc in plan.get("derived_columns", [])]:
            select_parts.append(_quote_col(cat_col))
        select_parts.append(agg_expr)
        select = ", ".join(select_parts)
        group  = ", ".join(
            [idx_col] + [dc.get("name", "") for dc in plan.get("derived_columns", [])]
        )

    elif group_by:
        # GROUP BY query: SELECT group_cols, CASE..., COUNT(*)
        group_parts = [_quote_col(c) for c in group_by]
        agg_parts   = []
        for a in aggs:
            col   = a.get("column", "*")
            atype = a.get("type", "count").upper()
            col_sql = "*" if col == "*" else _quote_col(col)
            alias   = "count" if col == "*" else f"{atype.lower()}_{col.lower().replace(' ', '_')}"
            agg_parts.append(f"{atype}({col_sql}) AS {alias}")
        if not agg_parts:
            agg_parts = ["COUNT(*) AS count"]
        select = ", ".join(group_parts + case_parts + agg_parts)
        group  = ", ".join(group_parts + [dc.get("name", "") for dc in plan.get("derived_columns", [])])

    elif select_cols:
        select = ", ".join(_quote_col(c) for c in select_cols if c != "*")
        select = select or "*"
        group  = ""

    else:
        select = "*"
        group  = ""

    # ── WHERE clause ──────────────────────────────────────────────────────
    where_parts = [_filter_to_sql(f) for f in plan.get("filters", [])]

    # Auto-add IS NOT NULL for selected columns when distinct is used
    if plan.get("distinct") and plan.get("select"):
        for col in plan["select"]:
            if col != "*":
                null_clause = f"{_quote_col(col)} IS NOT NULL"
                if null_clause not in where_parts:
                    where_parts.append(null_clause)

    where = " AND ".join(where_parts)

    # ── Assemble ──────────────────────────────────────────────────────────
    parts = [f"SELECT {'DISTINCT ' if plan.get('distinct') else ''}{select} FROM {frm}"]
    if where:
        parts.append(f"WHERE {where}")
    if group:
        parts.append(f"GROUP BY {group}")

    order = plan.get("order_by")
    if order and order.get("column"):
        direction = "ASC" if order.get("ascending", True) else "DESC"
        parts.append(f"ORDER BY {_quote_col(order['column'])} {direction}")

    if plan.get("limit"):
        parts.append(f"LIMIT {plan['limit']}")

    raw_sql = " ".join(parts) + ";"

    # Append pivot comment if applicable
    if pivot_spec:
        cat = pivot_spec.get("columns", "")
        raw_sql += f"\n-- Pivot on '{cat}' column to spread categories into series"

    return normalize_sql(raw_sql)


# ---------------------------------------------------------------------------
# Convenience: build flat DataFrame from stored structured_data
# ---------------------------------------------------------------------------

def detect_dual_axis_from_rows(rows: list[dict], x_key: str) -> dict:
    """
    Compute scale-aware chart_config.series from serialised row data.
    Mirrors _detect_scale_groups but works on rows (post-serialisation).

    Returns a dict with keys:
      dual_axis: bool
      series:    list[str] | list[{key, axis}]
    """
    if not rows:
        return {"dual_axis": False, "series": []}

    # Collect numeric series from first row (scan up to 5 for non-null)
    sample     = rows[0]
    candidates = [k for k in sample if k != x_key]
    series     = []
    for key in candidates:
        for row in rows[:5]:
            val = row.get(key)
            if val is None:
                continue
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                series.append(key)
            break

    if len(series) < 2:
        return {"dual_axis": False, "series": series}

    # Compute max absolute value per series
    maxes: dict[str, float] = {}
    for key in series:
        vals = [row.get(key) for row in rows
                if isinstance(row.get(key), (int, float))
                and not isinstance(row.get(key), bool)]
        maxes[key] = max((abs(v) for v in vals), default=0.0)

    positive = [v for v in maxes.values() if v > 0]
    if not positive:
        return {"dual_axis": False, "series": series}

    overall_max = max(positive)
    overall_min = min(positive)

    # Guard division by zero
    ratio = (overall_max / overall_min) if overall_min > 0 else 1

    if ratio <= 10:
        logging.info("detect_dual_axis_from_rows: ratio=%.1f → single axis", ratio)
        return {"dual_axis": False, "series": series}

    # Assign largest-magnitude series to right axis, rest to left
    threshold = overall_min * 10
    left  = [c for c in series if maxes[c] <= threshold]
    right = [c for c in series if maxes[c] >  threshold]

    # Safety: ensure both sides non-empty
    if not left:
        left, right = [series[0]], series[1:]
    if not right:
        left, right = series[:-1], [series[-1]]

    series_config = (
        [{"key": c, "type": "bar",  "axis": "left"}  for c in left] +
        [{"key": c, "type": "line", "axis": "right"} for c in right]
    )
    logging.info("detect_dual_axis_from_rows: ratio=%.1f → composed chart left=%s right=%s",
                 ratio, left, right)
    return {"dual_axis": True, "chart_type": "composed", "series": series_config}


def get_series_from_data(data: list[dict], x_key: str = "") -> list[str]:
    """
    Derive chart series dynamically from serialised row data.
    Returns all keys whose values are numeric (int or float), excluding x_key.
    Scans up to 5 rows per key to handle leading nulls.
    """
    if not data:
        return []

    sample     = data[0]
    candidates = [k for k in sample if k != x_key]

    series = []
    for key in candidates:
        for row in data[:5]:
            val = row.get(key)
            if val is None:
                continue
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                series.append(key)
            break

    logging.info("get_series_from_data: xKey=%s series=%s", x_key, series)
    return series


def structured_to_df(structured: dict) -> pd.DataFrame:
    """
    Convert stored structured_data (with optional 'sheets' key) into a
    single flat DataFrame. Adds a '_sheet' column when merging multiple sheets.
    """
    if not structured:
        return pd.DataFrame()

    sheets = structured.get("sheets", {})
    if sheets:
        frames = []
        for sname, sd in sheets.items():
            df_s = pd.DataFrame(sd.get("rows", []))
            if not df_s.empty:
                df_s["_sheet"] = sname
                frames.append(df_s)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    rows = structured.get("rows", [])
    if rows:
        return pd.DataFrame(rows)

    if isinstance(structured, list):
        return pd.DataFrame(structured)

    return pd.DataFrame()
