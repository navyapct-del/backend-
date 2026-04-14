"""
Dynamic DataFrame cleaner — works for ANY messy Excel or CSV file.

No hardcoded column names. No file-specific logic.

Public API:
    read_csv_clean(file_bytes, source_label)              -> pd.DataFrame
    read_excel_clean(file_bytes, sheet_name, source_label) -> pd.DataFrame
"""

import io
import logging
import pandas as pd


# ---------------------------------------------------------------------------
# Step 1 — Header row detection
# ---------------------------------------------------------------------------

def detect_header_row(df_raw: pd.DataFrame, scan_rows: int = 20) -> int:
    """
    Scan the first `scan_rows` rows of a raw (header=None) DataFrame.
    Score each row by counting non-null string values with len > 0.
    The row with the highest score is the header row.

    Returns the integer row index (0-based).
    """
    best_row  = 0
    max_score = 0

    n = min(scan_rows, len(df_raw))
    for i in range(n):
        row   = df_raw.iloc[i]
        score = sum(
            isinstance(x, str) and len(x.strip()) > 0
            for x in row
        )
        logging.debug("detect_header_row: row %d score=%d", i, score)
        if score > max_score:
            max_score = score
            best_row  = i

    logging.info("detect_header_row: selected row %d (score=%d)", best_row, max_score)
    return best_row


# ---------------------------------------------------------------------------
# Step 2 — Column cleaning
# ---------------------------------------------------------------------------

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise column names:
      - Cast to str, strip whitespace, collapse newlines
      - Drop columns whose name starts with 'Unnamed'
      - Drop columns where ALL values are null
    """
    # Step 4: clean names
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace("\n", " ")
    )

    # Step 5a: drop Unnamed columns
    before = list(df.columns)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False, regex=True)]

    # Step 5b: drop all-null columns
    df = df.dropna(axis=1, how="all")

    dropped = set(before) - set(df.columns)
    if dropped:
        logging.info("_clean_columns: dropped columns: %s", sorted(dropped))

    return df


# ---------------------------------------------------------------------------
# Step 3 — Row cleaning
# ---------------------------------------------------------------------------

def _clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where every cell is null."""
    before = len(df)
    df     = df.dropna(how="all").reset_index(drop=True)
    if len(df) < before:
        logging.info("_clean_rows: dropped %d all-null rows", before - len(df))
    return df


# ---------------------------------------------------------------------------
# Step 3b — Value cleaning (trim + deduplicate)
# ---------------------------------------------------------------------------

def _clean_values(df: pd.DataFrame, source_label: str = "") -> pd.DataFrame:
    """
    1. Strip leading/trailing whitespace from all string cells
    2. Strip column names (safety pass after _clean_columns)
    3. Drop exact duplicate rows
    Logs before/after counts.
    """
    # Trim all string values — leaves numbers, dates, NaN untouched
    df = df.apply(
        lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x)
    )

    # Normalise column names one more time (handles any whitespace reintroduced)
    df.columns = df.columns.str.strip()

    # Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dupes = before - len(df)
    if dupes:
        logging.info("_clean_values [%s]: removed %d duplicate rows (%d → %d)",
                     source_label, dupes, before, len(df))
    else:
        logging.info("_clean_values [%s]: no duplicates found (%d rows)", source_label, len(df))

    return df


# ---------------------------------------------------------------------------
# Step 4 — Validation
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame, source_label: str) -> None:
    """Step 7: ensure the result is a usable table."""
    if len(df.columns) < 2:
        raise ValueError(
            f"Invalid table structure in '{source_label}': "
            f"only {len(df.columns)} column(s) after cleaning. "
            "Ensure the file has a proper header row."
        )
    if len(df) == 0:
        raise ValueError(f"No data rows found in '{source_label}' after cleaning.")


# ---------------------------------------------------------------------------
# Public readers
# ---------------------------------------------------------------------------

def read_csv_clean(file_bytes: bytes, source_label: str = "csv") -> pd.DataFrame:
    """
    Read a CSV from bytes with automatic header detection and cleaning.

    Steps:
      1. Load raw (header=None, dtype=str) to inspect structure
      2. Detect header row
      3. Reload with correct header
      4. Clean columns → drop rows → validate
    """
    # Step 1: raw load — keep as native types so detect_header_row
    # can distinguish strings (labels) from numbers (data values)
    try:
        df_raw = pd.read_csv(io.BytesIO(file_bytes), header=None)
    except Exception as exc:
        raise RuntimeError(f"CSV parse failed for '{source_label}': {exc}") from exc

    # Step 2: detect header
    header_idx = detect_header_row(df_raw)
    logging.info("read_csv_clean [%s]: header row = %d", source_label, header_idx)

    # Step 3: reload with correct header (pandas skips rows 0..header_idx-1 automatically)
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), header=header_idx)
    except Exception as exc:
        raise RuntimeError(f"CSV reload failed for '{source_label}': {exc}") from exc

    # Steps 4-7
    df = _clean_columns(df)
    df = _clean_rows(df)
    df = _clean_values(df, source_label)
    _validate(df, source_label)

    logging.info("read_csv_clean [%s]: %d rows × %d cols | columns=%s",
                 source_label, len(df), len(df.columns), df.columns.tolist())
    return df


def read_excel_clean(
    file_bytes:   bytes,
    sheet_name:   str | int = 0,
    source_label: str = "excel",
) -> pd.DataFrame:
    """
    Read one Excel sheet from bytes with automatic header detection and cleaning.

    Steps:
      1. Load raw (header=None) to inspect structure
      2. Detect header row
      3. Reload with correct header
      4. Clean columns → drop rows → validate
    """
    label = f"{source_label}[{sheet_name}]"

    # Step 1: raw load — native types for accurate header scoring
    try:
        df_raw = pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name = sheet_name,
            header     = None,
            engine     = "openpyxl",
        )
    except Exception as exc:
        raise RuntimeError(f"Excel parse failed for '{label}': {exc}") from exc

    # Step 2: detect header
    header_idx = detect_header_row(df_raw)
    logging.info("read_excel_clean [%s]: header row = %d", label, header_idx)

    # Step 3: reload with correct header
    try:
        df = pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name = sheet_name,
            header     = header_idx,
            engine     = "openpyxl",
        )
    except Exception as exc:
        raise RuntimeError(f"Excel reload failed for '{label}': {exc}") from exc

    # Steps 4-7
    df = _clean_columns(df)
    df = _clean_rows(df)
    df = _clean_values(df, label)
    _validate(df, label)

    logging.info("read_excel_clean [%s]: %d rows × %d cols | columns=%s",
                 label, len(df), len(df.columns), df.columns.tolist())
    return df
