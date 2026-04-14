-- Run against Azure SQL Database to add AI processing columns
-- Safe to run multiple times (checks column existence first)

IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = 'documents_metadata' AND COLUMN_NAME = 'extracted_text'
)
    ALTER TABLE documents_metadata ADD extracted_text NVARCHAR(MAX) NULL;

IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = 'documents_metadata' AND COLUMN_NAME = 'summary'
)
    ALTER TABLE documents_metadata ADD summary NVARCHAR(MAX) NULL;

-- 'tags' column already exists from initial schema.
-- This statement overwrites it with AI-generated key phrases after blob processing.
-- No ALTER needed — UPDATE is handled in sql_service.update_ai_fields().
