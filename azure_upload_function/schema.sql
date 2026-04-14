-- Run this against your Azure SQL Database before deploying the function

CREATE TABLE documents_metadata (
    id          NVARCHAR(36)   NOT NULL PRIMARY KEY,   -- UUID stored as string
    filename    NVARCHAR(255)  NOT NULL,
    blob_url    NVARCHAR(1024) NOT NULL,
    description NVARCHAR(1000) NULL,
    tags        NVARCHAR(500)  NULL,
    created_at  DATETIME2      NOT NULL DEFAULT GETUTCDATE()
);

CREATE INDEX idx_documents_metadata_filename ON documents_metadata (filename);
CREATE INDEX idx_documents_metadata_created_at ON documents_metadata (created_at DESC);
