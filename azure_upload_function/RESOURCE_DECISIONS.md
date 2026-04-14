# Resource Decisions & Architecture Notes

All resource choices, alternatives considered, and reasoning documented in one place.
Updated every time a new file or resource is added to this project.

---

## 1. function_app.py — Azure Function (HTTP Trigger)

**Created:** Initial migration from AWS Lambda
**Replaces:** `backend/lambda_function.py` (AWS Lambda, DynamoDB Stream trigger)

### What it does
Entry point for the upload service. Accepts a `POST /upload` multipart request,
delegates to BlobService and SQLService, returns JSON with the record ID and blob URL.

### Why Azure Functions
- Serverless, scales to zero — direct equivalent of AWS Lambda
- Pay-per-execution, no idle cost
- Native HTTP trigger with multipart/form-data support
- Minimal infrastructure overhead

### Alternatives considered
| Option | Why not chosen |
|---|---|
| Azure App Service (FastAPI/Flask) | Always-on, costs money at idle, overkill for one endpoint |
| Azure Container Apps | More control but requires Docker, registry, higher ops overhead |
| Azure API Management + Logic Apps | Low-code, not suited for binary file handling |

### Auth level
Set to `ANONYMOUS` per requirements. For production switch to `AuthLevel.FUNCTION`
(API key header) or front with Azure API Management + OAuth2.

---

## 2. services/blob_service.py — Azure Blob Storage

**Created:** Initial migration from AWS S3
**Replaces:** S3 bucket `data-orch-upload-files-v1`
**Container:** `documents`

### What it does
Connects to Blob Storage via connection string, auto-creates the `documents` container
if missing, prefixes every blob with a UUID hex to prevent collisions, sets Content-Type,
and returns the full blob URL.

### Why Azure Blob Storage
- Direct equivalent of AWS S3 — same flat object storage model
- Cheapest Azure option for unstructured binary files
- Native SDK (`azure-storage-blob`) with SAS token support built in
- Hot / Cool / Archive tiering available for cost optimisation later
- Scales to petabytes with zero configuration

### Alternatives considered
| Option | Why not chosen |
|---|---|
| Azure Data Lake Storage Gen2 | Built on Blob but adds hierarchical namespace — better for analytics, overkill here |
| Azure Files (SMB/NFS) | Designed for lift-and-shift file shares, not HTTP object storage |
| Azure NetApp Files | Enterprise NFS/HPC workloads, extremely expensive |

### Upload strategy
Server-side upload through the Function using a connection string.
The original AWS flow used presigned S3 URLs (browser → S3 directly).
If client-side upload is needed later, swap to `generate_blob_sas()` to issue
a short-lived SAS token to the frontend instead.

### Container access
Private by default. Blobs are not publicly accessible by URL.
For production, serve files via SAS tokens or Azure CDN with token auth.

---

## 3. services/sql_service.py — Azure SQL Database

**Created:** Initial migration from AWS DynamoDB
**Replaces:** DynamoDB table `data_orchestration_backend`
**Table:** `documents_metadata`

### What it does
Opens a `pyodbc` connection to Azure SQL, inserts a metadata row (UUID, filename,
blob URL, description, tags, created_at), commits, and returns the generated UUID.
Rolls back on failure and closes the connection on object destruction.

### Why Azure SQL Database
- Structured relational schema — better fit than DynamoDB for metadata with
  fixed columns (id, filename, blob_url, description, tags, created_at)
- Full SQL query support for filtering, sorting, and joining later
- Managed service — automatic backups, patching, high availability
- ODBC Driver 18 works natively in Azure Functions Python runtime

### Alternatives considered
| Option | Why not chosen |
|---|---|
| Azure Cosmos DB (NoSQL) | Closest to DynamoDB, but schema-less and more expensive at low scale |
| Azure Table Storage | Key-value only, very limited query capability |
| Azure Database for PostgreSQL | Great alternative, but SQL Server is more common in Azure-native stacks |
| Azure Database for MySQL | Same as above — valid choice, slightly less Azure-native tooling |

### Connection management
Currently opens one connection per Function invocation. For high-throughput
scenarios, replace with a connection pool using `pyodbc` + a module-level
singleton, or switch to `aioodbc` for async support.

---

## 4. schema.sql — SQL Table Definition

**Created:** Initial migration
**Replaces:** DynamoDB schema (schemaless)

### What it does
Defines the `documents_metadata` table with typed columns and two indexes:
- Primary key on `id` (UUID as NVARCHAR(36))
- Index on `filename` for lookup queries
- Index on `created_at DESC` for pagination / recent-first listing

### Why NVARCHAR(36) for UUID
Azure SQL does not have a native UUID/GUID auto-generation that maps cleanly
to Python's `uuid.uuid4()` string output. Storing as NVARCHAR(36) keeps it
readable and avoids binary UNIQUEIDENTIFIER conversion complexity.
If performance becomes a concern, switch to `UNIQUEIDENTIFIER` with `NEWID()`.

---

## 5. requirements.txt — Python Dependencies

**Created:** Initial setup

| Package | Version | Purpose |
|---|---|---|
| `azure-functions` | 1.19.0 | Azure Functions Python worker SDK |
| `azure-storage-blob` | 12.19.1 | Blob Storage client (replaces `boto3` S3) |
| `pyodbc` | 5.1.0 | ODBC driver bridge for Azure SQL (replaces `boto3` DynamoDB) |

### Why not boto3
`boto3` is AWS-specific. All three packages above are the direct Azure equivalents.

---

## 6. host.json — Functions Runtime Configuration

**Created:** Initial setup

### What it does
- Sets Functions runtime to v2
- Enables Application Insights sampling (excludes Request type to reduce noise)
- Pins the extension bundle to v4.x for HTTP trigger support

### Why extension bundle v4
v4 is the current stable bundle for Python v2 programming model.
v3 bundles are compatible but miss newer binding types.

---

## 7. local.settings.json — Local Environment Variables

**Created:** Initial setup
**Never commit to git** — add to `.gitignore`

### Variables
| Key | Purpose |
|---|---|
| `AZURE_STORAGE_CONNECTION_STRING` | Authenticates BlobServiceClient — replaces AWS IAM role + boto3 session |
| `SQL_CONNECTION_STRING` | ODBC connection string for Azure SQL — replaces DynamoDB resource ARN |
| `AzureWebJobsStorage` | Required by Functions runtime for internal state (use Azurite locally) |
| `FUNCTIONS_WORKER_RUNTIME` | Tells the host to use the Python worker |

### Why connection strings instead of Managed Identity
Per requirements: no Azure AD, Key Vault, or RBAC.
Connection strings are the simplest auth path.
When requirements change, swap to `DefaultAzureCredential` + Managed Identity
with zero code changes — just remove the connection string env vars.

---

## 8. process_document.py — Azure Function (Blob Trigger)

**Created:** AI processing pipeline
**Replaces:** AWS Lambda triggered by DynamoDB Stream (`lambda_function.py`)

### What it does
Fires automatically when any file lands in the `documents` Blob container.
Reads the blob bytes, passes them through DocIntelligenceService and LanguageService,
then calls `SQLService.update_ai_fields()` to write results back to SQL.

### Why Blob Trigger instead of HTTP Trigger
- The original AWS pipeline used DynamoDB Streams to trigger Lambda after metadata
  was written. Here the file itself landing in Blob Storage is the natural event.
- Blob trigger is event-driven and serverless — no polling, no queue management needed
  for straightforward single-file processing.

### Alternatives considered
| Option | Why not chosen |
|---|---|
| Azure Event Grid + HTTP Trigger | More flexible routing, but adds an extra service and wiring for no benefit here |
| Azure Storage Queue Trigger | Good for retry/backoff on failures, slightly more complex setup |
| Azure Service Bus Trigger | Enterprise messaging — overkill for a single container trigger |

### Failure behaviour
Re-raises exceptions after logging so the Functions runtime marks the invocation
as failed and retries up to the configured `maxDequeueCount` (default 5).

---

## 9. services/doc_intelligence_service.py — Azure Document Intelligence

**Created:** AI processing pipeline
**Replaces:** AWS Comprehend (text extraction) + pdfminer (PDF parsing) in `comprehend.py`

### What it does
Uses the `prebuilt-read` model to extract all text lines from a PDF document.
Accepts raw bytes directly — no need to write to disk or generate a SAS URL.
Returns a single newline-joined string of all extracted lines.

### Why Azure Document Intelligence
- Handles complex PDF layouts (tables, columns, headers) that pdfminer struggles with
- Managed cloud service — no Lambda layer or pip package size concerns
- `prebuilt-read` model is the fastest and cheapest model for plain text extraction
- Supports PDF, JPEG, PNG, TIFF, BMP out of the box

### Alternatives considered
| Option | Why not chosen |
|---|---|
| pdfminer (Python library) | Used in original codebase, fragile on complex layouts, requires Lambda layer |
| PyPDF2 / pypdf | Pure Python, poor table/column handling |
| Azure Cognitive Search built-in extraction | Tied to a Search index, not a standalone extraction API |
| OpenAI GPT-4 Vision | Can extract text from images but expensive and non-deterministic |

### Model choice: prebuilt-read
- Cheapest Document Intelligence model (pay per page)
- Optimised purely for text extraction — no form field or table parsing overhead
- For structured forms or invoices, swap to `prebuilt-document` or a custom model

---

## 10. services/language_service.py — Azure AI Language (Key Phrase Extraction)

**Created:** AI processing pipeline
**Replaces:** AWS Comprehend `detect_entities` + NLTK lemmatization in `comprehend.py`

### What it does
Calls the Azure AI Language `extract_key_phrases` API on the extracted text.
Handles the 5,120-character API limit by chunking long documents automatically.
Deduplicates phrases (case-insensitive) and returns an ordered list.

### Why Azure AI Language
- Direct equivalent of AWS Comprehend for NLP tasks
- Key phrase extraction is a single API call — no NLTK downloads, no Lambda /tmp hacks
- Managed, versioned, no dependency on NLTK data files in the deployment package
- Same endpoint supports sentiment, entity recognition, PII detection — easy to extend

### Alternatives considered
| Option | Why not chosen |
|---|---|
| AWS Comprehend (keep as-is) | AWS-specific, not available in Azure |
| NLTK + spaCy (self-hosted) | Requires model files in deployment package, adds cold start time |
| Azure OpenAI (GPT summarisation) | More powerful but expensive and non-deterministic for tagging |
| Azure Cognitive Search semantic ranking | Tied to a Search index, not a standalone NLP API |

### Chunking strategy
Azure AI Language accepts max 5,120 characters per document per request.
Text is split into fixed-size chunks. Key phrases from all chunks are merged
and deduplicated. This is the same approach used in the original `comprehend.py`
(2,000-character chunks to Comprehend).

---

## 11. alter_schema.sql — SQL Schema Migration

**Created:** AI processing pipeline

### What it does
Adds two new nullable columns to `documents_metadata`:
- `extracted_text NVARCHAR(MAX)` — full text from Document Intelligence
- `summary NVARCHAR(MAX)` — first 300 characters of extracted text

Uses existence checks so the script is safe to re-run without errors.
The `tags` column already exists and is overwritten in-place by `update_ai_fields()`.

### Why NVARCHAR(MAX)
Extracted text from multi-page PDFs can easily exceed thousands of characters.
`NVARCHAR(MAX)` stores up to 2 GB — no truncation risk.
For search use cases, consider adding a Full-Text Search index on `extracted_text`.

---

## 12. Consolidation Fix — Single FunctionApp Instance

**Changed:** `process_document.py` deleted, blob trigger merged into `function_app.py`

### Root cause
The Azure Functions v2 Python programming model requires exactly **one**
`func.FunctionApp()` instance across the entire project. Having a second instance
in `process_document.py` caused the runtime to only register whichever file it
loaded first — the blob trigger was silently dropped.

### Fix applied
- Deleted `process_document.py`
- Moved `process_document` blob trigger function into `function_app.py` under the
  existing `app` instance
- Changed `connection` from `"AZURE_STORAGE_CONNECTION_STRING"` to
  `"AzureWebJobsStorage"` — the blob trigger `connection` parameter must reference
  the **app setting name**, not the env var value. `AzureWebJobsStorage` is the
  standard setting name the runtime uses for storage bindings.

### Rule going forward
All functions (HTTP, blob, queue, timer, etc.) must be registered on the **same**
`app = func.FunctionApp()` instance inside `function_app.py`.

---

## 13. services/table_service.py — Azure Table Storage (replaces SQL)

**Created:** SQL → Table Storage migration
**Replaces:** `services/sql_service.py` + `pyodbc` + Azure SQL Database

### What it does
- `insert_entity()` — creates a new entity with PartitionKey=`"documents"`, RowKey=UUID,
  and all metadata fields. Called by the HTTP upload trigger.
- `update_ai_fields()` — queries by filename, then issues a `MERGE` update so only
  `extracted_text`, `summary`, and `tags` are overwritten. Called by the blob trigger.
- Auto-creates the `documentsmetadata` table on first run if it doesn't exist.

### Why Azure Table Storage
- Already using `AZURE_STORAGE_CONNECTION_STRING` for Blob — Table Storage lives in
  the same account, zero extra infrastructure
- No schema migrations needed — schemaless entities match the evolving AI output fields
- Significantly cheaper than Azure SQL at low-to-medium scale
- `azure-data-tables` SDK is lightweight, no ODBC driver required

### Alternatives considered
| Option | Why not chosen |
|---|---|
| Azure SQL Database (previous) | Requires ODBC driver, schema migrations, higher cost |
| Azure Cosmos DB (Table API) | Compatible API but higher cost, overkill for this scale |
| Azure Cosmos DB (NoSQL) | More powerful querying but adds a new service dependency |

### Update mode: MERGE
`UpdateMode.MERGE` patches only the fields present in the patch entity.
`UpdateMode.REPLACE` would wipe all fields not included — dangerous for partial updates.

### Query strategy
`update_ai_fields` queries by `filename` within the `documents` partition.
If filenames are not unique, the first match is updated. For strict uniqueness,
pass the RowKey (UUID) through the blob metadata and query by RowKey directly.

---

## 14. services/search_service.py — Azure AI Search

**Created:** RAG pipeline
**Index:** `documents-index`

### What it does
- `ensure_index()` — creates the search index on first run if missing. Safe to call repeatedly.
- `index_document()` — pushes a single document into the index using `@search.action: upload`.
  Retries up to 3 times with exponential back-off (2s, 4s) on transient failures.
- `search()` — full-text search across `summary`, `extracted_text`, `tags`, `filename`.
  Returns top 3 results by default to keep LLM context small and cost low.

### Why Azure AI Search (REST API, not SDK)
- No extra SDK package needed — `requests` is already a standard dependency
- REST API is stable, versioned, and well-documented
- Avoids `azure-search-documents` package adding ~10MB to deployment size

### Index field design
| Field | Type | Searchable | Retrievable | Notes |
|---|---|---|---|---|
| id | Edm.String (key) | No | Yes | RowKey from Table Storage — ensures consistency |
| filename | Edm.String | Yes | Yes | Filterable for exact match |
| extracted_text | Edm.String | Yes | Yes | en.microsoft analyzer for English stemming |
| summary | Edm.String | Yes | Yes | en.microsoft analyzer |
| tags | Collection(Edm.String) | Yes | Yes | Filterable for tag-based queries |
| blob_url | Edm.String | No | Yes | Returned to frontend as source link |

### Alternatives considered
| Option | Why not chosen |
|---|---|
| Azure Cognitive Search SDK (`azure-search-documents`) | Adds package weight, REST is sufficient |
| Azure AI Search semantic ranking | Higher cost per query, not needed at this scale |
| Elasticsearch on AKS | Requires container infrastructure, much higher ops overhead |
| Azure Cosmos DB full-text search | Limited full-text capability compared to dedicated search |

---

## 15. services/rag_service.py — Azure OpenAI (RAG Answer Generation)

**Created:** RAG pipeline
**Model:** `gpt-4o-mini` (configurable via `OPENAI_DEPLOYMENT` env var)

### What it does
Builds a grounded prompt from up to 3 retrieved documents (summary + 1000 chars of
extracted text each), calls Azure OpenAI chat completions, and returns a concise answer.
System prompt explicitly instructs the model not to hallucinate beyond the context.

### Cost controls
- Max 3 documents passed to LLM
- 1000 chars per document text limit
- `max_tokens=512` on the response
- `temperature=0.2` for factual, low-variance output
- `gpt-4o-mini` is the cheapest capable model in the Azure OpenAI lineup

### Alternatives considered
| Option | Why not chosen |
|---|---|
| GPT-4o (full) | 10x more expensive per token, not needed for document Q&A |
| GPT-3.5-turbo | Cheaper but weaker reasoning on complex documents |
| Azure AI Studio prompt flow | Adds orchestration overhead, overkill for a single RAG chain |
| LangChain / LlamaIndex | External framework dependency, adds complexity and package size |

---

## 16. services/router_service.py — Query Router

**Created:** RAG pipeline — intelligent query routing

### What it does
`route_query(query)` classifies a user query as `"sql"` or `"rag"` using keyword matching.
SQL keywords include: count, total, sum, average, how many, compare, trend, group by, etc.
If any SQL keyword is found → routes to SQL system. Otherwise → routes to RAG pipeline.

### Why keyword matching instead of LLM classification
- Zero latency, zero cost — no extra API call needed
- Deterministic and debuggable
- Sufficient for the current query patterns (analytical vs. document Q&A)
- Can be upgraded to LLM-based classification later by replacing `route_query()` body

### Alternatives considered
| Option | Why not chosen |
|---|---|
| LLM-based classifier (GPT call) | Adds latency + cost to every query, overkill for keyword-level routing |
| Fine-tuned classifier model | Requires training data and model hosting |
| Azure Language custom classification | Managed but requires labelled training data and additional service |

---

## 17. /query Endpoint — Unified Query API

**Created:** RAG pipeline
**Route:** `GET/POST /query?q=<user_query>`

### What it does
1. Receives user query
2. Calls `route_query()` to classify as `sql` or `rag`
3. SQL path: returns `{ route: "sql" }` signal for the existing SQL frontend to handle
4. RAG path: calls `SearchService.search()` → `RAGService.generate_answer()` → returns
   `{ route: "rag", answer: "...", sources: [...] }`

### Response contract
```json
{
  "route":   "rag | sql",
  "query":   "user query string",
  "answer":  "LLM-generated answer or null",
  "sources": [
    { "filename": "...", "summary": "...", "blob_url": "..." }
  ]
}
```

### Frontend integration
- If `route == "sql"` → pass query to existing SQL/chart rendering system
- If `route == "rag"` → display `answer` text + render `sources` as document links
- Both paths return the same response shape so the frontend can handle them uniformly

---

## 18. Dependency & Env Var Standardization Fix

**Changed:** `requirements.txt`, `rag_service.py`, `local.settings.json`

### Root causes fixed

1. `openai` version pinned to `>=1.0.0,<2.0.0` — the `AzureOpenAI` class was
   introduced in v1.0.0. An unpinned or `==1.30.1` pin can resolve to a broken
   pre-release on some pip resolvers. The range pin is safer for Python 3.10.

2. Env var names were inconsistent — `local.settings.json` used `OPENAI_ENDPOINT`,
   `OPENAI_API_KEY`, `OPENAI_DEPLOYMENT` but the standard Azure OpenAI SDK convention
   and the task requirement is `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`,
   `AZURE_OPENAI_DEPLOYMENT_NAME`. All three files now use the `AZURE_OPENAI_*` names.

3. `rag_service.py` now uses `os.getenv()` with an explicit `EnvironmentError` guard
   instead of `os.environ[]` which raises a cryptic `KeyError` on missing vars.
   This gives a clear error message at cold start rather than inside a request.

4. Module-level `_DEPLOYMENT = os.environ.get(...)` removed — reading env vars at
   import time causes issues on cold starts before the Functions host injects settings.
   Moved into `__init__` so it only runs when the class is instantiated inside a request.

---

## 19. frontend/ — React + Vite + Tailwind UI

**Created:** Full frontend for all three modules

### Structure
```
frontend/
├── index.html
├── vite.config.js          — dev proxy: /api → localhost:7071
├── tailwind.config.js
├── postcss.config.js
├── package.json
└── src/
    ├── main.jsx            — app entry, BrowserRouter, Toaster
    ├── App.jsx             — layout shell, dark mode toggle, routes
    ├── index.css           — Tailwind directives + scrollbar utility
    ├── services/
    │   └── api.js          — uploadFile(), searchDocuments(), askQuestion()
    ├── components/
    │   ├── Sidebar.jsx     — NavLink-based nav with active state
    │   ├── Header.jsx      — page title + dark mode button
    │   ├── UploadBox.jsx   — drag & drop + file staging list
    │   ├── FileCard.jsx    — uploaded doc card with tags + blob link
    │   ├── SearchResultCard.jsx — result card with keyword highlighting
    │   └── ChatMessage.jsx — chat bubble, copy button, sources list
    └── pages/
        ├── FilesPage.jsx   — upload form + uploaded docs grid
        ├── SearchPage.jsx  — search bar + results list
        └── ChatPage.jsx    — chat interface with auto-scroll + typing indicator
```

### Why Vite over CRA
- 10-20x faster cold start and HMR
- Native ESM, no webpack overhead
- Built-in proxy config for local API calls

### Why react-hot-toast over custom toasts
- Zero-config, lightweight (~3KB)
- Accessible by default
- Consistent API across all three pages

### Dev proxy
`vite.config.js` proxies `/api/*` to `http://localhost:7071` so the frontend
calls `/api/upload` and `/api/query` without CORS issues during local development.
In production, set `VITE_API_BASE_URL` and update `api.js` baseURL accordingly.
