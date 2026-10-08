# jeen-doc-indexer

A small, production-minded Python module that turns PDF and DOCX documents into a **semantic search index**:
it extracts clean text, splits it into chunks with one of three strategies, embeds each chunk with
Google's `gemini-embedding-001`, stores everything in **PostgreSQL + pgvector**, and answers natural-language
queries by vector similarity.

Built for Part 2 of the Jeen AI Solution home assignment. In the overall solution it is the knowledge layer
behind *TriageBot*, the contact-center agent from Part 1: the agent's `search_knowledge_base` tool is exactly
the search implemented here, and the sample document is the (fictional) knowledge base the agent consults.

```mermaid
flowchart LR
    A[PDF / DOCX] --> B[Extract<br/>pypdf · python-docx]
    B --> C[Clean<br/>NFKC · NUL bytes · whitespace]
    C --> D{Chunk}
    D -->|fixed| E[Embed<br/>gemini-embedding-001<br/>768 dims · RETRIEVAL_DOCUMENT]
    D -->|sentence| E
    D -->|paragraph| E
    E --> F[(PostgreSQL<br/>pgvector · HNSW)]
    Q[Query] --> G[Embed<br/>RETRIEVAL_QUERY] --> H[Cosine search<br/>top-k] --> F
```

## Contents

- [Quick start](#quick-start)
- [Environment variables](#environment-variables)
- [Usage](#usage)
- [Chunking strategies](#chunking-strategies)
- [Database schema](#database-schema)
- [Error handling](#error-handling)
- [Design decisions](#design-decisions)
- [Example runs](#example-runs)
- [Project structure](#project-structure)
- [Tests](#tests)

## Quick start

Requirements: **Python 3.10+**, **Docker** (for PostgreSQL with pgvector) and a free **Gemini API key**
from [Google AI Studio](https://aistudio.google.com).

```bash
git clone https://github.com/<your-user>/jeen-doc-indexer.git
cd jeen-doc-indexer

# 1. Python environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate.bat (cmd) or .venv\Scripts\Activate.ps1 (PowerShell)
pip install -r requirements.txt

# 2. Configuration
cp .env.example .env                 # Windows: copy .env.example .env
# edit .env and set GEMINI_API_KEY

# 3. Database (PostgreSQL 17 + pgvector)
docker compose up -d

# 4. Index and search
python index_documents.py --file ./docs/example.pdf --strategy paragraph
python search.py --query "login issue"
```

The table and the HNSW index are created automatically on the first run.

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | yes | Google AI Studio API key, used for `gemini-embedding-001` |
| `POSTGRES_URL` | yes | `postgresql://user:password@host:port/database` |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_PORT` | for Docker only | Used by `docker-compose.yml` to create the local database; must match `POSTGRES_URL` |

Values are read from the environment or from a local `.env` file (git-ignored). No key, password or path is
hard-coded, and the CLIs never print secrets: database errors show only `host:port/database`.

## Usage

### Index a document

```bash
python index_documents.py --file ./docs/example.pdf --strategy paragraph
python index_documents.py --file ./docs/example.pdf --strategy all            # all three strategies
python index_documents.py --file ./docs/example.pdf --strategy fixed --chunk-size 800 --overlap 150
python index_documents.py --file ./docs/technician_visits_he.docx --strategy sentence
python index_documents.py --file ./docs/example.pdf --strategy sentence --dry-run   # no API, no DB
```

| Option | Description |
|---|---|
| `--file` | Path to a `.pdf` or `.docx` file (required) |
| `--strategy` | `fixed`, `sentence`, `paragraph`, or `all` (required) |
| `--chunk-size` | Target chunk size in characters (defaults: fixed 1000, sentence 800, paragraph 1500) |
| `--overlap` | Overlap in characters for `fixed` (default 200) |
| `--replace` | Delete the file's existing chunks for the strategy before indexing |
| `--dry-run` | Extract and chunk only, then print a preview |
| `-v` | Debug logging |

Re-running the same command is safe: each chunk's SHA-256 hash is stored, chunks that already exist are
skipped **before** they are sent to the API, and the database enforces uniqueness as a second guard.

### Search

```bash
python search.py --query "login issue"
python search.py --query "customer was charged twice" --top-k 3 --strategy paragraph
python search.py --query "router keeps disconnecting" --filename example.pdf --min-score 0.6
python search.py --query "מתי שולחים טכנאי?"          # any language: embeddings are multilingual
python search.py --query "login issue" --json          # machine-readable output
python search.py --list                                # indexed files and chunk counts
```

| Option | Description |
|---|---|
| `--query` | Natural-language query |
| `--top-k` | Number of results, 1–50 (default 5) |
| `--strategy` | Only search chunks created with this strategy (useful for comparing strategies) |
| `--filename` | Only search chunks from one file |
| `--min-score` | Drop results below this cosine similarity |
| `--json` | Print results as JSON |
| `--list` | List indexed files per strategy |

## Chunking strategies

| Strategy | How it works | Best for |
|---|---|---|
| `fixed` | Sliding window of up to `chunk-size` characters with `overlap`; cut points move to the nearest space so words are never split | Uniform, predictable chunks; documents without clear structure |
| `sentence` | Whole sentences packed up to `chunk-size`; headings start a new chunk; abbreviations ("e.g.") and section numbers ("2.") don't end a sentence | Precise, fact-style questions |
| `paragraph` | One chunk per paragraph; short paragraphs (such as headings) merge into the next one, long ones are split by sentence | Procedures and "how do I…" questions, where a topic should stay together |

PDFs have no paragraph markers, only visual lines, so the extractor rebuilds paragraphs: a line break
becomes a paragraph break when the line is clearly shorter than a full line (a heading or the end of a
paragraph) or starts a bullet. Sentence splitting works for English and Hebrew.

## Database schema

Defined in [`sql/schema.sql`](sql/schema.sql) and created automatically by the code.

| Column | Type | Notes |
|---|---|---|
| `id` | `BIGSERIAL` | primary key |
| `chunk_text` | `TEXT` | the chunk |
| `embedding` | `VECTOR(768)` | L2-normalized |
| `filename` | `TEXT` | base name only, never a local path |
| `split_strategy` | `TEXT` | `fixed` / `sentence` / `paragraph` (CHECK constraint) |
| `chunk_index` | `INTEGER` | position in the document |
| `content_hash` | `TEXT` | SHA-256 of `chunk_text`; `UNIQUE (filename, split_strategy, content_hash)` |
| `created_at` | `TIMESTAMPTZ` | default `now()` |

Indexes: HNSW on `embedding` with `vector_cosine_ops`, and a B-tree on `(filename, split_strategy)`.
Search ranks by cosine distance (`<=>`) and reports `score = 1 - distance`.

## Error handling

Expected failures print one clear line (no traceback) and exit with a stable code:

| Situation | Message (example) | Exit |
|---|---|---|
| Missing environment variable | `Missing GEMINI_API_KEY. Copy .env.example to .env and fill in the value(s).` | 1 |
| File not found | `File not found: ./docs/x.pdf` | 2 |
| Unsupported file type | `Unsupported file type '.txt'. Supported: .pdf, .docx` | 3 |
| Content doesn't match extension | `fake.pdf has a '.pdf' extension but its content is not a valid PDF file` | 3 |
| Encrypted or corrupted file | `report.pdf is password-protected and cannot be read` | 3 |
| Document with no text | `No extractable text in scan.pdf (is it a scanned PDF? OCR is not supported)` | 4 |
| Embedding failure | `Embedding failed after 5 attempts: HTTP 429: …` / `Embedding request rejected (HTTP 400 …). Check that GEMINI_API_KEY in .env is valid.` | 5 |
| Database unreachable | `Cannot connect to PostgreSQL at localhost:5432/doc_index. Is the database running? (docker compose up -d)` | 6 |
| Invalid argument | `--overlap must be >= 0 and smaller than --chunk-size` | 7 |
| Empty search results | `No results for "…". Try a broader query or remove filters.` (or a hint that the index is empty) | 0 |

Rate-limit (429), timeout and server errors are retried with exponential backoff and jitter; invalid-key and
other client errors fail immediately with a hint. The database connection is opened **before** any embedding
call, so a stopped database never wastes API quota.

## Design decisions

| Decision | Why |
|---|---|
| **768 dimensions** instead of the 3,072 default | pgvector's HNSW index supports up to 2,000 dimensions for `vector`; 768 is 4x smaller and faster, with a marginal quality cost for retrieval |
| **Normalize vectors in code** | Google normalizes only the full 3,072-dimension output; truncated vectors must be normalized for cosine similarity to be correct |
| **Different task types** for documents and queries | `RETRIEVAL_DOCUMENT` for chunks, `RETRIEVAL_QUERY` for searches, as recommended for asymmetric retrieval |
| **Batching + exponential backoff** | The free tier rate-limits quickly; batches of 50 reduce requests, and retries make runs resilient |
| **Remove `\x00` during cleaning** | PDFs sometimes contain NUL bytes, which PostgreSQL rejects in `TEXT` columns |
| **Magic-byte check** in addition to the extension | A renamed file gets a clear message instead of a parser crash |
| **Content hash + `UNIQUE`** | Re-indexing is idempotent and doesn't spend API quota on chunks already stored |
| **HNSW index** with `vector_cosine_ops` | Fast approximate nearest-neighbour search that keeps working as the index grows |
| **Regex sentence splitting** | No heavy NLP dependency; handles English and Hebrew punctuation, abbreviations and numbered headings |
| **OS certificate store** via `truststore` | Corporate proxies often re-sign HTTPS; Python's bundled CA list rejects them (`CERTIFICATE_VERIFY_FAILED`). Using the OS store fixes this without turning verification off, and certificate errors fail fast instead of being retried |
| **Scanned PDFs are rejected**, not OCR'd | Out of scope; reported explicitly. OCR (e.g. Tesseract or a vision model) is a natural extension |

Possible extensions: hybrid search (pgvector + PostgreSQL full-text), metadata such as page numbers per
chunk, an evaluation set of question → expected chunk to compare strategies objectively, and OCR.

## Example runs

> Real output from a run against the Gemini API and the local pgvector database (Windows, Python 3.10).

### Indexing

```text
$ python index_documents.py --file ./docs/example.pdf --strategy all
✓ Extracted 6,730 characters from example.pdf (3 pages)
✓ Split into 9 chunks (strategy=fixed, target<=1000 chars, overlap=200, avg 917 chars)
✓ Split into 22 chunks (strategy=sentence, target<=800 chars, avg 303 chars)
✓ Split into 20 chunks (strategy=paragraph, target<=1500 chars, avg 333 chars)
✓ Embedded 9 chunks with gemini-embedding-001 (768 dims) in 1 batch(es)
✓ Stored 9 rows in document_chunks (0 already indexed, skipped)
✓ Embedded 22 chunks with gemini-embedding-001 (768 dims) in 1 batch(es)
✓ Stored 22 rows in document_chunks (0 already indexed, skipped)
✓ Embedded 20 chunks with gemini-embedding-001 (768 dims) in 1 batch(es)
✓ Stored 20 rows in document_chunks (0 already indexed, skipped)
✓ Done in 4.4s. Database: localhost:5432/doc_index

$ python index_documents.py --file ./docs/technician_visits_he.docx --strategy paragraph
✓ Extracted 965 characters from technician_visits_he.docx
✓ Split into 4 chunks (strategy=paragraph, target<=1500 chars, avg 237 chars)
✓ Embedded 4 chunks with gemini-embedding-001 (768 dims) in 1 batch(es)
✓ Stored 4 rows in document_chunks (0 already indexed, skipped)
✓ Done in 1.8s. Database: localhost:5432/doc_index
```

### Search

```text
$ python search.py --query "login issue" --top-k 3
Top 3 results for "login issue"

#1  score=0.698  example.pdf  [paragraph]  chunk 3
    Symptoms: the customer cannot log in to the personal area on the website or in the mobile app.
    Typical messages are "incorrect password", "user not found" or a reset code that never arrives.
    Steps: first verify that the mobile number on the account is …

#2  score=0.693  example.pdf  [fixed]  chunk 1
    login failure Symptoms: the customer cannot log in to the personal area on the website or in the
    mobile app. Typical messages are "incorrect password", "user not found" or a reset code that
    never arrives. Steps: first verify that the mobile number on the …

#3  score=0.690  example.pdf  [sentence]  chunk 2
    2. KB-112 Personal area: login failure Symptoms: the customer cannot log in to the personal area
    on the website or in the mobile app. Typical messages are "incorrect password", "user not found"
    or a reset code that never arrives. Steps: first verify that the …
```

All three strategies return the KB-112 login procedure first. The next query shares almost no words with
the document ("paid two times" vs. "charged twice") and still finds the double-charge procedure:

```text
$ python search.py --query "the customer says he paid two times this month" --top-k 2
Top 2 results for "the customer says he paid two times this month"

#1  score=0.752  example.pdf  [sentence]  chunk 6
    6. KB-305 Billing: double charge and refund requests When a customer reports being charged
    twice, open the last two invoices and the payment history. A duplicate charge appears as two
    identical amounts on the same billing date. If a duplicate charge is …

#2  score=0.751  example.pdf  [paragraph]  chunk 13
    If a duplicate charge is confirmed, open a Billing ticket with priority Medium and attach both
    transaction references. The refund is issued by the Billing department to the original payment
    method within two billing cycles. Representatives may tell the …
```

A Hebrew question ("When may a technician be sent to a customer?") retrieves the Hebrew procedure:

```text
$ python search.py --query "מתי מותר לשלוח טכנאי ללקוח?" --top-k 2
Top 2 results for "מתי מותר לשלוח טכנאי ללקוח?"

#1  score=0.772  technician_visits_he.docx  [paragraph]  chunk 0
    נוהל ביקור טכנאי בבית הלקוח מסמך הדגמה פיקטיבי שנוצר עבור מטלת הבית של Jeen AI. אינו מסמך של
    חברה אמיתית. מתי מזמינים טכנאי מזמינים ביקור טכנאי רק אחרי שבוצעה בדיקת תקלות אזוריות, ובדיקת קו
    מרחוק נכשלה. אם קיימת תקלה פעילה באזור הלקוח, אין לתאם ביקור, ויש …

#2  score=0.748  technician_visits_he.docx  [paragraph]  chunk 1
    תיאום הביקור יש להציע ללקוח את שני חלונות הזמן הקרובים ביותר, בבוקר ובאחר הצהריים. הביקור נקבע
    רק אחרי שהלקוח אישר את הכתובת ואת מספר הטלפון ליצירת קשר. ביום הביקור נשלחת ללקוח הודעת SMS עם
    שעת הגעה משוערת.
```

Comparing strategies on the same query:

```text
$ python search.py --query "login issue" --top-k 1 --strategy fixed
#1  score=0.693  example.pdf  [fixed]  chunk 1

$ python search.py --query "login issue" --top-k 1 --strategy paragraph
#1  score=0.698  example.pdf  [paragraph]  chunk 3
```

### Rows stored in the database

```text
$ docker exec -it jeen-doc-index-db psql -U jeen -d doc_index -c "SELECT id, filename, split_strategy, chunk_index, left(chunk_text, 60) AS preview, created_at FROM document_chunks ORDER BY id LIMIT 5;"
 id |  filename   | split_strategy | chunk_index |                           preview                            |          created_at
----+-------------+----------------+-------------+--------------------------------------------------------------+-------------------------------
  1 | example.pdf | fixed          |           0 | Contact Center Knowledge Base Fictional telecom operator · V | 2026-10-08 23:15:48.187185+00
  2 | example.pdf | fixed          |           1 | login failure Symptoms: the customer cannot log in to the pe | 2026-10-08 23:15:48.187185+00
  3 | example.pdf | fixed          |           2 | account is locked automatically after five failed login atte | 2026-10-08 23:15:48.187185+00
  4 | example.pdf | fixed          |           3 | phone line or TV), check the outage map for the customer's c | 2026-10-08 23:15:48.187185+00
  5 | example.pdf | fixed          |           4 | light usually indicates a line problem; normal lights with o | 2026-10-08 23:15:48.187185+00
(5 rows)
```

### Error handling in practice

```text
$ python index_documents.py --file ./docs/missing.pdf --strategy fixed
✗ Error: File not found: ./docs/missing.pdf

$ python index_documents.py --file ./requirements.txt --strategy fixed
✗ Error: Unsupported file type '.txt'. Supported: .pdf, .docx
```

### Dry run (no API key needed)

```text
$ python index_documents.py --file ./docs/example.pdf --strategy all --dry-run
✓ Extracted 6,730 characters from example.pdf (3 pages)
✓ Split into 9 chunks (strategy=fixed, target<=1000 chars, overlap=200, avg 917 chars)
✓ Split into 22 chunks (strategy=sentence, target<=800 chars, avg 303 chars)
✓ Split into 20 chunks (strategy=paragraph, target<=1500 chars, avg 333 chars)
    [fixed #0] Contact Center Knowledge Base Fictional telecom operator · Version 3.2 · Demo document created for the Jeen AI home assignment. Not a real c…
    [fixed #1] login failure Symptoms: the customer cannot log in to the personal area on the website or in the mobile app. Typical messages are "incorrect…
    [fixed #2] account is locked automatically after five failed login attempts within 15 minutes, or manually after a fraud alert. A locked account cannot…
    [sentence #0] Contact Center Knowledge Base Fictional telecom operator · Version 3.2 · Demo document created for the Jeen AI home assignment. Not a real c…
    [sentence #1] 1. Purpose and scope This knowledge base supports first-line representatives in the contact center of a fictional national telecom operator.…
    [sentence #2] 2. KB-112 Personal area: login failure Symptoms: the customer cannot log in to the personal area on the website or in the mobile app. Typica…
    [paragraph #0] Contact Center Knowledge Base Fictional telecom operator · Version 3.2 · Demo document created for the Jeen AI home assignment. Not a real c…
    [paragraph #1] 1. Purpose and scope This knowledge base supports first-line representatives in the contact center of a fictional national telecom operator.…
    [paragraph #2] Representatives must always verify the customer's identity before discussing account details, and must never promise refunds, credits or dat…
✓ Dry run: nothing was embedded or stored
```

## Project structure

```text
jeen-doc-indexer/
├── index_documents.py        # CLI: extract → chunk → embed → store
├── search.py                 # CLI: semantic search
├── doc_indexer/
│   ├── config.py             # .env loading and validation
│   ├── errors.py             # exceptions with exit codes
│   ├── extract.py            # PDF/DOCX extraction, cleaning, paragraph rebuilding
│   ├── chunking.py           # fixed / sentence / paragraph strategies
│   ├── embeddings.py         # Gemini client: batching, retries, normalization
│   ├── db.py                 # schema, inserts, similarity search
│   └── cli.py                # logging and clean error output
├── sql/schema.sql            # documented schema
├── tests/                    # pytest: chunking, extraction, embeddings (no API or DB)
├── docs/
│   ├── example.pdf                   # fictional contact-center knowledge base (English)
│   └── technician_visits_he.docx     # fictional procedure in Hebrew, with a table
├── docker-compose.yml        # PostgreSQL 17 + pgvector
├── .env.example              # variable names, no secrets
├── requirements.txt
└── pytest.ini
```

## Tests

```bash
pytest
```

43 tests cover the three chunking strategies (sizes, overlap, word boundaries, Hebrew sentences,
abbreviations, headings), text cleaning, extraction of both sample files and every file-error path, and the
embedding client (normalization, batching, task types, retry and give-up behaviour, certificate errors) using a fake API client.
They need neither an API key nor a database.

---

The sample documents are fictional and were written for this assignment; they do not describe any real company.