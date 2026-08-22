-- ============================================================
-- ParcelPilot AI Agent - Supabase Setup
-- Run this in: Supabase Dashboard → SQL Editor
-- ============================================================

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;


-- ============================================================
-- 2. STRUCTURED DATA TABLES
-- ============================================================

CREATE TABLE IF NOT EXISTS accounts (
    account_id      TEXT PRIMARY KEY,
    account_name    TEXT NOT NULL,
    plan            TEXT,           -- Standard | Growth | Enterprise
    status          TEXT,           -- active | inactive
    csm             TEXT,
    contract_file   TEXT,           -- filename of agreement PDF, if any
    premium_support BOOLEAN DEFAULT FALSE,
    notes           TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS orders (
    order_id                  TEXT PRIMARY KEY,
    account_id                TEXT REFERENCES accounts(account_id),
    carrier                   TEXT,
    status                    TEXT,  -- DRAFT | BOOKED | PICKED_UP | DELIVERED
    booked_at                 TIMESTAMPTZ,
    pickup_window_start       TIMESTAMPTZ,
    pickup_window_end         TIMESTAMPTZ,
    pickup_actual_at          TIMESTAMPTZ,
    shipment_fee_inr          NUMERIC(10,2),
    carrier_fault             BOOLEAN DEFAULT FALSE,
    customer_fault            BOOLEAN DEFAULT FALSE,
    cancellation_requested_at TIMESTAMPTZ,
    notes                     TEXT,
    created_at                TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id                TEXT PRIMARY KEY,
    account_id               TEXT REFERENCES accounts(account_id),
    created_at               TIMESTAMPTZ,
    status                   TEXT,  -- open | closed
    subject                  TEXT,
    description              TEXT,
    channel                  TEXT,  -- email | chat
    assigned_to              TEXT,
    last_customer_message_at TIMESTAMPTZ,
    historical_resolution    TEXT,  -- CONTEXT ONLY - may be incorrect
    severity                 TEXT   -- P1 | P2 | P3 (computed at ingestion)
);


-- ============================================================
-- 3. VECTOR STORE TABLE (replaces ChromaDB)
-- ============================================================

CREATE TABLE IF NOT EXISTS document_chunks (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    source_file     TEXT NOT NULL,
    authority_level INT  NOT NULL CHECK (authority_level BETWEEN 1 AND 4),
    -- 1 = customer agreement, 2 = current policy, 3 = SOP/product doc, 4 = historical tickets
    is_deprecated   BOOLEAN DEFAULT FALSE,
    account_scope   TEXT,   -- NULL = general doc; 'ACCT-001' = specific to that account
    doc_type        TEXT,   -- agreement | policy | sop | product_doc
    page_num        INT,
    chunk_index     INT,
    content         TEXT NOT NULL,
    embedding       VECTOR(384),  -- BAAI/bge-small-en-v1.5 dimensions
    ingested_at     TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast vector search
CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding
    ON document_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);

-- Index for metadata filtering
CREATE INDEX IF NOT EXISTS idx_doc_chunks_meta
    ON document_chunks (authority_level, is_deprecated, account_scope);


-- ============================================================
-- 4. VECTOR SEARCH FUNCTION
-- ============================================================

CREATE OR REPLACE FUNCTION search_document_chunks(
    query_embedding   VECTOR(384),
    p_account_scope   TEXT    DEFAULT NULL,    -- NULL = no customer filter (internal)
    p_top_k           INT     DEFAULT 5,
    p_threshold       FLOAT   DEFAULT 0.35
)
RETURNS TABLE (
    id              UUID,
    content         TEXT,
    source_file     TEXT,
    authority_level INT,
    is_deprecated   BOOLEAN,
    account_scope   TEXT,
    doc_type        TEXT,
    page_num        INT,
    similarity      FLOAT
)
LANGUAGE SQL STABLE
AS $$
    SELECT
        id, content, source_file, authority_level, is_deprecated,
        account_scope, doc_type, page_num,
        1 - (embedding <=> query_embedding) AS similarity
    FROM document_chunks
    WHERE
        is_deprecated = FALSE
        -- Customer scope: general docs (NULL scope) OR their own agreement
        AND (p_account_scope IS NULL
             OR account_scope IS NULL
             OR account_scope = p_account_scope)
        AND 1 - (embedding <=> query_embedding) > p_threshold
    ORDER BY embedding <=> query_embedding
    LIMIT p_top_k;
$$;


-- ============================================================
-- 5. ROW LEVEL SECURITY
-- ============================================================

-- Orders
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;

CREATE POLICY orders_customer ON orders
    FOR SELECT
    USING (
        current_setting('app.role', true) = 'internal'
        OR account_id = current_setting('app.account_id', true)
    );

-- Tickets
ALTER TABLE tickets ENABLE ROW LEVEL SECURITY;

CREATE POLICY tickets_customer ON tickets
    FOR SELECT
    USING (
        current_setting('app.role', true) = 'internal'
        OR account_id = current_setting('app.account_id', true)
    );

-- Accounts: customers can only see their own row
ALTER TABLE accounts ENABLE ROW LEVEL SECURITY;

CREATE POLICY accounts_customer ON accounts
    FOR SELECT
    USING (
        current_setting('app.role', true) = 'internal'
        OR account_id = current_setting('app.account_id', true)
    );

-- Document chunks: no RLS (filtered in search function by account_scope)
-- Service role key bypasses RLS for ingestion


-- ============================================================
-- 6. HELPER: set session role/account (called before every query)
-- ============================================================

CREATE OR REPLACE FUNCTION set_session_context(p_role TEXT, p_account_id TEXT DEFAULT '')
RETURNS VOID LANGUAGE plpgsql AS $$
BEGIN
    PERFORM set_config('app.role',       p_role,       true);
    PERFORM set_config('app.account_id', p_account_id, true);
END;
$$;


-- ============================================================
-- 7. ACTIONS LOG (state-changing action audit trail)
-- ============================================================

CREATE TABLE IF NOT EXISTS actions_log (
    id          BIGSERIAL PRIMARY KEY,
    action_id   TEXT NOT NULL,
    action_type TEXT NOT NULL,           -- escalate_ticket | update_ticket_status | create_followup_task
    payload     JSONB,
    actor_name  TEXT,
    actor_role  TEXT,
    summary     TEXT,
    executed_at TIMESTAMPTZ DEFAULT NOW(),
    status      TEXT DEFAULT 'completed'
);
