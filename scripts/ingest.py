"""
Ingestion script — run once to populate Supabase with:
  1. PDF documents → chunked → embedded → document_chunks table
  2. Excel structured data → accounts / orders / tickets tables

Usage:
    python scripts/ingest.py

Requires SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, HF_TOKEN in .env
"""

import os
import sys
import time
import json
import math
import httpx
import fitz          # PyMuPDF
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL        = os.environ["SUPABASE_URL"]
SUPABASE_SVC_KEY    = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
HF_TOKEN            = os.environ["HF_TOKEN"]
HF_EMBED_MODEL      = "BAAI/bge-small-en-v1.5"
HF_EMBED_URL        = f"https://api-inference.huggingface.co/models/{HF_EMBED_MODEL}"

DATA_DIR  = Path(__file__).parent.parent / "data"
DOCS_DIR  = DATA_DIR / "docs"
XLSX_PATH = DATA_DIR / "ParcelPilot_Assessment_Data.xlsx"

CHUNK_SIZE    = 400   # tokens (approx chars / 4)
CHUNK_OVERLAP = 60

# Document metadata — authority and deprecation flags
DOC_META = {
    "01_Support_Policy_v3_CURRENT.pdf":                   {"authority_level": 2, "is_deprecated": False, "doc_type": "policy",      "account_scope": None},
    "02_Support_Policy_v2_DEPRECATED.pdf":                {"authority_level": 2, "is_deprecated": True,  "doc_type": "policy",      "account_scope": None},
    "03_Cancellation_and_Service_Credit_SOP_v4.pdf":      {"authority_level": 3, "is_deprecated": False, "doc_type": "sop",         "account_scope": None},
    "04_Product_Operations_Guide_and_Known_Issues.pdf":   {"authority_level": 3, "is_deprecated": False, "doc_type": "product_doc", "account_scope": None},
    "05_Northstar_Logistics_Enterprise_Agreement.pdf":    {"authority_level": 1, "is_deprecated": False, "doc_type": "agreement",   "account_scope": "ACCT-001"},
    "06_LumenWorks_Service_Agreement.pdf":                {"authority_level": 1, "is_deprecated": False, "doc_type": "agreement",   "account_scope": "ACCT-002"},
}

sb = create_client(SUPABASE_URL, SUPABASE_SVC_KEY)


# ── Embedding via HuggingFace Inference API ─────────────────────────────────

def embed_batch(texts: list[str]) -> list[list[float]]:
    """Call HF Inference API for a batch of texts. Returns list of embeddings."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = {"inputs": texts, "options": {"wait_for_model": True}}
    for attempt in range(3):
        resp = httpx.post(HF_EMBED_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 503:   # model loading
            print(f"  Model loading, waiting 20s… (attempt {attempt+1})")
            time.sleep(20)
        else:
            resp.raise_for_status()
    raise RuntimeError("HF Inference API failed after 3 attempts")


# ── PDF chunking ─────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by approximate token count."""
    words = text.split()
    chunks, i = [], 0
    step = max(1, chunk_size - overlap)
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += step
    return chunks


def ingest_pdfs():
    print("\n=== Ingesting PDFs ===")

    # Clear existing chunks
    sb.table("document_chunks").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
    print("  Cleared existing document_chunks.")

    all_rows = []
    for pdf_file, meta in DOC_META.items():
        pdf_path = DOCS_DIR / pdf_file
        if not pdf_path.exists():
            print(f"  SKIP (not found): {pdf_file}")
            continue

        doc = fitz.open(str(pdf_path))
        full_text_by_page = []
        for page_num, page in enumerate(doc, start=1):
            full_text_by_page.append((page_num, page.get_text("text")))
        doc.close()

        chunk_idx = 0
        for page_num, page_text in full_text_by_page:
            for chunk in chunk_text(page_text):
                all_rows.append({
                    "source_file":     pdf_file,
                    "authority_level": meta["authority_level"],
                    "is_deprecated":   meta["is_deprecated"],
                    "account_scope":   meta["account_scope"],
                    "doc_type":        meta["doc_type"],
                    "page_num":        page_num,
                    "chunk_index":     chunk_idx,
                    "content":         chunk,
                })
                chunk_idx += 1

        print(f"  Parsed: {pdf_file} → {chunk_idx} chunks")

    # Embed in batches of 20
    BATCH = 20
    print(f"\n  Embedding {len(all_rows)} chunks via HuggingFace ({HF_EMBED_MODEL})…")
    for i in range(0, len(all_rows), BATCH):
        batch = all_rows[i : i + BATCH]
        texts = [r["content"] for r in batch]
        embeddings = embed_batch(texts)
        for row, emb in zip(batch, embeddings):
            row["embedding"] = emb
        print(f"  Embedded {min(i + BATCH, len(all_rows))}/{len(all_rows)}")
        time.sleep(0.5)   # rate limit courtesy

    # Insert into Supabase
    print("  Inserting into Supabase…")
    for i in range(0, len(all_rows), 50):
        sb.table("document_chunks").insert(all_rows[i : i + 50]).execute()
    print(f"  Done — {len(all_rows)} chunks inserted.")


# ── Structured data ──────────────────────────────────────────────────────────

def ingest_structured():
    print("\n=== Ingesting Structured Data (xlsx) ===")

    xls = pd.ExcelFile(XLSX_PATH)

    # Accounts
    accounts_df = xls.parse("accounts").dropna(subset=["account_id"])
    accounts = []
    for _, row in accounts_df.iterrows():
        if not str(row.get("account_id", "")).startswith("ACCT"):
            continue
        accounts.append({
            "account_id":      str(row["account_id"]),
            "account_name":    str(row.get("account_name", "")),
            "plan":            str(row.get("plan", "")),
            "status":          str(row.get("status", "")),
            "csm":             str(row.get("csm", "")),
            "contract_file":   str(row.get("contract_file", "")) if pd.notna(row.get("contract_file")) else None,
            "premium_support": bool(row.get("premium_support", False)),
            "notes":           str(row.get("notes", "")) if pd.notna(row.get("notes")) else None,
        })
    sb.table("accounts").upsert(accounts).execute()
    print(f"  Accounts: {len(accounts)} rows")

    # Orders
    orders_df = xls.parse("orders").dropna(subset=["order_id"])
    orders = []
    for _, row in orders_df.iterrows():
        if not str(row.get("order_id", "")).startswith("ORD"):
            continue
        def ts(val):
            if pd.isna(val) or val == "": return None
            return pd.Timestamp(val).isoformat()
        orders.append({
            "order_id":                  str(row["order_id"]),
            "account_id":                str(row["account_id"]),
            "carrier":                   str(row.get("carrier", "")),
            "status":                    str(row.get("status", "")),
            "booked_at":                 ts(row.get("booked_at")),
            "pickup_window_start":       ts(row.get("pickup_window_start")),
            "pickup_window_end":         ts(row.get("pickup_window_end")),
            "pickup_actual_at":          ts(row.get("pickup_actual_at")),
            "shipment_fee_inr":          float(row["shipment_fee_inr"]) if pd.notna(row.get("shipment_fee_inr")) else None,
            "carrier_fault":             bool(row.get("carrier_fault", False)),
            "customer_fault":            bool(row.get("customer_fault", False)),
            "cancellation_requested_at": ts(row.get("cancellation_requested_at")),
            "notes":                     str(row.get("notes", "")) if pd.notna(row.get("notes")) else None,
        })
    sb.table("orders").upsert(orders).execute()
    print(f"  Orders: {len(orders)} rows")

    # Tickets
    tickets_df = xls.parse("tickets").dropna(subset=["ticket_id"])
    tickets = []
    for _, row in tickets_df.iterrows():
        if not str(row.get("ticket_id", "")).startswith("TKT"):
            continue
        def ts(val):
            if pd.isna(val) or val == "": return None
            return pd.Timestamp(val).isoformat()
        tickets.append({
            "ticket_id":                str(row["ticket_id"]),
            "account_id":               str(row["account_id"]),
            "created_at":               ts(row.get("created_at")),
            "status":                   str(row.get("status", "")),
            "subject":                  str(row.get("subject", "")),
            "description":              str(row.get("description", "")),
            "channel":                  str(row.get("channel", "")),
            "assigned_to":              str(row.get("assigned_to", "")),
            "last_customer_message_at": ts(row.get("last_customer_message_at")),
            "historical_resolution":    str(row.get("historical_resolution", "")) if pd.notna(row.get("historical_resolution")) else None,
            "severity":                 None,  # inferred by agent, not pre-coded
        })
    sb.table("tickets").upsert(tickets).execute()
    print(f"  Tickets: {len(tickets)} rows")


if __name__ == "__main__":
    ingest_pdfs()
    ingest_structured()
    print("\nIngestion complete.")
