"""
Tool 1: search_documents
Semantic search over document_chunks in Supabase (pgvector).
Embeds the query via HuggingFace Inference API, calls the search_document_chunks
SQL function, re-ranks by authority_level, detects conflicts.
"""

import os
import time
import httpx
from langchain_core.tools import tool
from langsmith import traceable
from supabase import create_client
from dotenv import load_dotenv

from pathlib import Path as _Path
load_dotenv(_Path(__file__).parent.parent / ".env" if (_Path(__file__).parent.parent / ".env").exists() else _Path(__file__).parent.parent.parent / ".env", override=True)

SUPABASE_URL  = os.environ["SUPABASE_URL"]
SUPABASE_KEY  = os.environ["SUPABASE_SERVICE_ROLE_KEY"]   # service role needed for search RPC
HF_TOKEN      = os.environ["HF_TOKEN"]
HF_EMBED_URL  = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5"
TOP_K         = int(os.getenv("RETRIEVAL_TOP_K", "5"))

sb = create_client(SUPABASE_URL, SUPABASE_KEY)


def _embed(text: str) -> list[float]:
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    for attempt in range(3):
        resp = httpx.post(HF_EMBED_URL, headers=headers,
                          json={"inputs": text, "options": {"wait_for_model": True}},
                          timeout=30)
        if resp.status_code == 200:
            result = resp.json()
            # HF returns list of embeddings for batch; single text â†' first element
            return result[0] if isinstance(result[0], list) else result
        if resp.status_code == 503:
            time.sleep(15)
        else:
            resp.raise_for_status()
    raise RuntimeError("HF embed failed")


def _detect_conflict(chunks: list[dict]) -> bool:
    """True if chunks from authority levels 1 and 2+ cover overlapping topics."""
    has_agreement = any(c["authority_level"] == 1 for c in chunks)
    has_lower     = any(c["authority_level"] > 1  for c in chunks)
    if not (has_agreement and has_lower):
        return False
    # Simple heuristic: any shared keyword between agreement chunk and policy chunk
    agreement_words = set()
    for c in chunks:
        if c["authority_level"] == 1:
            agreement_words.update(c["content"].lower().split())
    for c in chunks:
        if c["authority_level"] > 1:
            overlap = agreement_words & set(c["content"].lower().split())
            if len(overlap) > 5:   # more than 5 shared content words â†' likely conflict
                return True
    return False


@traceable(name="Tool: Document Search", run_type="retriever")
def search_documents_fn(query: str, account_scope: str | None = None) -> dict:
    """
    Search document chunks relevant to `query`.

    Args:
        query:         Natural language question / topic.
        account_scope: If customer role, pass account_id to restrict agreement access.
                       Pass None for internal (all docs visible).

    Returns dict with:
        chunks:           list of matching chunks (sorted by authority_level asc, similarity desc)
        conflict_detected: bool
        sources:          list[SourceRef] for UI panel
    """
    embedding = _embed(query)

    result = sb.rpc("search_document_chunks", {
        "query_embedding": embedding,
        "p_account_scope": account_scope,
        "p_top_k":         TOP_K,
        "p_threshold":     0.30,
    }).execute()

    chunks = result.data or []

    # Re-rank: authority_level ascending (1 first), then similarity descending
    chunks.sort(key=lambda c: (c["authority_level"], -c.get("similarity", 0)))

    conflict = _detect_conflict(chunks)

    def _fmt(filename: str) -> str:
        import re
        name = re.sub(r"^[\d]+_", "", filename)
        name = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
        return name.replace("_", " ")

    sources = [
        {
            "source_file":     _fmt(c["source_file"]),
            "authority_level": c["authority_level"],
            "doc_type":        c.get("doc_type", ""),
            "page_num":        c.get("page_num", 0),
            "preview":         c["content"][:120],
        }
        for c in chunks
    ]

    return {
        "chunks":            chunks,
        "conflict_detected": conflict,
        "sources":           sources,
    }

