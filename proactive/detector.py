"""
Proactive issue detector - for Internal Ops mode.

Scans open tickets and orders at startup (or on demand) to surface:
  1. SLA breaches (open tickets past their response target)
  2. Failed-pickup orders stuck too long
  3. Ticket topic clusters (similar complaints recurring)
  4. Multi-customer patterns (same issue affecting different accounts)
"""

import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from supabase import create_client
from dotenv import load_dotenv

from pathlib import Path as _Path
load_dotenv(_Path(__file__).parent.parent / ".env" if (_Path(__file__).parent.parent / ".env").exists() else _Path(__file__).parent.parent.parent / ".env", override=True)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]   # service role -- bypasses RLS
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

SLA_DEFAULTS = {
    "Enterprise": {"P1": 0.5,  "P2": 2.0,  "P3": 24.0},
    "Growth":     {"P1": 2.0,  "P2": 4.0,  "P3": 48.0},
    "Standard":   {"P1": 4.0,  "P2": 24.0, "P3": 48.0},
}
SLA_OVERRIDES = {
    "ACCT-001": {"P1": 0.25, "P2": 1.0,  "P3": 8.0},
    "ACCT-002": {"P1": 2.0,  "P2": 4.0,  "P3": 48.0},
}

FAILED_PICKUP_THRESHOLD_HOURS = 6.0
CLUSTER_JACCARD_THRESHOLD     = 0.25  # min word-overlap for two tickets to be in same cluster
CLUSTER_LOOKBACK_DAYS         = 30    # include recent-closed tickets in cluster scan
MULTI_CUSTOMER_MIN_ACCOUNTS   = 2     # minimum distinct accounts for a multi-customer flag

_STOP_WORDS = {
    "the", "and", "for", "that", "this", "with", "from", "they", "have",
    "been", "when", "what", "about", "still", "after", "their", "also",
    "will", "which", "your", "into", "them", "does", "each", "would",
    "there", "more", "some", "were", "how", "all", "our", "has", "but",
    "not", "are", "can", "any", "its", "was", "you", "out", "who", "got",
    "did", "just", "very", "want", "asks", "using", "while", "possible",
    "shows", "tells", "asked", "change", "please",
}


def _elapsed_hours(dt_str: str | None) -> float | None:
    if not dt_str:
        return None
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ref = datetime.now(timezone.utc)
    return max(0.0, (ref - dt).total_seconds() / 3600)


def _keywords(text: str) -> set[str]:
    """Extract significant lowercase words (4+ chars, non-stop-word) from text."""
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    return {w for w in words if w not in _STOP_WORDS}


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _detect_clusters(tickets: list[dict]) -> list[dict]:
    """
    Cluster tickets by subject/description similarity (Jaccard on keywords).
    Returns list of cluster dicts, each with a representative topic label and
    a flag indicating whether the cluster spans multiple accounts.
    """
    # Build keyword sets for each ticket
    ticket_kw = []
    for t in tickets:
        # Use subject only - descriptions add noise that lowers Jaccard below threshold
        text = t.get("subject", "") or t.get("issue_summary", "")
        ticket_kw.append((t, _keywords(text)))

    # Union-Find for grouping similar tickets
    n = len(ticket_kw)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if _jaccard(ticket_kw[i][1], ticket_kw[j][1]) >= CLUSTER_JACCARD_THRESHOLD:
                union(i, j)

    # Group by cluster root
    groups: dict[int, list] = defaultdict(list)
    for i, (ticket, kws) in enumerate(ticket_kw):
        groups[find(i)].append((ticket, kws))

    clusters = []
    for root, members in groups.items():
        if len(members) < 2:
            continue

        # Derive topic label from most common shared keywords
        all_kw: list[set] = [kws for _, kws in members]
        shared = all_kw[0].copy()
        for kws in all_kw[1:]:
            shared &= kws
        # Fall back to union top-frequency if no shared words
        if not shared:
            freq: dict[str, int] = defaultdict(int)
            for kws in all_kw:
                for w in kws:
                    freq[w] += 1
            shared = {w for w, c in freq.items() if c >= 2}
        topic = " / ".join(sorted(shared)[:4]) if shared else "mixed topic"

        ticket_list = [t for t, _ in members]
        accounts = {t.get("account_id") for t in ticket_list}
        open_count = sum(1 for t in ticket_list if t.get("status") == "open")

        clusters.append({
            "topic":          topic,
            "ticket_count":   len(ticket_list),
            "open_count":     open_count,
            "account_count":  len(accounts),
            "is_multi_customer": len(accounts) >= MULTI_CUSTOMER_MIN_ACCOUNTS,
            "tickets": [
                {
                    "ticket_id":  t["ticket_id"],
                    "account_id": t.get("account_id"),
                    "status":     t.get("status"),
                    "subject":    t.get("subject", ""),
                }
                for t in ticket_list
            ],
        })

    # Multi-customer first, then by ticket count
    clusters.sort(key=lambda c: (-int(c["is_multi_customer"]), -c["ticket_count"]))
    return clusters


def detect_issues() -> dict:
    """
    Full proactive scan. Returns:
        {
          sla_breaches:     list[dict],
          failed_pickups:   list[dict],
          ticket_clusters:  list[dict],
          summary:          str,
        }
    """
    # Fetch all accounts for plan lookup
    accounts = {
        a["account_id"]: a
        for a in (sb.table("accounts").select("*").execute().data or [])
    }

    # -- SLA breach scan -------------------------------------------------------
    open_tickets = sb.table("tickets").select("*").eq("status", "open").execute().data or []
    sla_breaches = []

    for ticket in open_tickets:
        acct_id  = ticket.get("account_id", "")
        acct     = accounts.get(acct_id, {})
        plan     = acct.get("plan", "Standard")
        severity = ticket.get("severity") or "P3"

        sla_map     = SLA_OVERRIDES.get(acct_id, SLA_DEFAULTS.get(plan, SLA_DEFAULTS["Standard"]))
        target_hrs  = sla_map.get(severity, 48.0)
        elapsed_hrs = _elapsed_hours(ticket.get("created_at"))

        if elapsed_hrs is not None and elapsed_hrs > target_hrs:
            sla_breaches.append({
                "ticket_id":      ticket["ticket_id"],
                "account_id":     acct_id,
                "account_name":   acct.get("account_name", "Unknown"),
                "severity":       severity,
                "sla_target_hrs": target_hrs,
                "elapsed_hrs":    round(elapsed_hrs, 1),
                "overage_hrs":    round(elapsed_hrs - target_hrs, 1),
                "issue":          ticket.get("subject") or ticket.get("issue_summary", ""),
            })

    severity_order = {"P1": 0, "P2": 1, "P3": 2}
    sla_breaches.sort(key=lambda x: (severity_order.get(x["severity"], 9), -x["overage_hrs"]))

    # -- Failed-pickup scan ----------------------------------------------------
    stuck_orders = (
        sb.table("orders")
          .select("*")
          .in_("status", ["failed_pickup", "PICKUP_FAILED"])
          .execute()
          .data or []
    )
    failed_pickups = []

    for order in stuck_orders:
        hours_stuck = _elapsed_hours(order.get("created_at"))
        if hours_stuck and hours_stuck > FAILED_PICKUP_THRESHOLD_HOURS:
            acct_id = order.get("account_id", "")
            acct    = accounts.get(acct_id, {})
            failed_pickups.append({
                "order_id":     order["order_id"],
                "account_id":   acct_id,
                "account_name": acct.get("account_name", "Unknown"),
                "carrier":      order.get("carrier", "Unknown"),
                "hours_stuck":  round(hours_stuck, 1),
                "status":       order.get("status"),
            })

    failed_pickups.sort(key=lambda x: -x["hours_stuck"])

    # -- Ticket clustering & multi-customer detection --------------------------
    # Include open tickets + recently closed (within CLUSTER_LOOKBACK_DAYS)
    lookback_cutoff = datetime.now(timezone.utc).timestamp() - CLUSTER_LOOKBACK_DAYS * 86400
    all_tickets_raw = sb.table("tickets").select("*").execute().data or []
    cluster_tickets = [
        t for t in all_tickets_raw
        if (
            t.get("status") == "open"
            or (
                t.get("created_at")
                and datetime.fromisoformat(t["created_at"]).replace(tzinfo=timezone.utc).timestamp() >= lookback_cutoff
            )
        )
    ]

    ticket_clusters = _detect_clusters(cluster_tickets)

    # Separate multi-customer from single-customer clusters
    multi_customer = [c for c in ticket_clusters if c["is_multi_customer"]]
    same_customer  = [c for c in ticket_clusters if not c["is_multi_customer"]]

    # -- Format summary --------------------------------------------------------
    lines = ["## Proactive Issue Scan\n"]

    if sla_breaches:
        lines.append(f"### [SLA BREACH] SLA Breaches ({len(sla_breaches)} open tickets)\n")
        for b in sla_breaches:
            lines.append(
                f"- **{b['ticket_id']}** | {b['account_name']} | {b['severity']} | "
                f"Elapsed: {b['elapsed_hrs']}h (target {b['sla_target_hrs']}h) | +{b['overage_hrs']}h overdue\n"
                f"  Issue: {b['issue'][:80]}"
            )
    else:
        lines.append(". No SLA breaches detected.\n")

    if failed_pickups:
        lines.append(f"\n### [WARNING] Stuck Failed-Pickup Orders ({len(failed_pickups)} orders)\n")
        for fp in failed_pickups:
            lines.append(
                f"- **{fp['order_id']}** | {fp['account_name']} | Carrier: {fp['carrier']} | "
                f"Stuck {fp['hours_stuck']}h"
            )
    else:
        lines.append(". No orders stuck in failed-pickup state.\n")

    if multi_customer:
        lines.append(f"\n### [PATTERN] Multi-Customer Issue Patterns ({len(multi_customer)} patterns)\n")
        for c in multi_customer:
            ticket_ids = ", ".join(t["ticket_id"] for t in c["tickets"])
            acct_ids   = ", ".join({t["account_id"] for t in c["tickets"]})
            lines.append(
                f"- **Topic: {c['topic']}** | {c['ticket_count']} tickets across "
                f"{c['account_count']} accounts ({acct_ids})\n"
                f"  Tickets: {ticket_ids}"
            )
    else:
        lines.append(". No multi-customer patterns detected.\n")

    if same_customer:
        lines.append(f"\n### [CLUSTER] Recurring Single-Account Issues ({len(same_customer)} clusters)\n")
        for c in same_customer:
            ticket_ids = ", ".join(t["ticket_id"] for t in c["tickets"])
            acct_id    = next(iter({t["account_id"] for t in c["tickets"]}), "?")
            acct_name  = accounts.get(acct_id, {}).get("account_name", acct_id)
            open_label = f"{c['open_count']} open" if c["open_count"] else "all closed"
            lines.append(
                f"- **Topic: {c['topic']}** | {acct_name} | {c['ticket_count']} tickets ({open_label})\n"
                f"  Tickets: {ticket_ids}"
            )
    else:
        lines.append(". No recurring single-account clusters detected.\n")

    return {
        "sla_breaches":    sla_breaches,
        "failed_pickups":  failed_pickups,
        "ticket_clusters": ticket_clusters,
        "summary":         "\n".join(lines),
    }


if __name__ == "__main__":
    result = detect_issues()
    print(result["summary"])
