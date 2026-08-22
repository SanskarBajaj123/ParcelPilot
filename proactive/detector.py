"""
Proactive issue detector — for Internal Ops mode.

Scans open tickets and orders at startup (or on demand) to surface
P1 escalations, SLA breaches, and orders stuck in failed-pickup state.
Returns a formatted alert summary for internal staff.
"""

import os
from datetime import datetime, timezone, timedelta
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]   # service role — bypasses RLS
SNAPSHOT_ISO = "2026-08-16T11:00:00+05:30"

sb = create_client(SUPABASE_URL, SUPABASE_KEY)
SNAPSHOT_DT = datetime.fromisoformat(SNAPSHOT_ISO)

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


def _elapsed_hours(dt_str: str | None) -> float | None:
    if not dt_str:
        return None
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ref = SNAPSHOT_DT
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    return max(0.0, (ref - dt).total_seconds() / 3600)


def detect_issues() -> dict:
    """
    Scan all open tickets and stuck orders.
    Returns:
        {
          sla_breaches: list[dict],
          failed_pickups: list[dict],
          summary: str,
        }
    """
    # Fetch all accounts for plan lookup
    accounts = {
        a["account_id"]: a
        for a in (sb.table("accounts").select("*").execute().data or [])
    }

    # ── SLA breach scan ───────────────────────────────────────────────────────
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
                "ticket_id":        ticket["ticket_id"],
                "account_id":       acct_id,
                "account_name":     acct.get("account_name", "Unknown"),
                "severity":         severity,
                "sla_target_hrs":   target_hrs,
                "elapsed_hrs":      round(elapsed_hrs, 1),
                "overage_hrs":      round(elapsed_hrs - target_hrs, 1),
                "issue":            ticket.get("issue_summary", ""),
            })

    # Sort by severity (P1 first), then by overage descending
    severity_order = {"P1": 0, "P2": 1, "P3": 2}
    sla_breaches.sort(key=lambda x: (severity_order.get(x["severity"], 9), -x["overage_hrs"]))

    # ── Failed-pickup scan ────────────────────────────────────────────────────
    stuck_orders = (
        sb.table("orders")
          .select("*")
          .in_("status", ["failed_pickup", "PICKUP_FAILED"])
          .execute()
          .data or []
    )
    failed_pickups = []

    for order in stuck_orders:
        hours_stuck = _elapsed_hours(order.get("updated_at") or order.get("created_at"))
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

    # ── Format summary ────────────────────────────────────────────────────────
    lines = ["## Proactive Issue Scan\n"]

    if sla_breaches:
        lines.append(f"### 🔴 SLA Breaches ({len(sla_breaches)} open tickets)\n")
        for b in sla_breaches:
            lines.append(
                f"- **{b['ticket_id']}** | {b['account_name']} | {b['severity']} | "
                f"Elapsed: {b['elapsed_hrs']}h (target {b['sla_target_hrs']}h) | +{b['overage_hrs']}h overdue\n"
                f"  Issue: {b['issue'][:80]}"
            )
    else:
        lines.append("✅ No SLA breaches detected.\n")

    if failed_pickups:
        lines.append(f"\n### ⚠️ Stuck Failed-Pickup Orders ({len(failed_pickups)} orders)\n")
        for fp in failed_pickups:
            lines.append(
                f"- **{fp['order_id']}** | {fp['account_name']} | Carrier: {fp['carrier']} | "
                f"Stuck {fp['hours_stuck']}h"
            )
    else:
        lines.append("\n✅ No orders stuck in failed-pickup state.\n")

    return {
        "sla_breaches":   sla_breaches,
        "failed_pickups": failed_pickups,
        "summary":        "\n".join(lines),
    }


if __name__ == "__main__":
    result = detect_issues()
    print(result["summary"])
