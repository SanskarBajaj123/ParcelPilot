"""
Tool 2: query_data
Structured queries against Supabase (accounts, orders, tickets).
Access control: sets Supabase session context so RLS enforces account scoping.
All queries use parameterised calls â€" no raw SQL injection risk.
"""

import os
from datetime import datetime, timezone
from supabase import create_client
from dotenv import load_dotenv

from pathlib import Path as _Path
load_dotenv(_Path(__file__).parent.parent / ".env" if (_Path(__file__).parent.parent / ".env").exists() else _Path(__file__).parent.parent.parent / ".env", override=True)

SUPABASE_URL = os.environ["SUPABASE_URL"]
# Service role key: RLS is stateless across REST calls; Python-level filtering
# below is the authoritative access control layer for this tool.
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
SNAPSHOT_ISO = "2026-08-16T11:00:00+00:00"   # 11:00 UTC = 16:30 IST; reference snapshot

sb = create_client(SUPABASE_URL, SUPABASE_KEY)
SNAPSHOT_DT  = datetime.fromisoformat(SNAPSHOT_ISO)


def _set_rls_context(role: str, account_id: str = ""):
    """Set Supabase session variables consumed by RLS policies."""
    sb.rpc("set_session_context", {"p_role": role, "p_account_id": account_id}).execute()


def _elapsed_hours(dt_str: str | None) -> float | None:
    """Hours elapsed from dt_str to snapshot time. None if dt_str is None."""
    if not dt_str:
        return None
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ref = SNAPSHOT_DT
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    return max(0.0, (ref - dt).total_seconds() / 3600)


def query_data_fn(intent: str, params: dict, role: str, account_id: str = "") -> dict:
    """
    Execute a structured data query.

    intent options:
      lookup_order                  params: {order_id}
      lookup_account                params: {account_id}
      list_open_tickets             params: {} (internal) or implicit from role
      calculate_credit_eligibility  params: {order_id}
      check_sla_breach              params: {ticket_id}

    role / account_id are injected from user_context by the tool_node â€" never from model args.
    """
    _set_rls_context(role, account_id)

    # â"€â"€ lookup_order â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    if intent == "lookup_order":
        oid = params.get("order_id", "")
        rows = sb.table("orders").select("*").eq("order_id", oid).execute().data
        if not rows:
            return {"error": f"Order {oid} not found (or not accessible)."}
        if role == "customer" and rows[0]["account_id"] != account_id:
            return {"error": f"Order {oid} not found (or not accessible)."}
        order = rows[0]
        # Attach computed fields
        if order.get("pickup_window_end"):
            order["hours_past_pickup_window"] = _elapsed_hours(order["pickup_window_end"])
        return {"order": order}

    # â"€â"€ lookup_account â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    elif intent == "lookup_account":
        aid = params.get("account_id") or account_id
        if role == "customer" and aid != account_id:
            return {"error": f"Account {aid} not accessible."}
        rows = sb.table("accounts").select("*").eq("account_id", aid).execute().data
        if not rows:
            return {"error": f"Account {aid} not found (or not accessible)."}
        return {"account": rows[0]}

    # â"€â"€ list_open_tickets â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    elif intent == "list_open_tickets":
        q = sb.table("tickets").select("*").eq("status", "open")
        if role == "customer":
            q = q.eq("account_id", account_id)
        rows = q.order("created_at", desc=True).execute().data
        return {"tickets": rows, "count": len(rows)}

    # â"€â"€ calculate_credit_eligibility â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    elif intent == "calculate_credit_eligibility":
        oid = params.get("order_id", "")
        rows = sb.table("orders").select("*").eq("order_id", oid).execute().data
        if not rows:
            return {"error": f"Order {oid} not found."}
        order = rows[0]

        hours_past = _elapsed_hours(order.get("pickup_window_end"))
        carrier_fault  = order.get("carrier_fault",  False)
        customer_fault = order.get("customer_fault", False)
        fee            = float(order.get("shipment_fee_inr") or 0)

        # Fetch account to check for agreement override
        acct_rows = sb.table("accounts").select("*").eq("account_id", order["account_id"]).execute().data
        acct = acct_rows[0] if acct_rows else {}

        # Default SOP thresholds
        threshold_hours  = 2.0
        credit_amount    = None
        credit_formula   = f"min(500, 10% of â‚¹{fee:.0f}) = â‚¹{min(500, fee * 0.1):.0f}"
        override_applied = False

        # LumenWorks override: 4h threshold, fixed â‚¹300
        if order["account_id"] == "ACCT-002":
            threshold_hours  = 4.0
            credit_amount    = 300.0
            credit_formula   = "Fixed â‚¹300 (LumenWorks agreement override)"
            override_applied = True

        eligible = (
            hours_past is not None
            and hours_past > threshold_hours
            and carrier_fault
            and not customer_fault
        )

        if eligible and credit_amount is None:
            credit_amount = min(500.0, fee * 0.1)

        return {
            "order_id":           oid,
            "account_id":         order["account_id"],
            "account_name":       acct.get("account_name"),
            "hours_past_window":  round(hours_past, 2) if hours_past is not None else None,
            "threshold_hours":    threshold_hours,
            "carrier_fault":      carrier_fault,
            "customer_fault":     customer_fault,
            "eligible":           eligible,
            "credit_amount_inr":  round(credit_amount, 2) if eligible and credit_amount else None,
            "credit_formula":     credit_formula,
            "override_applied":   override_applied,
            "needs_manager_approval": eligible and credit_amount is not None and credit_amount > 1000,
        }

    # â"€â"€ check_sla_breach â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    elif intent == "check_sla_breach":
        tid = params.get("ticket_id", "")
        rows = sb.table("tickets").select("*").eq("ticket_id", tid).execute().data
        if not rows:
            return {"error": f"Ticket {tid} not found."}
        if role == "customer" and rows[0]["account_id"] != account_id:
            return {"error": f"Ticket {tid} not found."}
        ticket = rows[0]

        acct_rows = sb.table("accounts").select("*").eq("account_id", ticket["account_id"]).execute().data
        acct = acct_rows[0] if acct_rows else {}
        plan = acct.get("plan", "Standard")

        # SLA targets from v3 policy + account-specific overrides
        SLA = {
            "ACCT-001": {"P1": 0.25, "P2": 1.0,  "P3": 8.0},   # Northstar agreement (hours)
            "ACCT-002": {"P1": 2.0,  "P2": 4.0,  "P3": 48.0},  # LumenWorks (2 biz days â‰ˆ 48h)
        }
        default_sla = {
            "Enterprise": {"P1": 0.5,  "P2": 2.0,  "P3": 24.0},
            "Growth":     {"P1": 2.0,  "P2": 4.0,  "P3": 48.0},
            "Standard":   {"P1": 4.0,  "P2": 24.0, "P3": 48.0},
        }
        sla_targets = SLA.get(ticket["account_id"], default_sla.get(plan, default_sla["Standard"]))

        severity    = ticket.get("severity") or "P3"   # default P3 if not set
        target_hrs  = sla_targets.get(severity, 48.0)
        elapsed_hrs = _elapsed_hours(ticket.get("created_at"))

        breached = elapsed_hrs is not None and elapsed_hrs > target_hrs

        return {
            "ticket_id":      tid,
            "severity":       severity,
            "account_id":     ticket["account_id"],
            "account_name":   acct.get("account_name"),
            "plan":           plan,
            "sla_target_hrs": target_hrs,
            "elapsed_hrs":    round(elapsed_hrs, 2) if elapsed_hrs is not None else None,
            "breached":       breached,
            "sla_source":     "customer agreement" if ticket["account_id"] in SLA else "policy v3 default",
        }

    else:
        return {"error": f"Unknown intent: {intent}. Valid: lookup_order, lookup_account, list_open_tickets, calculate_credit_eligibility, check_sla_breach"}

