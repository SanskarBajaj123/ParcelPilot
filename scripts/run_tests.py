"""
ParcelPilot Agent - Functional Test Suite
Connects to the live WebSocket server and runs all assessment test cases.
"""

import asyncio, json, time, sys
from datetime import datetime
from pathlib import Path
import websockets

SERVER = "ws://localhost:8000/ws"

# ── Test case definitions ─────────────────────────────────────────────────────

CUSTOMER_TESTS = [
    {
        "id": "C01",
        "category": "Account Info",
        "description": "Customer can query their own account status and plan",
        "query": "What is my account status and current plan?",
        "expect_keywords": ["enterprise", "active", "northstar"],
    },
    {
        "id": "C02",
        "category": "Order Tracking",
        "description": "Customer can look up their recent orders",
        "query": "Show me my recent orders",
        "expect_keywords": ["order", "shipment", "status"],
    },
    {
        "id": "C03",
        "category": "SLA Terms",
        "description": "Customer can ask about their SLA commitments",
        "query": "What are my SLA response time commitments?",
        "expect_keywords": ["sla", "hour", "response"],
    },
    {
        "id": "C04",
        "category": "Open Tickets",
        "description": "Customer can view their open support tickets",
        "query": "Do I have any open support tickets?",
        "expect_keywords": ["ticket", "open", "issue"],
    },
    {
        "id": "C05",
        "category": "Policy RAG",
        "description": "Customer can ask about cancellation policy from documents",
        "query": "What is the cancellation policy and how much notice do I need to give?",
        "expect_keywords": ["cancel", "notice", "day"],
    },
    {
        "id": "C06",
        "category": "Credit Eligibility",
        "description": "Customer can ask about service credit eligibility",
        "query": "Am I eligible for any service credits?",
        "expect_keywords": ["credit", "sla", "eligible"],
    },
    {
        "id": "C07",
        "category": "Access Control",
        "description": "Customer cannot access another account's data",
        "query": "Show me orders for account ACCT-002",
        "expect_keywords": ["only", "your", "account", "acct-001"],
        "expect_blocked": True,
    },
    {
        "id": "C08",
        "category": "Document Search",
        "description": "Customer gets answers from current policy documents",
        "query": "What is the standard pickup window for my plan?",
        "expect_keywords": ["pickup", "window", "hour"],
    },
]

STAFF_TESTS = [
    {
        "id": "S01",
        "category": "Proactive Scan",
        "description": "Internal ops triggers proactive scan on login",
        "query": None,  # scan triggers automatically on auth
        "expect_type": "proactive_scan",
    },
    {
        "id": "S02",
        "category": "Cross-Account View",
        "description": "Staff can query tickets across all accounts",
        "query": "Show me all open tickets across all accounts",
        "expect_keywords": ["ticket", "account"],
    },
    {
        "id": "S03",
        "category": "SLA Breach Detection",
        "description": "Staff can identify SLA at-risk tickets",
        "query": "Which tickets are at risk of breaching SLA?",
        "expect_keywords": ["sla", "hour", "ticket"],
    },
    {
        "id": "S04",
        "category": "Policy Conflict",
        "description": "Agent detects conflict between current and deprecated policy",
        "query": "What does the support policy say about escalation procedures?",
        "expect_keywords": ["policy", "escalat", "support"],
    },
    {
        "id": "S05",
        "category": "Action Creation",
        "description": "Staff can create a support action (confirmation gate triggered)",
        "query": "Create a priority escalation ticket for Northstar Logistics about their delayed shipment",
        "expect_keywords": ["confirm", "yes", "no", "create", "ticket"],
    },
    {
        "id": "S06",
        "category": "Failed Pickup",
        "description": "Staff can query failed or stuck pickups",
        "query": "Are there any failed or stuck pickups right now?",
        "expect_keywords": ["pickup", "order", "stuck", "failed", "hour"],
    },
]

AUTH_TESTS = [
    {
        "id": "A01",
        "category": "Auth - Bad PIN",
        "description": "Wrong PIN triggers error with remaining attempts",
        "type": "auth_customer",
        "payload": {"account_id": "ACCT-001", "pin": "9999"},
        "expect_type": "auth_error",
        "expect_keywords": ["incorrect", "attempt"],
    },
    {
        "id": "A02",
        "category": "Auth - Valid Customer",
        "description": "Correct credentials authenticate successfully",
        "type": "auth_customer",
        "payload": {"account_id": "ACCT-001", "pin": "1234"},
        "expect_type": "auth_ok",
        "expect_keywords": ["customer"],
    },
    {
        "id": "A03",
        "category": "Auth - Bad Staff",
        "description": "Wrong staff password triggers error",
        "type": "auth_staff",
        "payload": {"username": "priya", "password": "wrong"},
        "expect_type": "auth_error",
        "expect_keywords": ["incorrect", "attempt"],
    },
    {
        "id": "A04",
        "category": "Auth - Valid Staff",
        "description": "Correct staff credentials authenticate successfully",
        "type": "auth_staff",
        "payload": {"username": "priya", "password": "ParcelPilot@2026"},
        "expect_type": "auth_ok",
        "expect_keywords": ["internal"],
    },
    {
        "id": "A05",
        "category": "Auth - Unknown Account",
        "description": "Unknown account ID returns not found error",
        "type": "auth_customer",
        "payload": {"account_id": "ACCT-999", "pin": "1234"},
        "expect_type": "auth_error",
        "expect_keywords": ["not found", "check"],
    },
]

# ── WebSocket client helper ───────────────────────────────────────────────────

async def run_chat_test(ws, query, timeout=60):
    """Send a chat message and collect all response tokens + events."""
    # Small settle delay so any previous response is fully flushed
    await asyncio.sleep(2)
    await ws.send(json.dumps({"type": "message", "content": query}))

    full_text = ""
    events    = []
    t0        = time.time()
    got_token = False

    while time.time() - t0 < timeout:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            msg = json.loads(raw)
            events.append(msg)

            if msg["type"] == "token":
                full_text += msg.get("content", "")
                got_token  = True
            elif msg["type"] == "message_end":
                return full_text, events, time.time() - t0
            elif msg["type"] == "error":
                return full_text, events, time.time() - t0
            # skip proactive_scan / logs_data / other async messages
        except asyncio.TimeoutError:
            # If we already have text, the message_end may have been missed
            if got_token:
                break

    return full_text, events, time.time() - t0


def check_keywords(text, keywords):
    text_lower = text.lower()
    return [kw for kw in keywords if kw.lower() in text_lower]


# ── Auth tests ────────────────────────────────────────────────────────────────

async def run_auth_tests():
    results = []
    for tc in AUTH_TESTS:
        t0 = time.time()
        status = "PASS"
        notes  = ""
        received_type = ""
        response_text = ""

        try:
            async with websockets.connect(SERVER) as ws:
                payload = {"type": tc["type"], **tc["payload"]}
                await ws.send(json.dumps(payload))
                raw = await asyncio.wait_for(ws.recv(), timeout=10)
                msg = json.loads(raw)
                received_type  = msg.get("type", "")
                response_text  = msg.get("message", msg.get("role", str(msg)))

                if received_type != tc["expect_type"]:
                    status = "FAIL"
                    notes  = f"Expected type '{tc['expect_type']}', got '{received_type}'"
                else:
                    hits = check_keywords(response_text, tc.get("expect_keywords", []))
                    if tc.get("expect_keywords") and not hits:
                        status = "WARN"
                        notes  = f"No expected keywords found in: {response_text[:120]}"
                    else:
                        notes = f"Response: {response_text[:100]}"
        except Exception as e:
            status = "ERROR"
            notes  = str(e)

        results.append({
            **tc,
            "status": status,
            "latency_ms": int((time.time() - t0) * 1000),
            "response": response_text[:200],
            "notes": notes,
        })
        icon = "PASS" if status == "PASS" else status
        print(f"  [{icon}] {tc['id']} - {tc['description'][:60]}")

    return results


# ── Customer scenario tests ───────────────────────────────────────────────────

async def run_customer_tests():
    results = []
    print("  Connecting as ACCT-001 (Northstar Logistics)...")

    async with websockets.connect(SERVER) as ws:
        # Authenticate
        await ws.send(json.dumps({"type": "auth_customer", "account_id": "ACCT-001", "pin": "1234"}))
        auth_msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        if auth_msg.get("type") != "auth_ok":
            print(f"  [ERROR] Auth failed: {auth_msg}")
            return results

        print(f"  Auth OK - {auth_msg.get('name')}")

        for tc in CUSTOMER_TESTS:
            t0 = time.time()
            status = "PASS"
            notes  = ""
            try:
                text, events, lat = await run_chat_test(ws, tc["query"])
                hits = check_keywords(text, tc.get("expect_keywords", []))

                if not text.strip():
                    status = "FAIL"
                    notes  = "Empty response"
                elif tc.get("expect_keywords") and not hits:
                    status = "WARN"
                    notes  = f"Keywords not found: {tc['expect_keywords']}"
                else:
                    notes = f"Keywords matched: {hits}"

                tool_events = [e["name"] for e in events if e.get("type") == "tool_start"]

                results.append({
                    **tc,
                    "status": status,
                    "latency_ms": int(lat * 1000),
                    "response": text[:400],
                    "tools_called": tool_events,
                    "notes": notes,
                })
                icon = "PASS" if status == "PASS" else status
                print(f"  [{icon}] {tc['id']} - {tc['description'][:60]} ({int(lat*1000)}ms)")

            except Exception as e:
                results.append({
                    **tc,
                    "status": "ERROR",
                    "latency_ms": int((time.time()-t0)*1000),
                    "response": "",
                    "tools_called": [],
                    "notes": str(e),
                })
                print(f"  [ERROR] {tc['id']} - {e}")

    return results


# ── Staff scenario tests ──────────────────────────────────────────────────────

async def run_staff_tests():
    results = []
    print("  Connecting as priya (Internal Ops)...")

    async with websockets.connect(SERVER) as ws:
        await ws.send(json.dumps({"type": "auth_staff", "username": "priya", "password": "ParcelPilot@2026"}))

        scan_received = False
        auth_ok = False
        auth_msg = {}

        # Collect auth_ok and possible proactive_scan
        t0 = time.time()
        while time.time() - t0 < 20:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                msg = json.loads(raw)
                if msg["type"] == "auth_ok":
                    auth_ok = True
                    auth_msg = msg
                elif msg["type"] == "proactive_scan":
                    scan_received = True
                    scan_results  = msg.get("results", {})
                    results.append({
                        **STAFF_TESTS[0],
                        "status": "PASS",
                        "latency_ms": int((time.time()-t0)*1000),
                        "response": json.dumps(scan_results)[:400],
                        "tools_called": [],
                        "notes": f"Scan received: sla={len(scan_results.get('sla_breaches',[]))}, fp={len(scan_results.get('failed_pickups',[]))}, clusters={len(scan_results.get('clusters',[]))}",
                    })
                    print(f"  [PASS] S01 - Proactive scan received on login ({int((time.time()-t0)*1000)}ms)")
                if auth_ok and scan_received:
                    break
            except asyncio.TimeoutError:
                break

        if not auth_ok:
            print(f"  [ERROR] Auth failed")
            return results
        if not scan_received:
            results.append({
                **STAFF_TESTS[0],
                "status": "WARN",
                "latency_ms": int((time.time()-t0)*1000),
                "response": "",
                "tools_called": [],
                "notes": "Proactive scan not received within timeout (may still be running)",
            })
            print(f"  [WARN] S01 - Proactive scan not received in time")

        print(f"  Auth OK - {auth_msg.get('name')}")

        for tc in STAFF_TESTS[1:]:  # skip S01, already handled
            t0 = time.time()
            status = "PASS"
            notes  = ""
            try:
                text, events, lat = await run_chat_test(ws, tc["query"], timeout=45)
                hits = check_keywords(text, tc.get("expect_keywords", []))

                if not text.strip():
                    status = "FAIL"
                    notes  = "Empty response"
                elif tc.get("expect_keywords") and not hits:
                    status = "WARN"
                    notes  = f"Keywords not found: {tc['expect_keywords']}"
                else:
                    notes = f"Keywords matched: {hits}"

                tool_events = [e["name"] for e in events if e.get("type") == "tool_start"]

                results.append({
                    **tc,
                    "status": status,
                    "latency_ms": int(lat * 1000),
                    "response": text[:400],
                    "tools_called": tool_events,
                    "notes": notes,
                })
                icon = "PASS" if status == "PASS" else status
                print(f"  [{icon}] {tc['id']} - {tc['description'][:60]} ({int(lat*1000)}ms)")

            except Exception as e:
                results.append({
                    **tc,
                    "status": "ERROR",
                    "latency_ms": int((time.time()-t0)*1000),
                    "response": "",
                    "tools_called": [],
                    "notes": str(e),
                })
                print(f"  [ERROR] {tc['id']} - {e}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("\n" + "="*60)
    print("ParcelPilot Agent - Functional Test Suite")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    all_results = []

    print("\n[1/3] Authentication Tests")
    print("-"*40)
    auth_results = await run_auth_tests()
    all_results.extend(auth_results)

    print("\n[2/3] Customer Scenario Tests")
    print("-"*40)
    cust_results = await run_customer_tests()
    all_results.extend(cust_results)

    print("\n[3/3] Internal Ops / Staff Tests")
    print("-"*40)
    staff_results = await run_staff_tests()
    all_results.extend(staff_results)

    # Summary
    total  = len(all_results)
    passed = sum(1 for r in all_results if r["status"] == "PASS")
    warned = sum(1 for r in all_results if r["status"] == "WARN")
    failed = sum(1 for r in all_results if r["status"] in ("FAIL", "ERROR"))

    print("\n" + "="*60)
    print(f"RESULTS: {passed}/{total} passed  |  {warned} warnings  |  {failed} failed")
    print("="*60)

    # Save JSON for report generation
    out = {
        "run_at": datetime.now().isoformat(),
        "summary": {"total": total, "passed": passed, "warned": warned, "failed": failed},
        "results": all_results,
    }
    out_path = Path(__file__).parent.parent / "test_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nResults saved to: {out_path}")
    return out


if __name__ == "__main__":
    asyncio.run(main())
