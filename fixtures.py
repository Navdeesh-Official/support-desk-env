"""Task fixtures for SupportDeskEnv.

Each fixture contains:
  - Public data visible to the agent (ticket thread, account info, tool results)
  - Hidden grading rubric (correct classification, required evidence, forbidden claims)
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Weight rubric (shared across tasks)
# ---------------------------------------------------------------------------
RUBRIC_WEIGHTS = {
    "evidence_retrieval": 0.20,
    "classification_priority": 0.20,
    "policy_decision": 0.25,
    "routing_escalation": 0.15,
    "reply_quality": 0.20,
}

# ---------------------------------------------------------------------------
# Tool result databases (simulated lookups)
# ---------------------------------------------------------------------------

ACCOUNT_DB = {
    "ACC-1042": {
        "account_id": "ACC-1042",
        "company": "NovaTech Solutions",
        "plan": "Business Pro",
        "status": "active",
        "users": [
            {
                "user_id": "USR-301",
                "name": "Sarah Chen",
                "email": "sarah.chen@novatech.io",
                "role": "admin",
                "mfa_enabled": True,
                "last_login": "2026-03-24T09:15:00Z",
            },
            {
                "user_id": "USR-302",
                "name": "James Wu",
                "email": "james.wu@novatech.io",
                "role": "member",
                "mfa_enabled": False,
                "last_login": "2026-03-20T14:30:00Z",
            },
        ],
        "contract_renewal_date": "2026-09-15",
        "support_tier": "priority",
    },
    "ACC-2087": {
        "account_id": "ACC-2087",
        "company": "Meridian Analytics",
        "plan": "Enterprise",
        "status": "active",
        "users": [
            {
                "user_id": "USR-510",
                "name": "Priya Sharma",
                "email": "priya@meridian.dev",
                "role": "admin",
                "mfa_enabled": True,
                "last_login": "2026-03-25T08:00:00Z",
            },
        ],
        "contract_renewal_date": "2026-12-01",
        "support_tier": "enterprise",
    },
    "ACC-3055": {
        "account_id": "ACC-3055",
        "company": "Greenleaf Digital",
        "plan": "Starter",
        "status": "active",
        "users": [
            {
                "user_id": "USR-720",
                "name": "Marco Rossi",
                "email": "marco@greenleaf.co",
                "role": "admin",
                "mfa_enabled": False,
                "last_login": "2026-03-22T11:45:00Z",
            },
        ],
        "contract_renewal_date": "2026-06-01",
        "support_tier": "standard",
    },
}

BILLING_DB = {
    "ACC-1042": {
        "invoices": [
            {
                "invoice_id": "INV-9001",
                "date": "2026-02-01",
                "amount": 499.00,
                "status": "paid",
                "description": "Business Pro - Monthly",
            },
            {
                "invoice_id": "INV-9002",
                "date": "2026-03-01",
                "amount": 499.00,
                "status": "paid",
                "description": "Business Pro - Monthly",
            },
        ],
        "recent_charges": [
            {
                "charge_id": "CHG-401",
                "date": "2026-03-15",
                "amount": 49.99,
                "description": "Additional storage 50GB",
                "status": "completed",
            },
        ],
        "payment_method": "Visa ending 4242",
        "balance_due": 0.00,
    },
    "ACC-2087": {
        "invoices": [
            {
                "invoice_id": "INV-9101",
                "date": "2026-02-01",
                "amount": 2499.00,
                "status": "paid",
                "description": "Enterprise - Monthly",
            },
            {
                "invoice_id": "INV-9102",
                "date": "2026-03-01",
                "amount": 2499.00,
                "status": "paid",
                "description": "Enterprise - Monthly",
            },
            {
                "invoice_id": "INV-9102-DUP",
                "date": "2026-03-02",
                "amount": 2499.00,
                "status": "paid",
                "description": "Enterprise - Monthly",
            },
        ],
        "recent_charges": [],
        "payment_method": "Amex ending 1111",
        "balance_due": 0.00,
    },
    "ACC-3055": {
        "invoices": [
            {
                "invoice_id": "INV-9201",
                "date": "2026-02-01",
                "amount": 49.00,
                "status": "paid",
                "description": "Starter - Monthly",
            },
            {
                "invoice_id": "INV-9202",
                "date": "2026-03-01",
                "amount": 49.00,
                "status": "paid",
                "description": "Starter - Monthly",
            },
        ],
        "recent_charges": [],
        "payment_method": "Visa ending 8888",
        "balance_due": 0.00,
    },
}

HEALTH_STATUS_DB = {
    "api-gateway": {
        "service": "api-gateway",
        "status": "degraded",
        "incidents": [
            {
                "incident_id": "INC-701",
                "started_at": "2026-03-25T06:30:00Z",
                "resolved_at": "2026-03-25T09:45:00Z",
                "duration_minutes": 195,
                "severity": "major",
                "description": "Elevated error rates on /v2 endpoints affecting 15% of requests. Root cause: misconfigured load balancer rule after routine maintenance.",
                "affected_regions": ["us-east-1", "eu-west-1"],
            }
        ],
    },
    "data-pipeline": {
        "service": "data-pipeline",
        "status": "operational",
        "incidents": [],
    },
    "auth-service": {
        "service": "auth-service",
        "status": "operational",
        "incidents": [
            {
                "incident_id": "INC-698",
                "started_at": "2026-03-20T14:00:00Z",
                "resolved_at": "2026-03-20T14:12:00Z",
                "duration_minutes": 12,
                "severity": "minor",
                "description": "Brief authentication latency spike in eu-west-1.",
                "affected_regions": ["eu-west-1"],
            }
        ],
    },
}

KB_ARTICLES = {
    "KB-001": {
        "id": "KB-001",
        "title": "How to Reset User MFA Tokens",
        "category": "access",
        "content": (
            "If a user is locked out due to MFA issues, an account admin can reset their "
            "MFA token from Settings > Users > select user > Reset MFA. The user will receive "
            "an email to re-enroll. If the admin is also locked out, contact L2 support with "
            "account verification details. Do NOT disable MFA globally to resolve individual issues."
        ),
    },
    "KB-002": {
        "id": "KB-002",
        "title": "Handling Duplicate Charges",
        "category": "billing",
        "content": (
            "When a customer reports a duplicate charge: 1) Verify in the billing ledger that "
            "two identical invoices exist with different IDs. 2) Confirm both were processed by "
            "the payment gateway. 3) Issue a refund for the duplicate invoice only via the billing "
            "admin panel. 4) Refunds take 5-10 business days. 5) Never refund more than the "
            "duplicated amount. 6) Always confirm the refund with the customer in writing."
        ),
    },
    "KB-003": {
        "id": "KB-003",
        "title": "SLA Credit Policy",
        "category": "sla",
        "content": (
            "Our SLA guarantees 99.9% uptime per calendar month. Credits are calculated as: "
            "- 99.0%-99.9%: 10% of monthly fee credit "
            "- 95.0%-99.0%: 25% of monthly fee credit "
            "- Below 95.0%: 50% of monthly fee credit "
            "Credits must be approved by L2 support before promising to the customer. "
            "Enterprise customers with custom SLA terms should be escalated to the account manager. "
            "Do NOT promise credits that exceed the calculated amount. Do NOT offer credits "
            "for incidents outside the SLA measurement window."
        ),
    },
    "KB-004": {
        "id": "KB-004",
        "title": "Account Lockout Recovery",
        "category": "access",
        "content": (
            "If a user reports being locked out: 1) Check if MFA is enabled on their account. "
            "2) If MFA is enabled and they lost their device, the admin can reset MFA from "
            "the admin panel. 3) If no admin is available, L2 support can verify identity via "
            "registered company email and reset. 4) Never share temporary passwords over chat. "
            "5) Always verify account ownership before making changes."
        ),
    },
}


# ---------------------------------------------------------------------------
# Task fixtures
# ---------------------------------------------------------------------------


def _access_reset_fixture() -> Dict[str, Any]:
    """Task 1: Account access reset - Easy difficulty."""
    return {
        "task_id": "access_reset",
        "difficulty": "easy",
        "description": (
            "A customer's team member is locked out of their account. Investigate the "
            "account, determine the cause, and draft a resolution reply."
        ),
        "max_steps": 8,
        "public": {
            "ticket_thread": [
                {
                    "role": "customer",
                    "sender": "Sarah Chen",
                    "timestamp": "2026-03-25T10:00:00Z",
                    "message": (
                        "Hi Support,\n\n"
                        "One of our team members, James Wu (james.wu@novatech.io), is unable to "
                        "log in to the platform. He's been trying since yesterday and keeps getting "
                        '"Authentication failed" errors. He says he\'s using the correct password.\n\n'
                        "Our account ID is ACC-1042. Can you help?\n\n"
                        "Thanks,\nSarah Chen\nAdmin, NovaTech Solutions"
                    ),
                }
            ],
            "account_id": "ACC-1042",
        },
        "hidden": {
            "correct_classification": "access",
            "correct_priority": "medium",
            "valid_route": "l1_support",
            "resolution_code": "resolved",
            "required_evidence": ["view_account", "search_kb"],
            "evidence_artifact_ids": ["ACC-1042", "KB-001", "KB-004"],
            "required_reply_facts": [
                "mfa",  # The reply should mention MFA as the likely cause
                "admin",  # Should mention the admin can reset
                "reset",  # Should mention reset process
            ],
            "forbidden_claims": [
                "disable mfa",  # Must NOT suggest disabling MFA globally
                "temporary password",  # Must NOT offer temp passwords
                "refund",  # Irrelevant to access issue
            ],
            "weights": RUBRIC_WEIGHTS,
        },
    }


def _duplicate_charge_refund_fixture() -> Dict[str, Any]:
    """Task 2: Duplicate charge refund - Medium difficulty."""
    return {
        "task_id": "duplicate_charge_refund",
        "difficulty": "medium",
        "description": (
            "A customer has been charged twice for the same service. Verify the billing "
            "data, determine the correct resolution, and draft a reply confirming the refund."
        ),
        "max_steps": 10,
        "public": {
            "ticket_thread": [
                {
                    "role": "customer",
                    "sender": "Priya Sharma",
                    "timestamp": "2026-03-25T11:30:00Z",
                    "message": (
                        "Hello,\n\n"
                        "I noticed we were charged $2,499.00 twice in March for our Enterprise plan. "
                        "Our account is ACC-2087. Can you please look into this and refund the duplicate?\n\n"
                        "Regards,\nPriya Sharma\nMeridian Analytics"
                    ),
                }
            ],
            "account_id": "ACC-2087",
        },
        "hidden": {
            "correct_classification": "billing",
            "correct_priority": "high",
            "valid_route": "billing_team",
            "resolution_code": "resolved",
            "required_evidence": ["view_account", "view_billing", "search_kb"],
            "evidence_artifact_ids": ["ACC-2087", "ACC-2087-billing", "KB-002"],
            "required_reply_facts": [
                "duplicate",  # Must acknowledge duplicate
                "refund",  # Must confirm refund
                "5-10 business days",  # Must state refund timeline
                "inv-9102-dup",  # Must reference the duplicate invoice
            ],
            "forbidden_claims": [
                "25%",  # Must NOT offer SLA credit for billing issue
                "50%",  # Must NOT offer excessive credit
                "free month",  # Must NOT promise free service
                "escalated to engineering",  # Wrong escalation path
            ],
            "weights": RUBRIC_WEIGHTS,
        },
    }


def _incident_sla_credit_fixture() -> Dict[str, Any]:
    """Task 3: Incident SLA credit - Hard difficulty."""
    return {
        "task_id": "incident_sla_credit",
        "difficulty": "hard",
        "description": (
            "A customer is upset about an outage and requesting compensation. You need to "
            "check service health, calculate SLA credits, escalate appropriately, and draft "
            "a careful reply without over-promising."
        ),
        "max_steps": 12,
        "public": {
            "ticket_thread": [
                {
                    "role": "customer",
                    "sender": "Marco Rossi",
                    "timestamp": "2026-03-25T12:00:00Z",
                    "message": (
                        "Support,\n\n"
                        "Our API integration was down for over 3 hours yesterday morning. This cost "
                        "us real revenue. We're on the Starter plan (ACC-3055) but I expect "
                        "compensation for this disruption.\n\n"
                        "I want a full month credit at minimum. If this isn't resolved quickly "
                        "I'll be looking at alternative providers.\n\n"
                        "Marco Rossi\nGreenleaf Digital"
                    ),
                }
            ],
            "account_id": "ACC-3055",
        },
        "hidden": {
            "correct_classification": "outage",
            "correct_priority": "high",
            "valid_route": "l2_support",
            "resolution_code": "escalated",
            "required_evidence": ["view_account", "view_health", "search_kb"],
            "evidence_artifact_ids": ["ACC-3055", "api-gateway", "KB-003"],
            "required_reply_facts": [
                "acknowledge",  # Must acknowledge the incident
                "incident",  # Must reference the incident
                "sla",  # Must mention SLA
                "credit",  # Must mention credit (even if escalating)
                "escalat",  # Must mention escalation (stem match)
            ],
            "forbidden_claims": [
                "full month",  # Must NOT promise full month credit (exceeds policy)
                "100%",  # Must NOT promise 100% credit
                "free",  # Must NOT promise free service
                "guarantee",  # Must NOT guarantee specific outcome before escalation
            ],
            "sla_credit_calculation": {
                "monthly_fee": 49.00,
                "downtime_minutes": 195,
                "month_minutes": 43800,  # 30.6875 days * 24 * 60
                "uptime_percentage": (43800 - 195) / 43800 * 100,  # ~99.56%
                "credit_tier": "10%",
                "credit_amount": 4.90,
            },
            "weights": RUBRIC_WEIGHTS,
        },
    }


# ---------------------------------------------------------------------------
# Public API
# -----------

TASK_IDS = ["access_reset", "duplicate_charge_refund", "incident_sla_credit"]

_TASK_BUILDERS = {
    "access_reset": _access_reset_fixture,
    "duplicate_charge_refund": _duplicate_charge_refund_fixture,
    "incident_sla_credit": _incident_sla_credit_fixture,
}


def get_fixture(task_id: str) -> Dict[str, Any]:
    """Return a deep-copied fixture for the given task_id."""
    if task_id not in _TASK_BUILDERS:
        raise ValueError(f"Unknown task_id: {task_id}. Valid: {TASK_IDS}")
    return deepcopy(_TASK_BUILDERS[task_id]())


def get_tool_result(action_type: str, identifier: str) -> Dict[str, Any]:
    """Simulate a tool lookup and return the result dict, or an error."""
    if action_type == "view_account":
        if identifier in ACCOUNT_DB:
            return {"found": True, "data": deepcopy(ACCOUNT_DB[identifier])}
        return {"found": False, "error": f"Account {identifier} not found"}

    if action_type == "view_billing":
        if identifier in BILLING_DB:
            return {"found": True, "data": deepcopy(BILLING_DB[identifier])}
        return {"found": False, "error": f"Billing data for {identifier} not found"}

    if action_type == "view_health":
        if identifier in HEALTH_STATUS_DB:
            return {"found": True, "data": deepcopy(HEALTH_STATUS_DB[identifier])}
        return {"found": False, "error": f"Service {identifier} not found"}

    if action_type == "search_kb":
        matches = []
        query_lower = identifier.lower()
        for article in KB_ARTICLES.values():
            if (
                query_lower in article["title"].lower()
                or query_lower in article["content"].lower()
                or query_lower in article["category"].lower()
            ):
                matches.append(article)
        return {"found": len(matches) > 0, "results": matches}

    return {"found": False, "error": f"Unknown action type: {action_type}"}


def get_task_summary(task_id: str) -> Dict[str, Any]:
    """Return a public-facing summary of a task (no hidden data)."""
    fixture = get_fixture(task_id)
    return {
        "task_id": fixture["task_id"],
        "difficulty": fixture["difficulty"],
        "description": fixture["description"],
        "max_steps": fixture["max_steps"],
        "ticket_preview": fixture["public"]["ticket_thread"][0]["message"][:200]
        + "...",
    }
