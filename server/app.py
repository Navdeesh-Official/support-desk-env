"""FastAPI server for SupportDeskEnv with OpenEnv + custom endpoints."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import HTTPException

from openenv.core.env_server import create_fastapi_app

from environment import SupportDeskEnv
from fixtures import TASK_IDS, get_task_summary
from models import SupportAction, SupportObservation


# ---------------------------------------------------------------------------
# Factory callable for create_fastapi_app (must be a callable returning
# a fresh Environment each time the server needs one)
# ---------------------------------------------------------------------------


def _env_factory() -> SupportDeskEnv:
    return SupportDeskEnv()


# ---------------------------------------------------------------------------
# Build the base OpenEnv app (auto-creates /reset, /step, /state, /health)
# ---------------------------------------------------------------------------

app = create_fastapi_app(
    env=_env_factory,
    action_cls=SupportAction,
    observation_cls=SupportObservation,
)


# ---------------------------------------------------------------------------
# Keep a reference to the current env so custom endpoints can inspect it
# (OpenEnv manages per-session environments internally, so we store the
# last-created instance for grading / baseline convenience)
# ---------------------------------------------------------------------------

_current_env: SupportDeskEnv | None = None


# Monkey-patch the env factory to track the latest instance
_original_factory = _env_factory


def _tracked_factory() -> SupportDeskEnv:
    global _current_env
    _current_env = _original_factory()
    return _current_env


# We can't easily swap the factory after app creation, so the extra
# endpoints will just create their own env instances for /baseline.


# ---------------------------------------------------------------------------
# Custom endpoints
# ---------------------------------------------------------------------------


@app.get("/tasks", tags=["Support Desk"])
def list_tasks() -> List[Dict[str, Any]]:
    """List the 3 canonical benchmark tasks with descriptions and difficulty."""
    return [get_task_summary(tid) for tid in TASK_IDS]


@app.get("/grader", tags=["Support Desk"])
def get_grader() -> Dict[str, Any]:
    """Return the grade breakdown for the current episode.

    Returns 409 if the episode is not yet complete.
    """
    if _current_env is None:
        raise HTTPException(
            status_code=409, detail="No active episode. Call /reset first."
        )

    breakdown = _current_env.get_grade_breakdown()
    if breakdown is None:
        raise HTTPException(
            status_code=409,
            detail="Episode not yet complete. Call /step with a resolve_ticket action.",
        )
    return breakdown


@app.post("/baseline", tags=["Support Desk"])
def run_baseline() -> Dict[str, Any]:
    """Run a deterministic rule-based baseline on all 3 tasks and return scores."""
    results = []
    for task_id in TASK_IDS:
        score = _run_rule_baseline(task_id)
        results.append({"task_id": task_id, "score": score})

    mean_score = sum(r["score"] for r in results) / len(results)
    return {
        "model": "rule_based_v1",
        "prompt_version": "v1",
        "tasks": results,
        "mean_score": round(mean_score, 4),
    }


# ---------------------------------------------------------------------------
# Simple rule-based baseline (no LLM, just to validate grading works)
# ---------------------------------------------------------------------------


def _run_rule_baseline(task_id: str) -> float:
    """Run a deterministic trajectory for a task and return the score."""
    env = SupportDeskEnv()
    env.reset(task_id=task_id)

    if task_id == "access_reset":
        actions = [
            SupportAction(action_type="view_account", account_id="ACC-1042"),
            SupportAction(action_type="search_kb", query="mfa reset"),
            SupportAction(action_type="classify_ticket", classification="access"),
            SupportAction(action_type="set_priority", priority="medium"),
            SupportAction(action_type="route_ticket", route_to="l1_support"),
            SupportAction(
                action_type="draft_reply",
                reply_text=(
                    "Hi Sarah,\n\n"
                    "Thank you for reaching out. I've investigated James Wu's account "
                    "(USR-302) and found that MFA is not enabled on his account, which "
                    "may be causing authentication issues. As the account admin, you can "
                    "reset his access from Settings > Users > select James Wu > Reset MFA.\n\n"
                    "If you need further assistance, our L1 support team can help.\n\n"
                    "Best regards,\nSupport Team"
                ),
            ),
            SupportAction(action_type="resolve_ticket", resolution_code="resolved"),
        ]
    elif task_id == "duplicate_charge_refund":
        actions = [
            SupportAction(action_type="view_account", account_id="ACC-2087"),
            SupportAction(action_type="view_billing", billing_account_id="ACC-2087"),
            SupportAction(action_type="search_kb", query="duplicate charge"),
            SupportAction(action_type="classify_ticket", classification="billing"),
            SupportAction(action_type="set_priority", priority="high"),
            SupportAction(action_type="route_ticket", route_to="billing_team"),
            SupportAction(
                action_type="draft_reply",
                reply_text=(
                    "Hi Priya,\n\n"
                    "Thank you for flagging this. I've confirmed that invoice INV-9102-DUP "
                    "is a duplicate charge of $2,499.00. I've initiated a refund for the "
                    "duplicate amount. Please allow 5-10 business days for the refund to "
                    "appear on your Amex ending 1111.\n\n"
                    "Our billing team has been notified. Sorry for the inconvenience.\n\n"
                    "Best regards,\nSupport Team"
                ),
            ),
            SupportAction(action_type="resolve_ticket", resolution_code="resolved"),
        ]
    elif task_id == "incident_sla_credit":
        actions = [
            SupportAction(action_type="view_account", account_id="ACC-3055"),
            SupportAction(action_type="view_health", service_name="api-gateway"),
            SupportAction(action_type="search_kb", query="sla credit"),
            SupportAction(action_type="classify_ticket", classification="outage"),
            SupportAction(action_type="set_priority", priority="high"),
            SupportAction(action_type="route_ticket", route_to="l2_support"),
            SupportAction(
                action_type="draft_reply",
                reply_text=(
                    "Hi Marco,\n\n"
                    "Thank you for reaching out. I acknowledge the api-gateway incident "
                    "on March 25th impacted your integration, and I sincerely apologize "
                    "for the disruption.\n\n"
                    "Per our SLA policy, the incident has been reviewed and we are "
                    "escalating this to our L2 support team who will calculate any "
                    "applicable SLA credit. They will follow up within 24 hours.\n\n"
                    "We take these incidents seriously and are committed to maintaining "
                    "reliable service.\n\n"
                    "Best regards,\nSupport Team"
                ),
            ),
            SupportAction(action_type="resolve_ticket", resolution_code="escalated"),
        ]
    else:
        return 0.0

    for action in actions:
        obs = env.step(action)
        if obs.done:
            break

    return obs.reward or 0.0
