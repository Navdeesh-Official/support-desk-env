"""Mistral baseline agent for SupportDeskEnv.

Usage:
    python baseline_mistral.py [--base-url URL] [--model MODEL] [--task TASK_ID]

Requires MISTRAL_API_KEY environment variable.
Optional MISTRAL_MODEL environment variable (default: mistral-small-latest).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from mistralai import Mistral

from environment import SupportDeskEnv
from fixtures import TASK_IDS
from models import SupportAction

SYSTEM_PROMPT = """\
You are a B2B SaaS support agent. You receive a customer ticket and must resolve it.

Available actions (respond with EXACTLY one JSON object per turn):
- {"action_type": "view_account", "account_id": "ACC-XXXX"} — look up account info
- {"action_type": "view_billing", "billing_account_id": "ACC-XXXX"} — look up billing data
- {"action_type": "view_health", "service_name": "SERVICE"} — check service health status
- {"action_type": "search_kb", "query": "SEARCH TERMS"} — search knowledge base
- {"action_type": "classify_ticket", "classification": "access|billing|outage|general"} — classify
- {"action_type": "set_priority", "priority": "low|medium|high|critical"} — set priority
- {"action_type": "route_ticket", "route_to": "l1_support|l2_support|billing_team|engineering"} — route
- {"action_type": "draft_reply", "reply_text": "YOUR REPLY HERE"} — draft customer reply
- {"action_type": "resolve_ticket", "resolution_code": "resolved|escalated|closed"} — resolve and end

Rules:
1. Investigate before acting. Use view_account, view_billing, view_health, search_kb to gather info.
2. Classify, set priority, and route the ticket based on evidence.
3. Draft a professional reply that addresses the customer's issue with facts from your investigation.
4. Resolve the ticket when done.
5. Respond with ONLY the JSON action object. No markdown, no explanation.
"""


def run_baseline(
    base_url: str = "http://localhost:8000",
    model: str | None = None,
    task_id: str = "access_reset",
) -> dict[str, Any]:
    """Run the Mistral baseline on a single task and return results."""
    model = model or os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("ERROR: MISTRAL_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    client = Mistral(api_key=api_key)

    env = SupportDeskEnv()
    obs = env.reset(task_id=task_id)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Task: {task_id}\n\n"
                f"Ticket thread:\n{_format_thread(obs.ticket_thread)}\n\n"
                f"Take your first action."
            ),
        },
    ]

    step_count = 0
    while not obs.done:
        response = client.chat.complete(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=500,
        )
        reply_text = response.choices[0].message.content.strip()

        # Parse the action JSON
        try:
            action_dict = _extract_json(reply_text)
            action = SupportAction(**action_dict)
        except Exception as e:
            messages.append({"role": "assistant", "content": reply_text})
            messages.append(
                {
                    "role": "user",
                    "content": f"Invalid action: {e}. Respond with valid JSON only.",
                }
            )
            continue

        obs = env.step(action)
        step_count += 1

        # Feed observation back
        messages.append({"role": "assistant", "content": reply_text})
        messages.append(
            {
                "role": "user",
                "content": _format_observation(obs, step_count),
            }
        )

    grade = env.get_grade_breakdown()
    return {
        "task_id": task_id,
        "model": model,
        "score": obs.reward,
        "steps": step_count,
        "grade": grade,
    }


def _format_thread(thread: list[dict]) -> str:
    parts = []
    for msg in thread:
        parts.append(
            f"[{msg['role'].upper()}] {msg.get('sender', 'System')}: {msg['message']}"
        )
    return "\n\n".join(parts)


def _format_observation(obs: Any, step: int) -> str:
    lines = [f"Step {step}: {obs.status_message}"]
    if obs.latest_tool_result:
        lines.append(
            f"Tool result: {json.dumps(obs.latest_tool_result, indent=2)[:1000]}"
        )
    if obs.retrieved_artifacts:
        lines.append(f"Retrieved: {', '.join(obs.retrieved_artifacts)}")
    if obs.current_classification:
        lines.append(f"Classification: {obs.current_classification}")
    if obs.current_priority:
        lines.append(f"Priority: {obs.current_priority}")
    if obs.current_route:
        lines.append(f"Route: {obs.current_route}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    """Extract a JSON object from text, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mistral baseline for SupportDeskEnv")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default=None)
    parser.add_argument("--task", default=None, help="Run a single task (default: all)")
    args = parser.parse_args()

    tasks = [args.task] if args.task else TASK_IDS
    results = []

    for task_id in tasks:
        print(f"\n{'=' * 60}")
        print(f"Running task: {task_id}")
        print(f"{'=' * 60}")
        result = run_baseline(base_url=args.base_url, model=args.model, task_id=task_id)
        results.append(result)
        print(f"Score: {result['score']:.4f} ({result['steps']} steps)")

    mean = sum(r["score"] for r in results) / len(results)
    print(f"\n{'=' * 60}")
    print(f"Mean score: {mean:.4f}")
    print(f"Model: {results[0]['model']}")
    print(f"{'=' * 60}")

    # Output JSON summary
    summary = {
        "model": results[0]["model"],
        "prompt_version": "v1",
        "tasks": [{"task_id": r["task_id"], "score": r["score"]} for r in results],
        "mean_score": round(mean, 4),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
