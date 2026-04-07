"""OpenAI baseline agent for SupportDeskEnv serving as inference.py."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

try:
    from openai import OpenAI
    OPENAI_IMPORT_ERROR: Exception | None = None
except Exception as e:
    OpenAI = None  # type: ignore[assignment]
    OPENAI_IMPORT_ERROR = e

DEFAULT_TASK_IDS = ["access_reset"]

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

SYSTEM_PROMPT = """\
You are a B2B SaaS support agent. You receive a customer ticket and must resolve it.

Available actions (respond with EXACTLY one JSON object per turn):
- {"action_type": "view_account", "account_id": "ACC-XXXX"}
- {"action_type": "view_billing", "billing_account_id": "ACC-XXXX"}
- {"action_type": "view_health", "service_name": "SERVICE"}
- {"action_type": "search_kb", "query": "SEARCH TERMS"}
- {"action_type": "classify_ticket", "classification": "access|billing|outage|general"}
- {"action_type": "set_priority", "priority": "low|medium|high|critical"}
- {"action_type": "route_ticket", "route_to": "l1_support|l2_support|billing_team|engineering"}
- {"action_type": "draft_reply", "reply_text": "YOUR REPLY HERE"}
- {"action_type": "resolve_ticket", "resolution_code": "resolved|escalated|closed"}

Rules:
1. Investigate before acting.
2. Respond with ONLY the JSON action object.
"""


def _error_result(task_id: str, message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "task_id": task_id,
        "model": MODEL_NAME,
        "message": message,
        "score": 0.0,
        "steps": 0,
        "grade": None,
    }


def _format_thread(thread: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for msg in thread:
        role = str(msg.get("role", "system")).upper()
        sender = str(msg.get("sender", "System"))
        text = str(msg.get("message", ""))
        parts.append(f"[{role}] {sender}: {text}")
    return "\n\n".join(parts)


def _format_observation(obs: Any, step: int) -> str:
    lines = [f"Step {step}: {getattr(obs, 'status_message', '')}"]
    latest_tool_result = getattr(obs, "latest_tool_result", None)
    if latest_tool_result:
        lines.append(f"Tool result: {json.dumps(latest_tool_result, indent=2)[:1000]}")
    retrieved_artifacts = getattr(obs, "retrieved_artifacts", None)
    if retrieved_artifacts:
        lines.append(f"Retrieved: {', '.join(retrieved_artifacts)}")
    current_classification = getattr(obs, "current_classification", None)
    if current_classification:
        lines.append(f"Classification: {current_classification}")
    current_priority = getattr(obs, "current_priority", None)
    if current_priority:
        lines.append(f"Priority: {current_priority}")
    current_route = getattr(obs, "current_route", None)
    if current_route:
        lines.append(f"Route: {current_route}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict[str, Any]:
    candidate = (text or "").strip()
    if candidate.startswith("```"):
        lines = [line for line in candidate.split("\n") if not line.strip().startswith("```")]
        candidate = "\n".join(lines)
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Model response must be a JSON object")
    return parsed


def run_baseline(task_id: str = "access_reset") -> dict[str, Any]:
    """Run the OpenAI baseline on a single task and return results."""
    if not task_id or not isinstance(task_id, str):
        return _error_result("invalid_task", "Invalid task_id input")

    if OPENAI_IMPORT_ERROR is not None:
        return _error_result(task_id, f"OpenAI import failed: {OPENAI_IMPORT_ERROR}")

    try:
        from environment import SupportDeskEnv
        from models import SupportAction
    except Exception as e:
        return _error_result(task_id, f"Environment import failed: {e}")

    try:
        env = SupportDeskEnv()
        obs = env.reset(task_id=task_id)
    except Exception as e:
        return _error_result(task_id, f"Environment setup failed: {e}")

    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN,
        )
    except Exception as e:
        return _error_result(task_id, f"Client initialization failed: {e}")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Task: {task_id}\n\n"
                f"Ticket thread:\n{_format_thread(obs.ticket_thread)}\n\n"
                "Take your first action."
            ),
        },
    ]

    step_count = 0
    max_steps = 12

    while not obs.done and step_count < max_steps:
        print(f"[STEP] task={task_id} step={step_count + 1} action=calling_model", flush=True)
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0,
                max_tokens=500,
            )
        except Exception as e:
            print(
                f"[STEP] task={task_id} step={step_count + 1} action=error message={str(e).replace(' ', '_')}",
                flush=True,
            )
            return {
                "status": "error",
                "task_id": task_id,
                "model": MODEL_NAME,
                "message": str(e),
                "score": float(getattr(obs, "reward", 0.0) or 0.0),
                "steps": step_count,
                "grade": None,
            }

        reply_text = ""
        try:
            content = response.choices[0].message.content
            reply_text = (content or "").strip()
            action_dict = _extract_json(reply_text)
            action = SupportAction(**action_dict)
        except Exception as e:
            messages.append({"role": "assistant", "content": reply_text or "{}"})
            messages.append(
                {
                    "role": "user",
                    "content": f"Invalid action: {e}. Respond with valid JSON only.",
                }
            )
            step_count += 1
            continue

        try:
            obs = env.step(action)
        except Exception as e:
            return _error_result(task_id, f"Environment step failed: {e}")

        step_count += 1
        messages.append({"role": "assistant", "content": reply_text})
        messages.append({"role": "user", "content": _format_observation(obs, step_count)})

    try:
        grade = env.get_grade_breakdown()
    except Exception:
        grade = None

    return {
        "status": "ok",
        "task_id": task_id,
        "model": MODEL_NAME,
        "score": float(getattr(obs, "reward", 0.0) or 0.0),
        "steps": step_count,
        "grade": grade,
    }


def main() -> None:
    print(f"[START] model={MODEL_NAME}", flush=True)
    summary: dict[str, Any]

    try:
        parser = argparse.ArgumentParser(description="Inference baseline for SupportDeskEnv")
        parser.add_argument("--task", default=None, help="Run a single task (default: all)")
        args = parser.parse_args()

        if args.task:
            tasks = [args.task]
        else:
            try:
                from fixtures import TASK_IDS

                tasks = list(TASK_IDS)
            except Exception:
                tasks = list(DEFAULT_TASK_IDS)
        results: list[dict[str, Any]] = []

        for task_id in tasks:
            print(f"[STEP] task={task_id} action=run_baseline", flush=True)
            result = run_baseline(task_id=task_id)
            results.append(result)

        task_scores = [{"task_id": r.get("task_id"), "score": r.get("score", 0.0)} for r in results]
        mean_score = sum(float(r.get("score", 0.0) or 0.0) for r in results) / len(results)
        summary = {
            "status": "ok",
            "model": MODEL_NAME,
            "prompt_version": "v1",
            "tasks": task_scores,
            "mean_score": round(mean_score, 4),
            "results": results,
        }
    except Exception as e:
        print("STEP: error occurred")
        summary = {
            "status": "error",
            "model": MODEL_NAME,
            "message": str(e),
            "tasks": [],
            "mean_score": 0.0,
            "results": [],
        }
    finally:
        final_score = summary.get("mean_score", 0.0) if isinstance(summary, dict) else 0.0
        task_count = len(summary.get("tasks", [])) if isinstance(summary, dict) else 0
        print(f"[END] tasks={task_count} score={final_score}", flush=True)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
