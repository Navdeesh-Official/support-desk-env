"""Typed client for SupportDeskEnv.

Wraps GenericEnvClient to provide convenience methods that accept and
return typed Pydantic models rather than raw dicts.
"""

from __future__ import annotations

from typing import Any, Optional

from openenv import GenericEnvClient

from models import ActionType, SupportAction


class SupportDeskClient:
    """Convenience wrapper around GenericEnvClient for support desk tasks."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self._base_url = base_url
        self._client = GenericEnvClient(base_url=base_url)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> "SupportDeskClient":
        self._client.connect()
        return self

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SupportDeskClient":
        return self.connect()

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Core env operations
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "access_reset") -> dict[str, Any]:
        """Reset the environment and return the initial observation."""
        result = self._client.reset(task_id=task_id)
        return result.observation

    def step(self, action: SupportAction) -> dict[str, Any]:
        """Send a typed action and return the observation dict."""
        result = self._client.step(action.model_dump())
        return result.observation

    def state(self) -> dict[str, Any]:
        """Return the current environment state."""
        return self._client.state()

    # ------------------------------------------------------------------
    # Typed action helpers
    # ------------------------------------------------------------------

    def view_account(self, account_id: str) -> dict[str, Any]:
        return self.step(
            SupportAction(action_type=ActionType.VIEW_ACCOUNT, account_id=account_id)
        )

    def view_billing(self, account_id: str) -> dict[str, Any]:
        return self.step(
            SupportAction(
                action_type=ActionType.VIEW_BILLING, billing_account_id=account_id
            )
        )

    def view_health(self, service_name: str) -> dict[str, Any]:
        return self.step(
            SupportAction(action_type=ActionType.VIEW_HEALTH, service_name=service_name)
        )

    def search_kb(self, query: str) -> dict[str, Any]:
        return self.step(SupportAction(action_type=ActionType.SEARCH_KB, query=query))

    def classify_ticket(self, classification: str) -> dict[str, Any]:
        return self.step(
            SupportAction(
                action_type=ActionType.CLASSIFY_TICKET, classification=classification
            )
        )

    def set_priority(self, priority: str) -> dict[str, Any]:
        return self.step(
            SupportAction(action_type=ActionType.SET_PRIORITY, priority=priority)
        )

    def route_ticket(self, route_to: str) -> dict[str, Any]:
        return self.step(
            SupportAction(action_type=ActionType.ROUTE_TICKET, route_to=route_to)
        )

    def draft_reply(self, reply_text: str) -> dict[str, Any]:
        return self.step(
            SupportAction(action_type=ActionType.DRAFT_REPLY, reply_text=reply_text)
        )

    def resolve_ticket(self, resolution_code: str = "resolved") -> dict[str, Any]:
        return self.step(
            SupportAction(
                action_type=ActionType.RESOLVE_TICKET, resolution_code=resolution_code
            )
        )

    # ------------------------------------------------------------------
    # Custom endpoints
    # ------------------------------------------------------------------

    def list_tasks(self) -> list[dict[str, Any]]:
        import httpx

        r = httpx.get(f"{self._base_url}/tasks")
        r.raise_for_status()
        return r.json()

    def get_grader(self) -> dict[str, Any]:
        import httpx

        r = httpx.get(f"{self._base_url}/grader")
        r.raise_for_status()
        return r.json()

    def run_baseline(self) -> dict[str, Any]:
        import httpx

        r = httpx.post(f"{self._base_url}/baseline")
        r.raise_for_status()
        return r.json()
