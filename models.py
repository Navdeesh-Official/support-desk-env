"""Pydantic models for the SupportDeskEnv."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field

from openenv.core.env_server import Action, Observation, State


class ActionType(str, Enum):
    VIEW_ACCOUNT = "view_account"
    VIEW_BILLING = "view_billing"
    VIEW_HEALTH = "view_health"
    SEARCH_KB = "search_kb"
    CLASSIFY_TICKET = "classify_ticket"
    SET_PRIORITY = "set_priority"
    ROUTE_TICKET = "route_ticket"
    DRAFT_REPLY = "draft_reply"
    RESOLVE_TICKET = "resolve_ticket"


class SupportAction(Action):
    """A single support-desk action an agent may take each turn."""

    action_type: ActionType = Field(description="The kind of action to perform")

    # Optional typed payloads per action
    account_id: Optional[str] = Field(
        default=None, description="Account ID for view_account"
    )
    billing_account_id: Optional[str] = Field(
        default=None, description="Account ID for view_billing"
    )
    service_name: Optional[str] = Field(
        default=None, description="Service name for view_health"
    )
    query: Optional[str] = Field(default=None, description="Search query for search_kb")
    classification: Optional[str] = Field(
        default=None,
        description="Ticket classification for classify_ticket (e.g. access, billing, outage)",
    )
    priority: Optional[str] = Field(
        default=None,
        description="Priority level for set_priority (low, medium, high, critical)",
    )
    route_to: Optional[str] = Field(
        default=None,
        description="Routing target for route_ticket (e.g. l1_support, l2_support, billing_team, engineering)",
    )
    reply_text: Optional[str] = Field(
        default=None, description="Drafted reply text for draft_reply"
    )
    resolution_code: Optional[str] = Field(
        default=None,
        description="Resolution code for resolve_ticket (e.g. resolved, escalated, closed)",
    )


class SupportObservation(Observation):
    """Agent-visible observation returned after each step."""

    ticket_thread: List[Dict[str, str]] = Field(
        default_factory=list,
        description="The customer support ticket conversation thread",
    )
    latest_tool_result: Optional[Dict[str, Any]] = Field(
        default=None, description="Result from the most recent tool/action call"
    )
    retrieved_artifacts: List[str] = Field(
        default_factory=list,
        description="IDs of artifacts the agent has retrieved so far",
    )
    current_classification: Optional[str] = Field(
        default=None, description="Current ticket classification set by agent"
    )
    current_priority: Optional[str] = Field(
        default=None, description="Current priority set by agent"
    )
    current_route: Optional[str] = Field(
        default=None, description="Current routing target set by agent"
    )
    status_message: str = Field(
        default="", description="Short human-readable status of last action"
    )


class SupportState(State):
    """Internal episode state (returned via /state endpoint)."""

    task_id: str = Field(default="", description="Current task identifier")
    retrieved_artifact_ids: List[str] = Field(
        default_factory=list, description="All artifact IDs retrieved this episode"
    )
    chosen_classification: Optional[str] = Field(
        default=None, description="Agent-chosen classification"
    )
    chosen_priority: Optional[str] = Field(
        default=None, description="Agent-chosen priority"
    )
    chosen_route: Optional[str] = Field(
        default=None, description="Agent-chosen routing target"
    )
    drafted_reply: Optional[str] = Field(
        default=None, description="Agent-drafted reply text"
    )
    cumulative_reward: float = Field(
        default=0.0, description="Cumulative reward this episode"
    )
    completed: bool = Field(
        default=False, description="Whether the episode has completed"
    )
