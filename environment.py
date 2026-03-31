"""SupportDeskEnv - OpenEnv environment for B2B SaaS support ticket resolution."""

from __future__ import annotations

import re
import uuid
from copy import deepcopy
from typing import Any, Dict, Optional

from openenv.core.env_server import Environment

from fixtures import TASK_IDS, get_fixture, get_tool_result
from models import ActionType, SupportAction, SupportObservation, SupportState
from fixtures import RUBRIC_WEIGHTS


class SupportDeskEnv(Environment[SupportAction, SupportObservation, SupportState]):
    """Simulates one B2B SaaS support ticket per episode."""

    def __init__(self) -> None:
        self._fixture: Optional[Dict[str, Any]] = None
        self._state = SupportState()
        self._retrieved_artifact_ids: list[str] = []
        self._actions_taken: list[str] = []
        self._last_reward = 0.0
        self._grade_breakdown: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        **kwargs: Any,
    ) -> SupportObservation:
        task_id = task_id or "access_reset"
        if task_id not in TASK_IDS:
            raise ValueError(f"Unknown task_id: {task_id}. Valid: {TASK_IDS}")

        self._fixture = get_fixture(task_id)
        self._state = SupportState(
            episode_id=episode_id or uuid.uuid4().hex[:12],
            step_count=0,
            task_id=task_id,
            cumulative_reward=0.0,
            completed=False,
        )
        self._retrieved_artifact_ids = []
        self._actions_taken = []
        self._last_reward = 0.0
        self._grade_breakdown = None

        return self._build_observation(
            status_message=f"Episode started. Task: {task_id}",
        )

    def step(self, action: SupportAction, **kwargs: Any) -> SupportObservation:
        if self._fixture is None:
            raise RuntimeError("Call reset() before step()")

        self._state.step_count += 1
        self._actions_taken.append(action.action_type.value)

        status_msg = ""
        tool_result: Optional[Dict[str, Any]] = None

        # Dispatch action
        if action.action_type == ActionType.VIEW_ACCOUNT:
            aid = action.account_id or self._fixture["public"].get("account_id", "")
            tool_result = get_tool_result("view_account", aid)
            if tool_result.get("found"):
                self._retrieved_artifact_ids.append(aid)
            status_msg = f"Viewed account {aid}"

        elif action.action_type == ActionType.VIEW_BILLING:
            aid = action.billing_account_id or self._fixture["public"].get(
                "account_id", ""
            )
            tool_result = get_tool_result("view_billing", aid)
            if tool_result.get("found"):
                self._retrieved_artifact_ids.append(f"{aid}-billing")
            status_msg = f"Viewed billing for {aid}"

        elif action.action_type == ActionType.VIEW_HEALTH:
            svc = action.service_name or ""
            tool_result = get_tool_result("view_health", svc)
            if tool_result.get("found"):
                self._retrieved_artifact_ids.append(svc)
            status_msg = f"Viewed health status for {svc}"

        elif action.action_type == ActionType.SEARCH_KB:
            query = action.query or ""
            tool_result = get_tool_result("search_kb", query)
            if tool_result.get("found"):
                for r in tool_result.get("results", []):
                    if r["id"] not in self._retrieved_artifact_ids:
                        self._retrieved_artifact_ids.append(r["id"])
            status_msg = f"Searched KB for: {query}"

        elif action.action_type == ActionType.CLASSIFY_TICKET:
            self._state.chosen_classification = action.classification
            status_msg = f"Classified as: {action.classification}"

        elif action.action_type == ActionType.SET_PRIORITY:
            self._state.chosen_priority = action.priority
            status_msg = f"Priority set to: {action.priority}"

        elif action.action_type == ActionType.ROUTE_TICKET:
            self._state.chosen_route = action.route_to
            status_msg = f"Routed to: {action.route_to}"

        elif action.action_type == ActionType.DRAFT_REPLY:
            self._state.drafted_reply = action.reply_text
            status_msg = "Draft reply saved"

        elif action.action_type == ActionType.RESOLVE_TICKET:
            self._state.completed = True
            self._state.cumulative_reward = self._compute_final_score()
            self._last_reward = self._state.cumulative_reward
            status_msg = f"Ticket resolved with code: {action.resolution_code}"
            return self._build_observation(
                status_message=status_msg,
                done=True,
                reward=self._state.cumulative_reward,
                tool_result=tool_result,
            )

        else:
            status_msg = f"Unknown action: {action.action_type}"

        # Check max steps
        done = self._state.step_count >= self._fixture["max_steps"]
        if done and not self._state.completed:
            self._state.completed = True
            self._state.cumulative_reward = self._compute_final_score()
            self._last_reward = self._state.cumulative_reward
            status_msg += " (max steps reached - auto-resolving)"
            return self._build_observation(
                status_message=status_msg,
                done=True,
                reward=self._state.cumulative_reward,
                tool_result=tool_result,
            )

        # Incremental reward: 0 during episode, final at end
        return self._build_observation(
            status_message=status_msg, reward=0.0, tool_result=tool_result
        )

    def state(self) -> SupportState:
        return self._state.model_copy(deep=True)

    # ------------------------------------------------------------------
    # Grading
    # ------------------------------------------------------------------

    def _compute_final_score(self) -> float:
        hidden = self._fixture["hidden"]
        weights = hidden["weights"]

        scores: Dict[str, float] = {}

        # 1. Evidence retrieval (0.20)
        required = set(hidden["required_evidence"])
        retrieved_actions = set(self._actions_taken)
        evidence_count = sum(1 for r in required if r in retrieved_actions)
        scores["evidence_retrieval"] = (
            evidence_count / len(required) if required else 0.0
        )

        # 2. Classification + priority (0.20)
        cls_score = 0.0
        pri_score = 0.0
        if self._state.chosen_classification == hidden["correct_classification"]:
            cls_score = 1.0
        if self._state.chosen_priority == hidden["correct_priority"]:
            pri_score = 1.0
        scores["classification_priority"] = (cls_score + pri_score) / 2.0

        # 3. Policy decision - reply quality check (0.25)
        scores["policy_decision"] = self._grade_reply_policy(hidden)

        # 4. Routing/escalation (0.15)
        if self._state.chosen_route == hidden["valid_route"]:
            scores["routing_escalation"] = 1.0
        elif self._state.chosen_route is not None:
            scores["routing_escalation"] = 0.3  # Partial credit for routing somewhere
        else:
            scores["routing_escalation"] = 0.0

        # 5. Reply quality (0.20)
        scores["reply_quality"] = self._grade_reply_quality(hidden)

        # Weighted sum
        total = sum(scores[k] * weights[k] for k in weights)

        # Clamp to [0, 1]
        total = max(0.0, min(1.0, total))

        self._grade_breakdown = {k: round(v, 4) for k, v in scores.items()}
        return round(total, 4)

    def _grade_reply_policy(self, hidden: Dict[str, Any]) -> float:
        """Grade the drafted reply for policy compliance (forbidden claims)."""
        reply = (self._state.drafted_reply or "").lower()
        if not reply:
            return 0.0

        forbidden = hidden.get("forbidden_claims", [])
        penalties = sum(1 for f in forbidden if f.lower() in reply)
        if penalties == 0:
            return 1.0
        # Each forbidden claim reduces score
        return max(0.0, 1.0 - penalties * 0.35)

    def _grade_reply_quality(self, hidden: Dict[str, Any]) -> float:
        """Grade reply for required facts (deterministic string matching)."""
        reply = (self._state.drafted_reply or "").lower()
        if not reply:
            return 0.0

        required_facts = hidden.get("required_reply_facts", [])
        if not required_facts:
            return 1.0

        found = sum(1 for fact in required_facts if fact.lower() in reply)
        return found / len(required_facts)

    def get_grade_breakdown(self) -> Optional[Dict[str, float]]:
        """Return the grade breakdown if the episode is complete."""
        if not self._state.completed or self._grade_breakdown is None:
            return None
        return {
            "task_id": self._state.task_id,
            "total_score": self._state.cumulative_reward,
            "breakdown": self._grade_breakdown,
            "steps_taken": self._state.step_count,
            "max_steps": self._fixture["max_steps"] if self._fixture else 0,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        status_message: str = "",
        done: bool = False,
        reward: float | None = 0.0,
        tool_result: Optional[Dict[str, Any]] = None,
    ) -> SupportObservation:
        return SupportObservation(
            done=done,
            reward=reward,
            ticket_thread=self._fixture["public"]["ticket_thread"]
            if self._fixture
            else [],
            latest_tool_result=tool_result,
            retrieved_artifacts=list(self._retrieved_artifact_ids),
            current_classification=self._state.chosen_classification,
            current_priority=self._state.chosen_priority,
            current_route=self._state.chosen_route,
            status_message=status_message,
            metadata={
                "task_id": self._state.task_id,
                "step_count": self._state.step_count,
                "max_steps": self._fixture["max_steps"] if self._fixture else 0,
            },
        )
