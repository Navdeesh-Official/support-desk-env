"""Tests for SupportDeskEnv."""

from __future__ import annotations

import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import SupportDeskEnv
from fixtures import TASK_IDS, get_fixture, get_tool_result, get_task_summary
from models import ActionType, SupportAction, SupportObservation, SupportState


# ======================================================================
# Fixture tests
# ======================================================================


class TestFixtures:
    def test_all_task_ids_exist(self):
        assert set(TASK_IDS) == {
            "access_reset",
            "duplicate_charge_refund",
            "incident_sla_credit",
        }

    def test_get_fixture_returns_deep_copy(self):
        f1 = get_fixture("access_reset")
        f2 = get_fixture("access_reset")
        f1["public"]["ticket_thread"][0]["message"] = "TAMPERED"
        assert f2["public"]["ticket_thread"][0]["message"] != "TAMPERED"

    def test_fixture_structure(self):
        for tid in TASK_IDS:
            f = get_fixture(tid)
            assert f["task_id"] == tid
            assert "public" in f
            assert "hidden" in f
            assert "ticket_thread" in f["public"]
            assert "max_steps" in f
            assert f["max_steps"] > 0

    def test_hidden_has_grading_fields(self):
        for tid in TASK_IDS:
            h = get_fixture(tid)["hidden"]
            assert "correct_classification" in h
            assert "correct_priority" in h
            assert "valid_route" in h
            assert "required_evidence" in h
            assert "required_reply_facts" in h
            assert "forbidden_claims" in h
            assert "weights" in h

    def test_tool_result_account(self):
        r = get_tool_result("view_account", "ACC-1042")
        assert r["found"] is True
        assert r["data"]["company"] == "NovaTech Solutions"

    def test_tool_result_billing(self):
        r = get_tool_result("view_billing", "ACC-2087")
        assert r["found"] is True
        # Should have duplicate invoice
        invoices = r["data"]["invoices"]
        assert len(invoices) == 3

    def test_tool_result_health(self):
        r = get_tool_result("view_health", "api-gateway")
        assert r["found"] is True
        assert r["data"]["status"] == "degraded"

    def test_tool_result_kb_search(self):
        r = get_tool_result("search_kb", "mfa")
        assert r["found"] is True
        assert len(r["results"]) > 0

    def test_tool_result_not_found(self):
        r = get_tool_result("view_account", "ACC-9999")
        assert r["found"] is False

    def test_get_task_summary(self):
        s = get_task_summary("access_reset")
        assert s["task_id"] == "access_reset"
        assert s["difficulty"] == "easy"
        assert "description" in s


# ======================================================================
# Environment tests
# ======================================================================


class TestEnvironment:
    def _run_perfect_access(self) -> SupportDeskEnv:
        env = SupportDeskEnv()
        env.reset(task_id="access_reset")
        env.step(
            SupportAction(action_type=ActionType.VIEW_ACCOUNT, account_id="ACC-1042")
        )
        env.step(SupportAction(action_type=ActionType.SEARCH_KB, query="mfa reset"))
        env.step(
            SupportAction(
                action_type=ActionType.CLASSIFY_TICKET, classification="access"
            )
        )
        env.step(SupportAction(action_type=ActionType.SET_PRIORITY, priority="medium"))
        env.step(
            SupportAction(action_type=ActionType.ROUTE_TICKET, route_to="l1_support")
        )
        env.step(
            SupportAction(
                action_type=ActionType.DRAFT_REPLY,
                reply_text=(
                    "Hi Sarah, James needs to reset his MFA. The admin can "
                    "reset from Settings > Users > Reset MFA."
                ),
            )
        )
        env.step(
            SupportAction(
                action_type=ActionType.RESOLVE_TICKET, resolution_code="resolved"
            )
        )
        return env

    def test_reset_returns_observation(self):
        env = SupportDeskEnv()
        obs = env.reset(task_id="access_reset")
        assert isinstance(obs, SupportObservation)
        assert obs.done is False
        assert len(obs.ticket_thread) == 1

    def test_state_after_reset(self):
        env = SupportDeskEnv()
        env.reset(task_id="access_reset")
        s = env.state()
        assert isinstance(s, SupportState)
        assert s.task_id == "access_reset"
        assert s.step_count == 0
        assert s.completed is False

    def test_perfect_access_score(self):
        env = self._run_perfect_access()
        assert env.state().cumulative_reward == 1.0

    def test_perfect_access_grade_breakdown(self):
        env = self._run_perfect_access()
        g = env.get_grade_breakdown()
        assert g is not None
        assert g["total_score"] == 1.0
        assert all(v == 1.0 for v in g["breakdown"].values())

    def test_perfect_duplicate_charge(self):
        env = SupportDeskEnv()
        env.reset(task_id="duplicate_charge_refund")
        env.step(
            SupportAction(action_type=ActionType.VIEW_ACCOUNT, account_id="ACC-2087")
        )
        env.step(
            SupportAction(
                action_type=ActionType.VIEW_BILLING, billing_account_id="ACC-2087"
            )
        )
        env.step(
            SupportAction(action_type=ActionType.SEARCH_KB, query="duplicate charge")
        )
        env.step(
            SupportAction(
                action_type=ActionType.CLASSIFY_TICKET, classification="billing"
            )
        )
        env.step(SupportAction(action_type=ActionType.SET_PRIORITY, priority="high"))
        env.step(
            SupportAction(action_type=ActionType.ROUTE_TICKET, route_to="billing_team")
        )
        env.step(
            SupportAction(
                action_type=ActionType.DRAFT_REPLY,
                reply_text=(
                    "Hi Priya, I found the duplicate invoice INV-9102-DUP. "
                    "Refund initiated. Allow 5-10 business days."
                ),
            )
        )
        obs = env.step(
            SupportAction(
                action_type=ActionType.RESOLVE_TICKET, resolution_code="resolved"
            )
        )
        assert obs.reward == 1.0

    def test_perfect_incident_sla(self):
        env = SupportDeskEnv()
        env.reset(task_id="incident_sla_credit")
        env.step(
            SupportAction(action_type=ActionType.VIEW_ACCOUNT, account_id="ACC-3055")
        )
        env.step(
            SupportAction(
                action_type=ActionType.VIEW_HEALTH, service_name="api-gateway"
            )
        )
        env.step(SupportAction(action_type=ActionType.SEARCH_KB, query="sla credit"))
        env.step(
            SupportAction(
                action_type=ActionType.CLASSIFY_TICKET, classification="outage"
            )
        )
        env.step(SupportAction(action_type=ActionType.SET_PRIORITY, priority="high"))
        env.step(
            SupportAction(action_type=ActionType.ROUTE_TICKET, route_to="l2_support")
        )
        env.step(
            SupportAction(
                action_type=ActionType.DRAFT_REPLY,
                reply_text=(
                    "Hi Marco, I acknowledge the api-gateway incident. "
                    "Per our SLA, we are escalating to L2 support for credit calculation."
                ),
            )
        )
        obs = env.step(
            SupportAction(
                action_type=ActionType.RESOLVE_TICKET, resolution_code="escalated"
            )
        )
        assert obs.reward == 1.0

    def test_partial_score_wrong_classification(self):
        env = SupportDeskEnv()
        env.reset(task_id="access_reset")
        env.step(
            SupportAction(action_type=ActionType.VIEW_ACCOUNT, account_id="ACC-1042")
        )
        env.step(
            SupportAction(
                action_type=ActionType.CLASSIFY_TICKET, classification="billing"
            )
        )
        env.step(SupportAction(action_type=ActionType.SET_PRIORITY, priority="medium"))
        env.step(
            SupportAction(action_type=ActionType.ROUTE_TICKET, route_to="l1_support")
        )
        env.step(
            SupportAction(
                action_type=ActionType.DRAFT_REPLY,
                reply_text="Hi, we reset the MFA for the admin.",
            )
        )
        obs = env.step(
            SupportAction(
                action_type=ActionType.RESOLVE_TICKET, resolution_code="resolved"
            )
        )
        assert 0.0 < obs.reward < 1.0

    def test_forbidden_claim_penalty(self):
        env = SupportDeskEnv()
        env.reset(task_id="incident_sla_credit")
        env.step(
            SupportAction(action_type=ActionType.VIEW_ACCOUNT, account_id="ACC-3055")
        )
        env.step(
            SupportAction(
                action_type=ActionType.VIEW_HEALTH, service_name="api-gateway"
            )
        )
        env.step(SupportAction(action_type=ActionType.SEARCH_KB, query="sla credit"))
        env.step(
            SupportAction(
                action_type=ActionType.CLASSIFY_TICKET, classification="outage"
            )
        )
        env.step(SupportAction(action_type=ActionType.SET_PRIORITY, priority="high"))
        env.step(
            SupportAction(action_type=ActionType.ROUTE_TICKET, route_to="l2_support")
        )
        env.step(
            SupportAction(
                action_type=ActionType.DRAFT_REPLY,
                reply_text=(
                    "Hi Marco, I acknowledge the incident. "
                    "We guarantee a full month credit as compensation. "
                    "The SLA escalation is in progress."
                ),
            )
        )
        obs = env.step(
            SupportAction(
                action_type=ActionType.RESOLVE_TICKET, resolution_code="escalated"
            )
        )
        # Should be penalized for "guarantee" and "full month"
        assert obs.reward < 1.0

    def test_max_steps_auto_resolve(self):
        env = SupportDeskEnv()
        env.reset(task_id="access_reset")
        for _ in range(8):
            obs = env.step(
                SupportAction(
                    action_type=ActionType.VIEW_ACCOUNT, account_id="ACC-1042"
                )
            )
        assert obs.done is True
        assert "max steps" in obs.status_message

    def test_no_actions_score(self):
        env = SupportDeskEnv()
        env.reset(task_id="access_reset")
        for _ in range(8):
            obs = env.step(
                SupportAction(
                    action_type=ActionType.VIEW_ACCOUNT, account_id="ACC-1042"
                )
            )
        # No classification, no reply, no routing
        assert obs.reward < 0.5

    def test_invalid_task_id_raises(self):
        env = SupportDeskEnv()
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset(task_id="nonexistent")

    def test_step_before_reset_raises(self):
        env = SupportDeskEnv()
        with pytest.raises(RuntimeError, match="Call reset"):
            env.step(
                SupportAction(
                    action_type=ActionType.VIEW_ACCOUNT, account_id="ACC-1042"
                )
            )

    def test_grade_breakdown_none_before_done(self):
        env = SupportDeskEnv()
        env.reset(task_id="access_reset")
        env.step(
            SupportAction(action_type=ActionType.VIEW_ACCOUNT, account_id="ACC-1042")
        )
        assert env.get_grade_breakdown() is None


# ======================================================================
# API / Server tests
# ======================================================================


class TestServerEndpoints:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from server.app import app

        return TestClient(app)

    def test_tasks_endpoint(self, client):
        r = client.get("/tasks")
        assert r.status_code == 200
        tasks = r.json()
        assert len(tasks) == 3
        assert all("task_id" in t for t in tasks)

    def test_health_endpoint(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_grader_returns_409_when_no_episode(self, client):
        r = client.get("/grader")
        assert r.status_code == 409

    def test_baseline_endpoint(self, client):
        r = client.post("/baseline")
        assert r.status_code == 200
        data = r.json()
        assert "tasks" in data
        assert "mean_score" in data
        assert len(data["tasks"]) == 3

    def test_reset_endpoint(self, client):
        r = client.post("/reset", json={"task_id": "access_reset"})
        assert r.status_code == 200
        obs = r.json()
        assert obs["done"] is False
