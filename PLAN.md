# SupportDeskEnv OpenEnv Plan

## Summary
- Build a standalone OpenEnv repo named `support_desk_env` using the current `openenv init` single-environment layout, not the upstream monorepo layout.
- The environment simulates one B2B SaaS support ticket per episode and is optimized for deterministic grading, partial rewards, and easy Hugging Face Spaces deployment.
- Ship exactly 3 canonical benchmark tasks:
  - `access_reset`: account access issue resolved from account data + KB guidance
  - `duplicate_charge_refund`: billing dispute resolved from billing ledger + refund policy
  - `incident_sla_credit`: outage complaint requiring health-status lookup, SLA reasoning, escalation, and safe customer communication

## Public Interfaces
- `SupportAction` is a Pydantic action model with `action_type` plus typed payload fields. Supported `action_type` values: `view_account`, `view_billing`, `view_health`, `search_kb`, `classify_ticket`, `set_priority`, `route_ticket`, `draft_reply`, `resolve_ticket`.
- `SupportObservation` exposes only agent-visible data: current ticket thread, latest tool result, retrieved artifacts, current triage fields, last reward delta, done flag, and a short status message.
- `SupportState` exposes episode metadata and agent-produced state only: `episode_id`, `task_id`, `step_count`, retrieved artifact IDs, chosen classification/priority/route, drafted reply, cumulative reward, and completion status.
- Keep hidden answer keys and grading targets in internal-only fixture/rubric models; never leak them through `state()`.
- Standard OpenEnv API: `reset()`, `step()`, `state()`.
- Extra HTTP endpoints:
  - `/tasks`: returns the 3 task summaries, expected difficulty, and `SupportAction.model_json_schema()`
  - `/grader`: returns `409` until the episode is done, then returns a full grade breakdown
  - `/baseline`: runs the baseline agent on the 3 canonical tasks in fixed order and returns per-task score, mean score, model, and prompt version

## Implementation Changes
- Scaffold the repo with OpenEnv, then replace the template models/client/server with support-desk-specific versions and add a root `Dockerfile` for Hugging Face Spaces. Keep the manifest on the current scaffolded schema and point it at `server.app:app`.
- Add a small deterministic fixture system:
  - Public task data: customer thread, account summary, visible ticket metadata, max step budget
  - Hidden task data: required evidence sources, correct classification, correct priority, valid routing target, allowed resolution codes, required reply facts, forbidden claims, and per-dimension weights
- Implement task progression as fixed, in-memory fixtures with no external DB or retrieval service. `reset(task_id)` deep-copies a fixture into episode state; `step()` dispatches one action, updates visible state, computes reward delta, and terminates on `resolve_ticket` or max steps.
- Use one shared reward rubric across tasks with fixed weights:
  - evidence retrieval `0.20`
  - classification + priority `0.20`
  - policy decision `0.25`
  - routing/escalation `0.15`
  - final customer reply quality `0.20`
  - penalties subtract for unsupported promises, wrong credits/refunds, or missing required escalation
  - final score is clamped to `0.0–1.0`; per-step reward is the incremental change in cumulative score
- Grade `draft_reply` against required facts and forbidden claims using deterministic string/rule checks first; avoid subjective model-as-judge grading in v1.
- Baseline script uses the OpenAI Python SDK, reads `OPENAI_API_KEY`, supports optional `OPENAI_MODEL`, defaults to `gpt-4.1-mini`, uses temperature `0`, and produces one JSON `SupportAction` per turn until the episode ends.
- README must include motivation, action/observation/state definitions, the 3 tasks with difficulty notes, local setup, Docker/HF deployment steps, validator command, `/baseline` usage, and recorded baseline scores.

## Test Plan
- Unit-test each fixture and rubric:
  - perfect trajectory reaches expected near-1.0 score
  - partially correct trajectory gets intermediate score
  - unsafe/wrong-credit trajectory is penalized
- API-test `reset/step/state` plus `/tasks`, `/grader`, and `/baseline` contract behavior.
- Integration-test each canonical task end-to-end with hand-written action sequences so graders are proven non-constant and difficulty actually increases.
- Smoke-test Docker startup, health endpoint, and `openenv validate` before recording baseline scores.
- Run the baseline script twice against the same task order and confirm stable outputs within a tight tolerance; store the reported benchmark numbers in the README.

## Assumptions and Defaults
- v1 is intentionally limited to 3 polished tasks; no task generator, auth layer, database, or web UI beyond what FastAPI/OpenEnv already provides.
- One ticket per episode is the core abstraction; no inbox queue in v1.
- `draft_reply` is the only free-text action field; all other behavior stays strongly typed for stable grading.
- Server should honor `PORT` when present and otherwise default to local OpenEnv-friendly settings.
- Follow the current OpenEnv code path that uses Pydantic models even though some older docs still show dataclass examples.

## References
- [OpenEnv repository](https://github.com/meta-pytorch/OpenEnv)
- [OpenEnv environment build guide](https://github.com/meta-pytorch/OpenEnv/tree/main/envs)
- [OpenEnv course: Building Your Own Environment](https://github.com/raun/openenv-course/tree/main/module-4)
- [OpenAI model docs: GPT-4.1 mini](https://platform.openai.com/docs/models/gpt-4.1-mini)
