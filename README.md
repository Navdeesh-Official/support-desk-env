---
title: SupportDeskEnv
emoji: 🎧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - customer-support
---

# SupportDeskEnv

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment that simulates B2B SaaS support ticket resolution. Agents must investigate customer issues using account data, billing records, service health status, and a knowledge base, then classify, route, and draft professional replies.

## Motivation

Real-world support automation requires more than text generation — agents must reason over structured data, follow policy constraints, and avoid harmful claims. SupportDeskEnv provides a deterministic benchmark for evaluating these capabilities with partial-reward grading.

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| `access_reset` | Easy | A team member is locked out. Investigate the account, identify the MFA issue, and guide the admin through the reset process. |
| `duplicate_charge_refund` | Medium | A customer was billed twice for their Enterprise plan. Verify the duplicate invoice, issue a refund per policy, and confirm with the customer. |
| `incident_sla_credit` | Hard | A customer demands compensation after a service outage. Check health status, calculate SLA credits, escalate appropriately, and draft a careful reply without over-promising. |

## Actions

| Action Type | Payload Fields | Description |
|------------|---------------|-------------|
| `view_account` | `account_id` | Look up account information |
| `view_billing` | `billing_account_id` | View billing ledger and charges |
| `view_health` | `service_name` | Check service health status and incidents |
| `search_kb` | `query` | Search the knowledge base |
| `classify_ticket` | `classification` | Set ticket classification (access, billing, outage, general) |
| `set_priority` | `priority` | Set priority (low, medium, high, critical) |
| `route_ticket` | `route_to` | Route to team (l1_support, l2_support, billing_team, engineering) |
| `draft_reply` | `reply_text` | Draft the customer reply |
| `resolve_ticket` | `resolution_code` | Resolve and end episode (resolved, escalated, closed) |

## Scoring

Each episode is graded on five dimensions:

| Dimension | Weight | What's Measured |
|-----------|--------|-----------------|
| Evidence Retrieval | 0.20 | Did the agent use the right tools to gather required data? |
| Classification & Priority | 0.20 | Was the ticket correctly classified and prioritized? |
| Policy Decision | 0.25 | Does the reply avoid forbidden claims (wrong credits, unsupported promises)? |
| Routing/Escalation | 0.15 | Was the ticket routed to the correct team? |
| Reply Quality | 0.20 | Does the reply contain all required facts? |

Final score is clamped to `0.0–1.0`. Per-step reward is `0` during the episode and the final score at resolution.

## Local Setup

```bash
pip install -r requirements.txt
```

### Run the server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run tests

```bash
pytest tests/ -v
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new episode (pass `task_id` in body) |
| `/step` | POST | Execute an action (pass action in `{"action": {...}}`) |
| `/state` | GET | Get current environment state |
| `/health` | GET | Health check |
| `/tasks` | GET | List the 3 benchmark tasks |
| `/grader` | GET | Get grade breakdown (409 until episode is done) |
| `/baseline` | POST | Run rule-based baseline on all tasks |
| `/docs` | GET | Swagger UI |

## Baseline Scores

Rule-based baseline (deterministic, no LLM):

| Task | Score |
|------|-------|
| `access_reset` | 1.00 |
| `duplicate_charge_refund` | 1.00 |
| `incident_sla_credit` | 1.00 |
| **Mean** | **1.00** |

### OpenAI Baseline

```bash
export OPENAI_API_KEY="sk-..."
python baseline_openai.py --task access_reset
python baseline_openai.py  # runs all 3 tasks
```

### Mistral Baseline

```bash
export MISTRAL_API_KEY="..."
python baseline_mistral.py --task access_reset
python baseline_mistral.py  # runs all 3 tasks
```

## Docker / Hugging Face Spaces

### Local Docker

```bash
# Build with official OpenEnv base image (recommended)
docker build --build-arg BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest -t support-desk-env .

# Or build self-contained (no external base needed)
docker build -t support-desk-env .

# Run locally
docker run -p 8000:8000 support-desk-env
```

### Deploy to Hugging Face Spaces

#### Option A: Using OpenEnv CLI (recommended)

```bash
pip install openenv-core[cli]
openenv push --repo-id <your-username>/support-desk-env
```

#### Option B: Manual git push

```bash
# Create a new Space on huggingface.co/spaces/new with SDK: Docker
git clone https://huggingface.co/spaces/<your-username>/support-desk-env
cd support-desk-env
cp -r /path/to/OpenENV/* .
git add .
git commit -m "Deploy SupportDeskEnv"
git push
```

The Space will be live at `https://<your-username>-support-desk-env.hf.space`.

#### Verify deployment

```bash
curl https://<your-username>-support-desk-env.hf.space/health
# {"status":"healthy"}
```

#### Install as a client package

```bash
pip install git+https://huggingface.co/spaces/<your-username>/support-desk-env
```

## Project Structure

```
support_desk_env/
├── openenv.yaml          # OpenEnv manifest
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container build
├── models.py             # Pydantic Action, Observation, State
├── fixtures.py           # Task fixtures and simulated tool lookups
├── environment.py        # SupportDeskEnv (reset/step/state + grading)
├── client.py             # Typed client wrapper
├── baseline_openai.py    # OpenAI baseline agent
├── baseline_mistral.py   # Mistral baseline agent
├── server/
│   ├── __init__.py
│   └── app.py            # FastAPI app with OpenEnv + custom endpoints
├── tests/
│   ├── __init__.py
│   └── test_support_desk.py
└── README.md
```
