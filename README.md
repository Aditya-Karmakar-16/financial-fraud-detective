---
title: Financial Fraud Detective
emoji: 🕵️
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

# 🕵️ Financial Fraud Detective — OpenEnv

> **Meta PyTorch OpenEnv x Scaler Hackathon**  
> A real-world reinforcement learning environment where an AI agent acts as a financial fraud analyst.

---

## Overview

Banks lose **$485 billion per year** to fraud. This environment puts an AI agent in the seat of a fraud analyst reviewing live bank transactions across 5 currencies and 5 regions. At every step the agent must decide how to act — and every decision has consequences.

The agent reviews transactions one at a time, choosing from 4 actions:

| Action | When to use |
|--------|-------------|
| `flag` | Transaction is suspicious — unusual location, off-hours ATM, card skimming |
| `freeze` | Account is under coordinated attack — multiple rapid hits from same fraud IP |
| `escalate` | Transaction is part of a money-laundering chain — structured cross-border layering |
| `ignore` | Transaction is legitimate — no action needed |

Rewards are given at **every step**, not just end-of-episode, making this a rich dense-reward environment suitable for LLM agents, RL fine-tuning, and policy search.

---

## Currencies & Regions

| Currency | Region | Accounts |
|----------|--------|----------|
| INR — Indian Rupee | India | ACC_001, ACC_002, ACC_005, ACC_006 |
| USD — US Dollar | USA / Singapore | ACC_007 |
| GBP — British Pound | United Kingdom | ACC_003 |
| AED — UAE Dirham | UAE | ACC_004, ACC_008 |
| EUR — Euro | Germany, France | ACC_009, ACC_010 |

---

## Tasks

### 🟢 Easy — Single Fraud (5 transactions, perfect score: 1.2)
A fraudster uses a cloned card to make a USD ATM withdrawal in Nigeria at 3am on an INR account based in India. The other 4 transactions are legitimate. The agent must identify the single anomalous transaction without generating false positives.

### 🟡 Medium — Coordinated Card Testing (12 transactions, perfect score: 3.6)
A fraud ring runs a coordinated card-testing attack across three accounts (INR, GBP, AED) using the same fraud IP address. Multiple small transactions probe the accounts before a large withdrawal attempt. The agent must identify the attack pattern and freeze the affected accounts.

### 🔴 Hard — Money Laundering Chain (20 transactions, perfect score: 6.4)
A sophisticated money-laundering operation moves funds across 5 countries: INR → USD → AED → EUR, using structuring (smurfing) to stay below reporting thresholds. The agent must identify the layering chain and escalate the linked transactions while correctly ignoring the legitimate transactions interspersed throughout.

---

## Observation Space

Each step the agent receives a transaction observation:

```json
{
  "transaction_id": "TXN007",
  "amount": 1500.00,
  "currency": "USD",
  "account_id": "ACC_001",
  "merchant": "ATM Lagos Branch",
  "location": "Lagos, Nigeria",
  "timestamp": "2024-01-15T03:17:00Z",
  "risk_signals": ["unusual_location", "off_hours", "foreign_currency"]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `transaction_id` | string | Unique transaction ID |
| `amount` | float | Amount in local currency |
| `currency` | string | ISO 4217 code (INR/USD/GBP/AED/EUR) |
| `account_id` | string | Source account |
| `merchant` | string | Merchant or counterparty |
| `location` | string | Transaction location |
| `timestamp` | string | ISO 8601 timestamp |
| `risk_signals` | list | Pre-computed risk flags |

---

## Action Space

Discrete action space with 4 actions:

```json
{
  "action_type": "flag",
  "transaction_id": "TXN007",
  "reasoning": "USD ATM withdrawal in Nigeria at 3am on INR account — card skimming."
}
```

---

## Reward Function

Rewards are issued at every step:

| Outcome | Reward |
|---------|--------|
| Correctly flagged fraudulent transaction | **+0.4** |
| Correctly escalated money-laundering transaction | **+0.5** |
| Correctly ignored legitimate transaction | **+0.2** |
| Correctly froze account under active attack | **+0.1** |
| False positive (flagged/escalated legitimate) | **-0.3** |
| False negative (ignored fraud) | **-0.5** |
| Missed laundering (ignored money-laundering) | **-0.4** |
| Wrong action type (flag vs escalate) | **-0.2** |
| Repeated action on same transaction | **-0.1** |

Reward range: **-1.0 to +1.0** per step.

---

## API Endpoints

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/health` | GET | — | Health check |
| `/tasks` | GET | — | List all 3 tasks |
| `/reset` | POST | `{"task_id": "easy"}` | Start new episode |
| `/state` | GET | — | Current observation + running score |
| `/step` | POST | `{"action_type", "transaction_id", "reasoning"}` | Submit action, receive reward |

---

## Setup & Running

### Local (no Docker)

```bash
# Install dependencies
pip3 install fastapi uvicorn pydantic openai pyyaml requests

# Start the server
python3 server.py

# In a new terminal — run the baseline agent
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_openai_key_here
python3 inference.py
```

### Docker

```bash
# Build
docker build -t fraud-detective-env .

# Run
docker run -p 7860:7860 \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e HF_TOKEN=your_key_here \
  fraud-detective-env

# Test
curl http://localhost:7860/health
```

### HuggingFace Space

```bash
# Validate
openenv validate https://your-username-financial-fraud-detective.hf.space
```

---

## Baseline Agent Scores

The baseline agent uses `gpt-4o-mini` via the OpenAI API with a zero-shot prompt encoding the reward function.

| Task | Agent Score | Perfect Score | % of Perfect |
|------|-------------|---------------|--------------|
| Easy | 1.2 | 1.2 | 100% |
| Medium | 3.6 | 3.6 | 100% |
| Hard | 6.4 | 6.4 | 100% |

Baseline inference completes in under 5 minutes on 2 vCPU / 8GB RAM.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | LLM API base URL (e.g. `https://api.openai.com/v1`) |
| `MODEL_NAME` | Yes | Model name (e.g. `gpt-4o-mini`) |
| `HF_TOKEN` | Yes | API key for the LLM provider |
| `SERVER_URL` | No | Override server URL (default: `http://localhost:8000`) |

---

## File Structure

```
financial-fraud-detective/
├── models.py          # Observation, Action, Reward dataclasses
├── data.py            # 37 transactions across 5 currencies, 3 tasks
├── environment.py     # step(), reset(), state() + task graders
├── server.py          # FastAPI HTTP server
├── inference.py       # Baseline LLM agent (mandatory)
├── Dockerfile         # Container for HuggingFace Space
├── requirements.txt   # Python dependencies
├── openenv.yaml       # OpenEnv metadata
└── README.md          # This file
```

---

## Why This Environment

- **Real-world utility** — financial fraud costs $485B/year globally; this models actual analyst workflows
- **Dense rewards** — every step returns a signal, enabling efficient policy learning
- **Multi-currency complexity** — 5 currencies, 5 regions, realistic cross-border fraud patterns
- **Deterministic graders** — perfect answer keys ensure reproducible, fair scoring
- **Escalating difficulty** — easy → medium → hard tasks test increasingly sophisticated fraud reasoning
- **First multi-currency fraud environment in OpenEnv**

---

## License

MIT
# rebuild
