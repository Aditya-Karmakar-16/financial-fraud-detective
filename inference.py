"""
Financial Fraud Detective — Baseline Agent (inference.py)
=========================================================
Mandatory baseline script for the Meta PyTorch OpenEnv x Scaler Hackathon.

Structured stdout format (required by validator):
  [START] task=easy
  [STEP] step=1 reward=0.4 cumulative=0.4
  [END] task=easy score=1.2 steps=5
"""

import os
import sys
import json
import time
import requests

# ── Configuration ──────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN",     "").strip()
SERVER_URL   = os.getenv("SERVER_URL",   "http://localhost:7860")

TASKS     = ["easy", "medium", "hard"]
MAX_STEPS = 25
SLEEP_SECS = 0.3

# NOTE: OpenAI client is intentionally NOT created at module level.
# openai>=1.0 raises immediately on empty api_key, crashing before main() runs.
# Client is created lazily inside call_llm() only when actually needed.

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert financial fraud analyst reviewing bank transactions.

You must respond with a JSON object — no extra text, no markdown — containing exactly:
{
  "action_type": "<one of: flag | freeze | escalate | ignore>",
  "reasoning": "<one sentence explaining your decision>"
}

Action guide:
- flag      : suspicious / likely fraud (card skimming, unusual location)
- freeze    : account under active coordinated attack (multiple rapid hits same IP)
- escalate  : money-laundering chain (structured layering / integration)
- ignore    : transaction is legitimate

Reward signals:
  +0.4  correct flag       -0.5  missed fraud (worst)
  +0.5  correct escalate   -0.4  missed laundering
  +0.2  correct ignore     -0.3  false positive
  +0.1  correct freeze     -0.2  wrong action type
"""

# ── Server helpers ─────────────────────────────────────────────────────────────
def server_get(path: str) -> dict:
    r = requests.get(f"{SERVER_URL}{path}", timeout=10)
    r.raise_for_status()
    return r.json()

def server_post(path: str, body: dict) -> dict:
    r = requests.post(f"{SERVER_URL}{path}", json=body, timeout=10)
    r.raise_for_status()
    return r.json()

def observation_to_text(obs: dict) -> str:
    lines = ["=== Transaction for Review ==="]
    for k, v in obs.items():
        if v is not None:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)

# ── LLM call (lazy client init) ────────────────────────────────────────────────
def call_llm(conversation: list) -> dict:
    from openai import OpenAI
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=conversation,
        temperature=0.0,
        max_tokens=200,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)

# ── Episode runner ─────────────────────────────────────────────────────────────
def run_episode(task_id: str) -> float:
    # Required structured log: [START]
    print(f"[START] task={task_id}", flush=True)

    reset_resp = server_post("/reset", {"task_id": task_id})

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    seen_txns: set = set()
    step = 0
    cumulative_score = 0.0

    while step < MAX_STEPS:
        state = server_get("/state")

        if state.get("done", False):
            cumulative_score = state.get("cumulative_score", cumulative_score)
            break

        obs = state.get("current_observation")
        if obs is None:
            cumulative_score = state.get("cumulative_score", cumulative_score)
            break

        txn_id = obs.get("transaction_id", f"TXN_{step}")
        if txn_id in seen_txns:
            break
        seen_txns.add(txn_id)

        conversation.append({"role": "user", "content": observation_to_text(obs)})
        try:
            action = call_llm(conversation)
        except Exception as e:
            action = {"action_type": "ignore", "reasoning": f"fallback: {e}"}

        action_type = action.get("action_type", "ignore")
        reasoning   = action.get("reasoning", "")
        conversation.append({"role": "assistant", "content": json.dumps(action)})

        step_resp = server_post("/step", {
            "action_type": action_type,
            "transaction_id": txn_id,
            "reasoning": reasoning,
        })

        reward           = step_resp if "score" in step_resp else step_resp.get("reward", step_resp)
        score            = reward.get("score", 0.0)
        cumulative_score = reward.get("cumulative_score", cumulative_score)
        done             = reward.get("done", False)

        # Required structured log: [STEP]
        print(f"[STEP] step={step + 1} reward={round(score, 3)} cumulative={round(cumulative_score, 3)}", flush=True)

        step += 1
        time.sleep(SLEEP_SECS)

        if done:
            break

    # Required structured log: [END]
    print(f"[END] task={task_id} score={round(cumulative_score, 3)} steps={step}", flush=True)
    return cumulative_score


def main():
    # Health check
    try:
        server_get("/health")
    except Exception as e:
        print(f"[END] phase=health_check error={e}", flush=True)
        sys.exit(1)

    tasks_resp = server_get("/tasks")
    available  = [t["task_id"] for t in tasks_resp.get("tasks", [])]

    start_time = time.time()
    results: dict = {}

    for task_id in TASKS:
        if task_id not in available:
            continue
        results[task_id] = run_episode(task_id)

    elapsed = time.time() - start_time
    perfect = {"easy": 1.2, "medium": 3.6, "hard": 6.4}

    for task_id, score in results.items():
        perf = perfect.get(task_id, 1.0)
        pct  = round((score / perf * 100) if perf else 0, 1)
        print(f"[STEP] summary task={task_id} score={round(score,3)} perfect={perf} pct={pct}", flush=True)

    print(f"[END] phase=inference elapsed={round(elapsed,1)}s within_budget={elapsed <= 1200}", flush=True)


if __name__ == "__main__":
    main()
