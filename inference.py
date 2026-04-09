"""
Financial Fraud Detective — Baseline Agent (inference.py)
=========================================================
Mandatory baseline script for the Meta PyTorch OpenEnv x Scaler Hackathon.

STDOUT FORMAT (exact — any deviation fails evaluation):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
import time
from typing import List, Optional

import requests
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
# Support both HF_TOKEN and API_KEY — validator injects API_KEY via LiteLLM proxy
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
SERVER_URL   = os.getenv("SERVER_URL",   "http://localhost:7860")

TASKS      = ["easy", "medium", "hard"]
MAX_STEPS  = 25
SLEEP_SECS = 0.3
BENCHMARK  = "financial-fraud-detective"
SUCCESS_THRESHOLD = 0.5

# NOTE: OpenAI client created lazily inside call_llm() — never at module level.
# openai>=1.0 raises AuthenticationError immediately on empty api_key.

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

# ── Structured stdout loggers (exact format required) ─────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

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

# ── LLM call — lazy client init ────────────────────────────────────────────────
def call_llm(conversation: list) -> dict:
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
PERFECT = {"easy": 1.2, "medium": 3.6, "hard": 6.4}

def run_episode(task_id: str) -> float:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    server_post("/reset", {"task_id": task_id})

    conversation         = [{"role": "system", "content": SYSTEM_PROMPT}]
    seen_txns: set       = set()
    step                 = 0
    cumulative           = 0.0
    rewards: List[float] = []
    error_msg            = None

    try:
        while step < MAX_STEPS:
            state = server_get("/state")

            if state.get("done", False):
                cumulative = state.get("cumulative_score", cumulative)
                break

            obs = state.get("current_observation")
            if obs is None:
                cumulative = state.get("cumulative_score", cumulative)
                break

            txn_id = obs.get("transaction_id", f"TXN_{step}")
            if txn_id in seen_txns:
                break
            seen_txns.add(txn_id)

            conversation.append({"role": "user", "content": observation_to_text(obs)})
            try:
                action    = call_llm(conversation)
                error_msg = None
            except Exception as e:
                action    = {"action_type": "ignore", "reasoning": f"fallback: {e}"}
                error_msg = str(e)

            action_type = action.get("action_type", "ignore")
            reasoning   = action.get("reasoning", "")
            conversation.append({"role": "assistant", "content": json.dumps(action)})

            step_resp = server_post("/step", {
                "action_type":    action_type,
                "transaction_id": txn_id,
                "reasoning":      reasoning,
            })

            reward_block = step_resp if "score" in step_resp else step_resp.get("reward", step_resp)
            score        = reward_block.get("score", 0.0)
            cumulative   = reward_block.get("cumulative_score", cumulative)
            done         = reward_block.get("done", False)

            rewards.append(score)
            step += 1

            log_step(step=step, action=action_type, reward=score, done=done, error=error_msg)

            time.sleep(SLEEP_SECS)

            if done:
                break

    except Exception as e:
        error_msg = str(e)

    perfect  = PERFECT.get(task_id, 1.0)
    norm     = min(max(cumulative / perfect, 0.0), 1.0) if perfect else 0.0
    success  = norm >= SUCCESS_THRESHOLD

    log_end(success=success, steps=step, score=norm, rewards=rewards)
    return cumulative


def main():
    try:
        server_get("/health")
    except Exception as e:
        log_end(success=False, steps=0, score=0.0, rewards=[])
        sys.exit(1)

    tasks_resp = server_get("/tasks")
    available  = [t["task_id"] for t in tasks_resp.get("tasks", [])]

    for task_id in TASKS:
        if task_id not in available:
            continue
        run_episode(task_id)


if __name__ == "__main__":
    main()
