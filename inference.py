"""
Financial Fraud Detective — Baseline Agent (inference.py)
=========================================================
Mandatory baseline script for the Meta PyTorch OpenEnv x Scaler Hackathon.

Usage:
    export API_BASE_URL=https://api.openai.com/v1
    export MODEL_NAME=gpt-4o-mini
    export HF_TOKEN=your_hf_token_here        # optional if server is local
    python3 inference.py

Environment variables:
    API_BASE_URL  — Base URL of the LLM API  (default: https://api.openai.com/v1)
    MODEL_NAME    — Model to use             (default: gpt-4o-mini)
    HF_TOKEN      — HuggingFace token / API key for the LLM
    SERVER_URL    — Override the environment server URL (default: http://localhost:8000)
"""

import os
import sys
import json
import time
import requests
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
SERVER_URL   = os.getenv("SERVER_URL",   "http://localhost:8000")

TASKS        = ["easy", "medium", "hard"]
MAX_STEPS    = 25          # safety ceiling — hard task has 20 txns
SLEEP_SECS   = 0.3         # polite delay between LLM calls

# ── OpenAI client ──────────────────────────────────────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "sk-placeholder",   # some proxies ignore the key
)

# ── System prompt sent once per episode ───────────────────────────────────────
SYSTEM_PROMPT = """You are an expert financial fraud analyst reviewing bank transactions.

For each transaction you will receive details including: transaction ID, amount, currency,
account, merchant, location, timestamp, and any risk signals.

You must respond with a JSON object — no extra text, no markdown — containing exactly:
{
  "action_type": "<one of: flag | freeze | escalate | ignore>",
  "reasoning": "<one sentence explaining your decision>"
}

Action guide:
- flag      : transaction is suspicious / likely fraud (card skimming, unusual location, etc.)
- freeze    : account is under active coordinated attack (multiple rapid hits from same IP)
- escalate  : transaction is part of a money-laundering chain (structured layering / integration)
- ignore    : transaction is legitimate — do not act

Reward signals you should optimise for:
  +0.4  correct flag on fraudulent transaction
  +0.5  correct escalate on money-laundering transaction
  +0.2  correct ignore on legitimate transaction
  +0.1  correct freeze on account under attack
  -0.3  false positive (flag/escalate on legitimate)
  -0.5  missed fraud (ignore on fraudulent) — worst penalty
  -0.4  missed laundering (ignore on money-laundering)
  -0.2  wrong action type (flag when should escalate, or vice-versa)
  -0.1  repeated action on same transaction

Be conservative: if in doubt between flag and escalate, prefer escalate for large cross-border
transfers between multiple currencies. Prefer freeze when you see a coordinated IP pattern
across multiple accounts.
"""

# ── Helpers ────────────────────────────────────────────────────────────────────

def server_get(path: str) -> dict:
    r = requests.get(f"{SERVER_URL}{path}", timeout=10)
    r.raise_for_status()
    return r.json()


def server_post(path: str, body: dict) -> dict:
    r = requests.post(f"{SERVER_URL}{path}", json=body, timeout=10)
    r.raise_for_status()
    return r.json()


def observation_to_text(obs: dict) -> str:
    """Convert an observation dict to a readable string for the LLM."""
    lines = ["=== Transaction for Review ==="]
    for k, v in obs.items():
        if v is not None:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def call_llm(conversation: list[dict]) -> dict:
    """Call the LLM and return the parsed action dict."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=conversation,
        temperature=0.0,
        max_tokens=200,
    )
    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return json.loads(raw)


def run_episode(task_id: str) -> float:
    """Run a single episode for the given task. Returns the final cumulative score."""
    print(f"\n{'='*60}")
    print(f"  TASK: {task_id.upper()}")
    print(f"{'='*60}")

    # Reset environment
    reset_resp = server_post("/reset", {"task_id": task_id})
    print(f"  Episode started. "
          f"Transactions: {reset_resp.get('total_transactions', '?')}")

    # Build conversation — system prompt stays constant throughout the episode
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    seen_transactions: set[str] = set()
    step = 0
    cumulative_score = 0.0

    while step < MAX_STEPS:
        # Get current state
        state = server_get("/state")

        if state.get("done", False):
            cumulative_score = state.get("cumulative_score", cumulative_score)
            print(f"\n  ✓ Episode complete after {step} steps.")
            break

        obs = state.get("current_observation")
        if obs is None:
            print("  No observation returned — episode may be complete.")
            cumulative_score = state.get("cumulative_score", cumulative_score)
            break

        txn_id = obs.get("transaction_id", f"TXN_{step}")

        # Guard against infinite-loop penalty
        if txn_id in seen_transactions:
            print(f"  [WARN] Already acted on {txn_id} — skipping to avoid penalty.")
            break
        seen_transactions.add(txn_id)

        # Ask LLM
        user_msg = observation_to_text(obs)
        conversation.append({"role": "user", "content": user_msg})

        try:
            action = call_llm(conversation)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [ERROR] LLM parse error on step {step}: {e}. Defaulting to ignore.")
            action = {"action_type": "ignore", "reasoning": "Parse error fallback."}

        action_type = action.get("action_type", "ignore")
        reasoning   = action.get("reasoning",   "No reasoning provided.")

        # Add assistant reply to conversation for context
        conversation.append({
            "role": "assistant",
            "content": json.dumps(action),
        })

        # Submit action
        step_resp = server_post("/step", {
            "action_type":     action_type,
            "transaction_id":  txn_id,
            "reasoning":       reasoning,
        })

        reward           = step_resp.get("reward", {})
        score            = reward.get("score", 0.0)
        cumulative_score = reward.get("cumulative_score", cumulative_score)
        feedback         = reward.get("feedback", "")
        done             = reward.get("done", False)

        print(f"  [{step+1:02d}] {txn_id}  →  {action_type:<10}  "
              f"score: {score:+.1f}  cumulative: {cumulative_score:.2f}  | {feedback}")

        step += 1
        time.sleep(SLEEP_SECS)

        if done:
            print(f"\n  ✓ Episode complete after {step} steps.")
            break

    return cumulative_score


def main():
    # ── Preflight checks ──────────────────────────────────────────────────────
    print("\nFinancial Fraud Detective — Baseline Agent")
    print(f"  Server  : {SERVER_URL}")
    print(f"  Model   : {MODEL_NAME}")
    print(f"  API URL : {API_BASE_URL}")

    # Health check
    try:
        health = server_get("/health")
        print(f"  Server health: {health}")
    except Exception as e:
        print(f"\n[FATAL] Cannot reach server at {SERVER_URL}: {e}")
        print("  Start the server first:  python3 server.py")
        sys.exit(1)

    # List tasks
    tasks_resp = server_get("/tasks")
    available  = [t["task_id"] for t in tasks_resp.get("tasks", [])]
    print(f"  Available tasks: {available}\n")

    # ── Run all three tasks ───────────────────────────────────────────────────
    results: dict[str, float] = {}
    start_time = time.time()

    for task_id in TASKS:
        if task_id not in available:
            print(f"  [SKIP] Task '{task_id}' not found on server.")
            continue
        score = run_episode(task_id)
        results[task_id] = score

    elapsed = time.time() - start_time

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL SCORES")
    print(f"{'='*60}")
    perfect = {"easy": 1.2, "medium": 3.6, "hard": 6.4}
    for task_id, score in results.items():
        perf  = perfect.get(task_id, 1.0)
        pct   = (score / perf * 100) if perf else 0
        print(f"  {task_id:<8}  score: {score:.2f} / {perf:.2f}  ({pct:.1f}%)")
    print(f"\n  Total elapsed: {elapsed:.1f}s")

    if elapsed > 1200:   # 20 min
        print("  [WARN] Inference exceeded 20-minute budget.")
    else:
        print("  [OK]   Completed within 20-minute budget.")

    print()


if __name__ == "__main__":
    main()
