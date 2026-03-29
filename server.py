"""
server.py — Financial Fraud Detective OpenEnv
=============================================
FastAPI server that exposes the environment over HTTP.
This is what HuggingFace Space runs.
Any AI agent connects to these endpoints to interact with the environment.

Endpoints:
  POST /reset      → start a new episode, get first transaction
  POST /step       → submit an action, get reward back
  GET  /state      → see current episode state
  GET  /health     → confirm server is alive (required by HF Space)
  GET  /tasks      → list all available tasks
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

from data import TASKS


# ─── Pydantic models (full spec-compliant versions) ───────────
# These are the REAL Pydantic models used in production.
# They replace the dataclass versions used for local testing.

class ObservationModel(BaseModel):
    transaction_id: str
    amount: float
    currency: str
    merchant: str
    merchant_category: str
    country: str
    timestamp: str
    account_id: str
    account_home_country: str
    account_age_days: int
    prior_txn_countries: list[str]
    prior_actions: list[str]
    step_number: int
    total_steps: int
    task_id: str
    ip_address: Optional[str] = None
    context_note: Optional[str] = None


class ActionModel(BaseModel):
    action_type: str = Field(
        ...,
        description="One of: flag, freeze, escalate, ignore",
        examples=["flag"]
    )
    transaction_id: str = Field(
        ...,
        description="The transaction_id from the current observation",
        examples=["TXN003"]
    )
    reasoning: str = Field(
        ...,
        description="Why the agent made this decision",
        examples=["Foreign ATM withdrawal at 3am from a country the account has never used"]
    )


class RewardModel(BaseModel):
    score: float
    done: bool
    correct: bool
    feedback: str
    cumulative_score: float
    info: dict


class ResetRequest(BaseModel):
    task_id: str = Field(
        default="easy",
        description="Which task to run: easy, medium, or hard",
        examples=["easy"]
    )


# ─── Reward values ─────────────────────────────────────────────
REWARDS = {
    "correct_flag":      +0.4,
    "correct_escalate":  +0.5,
    "correct_ignore":    +0.2,
    "correct_freeze":    +0.1,
    "wrong_flag":        -0.3,
    "wrong_escalate":    -0.3,
    "missed_fraud":      -0.5,
    "missed_laundering": -0.4,
    "wrong_action_type": -0.2,
    "repeat_action":     -0.1,
}


# ─── Episode state (in-memory) ─────────────────────────────────
# Stores the current episode. One session at a time.
# In production you would use session IDs for multiple concurrent agents.
class EpisodeState:
    def __init__(self):
        self.task = None
        self.transactions = []
        self.current_step = 0
        self.cumulative_score = 0.0
        self.actions_taken = {}
        self.done = False
        self.initialized = False

    def reset(self, task_id: str):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose: easy, medium, hard")
        self.task = TASKS[task_id]
        self.transactions = self.task["transactions"]
        self.current_step = 0
        self.cumulative_score = 0.0
        self.actions_taken = {}
        self.done = False
        self.initialized = True

    def current_observation(self) -> dict:
        txn = self.transactions[self.current_step]
        return {
            "transaction_id": txn["transaction_id"],
            "amount": txn["amount"],
            "currency": txn["currency"],
            "merchant": txn["merchant"],
            "merchant_category": txn["merchant_category"],
            "country": txn["country"],
            "timestamp": txn["timestamp"],
            "account_id": txn["account_id"],
            "account_home_country": txn["account_home_country"],
            "account_age_days": txn["account_age_days"],
            "prior_txn_countries": txn["prior_txn_countries"],
            "ip_address": txn.get("ip_address"),
            "prior_actions": list(self.actions_taken.values()),
            "step_number": self.current_step + 1,
            "total_steps": len(self.transactions),
            "task_id": self.task["task_id"],
            "context_note": txn.get("context_note"),
        }

    def grade_action(self, action_type: str, txn: dict) -> tuple[float, bool, str]:
        correct_action = txn["_correct_action"]
        label = txn["_label"]
        txn_id = txn["transaction_id"]

        if action_type == correct_action:
            if action_type == "flag":
                return REWARDS["correct_flag"], True, f"Correct! {txn_id} was fraudulent. Good catch."
            elif action_type == "escalate":
                return REWARDS["correct_escalate"], True, f"Excellent! {txn_id} is money laundering. Correct escalation."
            elif action_type == "ignore":
                return REWARDS["correct_ignore"], True, f"Correct. {txn_id} was legitimate. No action needed."
            elif action_type == "freeze":
                return REWARDS["correct_freeze"], True, f"Good. {txn_id} account freeze was appropriate."

        if label in ("fraud", "laundering") and action_type in ("flag", "escalate", "freeze"):
            return REWARDS["wrong_action_type"], False, (
                f"Partially right — {txn_id} IS suspicious, but '{action_type}' "
                f"was used when '{correct_action}' was needed."
            )

        if label == "fraud" and action_type == "ignore":
            return REWARDS["missed_fraud"], False, f"Missed fraud! {txn_id} was fraudulent but you ignored it."

        if label == "laundering" and action_type == "ignore":
            return REWARDS["missed_laundering"], False, f"Missed laundering! {txn_id} needed escalation."

        if label == "legitimate" and action_type in ("flag", "escalate", "freeze"):
            return REWARDS["wrong_flag"], False, f"False positive — {txn_id} was legitimate."

        return -0.1, False, f"Unexpected action combination for {txn_id}."


# ─── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title="Financial Fraud Detective — OpenEnv",
    description=(
        "An OpenEnv environment where an AI agent acts as a financial fraud analyst. "
        "The agent reviews bank transactions and decides whether to flag, freeze, "
        "escalate, or ignore each one based on fraud indicators."
    ),
    version="1.0.0",
)

# Allow any origin (required for HuggingFace Space)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global episode (one agent at a time)
episode = EpisodeState()


# ─── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — HuggingFace Space pings this to confirm deployment."""
    return {"status": "ok", "environment": "financial-fraud-detective", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    """List all available tasks with descriptions."""
    return {
        "tasks": [
            {
                "task_id": "easy",
                "name": "Single Obvious Fraud",
                "difficulty": "easy",
                "num_transactions": 5,
                "description": "5 transactions, one blatantly fraudulent. Good for baseline evaluation.",
            },
            {
                "task_id": "medium",
                "name": "Coordinated Card Testing Attack",
                "difficulty": "medium",
                "num_transactions": 12,
                "description": "12 transactions across 3 accounts. Fraud pattern only visible across accounts.",
            },
            {
                "task_id": "hard",
                "name": "Sophisticated Money Laundering",
                "difficulty": "hard",
                "num_transactions": 20,
                "description": "20 transactions. Complex layering chain. Requires escalate, not just flag.",
            },
        ]
    }


@app.post("/reset", response_model=ObservationModel)
def reset(request: ResetRequest):
    """
    Start a new episode.
    Returns the first transaction for the agent to evaluate.
    Call this before starting any episode.
    """
    try:
        episode.reset(request.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ObservationModel(**episode.current_observation())


@app.get("/state")
def state():
    """
    Return the current state of the episode.
    Includes current transaction, actions taken so far, and running score.
    """
    if not episode.initialized:
        raise HTTPException(
            status_code=400,
            detail="No episode running. Call /reset first."
        )

    current = None
    if not episode.done:
        current = episode.current_observation()

    return {
        "task_id": episode.task["task_id"],
        "task_description": episode.task["description"],
        "current_step": episode.current_step,
        "total_steps": len(episode.transactions),
        "cumulative_score": round(episode.cumulative_score, 3),
        "done": episode.done,
        "actions_taken": episode.actions_taken,
        "current_transaction": current,
    }


@app.post("/step", response_model=RewardModel)
def step(action: ActionModel):
    """
    Submit an action for the current transaction.
    Returns a reward telling the agent how well it did.

    action_type must be one of: flag, freeze, escalate, ignore
    """
    if not episode.initialized:
        raise HTTPException(
            status_code=400,
            detail="No episode running. Call /reset first."
        )

    if episode.done:
        return RewardModel(
            score=0.0,
            done=True,
            correct=False,
            feedback="Episode finished. Call /reset to start a new one.",
            cumulative_score=round(episode.cumulative_score, 3),
            info={"error": "episode_done"},
        )

    valid_actions = {"flag", "freeze", "escalate", "ignore"}
    if action.action_type not in valid_actions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action '{action.action_type}'. Must be one of: {list(valid_actions)}"
        )

    txn = episode.transactions[episode.current_step]
    txn_id = txn["transaction_id"]

    # Penalize repeat actions
    if txn_id in episode.actions_taken:
        episode.cumulative_score += REWARDS["repeat_action"]
        return RewardModel(
            score=REWARDS["repeat_action"],
            done=episode.done,
            correct=False,
            feedback=f"Already acted on {txn_id}. Repeating actions is penalized.",
            cumulative_score=round(episode.cumulative_score, 3),
            info={"penalty": "repeat_action"},
        )

    # Grade the action
    score, correct, feedback = episode.grade_action(action.action_type, txn)
    episode.actions_taken[txn_id] = action.action_type
    episode.cumulative_score += score

    # Advance to next transaction
    episode.current_step += 1
    if episode.current_step >= len(episode.transactions):
        episode.done = True

    return RewardModel(
        score=round(score, 3),
        done=episode.done,
        correct=correct,
        feedback=feedback,
        cumulative_score=round(episode.cumulative_score, 3),
        info={
            "transaction_id": txn_id,
            "your_action": action.action_type,
            "correct_action": txn["_correct_action"],
            "transaction_label": txn["_label"],
            "your_reasoning": action.reasoning,
        },
    )


# ─── Run server ────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting Financial Fraud Detective server...")
    print("Docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
