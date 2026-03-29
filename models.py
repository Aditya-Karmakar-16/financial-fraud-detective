"""
models.py — Financial Fraud Detective OpenEnv
=============================================
Defines the 3 core data types that the environment and agent exchange.
Think of these as the "language" between the environment and the AI agent.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Observation:
    """What the agent SEES at each step — one transaction to evaluate."""
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
    prior_txn_countries: list
    prior_actions: list
    step_number: int
    total_steps: int
    task_id: str
    ip_address: Optional[str] = None
    context_note: Optional[str] = None

    def model_dump(self):
        return asdict(self)


@dataclass
class Action:
    """What the agent DOES — must pick one of: flag, freeze, escalate, ignore."""
    action_type: str
    transaction_id: str
    reasoning: str

    def model_dump(self):
        return asdict(self)


@dataclass
class Reward:
    """What the environment sends back after the agent acts."""
    score: float
    done: bool
    correct: bool
    feedback: str
    cumulative_score: float
    info: dict = field(default_factory=dict)

    def model_dump(self):
        return asdict(self)
