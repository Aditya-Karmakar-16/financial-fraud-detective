"""
environment.py — Financial Fraud Detective OpenEnv
==================================================
The brain of the environment.
Implements the 3 required OpenEnv methods: step(), reset(), state()
Also implements the reward function and graders for all 3 tasks.
"""

from models import Observation, Action, Reward
from data import TASKS


# ─── Reward values ────────────────────────────────────────────
# These are the exact values given for each type of decision.
REWARDS = {
    "correct_flag":     +0.4,   # caught a fraud
    "correct_escalate": +0.5,   # caught money laundering (harder, worth more)
    "correct_ignore":   +0.2,   # correctly left a legit transaction alone
    "correct_freeze":   +0.1,   # proactively froze an attacked account
    "wrong_flag":       -0.3,   # flagged a legitimate transaction
    "wrong_escalate":   -0.3,   # escalated a legitimate transaction
    "missed_fraud":     -0.5,   # ignored a fraud transaction
    "missed_laundering":-0.4,   # ignored a laundering transaction
    "wrong_action_type":-0.2,   # used flag instead of escalate or vice versa
    "repeat_action":    -0.1,   # acted on same transaction twice
}


class FraudDetectiveEnv:
    """
    The main environment class.
    One instance = one episode (one agent reviewing one task).
    """

    def __init__(self):
        self.task = None
        self.transactions = []
        self.current_step = 0
        self.cumulative_score = 0.0
        self.actions_taken = {}   # transaction_id → action_type taken
        self.done = False

    def reset(self, task_id: str = "easy") -> Observation:
        """
        Start a fresh episode.
        Loads the requested task and returns the first transaction.
        Called at the start of every new episode.
        """
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose: easy, medium, hard")

        self.task = TASKS[task_id]
        self.transactions = self.task["transactions"]
        self.current_step = 0
        self.cumulative_score = 0.0
        self.actions_taken = {}
        self.done = False

        return self._make_observation()

    def state(self) -> dict:
        """
        Return the current state of the episode.
        Useful for agents that want to review what's happened so far.
        """
        return {
            "task_id": self.task["task_id"] if self.task else None,
            "task_description": self.task["description"] if self.task else None,
            "current_step": self.current_step,
            "total_steps": len(self.transactions),
            "cumulative_score": round(self.cumulative_score, 3),
            "done": self.done,
            "actions_taken": self.actions_taken,
            "current_transaction": (
                self._make_observation().model_dump()
                if not self.done and self.task
                else None
            ),
        }

    def step(self, action: Action) -> Reward:
        """
        The agent submits a decision. We evaluate it and return a reward.
        This is the core of the environment.
        """
        if self.done:
            return Reward(
                score=0.0,
                done=True,
                correct=False,
                feedback="Episode already finished. Call reset() to start a new one.",
                cumulative_score=round(self.cumulative_score, 3),
                info={"error": "episode_done"},
            )

        if self.task is None:
            return Reward(
                score=0.0,
                done=False,
                correct=False,
                feedback="No task loaded. Call reset() first.",
                cumulative_score=0.0,
                info={"error": "no_task"},
            )

        # Validate action type
        valid_actions = {"flag", "freeze", "escalate", "ignore"}
        if action.action_type not in valid_actions:
            return Reward(
                score=-0.1,
                done=False,
                correct=False,
                feedback=f"Invalid action '{action.action_type}'. Must be one of: {valid_actions}",
                cumulative_score=round(self.cumulative_score, 3),
                info={"error": "invalid_action"},
            )

        # Get the current transaction
        txn = self.transactions[self.current_step]
        txn_id = txn["transaction_id"]

        # Check for repeat action on same transaction
        if txn_id in self.actions_taken:
            score = REWARDS["repeat_action"]
            self.cumulative_score += score
            return Reward(
                score=score,
                done=self.done,
                correct=False,
                feedback=f"You already acted on {txn_id}. Repeating actions is penalized.",
                cumulative_score=round(self.cumulative_score, 3),
                info={"penalty": "repeat_action", "transaction_id": txn_id},
            )

        # ── Core grading logic ─────────────────────────────────
        correct_action = txn["_correct_action"]
        label = txn["_label"]  # "legitimate", "fraud", or "laundering"

        score, correct, feedback = self._grade_action(
            action.action_type, correct_action, label, txn_id
        )

        # Record this action
        self.actions_taken[txn_id] = action.action_type
        self.cumulative_score += score

        # Move to next transaction
        self.current_step += 1
        if self.current_step >= len(self.transactions):
            self.done = True

        return Reward(
            score=round(score, 3),
            done=self.done,
            correct=correct,
            feedback=feedback,
            cumulative_score=round(self.cumulative_score, 3),
            info={
                "transaction_id": txn_id,
                "your_action": action.action_type,
                "correct_action": correct_action,
                "transaction_label": label,
                "your_reasoning": action.reasoning,
            },
        )

    def _grade_action(
        self, action_type: str, correct_action: str, label: str, txn_id: str
    ) -> tuple[float, bool, str]:
        """
        Compare agent's action to the correct answer.
        Returns: (score, is_correct, feedback_message)
        """

        # ── Perfect match ──────────────────────────────────────
        if action_type == correct_action:
            if action_type == "flag":
                return (
                    REWARDS["correct_flag"],
                    True,
                    f"Correct! {txn_id} was indeed fraudulent. Good catch.",
                )
            elif action_type == "escalate":
                return (
                    REWARDS["correct_escalate"],
                    True,
                    f"Excellent! {txn_id} is part of a money laundering chain. Correct escalation.",
                )
            elif action_type == "ignore":
                return (
                    REWARDS["correct_ignore"],
                    True,
                    f"Correct. {txn_id} was legitimate. No action needed.",
                )
            elif action_type == "freeze":
                return (
                    REWARDS["correct_freeze"],
                    True,
                    f"Good. {txn_id} account freeze was appropriate.",
                )

        # ── Wrong action type on a fraud transaction ───────────
        # e.g. used "flag" when should have "escalate"
        if label in ("fraud", "laundering") and action_type in ("flag", "escalate", "freeze"):
            return (
                REWARDS["wrong_action_type"],
                False,
                f"Partially right — {txn_id} IS suspicious, but you used '{action_type}' "
                f"when '{correct_action}' was needed. Action type matters.",
            )

        # ── Missed fraud entirely (ignored a fraud transaction) ─
        if label == "fraud" and action_type == "ignore":
            return (
                REWARDS["missed_fraud"],
                False,
                f"Missed fraud! {txn_id} was fraudulent but you ignored it. "
                f"Missing fraud is the costliest mistake.",
            )

        # ── Missed laundering entirely ──────────────────────────
        if label == "laundering" and action_type == "ignore":
            return (
                REWARDS["missed_laundering"],
                False,
                f"Missed laundering! {txn_id} is part of a money laundering chain. "
                f"You should have escalated.",
            )

        # ── False positive (flagged/escalated a legitimate txn) ─
        if label == "legitimate" and action_type in ("flag", "escalate", "freeze"):
            return (
                REWARDS["wrong_flag"],
                False,
                f"False positive — {txn_id} was legitimate but you marked it as '{action_type}'. "
                f"This causes real harm to customers.",
            )

        # ── Fallback (shouldn't normally reach here) ────────────
        return (
            -0.1,
            False,
            f"Unexpected action combination for {txn_id}. Review your decision logic.",
        )

    def _make_observation(self) -> Observation:
        """
        Build an Observation object from the current transaction.
        This is what the agent sees at each step.
        """
        txn = self.transactions[self.current_step]
        return Observation(
            transaction_id=txn["transaction_id"],
            amount=txn["amount"],
            currency=txn["currency"],
            merchant=txn["merchant"],
            merchant_category=txn["merchant_category"],
            country=txn["country"],
            timestamp=txn["timestamp"],
            account_id=txn["account_id"],
            account_home_country=txn["account_home_country"],
            account_age_days=txn["account_age_days"],
            prior_txn_countries=txn["prior_txn_countries"],
            ip_address=txn.get("ip_address"),
            prior_actions=list(self.actions_taken.values()),
            step_number=self.current_step + 1,
            total_steps=len(self.transactions),
            task_id=self.task["task_id"],
            context_note=txn.get("context_note"),
        )


# ─── Quick local test ─────────────────────────────────────────
# Run: python environment.py
# Should print results without any errors.
if __name__ == "__main__":
    print("=" * 55)
    print("Financial Fraud Detective — Environment Test")
    print("=" * 55)

    env = FraudDetectiveEnv()

    for task_id in ["easy", "medium", "hard"]:
        print(f"\n--- Task: {task_id.upper()} ---")
        obs = env.reset(task_id)
        print(f"First transaction: {obs.transaction_id} | "
              f"${obs.amount} | {obs.merchant} | {obs.country}")
        print(f"Total steps: {obs.total_steps}")

        # Simulate a perfect agent that always picks the correct answer
        total_score = 0
        steps = 0
        while not env.done:
            current_txn = env.transactions[env.current_step]
            correct = current_txn["_correct_action"]
            action = Action(
                action_type=correct,
                transaction_id=current_txn["transaction_id"],
                reasoning="Test agent using answer key",
            )
            reward = env.step(action)
            total_score += reward.score
            steps += 1

        print(f"Steps: {steps} | Final score: {round(total_score, 3)} | "
              f"Cumulative: {reward.cumulative_score}")

    print("\nAll tasks passed. Environment is working correctly.")
