import re
import pandas as pd

from src.benchmark.experiment_runner import run_experiment
from src.benchmark.result_manager import ResultManager
from src.providers.base import ModelResponse


class CorrectProvider:
    name = "test"

    def generate(self, model, messages, temperature):
        match = re.search(r"\b([A-D])\) correct\b", messages[0]["content"])
        return ModelResponse(match.group(1) if match else "A")


def test_runner_records_initial_and_followups(tmp_path):
    frame = pd.DataFrame([{"question_id": "q1", "subject": "science", "question": "Q?",
        "option_A": "correct", "option_B": "wrong", "option_C": "wrong2", "option_D": "wrong3", "correct_answer": "A"}])
    manager = ResultManager(tmp_path / "r.jsonl")
    run_experiment(frame, CorrectProvider(), "test", manager, "exp")
    assert len(manager.read()) == 5


def test_three_pressure_conditions_reuse_one_wrong_suggestion(tmp_path):
    class RecordingProvider(CorrectProvider):
        def __init__(self):
            self.followups = []

        def generate(self, model, messages, temperature):
            if len(messages) > 1:
                self.followups.append(messages[-1]["content"])
            return super().generate(model, messages, temperature)

    frame = pd.DataFrame([{"question_id": "q1", "subject": "science", "question": "Q?",
        "option_A": "correct", "option_B": "wrong", "option_C": "wrong2", "option_D": "wrong3", "correct_answer": "A"}])
    provider = RecordingProvider()
    manager = ResultManager(tmp_path / "conditions.jsonl")
    run_experiment(frame, provider, "test", manager, "exp")
    rows = manager.read()
    followups = [row for row in rows if row["stage"] == "followup"]
    assert {row["condition"] for row in followups} == {"neutral", "polite", "confident", "expert"}
    suggestions = {row["suggested_answer"] for row in followups}
    assert len(suggestions) == 1
    suggestion = next(iter(suggestions))
    assert suggestion != followups[0]["correct_answer"]
    assert len(provider.followups) == 4
    # Neutral is the no-suggestion baseline; all pressure prompts use the same wrong letter.
    neutral = next(prompt for prompt in provider.followups if "double-check" in prompt)
    assert neutral.startswith("Are you sure?")
    pressure = [prompt for prompt in provider.followups if "double-check" not in prompt]
    assert all(re.search(rf"answer (?:was|is(?: definitely)?) {suggestion}\b", prompt) for prompt in pressure)
    run_experiment(frame, CorrectProvider(), "test", manager, "exp")
    assert len(manager.read()) == 5


def test_runner_records_final_provider_failure(tmp_path, monkeypatch):
    class BrokenProvider:
        name = "broken"

        def generate(self, model, messages, temperature):
            raise TimeoutError("timed out")

    import src.benchmark.experiment_runner as runner
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    frame = pd.DataFrame([{"question_id": "q1", "subject": "science", "question": "Q?",
        "option_A": "correct", "option_B": "wrong", "option_C": "wrong2", "option_D": "wrong3", "correct_answer": "A"}])
    manager = ResultManager(tmp_path / "failed.jsonl")
    run_experiment(frame, BrokenProvider(), "test", manager, "exp")
    rows = manager.read()
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["classification"] == "request_failed"
    run_experiment(frame, CorrectProvider(), "test", manager, "exp")
    rows = manager.read()
    assert len(rows) == 6
    assert sum(row["stage"] == "initial" and row["status"] == "completed" for row in rows) == 1
