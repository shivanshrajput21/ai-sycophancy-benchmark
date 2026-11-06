import json

import pandas as pd

from src.dashboard_data import (
    condition_metrics,
    filter_experiment_ids,
    load_experiment_history,
    load_experiment_records,
    records_for_experiments,
)


def test_dashboard_loads_experiments_independently(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for experiment_id in ("one", "two"):
        (raw_dir / f"{experiment_id}.jsonl").write_text(
            json.dumps({"experiment_id": experiment_id, "question_id": "q1"}) + "\n",
            encoding="utf-8",
        )

    experiments = load_experiment_records(raw_dir)

    assert set(experiments) == {"one", "two"}
    assert records_for_experiments(experiments, ["two"])[0]["experiment_id"] == "two"


def test_experiment_filtering_requires_explicit_simulation_scope():
    history = pd.DataFrame(
        [
            {"experiment_id": "real", "simulated": False},
            {"experiment_id": "mock-a", "simulated": True},
            {"experiment_id": "mock-b", "simulated": True},
        ]
    )

    assert filter_experiment_ids(history, "Research providers only") == ["real"]
    assert filter_experiment_ids(history, "Simulated mock experiments only") == ["mock-a", "mock-b"]
    assert filter_experiment_ids(history, "All experiments (explicitly include simulated)") == ["real", "mock-a", "mock-b"]


def test_history_reads_config_and_report_availability(tmp_path):
    results_dir = tmp_path / "results"
    (results_dir / "configs").mkdir(parents=True)
    (results_dir / "raw").mkdir()
    (results_dir / "reports").mkdir()
    config = {
        "experiment_id": "exp",
        "provider": "mock",
        "model": "mock-deterministic",
        "question_ids": ["q1", "q2"],
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    (results_dir / "configs" / "exp.json").write_text(json.dumps(config), encoding="utf-8")
    (results_dir / "raw" / "exp.jsonl").write_text("", encoding="utf-8")
    (results_dir / "reports" / "exp.md").write_text("report", encoding="utf-8")

    history = load_experiment_history(results_dir)

    row = history.iloc[0]
    assert row["questions"] == 2
    assert bool(row["raw_available"])
    assert bool(row["report_available"])
    assert bool(row["simulated"])


def test_condition_metrics_pairs_followups_with_initial_answers():
    records = [
        {"experiment_id": "exp", "stage": "initial", "question_id": "q1", "answer": "A", "correct_answer": "A", "status": "completed"},
        {"experiment_id": "exp", "stage": "followup", "question_id": "q1", "condition": "expert", "answer": "B", "initial_answer": "A", "correct_answer": "A", "suggested_answer": "B", "status": "completed"},
        {"experiment_id": "exp", "stage": "followup", "question_id": "q1", "condition": "neutral", "answer": "A", "initial_answer": "A", "correct_answer": "A", "suggested_answer": "B", "status": "completed"},
    ]

    metrics = condition_metrics(records, "expert")

    assert metrics["conditional_flip_rate"] == 1.0
    assert metrics["conditional_followup_denominator"] == 1
