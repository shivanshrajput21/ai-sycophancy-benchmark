"""Data loading and filtering helpers for the Streamlit research dashboard."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.metrics import calculate_metrics
from src.analysis.classification import classify_followup
from src.benchmark.prompt_builder import followup_prompt
from src.config.experiment_config import ExperimentConfig

OUTCOME_LABELS = {
    "correct_answer_retained": "Correct answer retained",
    "changed_to_suggested_wrong_answer": "Changed to suggested wrong answer",
    "changed_to_another_incorrect_answer": "Changed to another incorrect answer",
    "correct_answer_recovered": "Correct answer recovered after reconsideration",
    "invalid_response": "Invalid response",
    "missing_response": "Missing response",
}


CONDITION_LABELS = {
    "neutral": "Neutral",
    "polite": "Polite disagreement",
    "confident": "Confident disagreement",
    "expert": "Expert authority",
}


def load_experiment_records(raw_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load each JSONL file independently, returning an empty mapping when absent."""
    if not raw_dir.exists():
        return {}
    experiments: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(raw_dir.glob("*.jsonl")):
        records: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path.name} line {line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected an object in {path.name} line {line_number}")
                records.append(record)
        experiments[path.stem] = records
    return experiments


def load_experiment_history(results_dir: Path) -> pd.DataFrame:
    """Combine configs, raw result availability, and report availability."""
    config_dir = results_dir / "configs"
    raw_dir = results_dir / "raw"
    report_dir = results_dir / "reports"
    rows: list[dict[str, Any]] = []
    if not config_dir.exists():
        return pd.DataFrame(
            columns=["experiment_id", "provider", "model", "questions", "created_at", "raw_available", "report_available"]
        )
    for path in sorted(config_dir.glob("*.json")):
        try:
            config = ExperimentConfig.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not read experiment config {path.name}: {exc}") from exc
        rows.append(
            {
                "experiment_id": config.experiment_id,
                "provider": config.provider,
                "model": config.model,
                "questions": len(config.question_ids),
                "created_at": config.created_at,
                "raw_available": (raw_dir / f"{config.experiment_id}.jsonl").exists(),
                "report_available": (report_dir / f"{config.experiment_id}.md").exists(),
                "simulated": config.provider == "mock",
            }
        )
    return pd.DataFrame(rows)


def filter_experiment_ids(
    history: pd.DataFrame,
    scope: str,
    selected_ids: list[str] | None = None,
) -> list[str]:
    """Apply the explicit real/simulated scope and optional experiment selection."""
    if history.empty:
        return []
    filtered = history.copy()
    if scope == "Research providers only":
        filtered = filtered[~filtered["simulated"]]
    elif scope == "Simulated mock experiments only":
        filtered = filtered[filtered["simulated"]]
    if selected_ids is not None:
        filtered = filtered[filtered["experiment_id"].isin(selected_ids)]
    return filtered["experiment_id"].tolist()


def records_for_experiments(
    experiments: dict[str, list[dict[str, Any]]], experiment_ids: list[str]
) -> list[dict[str, Any]]:
    """Flatten only the explicitly selected experiments, preserving isolation keys."""
    return [record for experiment_id in experiment_ids for record in experiments.get(experiment_id, [])]


def select_questions(questions: pd.DataFrame, count: int, seed: int = 42) -> pd.DataFrame:
    """Choose a reproducible, duplicate-free subset without mutating the dataset."""
    unique_questions = questions.drop_duplicates("question_id", keep="first")
    if count < 1 or count > len(unique_questions):
        raise ValueError(f"Question count must be between 1 and {len(unique_questions)}")
    return unique_questions.sample(n=count, random_state=seed).sort_index().reset_index(drop=True)


def response_inspection_rows(
    records: list[dict[str, Any]], questions: pd.DataFrame
) -> list[dict[str, Any]]:
    """Pair stored initial/follow-up records with source question context for inspection."""
    question_by_id = questions.drop_duplicates("question_id").set_index("question_id").to_dict("index")
    initials = {
        (record.get("experiment_id"), record.get("question_id")): record
        for record in records
        if record.get("stage") == "initial"
    }
    rows = []
    for followup in records:
        if followup.get("stage") != "followup":
            continue
        question = question_by_id.get(followup.get("question_id"), {})
        initial = initials.get((followup.get("experiment_id"), followup.get("question_id")), {})
        condition = followup.get("condition")
        suggested = followup.get("suggested_answer")
        challenge = None
        if condition:
            try:
                challenge = followup_prompt(condition, suggested or "")
            except ValueError:
                challenge = None
        final_answer = followup.get("answer")
        correct_answer = followup.get("correct_answer")
        initial_answer = initial.get("answer")
        outcome_code = classify_followup(initial, followup)
        outcome = OUTCOME_LABELS[outcome_code]
        rows.append(
            {
                "experiment_id": followup.get("experiment_id"),
                "question_id": followup.get("question_id"),
                "subject": question.get("subject", followup.get("subject")),
                "question": question.get("question"),
                "correct_answer": correct_answer,
                "initial_response": initial.get("raw_response"),
                "initial_answer": initial_answer,
                "condition": condition,
                "suggested_answer": suggested,
                "challenge": challenge,
                "second_response": followup.get("raw_response"),
                "final_answer": final_answer,
                "final_correct": outcome_code in {"correct_answer_retained", "correct_answer_recovered"},
                "answer_changed": outcome_code in {"changed_to_suggested_wrong_answer", "changed_to_another_incorrect_answer"},
                "suggested_wrong_adopted": outcome_code == "changed_to_suggested_wrong_answer",
                "outcome": outcome,
                "status": followup.get("status"),
            }
        )
    return rows


def subject_metrics(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Calculate existing benchmark metrics independently for each subject."""
    subjects = sorted({record.get("subject") for record in records if record.get("subject")})
    rows = []
    for subject in subjects:
        metrics = calculate_metrics([record for record in records if record.get("subject") == subject])
        rows.append(
            {
                "Subject": subject,
                "Initial accuracy": metrics.get("initial_accuracy"),
                "Initial numerator": metrics.get("initial_correct", 0),
                "Initial denominator": metrics.get("initial_denominator", 0),
                "Answer-flip rate": metrics.get("conditional_flip_rate"),
                "Follow-up denominator": metrics.get("conditional_followup_denominator", 0),
                "Suggested adoption": metrics.get("suggested_adoption_rate"),
                "Correct retention": metrics.get("correct_answer_retention_rate"),
                "Other answer change": metrics.get("other_change_rate"),
            }
        )
    return pd.DataFrame(rows)


def condition_metrics(records: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    """Calculate existing metrics for one condition while retaining its paired initials."""
    followups = [
        record for record in records
        if record.get("stage") == "followup" and record.get("condition") == condition
    ]
    keys = {(record.get("experiment_id"), record.get("question_id")) for record in followups}
    initials = [
        record for record in records
        if record.get("stage") == "initial"
        and (record.get("experiment_id"), record.get("question_id")) in keys
    ]
    return calculate_metrics(initials + followups)
