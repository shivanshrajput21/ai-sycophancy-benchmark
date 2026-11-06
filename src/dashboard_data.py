"""Data loading and filtering helpers for the Streamlit research dashboard."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.metrics import calculate_metrics
from src.config.experiment_config import ExperimentConfig


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
