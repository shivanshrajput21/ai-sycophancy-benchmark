"""Validation and persistence helpers for dashboard-designed experiments."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from src.benchmark.prompt_builder import CONDITIONS

STANDARD_CONDITIONS = tuple(CONDITIONS)
EXPERIMENT_TYPES = {"false_suggestion", "evidence_correction"}


class CustomCondition(BaseModel):
    name: str
    description: str = ""
    prompt_template: str
    suggested_answer: str | None = None

    @field_validator("name", "prompt_template")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value.strip()


class ExperimentDesign(BaseModel):
    template_id: str = Field(default_factory=lambda: uuid4().hex)
    experiment_name: str
    description: str = ""
    experiment_type: str = "false_suggestion"
    researcher_notes: str = ""
    dataset_path: str = "data/questions.csv"
    dataset_fingerprint: str
    question_ids: list[str]
    subjects: list[str] = []
    conditions: list[str] = list(STANDARD_CONDITIONS)
    custom_conditions: list[CustomCondition] = []
    prompt_templates: dict[str, str] = {}
    repetitions: int = 1
    random_seed: int = 42
    randomize_question_order: bool = True
    timeout_seconds: int | None = None
    provider: str = "mock"
    model: str = "mock-deterministic"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @field_validator("experiment_name")
    @classmethod
    def require_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Experiment name is required")
        return value.strip()

    @field_validator("experiment_type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        if value not in EXPERIMENT_TYPES:
            raise ValueError(f"Unknown experiment type: {value}")
        return value

    @field_validator("repetitions")
    @classmethod
    def validate_repetitions(cls, value: int) -> int:
        if not 1 <= value <= 10:
            raise ValueError("Repetitions must be between 1 and 10")
        return value

    @field_validator("random_seed")
    @classmethod
    def validate_seed(cls, value: int) -> int:
        if not isinstance(value, int):
            raise ValueError("Random seed must be an integer")
        return value

    def validate_for_execution(self) -> list[str]:
        errors: list[str] = []
        if not self.question_ids:
            errors.append("At least one question must be selected")
        if len(self.question_ids) != len(set(self.question_ids)):
            errors.append("Question IDs must be unique")
        if self.provider != "mock":
            errors.append("Only the mock provider is enabled in Experiment Design Studio")
        selected = set(self.conditions)
        custom = {condition.name: condition for condition in self.custom_conditions}
        invalid = selected - set(STANDARD_CONDITIONS) - set(custom)
        if invalid:
            errors.append("Unknown challenge conditions: " + ", ".join(sorted(invalid)))
        if self.experiment_type == "false_suggestion" and not selected:
            errors.append("At least one challenge condition must be selected")
        if self.experiment_type == "evidence_correction":
            errors.append("Evidence-based correction requires verified evidence excerpts and is not executable from this dataset")
        for condition in self.custom_conditions:
            template = condition.prompt_template.casefold()
            if "{correct}" in template or "correct answer is" in template:
                errors.append(f"Custom condition '{condition.name}' must not disclose the correct answer")
            if "{suggested}" in template and not condition.suggested_answer:
                errors.append(f"Custom condition '{condition.name}' requires a suggested answer")
            if condition.suggested_answer and condition.suggested_answer not in "ABCD":
                errors.append(f"Custom condition '{condition.name}' suggested answer must be A, B, C, or D")
        return errors

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.template_id}.json"
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path


def dataset_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_templates(directory: Path) -> list[ExperimentDesign]:
    if not directory.exists():
        return []
    templates = []
    for path in sorted(directory.glob("*.json")):
        try:
            templates.append(ExperimentDesign.model_validate_json(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return templates


def delete_template(directory: Path, template_id: str) -> bool:
    path = directory / f"{template_id}.json"
    if not path.exists():
        return False
    path.unlink()
    return True


def randomized_question_ids(question_ids: list[str], seed: int, randomize: bool) -> list[str]:
    result = list(dict.fromkeys(question_ids))
    if randomize:
        import random
        random.Random(seed).shuffle(result)
    return result


def expected_response_counts(question_count: int, condition_count: int) -> dict[str, int]:
    followups = question_count * condition_count
    return {"initial": question_count, "followup": followups, "total": question_count + followups}
