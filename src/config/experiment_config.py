from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    experiment_id: str = Field(default_factory=lambda: uuid4().hex)
    provider: str
    model: str
    question_ids: list[str]
    random_seed: int = 42
    temperature: float = 0.0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    dataset_path: str = "data/questions.csv"
    experiment_type: str = "false_suggestion"
    experiment_name: str = ""
    description: str = ""
    question_order: list[str] = []
    conditions: list[str] = []
    custom_conditions: dict[str, dict] = {}
    prompt_templates: dict[str, str] = {}
    randomize_question_order: bool = True
    template_id: str | None = None
    dataset_fingerprint: str | None = None
    dataset_version: str | None = None
    software_version: str | None = None
    repetition_number: int | None = None
    reproduced_from_experiment_id: str | None = None

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.experiment_id}.json"
        if path.exists():
            raise FileExistsError(f"Configuration snapshot already exists: {path.name}")
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path
