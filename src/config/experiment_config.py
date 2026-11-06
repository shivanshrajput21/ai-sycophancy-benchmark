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

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.experiment_id}.json"
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path
