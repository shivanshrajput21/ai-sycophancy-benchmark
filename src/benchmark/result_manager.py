import json
from pathlib import Path
from typing import Any


class ResultManager:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def completed(self) -> set[tuple[str, str, str]]:
        return {(r["question_id"], r["stage"], r.get("condition") or "") for r in self.read() if r.get("status") == "completed"}

    def append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
