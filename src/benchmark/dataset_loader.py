from pathlib import Path

import pandas as pd

from src.benchmark.dataset_validator import REQUIRED_COLUMNS, validate_dataset


def load_questions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    # Support the original starter CSV while retaining its answer keys verbatim.
    if "id" in frame.columns and "correct" in frame.columns:
        migrated = []
        for row in frame.to_dict("records"):
            options = [row[k] for k in ("correct", "wrong1", "wrong2", "wrong3")]
            migrated.append({"question_id": row["id"], "subject": row["category"], "question": row["question"],
                             **{f"option_{letter}": option for letter, option in zip("ABCD", options)},
                             "correct_answer": "A", "source": "", "verification_status": "unverified"})
        frame = pd.DataFrame(migrated)
    errors = validate_dataset(frame)
    if errors:
        raise ValueError("Invalid dataset: " + "; ".join(errors))
    # Keep only the public schema, allowing future metadata columns in input files.
    return frame[REQUIRED_COLUMNS].copy()
