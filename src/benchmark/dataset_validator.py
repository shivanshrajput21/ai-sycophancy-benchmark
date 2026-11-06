from collections import Counter

import pandas as pd

REQUIRED_COLUMNS = ["question_id", "subject", "question", "option_A", "option_B", "option_C", "option_D", "correct_answer", "source"]


def validate_dataset(frame: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        return [f"Missing columns: {', '.join(sorted(missing))}"]
    if frame["question_id"].isna().any() or frame["question_id"].duplicated().any():
        errors.append("question_id values must be present and unique")
    if not bool(frame["correct_answer"].fillna("").astype(str).str.upper().isin(list("ABCD")).all()):
        errors.append("correct_answer must be exactly one of A, B, C, D")
    for row in frame.to_dict("records"):
        qid = row.get("question_id", "?")
        values = [str(row.get(f"option_{letter}", "")).strip() for letter in "ABCD"]
        if any(not value for value in values):
            errors.append(f"{qid}: all four options must be non-empty")
        if len({value.casefold() for value in values}) != 4:
            errors.append(f"{qid}: answer options must be distinct")
        if not str(row.get("question", "")).strip():
            errors.append(f"{qid}: question text is empty")
    return errors
