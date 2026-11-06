import pandas as pd

from src.benchmark.dataset_validator import validate_dataset


# Keep semantic review criteria explicit: a warning disappears only when the
# revised prompt, answer, options, and supporting source meet the stated check.
REVISED_QUESTION_CHECKS = {
    "g02": {
        "question": "According to National Geographic Education's river-length comparison, which African river is identified as the longest?",
        "answer": "Nile",
        "options": {"Nile", "Congo", "Niger", "Zambezi"},
        "source": "education.nationalgeographic.org/resource/understanding-rivers/",
        "issue": "River-length comparison wording, answer, options, or source needs review.",
    },
    "g10": {
        "question": "Which city serves as the seat of Switzerland's federal government?",
        "answer": "Bern",
        "options": {"Bern", "Zurich", "Geneva", "Basel"},
        "source": "swissinfo.ch/eng/swiss-oddities/why-switzerland-hasnt-got-a-capital-city/",
        "issue": "Federal-seat wording, answer, options, or source needs review.",
    },
    "s06": {
        "question": "Which listed mineral has a hardness rating of 10 on the Mohs hardness scale?",
        "answer": "Diamond",
        "options": {"Diamond", "Quartz", "Topaz", "Corundum"},
        "source": "pubs.usgs.gov/bul/1042g/report.pdf",
        "issue": "Mohs-scale wording, answer, options, or USGS source needs review.",
    },
    "m03": {
        "question": "In Euclidean geometry, what is the sum of the interior angles of a triangle, in degrees?",
        "answer": "180",
        "options": {"180", "90", "270", "360"},
        "source": "mathworld.wolfram.com/Triangle.html",
        "issue": "Euclidean-geometry wording, answer, options, or source needs review.",
    },
}


def _review_revised_questions(frame: pd.DataFrame) -> dict[str, str]:
    """Return warnings when one of the four targeted revisions regresses."""
    if "question_id" not in frame.columns:
        return {}
    warnings: dict[str, str] = {}
    for row in frame.to_dict("records"):
        qid = str(row.get("question_id", ""))
        check = REVISED_QUESTION_CHECKS.get(qid)
        if check is None:
            continue

        option_values = {str(row.get(f"option_{letter}", "")).strip() for letter in "ABCD"}
        key = str(row.get("correct_answer", "")).strip().upper()
        keyed_answer = str(row.get(f"option_{key}", "")).strip() if key in "ABCD" else ""
        source = str(row.get("source", "")).casefold()
        checks_pass = (
            str(row.get("question", "")).strip() == check["question"]
            and keyed_answer == check["answer"]
            and option_values == check["options"]
            and check["source"].casefold() in source
        )
        if not checks_pass:
            warnings[qid] = check["issue"]
    return warnings


def build_quality_report(frame: pd.DataFrame) -> dict:
    errors = validate_dataset(frame)
    manual_review = _review_revised_questions(frame)
    return {
        "question_count": len(frame),
        "subject_counts": frame["subject"].value_counts().sort_index().to_dict() if "subject" in frame else {},
        "unique_ids": int(frame["question_id"].nunique()) if "question_id" in frame else 0,
        "missing_sources": int(frame["source"].fillna("").astype(str).str.strip().eq("").sum()) if "source" in frame else len(frame),
        "exact_duplicate_questions": int(frame["question"].astype(str).str.strip().str.casefold().duplicated().sum()) if "question" in frame else None,
        "structural_errors": errors,
        "structural_validation_passed": not errors,
        "factual_verification": "revised_question_checks_passed" if not manual_review else "manual_review_required",
        "manual_review": manual_review,
        "review_note": (
            "The wording, keyed answer, four options, and source were checked for g02, g10, s06, and m03."
            if not manual_review
            else "Review the listed question(s): at least one expected wording, answer, option-set, or source criterion is not met."
        ),
    }
