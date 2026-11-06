from pathlib import Path

from src.benchmark.dataset_loader import load_questions
from src.benchmark.dataset_quality_report import build_quality_report
from src.benchmark.dataset_validator import validate_dataset


def test_load_supplied_dataset():
    frame = load_questions(Path("data/questions.csv"))
    assert len(frame) == 50
    assert validate_dataset(frame) == []


def test_dataset_has_source_urls_for_every_question():
    frame = load_questions(Path("data/questions.csv"))
    assert frame.source.str.startswith("https://").all()


def test_quality_report_clears_review_flags_when_revised_questions_pass():
    report = build_quality_report(load_questions(Path("data/questions.csv")))
    assert report["question_count"] == 50
    assert report["missing_sources"] == 0
    assert report["factual_verification"] == "revised_question_checks_passed"
    assert report["manual_review"] == {}


def test_quality_report_warns_when_a_revised_question_regresses():
    frame = load_questions(Path("data/questions.csv"))
    frame.loc[frame.question_id == "g02", "question"] = "Which is the longest river in Africa?"
    report = build_quality_report(frame)
    assert set(report["manual_review"]) == {"g02"}
    assert report["factual_verification"] == "manual_review_required"
