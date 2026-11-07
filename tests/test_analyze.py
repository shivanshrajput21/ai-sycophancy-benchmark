import json
import subprocess
import sys

from src.analysis.report_generator import generate_pdf_report

EXPERIMENT_A = "experiment-a"
EXPERIMENT_B = "experiment-b"


def _record(experiment_id, question_id, stage, answer, correct="A", **extra):
    return {
        "experiment_id": experiment_id,
        "provider": "mock",
        "model": "mock-deterministic",
        "question_id": question_id,
        "stage": stage,
        "answer": answer,
        "correct_answer": correct,
        "status": "completed",
        "raw_response": answer or "",
        "elapsed_seconds": 1.0,
        **extra,
    }


def test_analyze_filters_and_writes_experiment_specific_report(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [
        _record(EXPERIMENT_A, "a1", "initial", "A"),
        _record(EXPERIMENT_A, "a1", "followup", "B", initial_answer="A", suggested_answer="C", condition="neutral"),
        _record(EXPERIMENT_B, "b1", "initial", "A"),
        _record(EXPERIMENT_B, "b1", "followup", "A", initial_answer="A", suggested_answer="C", condition="neutral"),
    ]
    (raw / "combined.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "analyze.py", "--results", str(raw), "--experiment-id", EXPERIMENT_B],
        capture_output=True, text=True, check=True,
    )
    report = (tmp_path / "reports" / f"{EXPERIMENT_B}.md").read_text(encoding="utf-8")
    assert "experiment-b" in report
    assert "experiment-a" not in report
    assert "Question count: 1" in report
    assert "Initial accuracy: 100.0% (1/1)" in report
    assert "Correct-answer retention after challenge: 100.0% (1/1)" in report
    assert f"Saved {tmp_path / 'reports' / (EXPERIMENT_B + '.md')}" in result.stdout


def test_analyze_combined_summary_keeps_experiments_separate(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [
        _record(EXPERIMENT_A, "q1", "initial", "A"),
        _record(EXPERIMENT_B, "q1", "initial", "B"),
    ]
    (raw / "a.jsonl").write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    (raw / "b.jsonl").write_text(json.dumps(rows[1]) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, "analyze.py", "--results", str(raw)], check=True, capture_output=True, text=True)
    summary = (tmp_path / "reports" / "summary.md").read_text(encoding="utf-8")
    assert "Experiment ID: `experiment-a`" in summary
    assert "Experiment ID: `experiment-b`" in summary
    assert "Initial accuracy: 100.0% (1/1)" in summary
    assert "Initial accuracy: 0.0% (0/1)" in summary
    report_a = (tmp_path / "reports" / "experiment-a.md").read_text(encoding="utf-8")
    report_b = (tmp_path / "reports" / "experiment-b.md").read_text(encoding="utf-8")
    assert "Experiment ID: `experiment-a`" in report_a
    assert "Experiment ID: `experiment-b`" not in report_a
    assert "Experiment ID: `experiment-b`" in report_b
    assert "Experiment ID: `experiment-a`" not in report_b


def test_report_pdf_export_has_pdf_signature():
    assert generate_pdf_report("# Report\n\nSIMULATED - NOT RESEARCH FINDINGS").startswith(b"%PDF-1.4")
