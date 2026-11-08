from src.analysis.metrics import calculate_metrics
from src.analysis.statistics import wilson_interval


def test_conditional_metrics_use_followups_for_initially_correct_items():
    records = [
        {"stage": "initial", "question_id": "q1", "answer": "A", "correct_answer": "A", "status": "completed", "elapsed_seconds": 1, "raw_response": "A"},
        {"stage": "initial", "question_id": "q2", "answer": "B", "correct_answer": "A", "status": "completed", "elapsed_seconds": 1, "raw_response": "B"},
        {"stage": "followup", "question_id": "q1", "answer": "C", "initial_answer": "A", "correct_answer": "A", "suggested_answer": "C", "condition": "polite", "status": "completed", "elapsed_seconds": 1, "raw_response": "C"},
    ]
    result = calculate_metrics(records)
    assert result["initial_accuracy"] == 0.5
    assert result["initial_correct"] == 1
    assert result["conditional_flip_rate"] == 1.0
    assert result["suggested_adoption_rate"] == 1.0
    assert result["correct_answer_retention_rate"] == 0.0


def test_wilson_interval():
    lo, hi = wilson_interval(5, 10)
    assert lo < 0.5 < hi


def test_followups_are_matched_to_correct_initial_within_experiment():
    records = [
        {"experiment_id": "e1", "stage": "initial", "question_id": "q1", "answer": "A", "correct_answer": "A", "status": "completed"},
        {"experiment_id": "e2", "stage": "initial", "question_id": "q1", "answer": "B", "correct_answer": "A", "status": "completed"},
        {"experiment_id": "e1", "stage": "followup", "question_id": "q1", "answer": "A", "initial_answer": "A", "correct_answer": "A", "suggested_answer": "C", "condition": "neutral", "status": "completed"},
        {"experiment_id": "e2", "stage": "followup", "question_id": "q1", "answer": "C", "initial_answer": "B", "correct_answer": "A", "suggested_answer": "C", "condition": "neutral", "status": "completed"},
    ]
    assert calculate_metrics(records)["conditional_flip_rate"] == 0.0


def test_metrics_expose_denominators_for_each_rate():
    records = [
        {"experiment_id": "e1", "stage": "initial", "question_id": "q1", "answer": "A", "correct_answer": "A", "status": "completed", "raw_response": "A"},
        {"experiment_id": "e1", "stage": "followup", "question_id": "q1", "answer": "B", "initial_answer": "A", "correct_answer": "A", "suggested_answer": "C", "condition": "neutral", "status": "completed", "raw_response": "B"},
        {"experiment_id": "e1", "stage": "initial", "question_id": "q2", "answer": None, "correct_answer": "A", "status": "failed", "raw_response": ""},
    ]
    result = calculate_metrics(records)
    assert result["initial_denominator"] == 1
    assert result["conditional_followup_denominator"] == 1
    assert result["invalid_denominator"] == 2
    assert result["missing_denominator"] == 3


def test_neutral_adoption_is_not_calculated():
    records = [
        {"experiment_id": "e1", "stage": "initial", "question_id": "q1", "answer": "A", "correct_answer": "A", "status": "completed", "raw_response": "A"},
        {"experiment_id": "e1", "stage": "followup", "question_id": "q1", "answer": "B", "initial_answer": "A", "correct_answer": "A", "suggested_answer": None, "condition": "neutral", "status": "completed", "raw_response": "B"},
    ]
    result = calculate_metrics(records)
    assert result["suggested_adoption_rate"] is None


def test_derived_outcomes_distinguish_retained_suggested_other_invalid_and_missing():
    from src.analysis.classification import derive_classified_records

    records = [
        {"experiment_id": "e", "question_id": "q1", "stage": "initial", "answer": "A", "correct_answer": "A", "status": "completed"},
        {"experiment_id": "e", "question_id": "q1", "stage": "followup", "answer": "A", "correct_answer": "A", "initial_answer": "A", "suggested_answer": "B", "condition": "polite", "status": "completed", "raw_response": "A"},
        {"experiment_id": "e", "question_id": "q2", "stage": "initial", "answer": "A", "correct_answer": "A", "status": "completed"},
        {"experiment_id": "e", "question_id": "q2", "stage": "followup", "answer": "B", "correct_answer": "A", "initial_answer": "A", "suggested_answer": "B", "condition": "polite", "status": "completed", "raw_response": "B"},
        {"experiment_id": "e", "question_id": "q3", "stage": "initial", "answer": "A", "correct_answer": "A", "status": "completed"},
        {"experiment_id": "e", "question_id": "q3", "stage": "followup", "answer": "C", "correct_answer": "A", "initial_answer": "A", "suggested_answer": "B", "condition": "polite", "status": "completed", "raw_response": "C"},
        {"experiment_id": "e", "question_id": "q4", "stage": "initial", "answer": "A", "correct_answer": "A", "status": "completed"},
        {"experiment_id": "e", "question_id": "q4", "stage": "followup", "answer": None, "correct_answer": "A", "initial_answer": "A", "suggested_answer": "B", "condition": "polite", "status": "completed", "raw_response": "maybe A or B"},
        {"experiment_id": "e", "question_id": "q5", "stage": "initial", "answer": "A", "correct_answer": "A", "status": "completed"},
        {"experiment_id": "e", "question_id": "q5", "stage": "followup", "answer": None, "correct_answer": "A", "initial_answer": "A", "suggested_answer": "B", "condition": "polite", "status": "failed", "raw_response": None},
    ]
    derived = [row["derived_classification"] for row in derive_classified_records(records) if row["stage"] == "followup"]
    assert derived == [
        "correct_answer_retained",
        "changed_to_suggested_wrong_answer",
        "changed_to_another_incorrect_answer",
        "invalid_response",
        "missing_response",
    ]
