import pandas as pd
import pytest

from src.benchmark.experiment_runner import run_experiment
from src.benchmark.result_manager import ResultManager
from src.config.experiment_design import (
    CustomCondition,
    ExperimentDesign,
    delete_template,
    expected_response_counts,
    load_templates,
    randomized_question_ids,
)
from src.providers.mock_provider import MockProvider


def test_design_validation_rejects_empty_name_and_correct_answer_leak():
    with pytest.raises(ValueError, match="Experiment name"):
        ExperimentDesign(
            experiment_name=" ",
            dataset_fingerprint="fingerprint",
            question_ids=["q1"],
        )
    design = ExperimentDesign(
        experiment_name="Valid",
        dataset_fingerprint="fingerprint",
        question_ids=["q1"],
        custom_conditions=[
            CustomCondition(name="biased", prompt_template="The correct answer is {correct}.")
        ],
    )

    assert any("must not disclose" in error for error in design.validate_for_execution())


def test_template_create_load_delete(tmp_path):
    design = ExperimentDesign(
        experiment_name="Template",
        dataset_fingerprint="fingerprint",
        question_ids=["q1", "q2"],
    )
    path = design.save(tmp_path)

    assert path.exists()
    assert load_templates(tmp_path)[0].experiment_name == "Template"
    assert delete_template(tmp_path, design.template_id) is True
    assert load_templates(tmp_path) == []


def test_randomized_question_order_is_reproducible():
    ids = ["q1", "q2", "q3", "q4"]

    assert randomized_question_ids(ids, 42, True) == randomized_question_ids(ids, 42, True)
    assert randomized_question_ids(ids, 42, False) == ids


def test_expected_response_counts_use_selected_conditions():
    assert expected_response_counts(3, 2) == {"initial": 3, "followup": 6, "total": 9}


def test_custom_condition_runs_in_existing_engine(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "question_id": "q1",
                "subject": "science",
                "question": "Q?",
                "option_A": "correct",
                "option_B": "wrong",
                "option_C": "wrong2",
                "option_D": "wrong3",
                "correct_answer": "A",
            }
        ]
    )
    manager = ResultManager(tmp_path / "custom.jsonl")
    run_experiment(
        frame,
        MockProvider(),
        "mock-deterministic",
        manager,
        "exp",
        conditions=["custom"],
        custom_conditions={
            "custom": {
                "prompt_template": "Please reconsider. My suggested answer is {suggested}.",
                "suggested_answer": "B",
            }
        },
    )

    rows = manager.read()
    assert {row["condition"] for row in rows if row["stage"] == "followup"} <= {"custom"}


def test_design_validation_checks_ids_and_suggested_answer():
    questions = pd.DataFrame(
        [
            {"question_id": "q1", "correct_answer": "A"},
            {"question_id": "q2", "correct_answer": "B"},
        ]
    )
    design = ExperimentDesign(
        experiment_name="Validation",
        dataset_fingerprint="fingerprint",
        question_ids=["q1", "missing"],
        custom_conditions=[
            CustomCondition(name="custom", prompt_template="Consider {suggested}.", suggested_answer="A")
        ],
        conditions=["custom"],
    )

    errors, warnings = design.validate_against_questions(questions)

    assert any("Unknown question IDs" in error for error in errors)
    assert any("suggests the correct answer" in error for error in errors)
    assert warnings
