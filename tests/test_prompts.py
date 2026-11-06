import pytest

from src.benchmark.prompt_builder import Question, initial_prompt, followup_prompt


def test_prompt_contains_question_and_all_choices():
    q = Question("q1", "science", "Which?", dict(A="one", B="two", C="three", D="four"), "A")
    prompt = initial_prompt(q)
    assert "Which?" in prompt and "D) four" in prompt


def test_same_suggested_option_can_be_reused():
    assert "B" in followup_prompt("polite", "B")


def test_rejects_invalid_condition():
    with pytest.raises(ValueError):
        followup_prompt("unknown", "A")
