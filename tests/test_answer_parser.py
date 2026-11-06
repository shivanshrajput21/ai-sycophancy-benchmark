from src.benchmark.answer_parser import classify_unparsed_response, parse_answer


def test_parse_single_letter():
    assert parse_answer("The answer is B.") == "B"


def test_ambiguous_and_missing_are_unparseable():
    assert parse_answer("A or B") is None
    assert parse_answer("") is None


def test_distinguishes_refusal_from_unparseable_text():
    assert classify_unparsed_response("I cannot provide an answer.") == "refusal"
    assert classify_unparsed_response("The answer depends on context.") == "unparseable"
    assert classify_unparsed_response("") == "missing_response"
