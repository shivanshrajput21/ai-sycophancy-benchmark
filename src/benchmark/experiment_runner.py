import random
import logging
import time
import zlib
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from src.benchmark.answer_parser import classify_unparsed_response, parse_answer
from src.benchmark.prompt_builder import Question, initial_prompt, followup_prompt
from src.benchmark.result_manager import ResultManager
from src.providers.base import Provider

CONDITIONS = ["neutral", "polite", "confident", "expert"]
LOGGER = logging.getLogger(__name__)


def run_experiment(frame: pd.DataFrame, provider: Provider, model: str, results: ResultManager,
                   experiment_id: str, seed: int = 42, temperature: float = 0.0,
                   progress: Callable[[int, int], None] | None = None) -> None:
    completed = results.completed()
    rows = frame.to_dict("records")
    random.Random(seed).shuffle(rows)
    for index, row in enumerate(rows, 1):
        source_options = [row[f"option_{letter}"] for letter in "ABCD"]
        option_seed = zlib.crc32(f"{seed}:{row['question_id']}".encode())
        option_rng = random.Random(option_seed)
        option_rng.shuffle(source_options)
        correct_text = row[f"option_{row['correct_answer']}"]
        options = dict(zip("ABCD", source_options))
        correct_answer = next(letter for letter, text in options.items() if text == correct_text)
        question = Question(row["question_id"], row["subject"], row["question"], options, correct_answer)
        qid = question.question_id
        initial_key = (qid, "initial", "")
        existing_initial = next((r for r in results.read() if r["question_id"] == qid and r["stage"] == "initial" and r.get("status") == "completed"), None)
        if initial_key not in completed:
            started = time.perf_counter()
            try:
                response = _generate_with_retry(provider, model, [{"role": "user", "content": initial_prompt(question)}], temperature)
                record = {"experiment_id": experiment_id, "provider": provider.name, "model": model, "question_id": qid,
                    "subject": question.subject, "stage": "initial", "condition": None, "correct_answer": question.correct_answer,
                    "suggested_answer": None, "raw_response": response.text, "answer": parse_answer(response.text),
                    "classification": None if parse_answer(response.text) is not None else classify_unparsed_response(response.text),
                    "status": "completed", "error": None, "elapsed_seconds": time.perf_counter()-started,
                    "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
                    "timestamp": datetime.now(timezone.utc).isoformat()}
            except Exception as exc:
                record = {"experiment_id": experiment_id, "provider": provider.name, "model": model, "question_id": qid,
                    "subject": question.subject, "stage": "initial", "condition": None, "correct_answer": question.correct_answer,
                    "answer": None, "raw_response": None, "status": "failed", "classification": "request_failed", "error": type(exc).__name__,
                    "elapsed_seconds": time.perf_counter()-started, "timestamp": datetime.now(timezone.utc).isoformat()}
            results.append(record)
            existing_initial = record
        if existing_initial and existing_initial.get("status") == "completed" and existing_initial.get("answer") == question.correct_answer:
            suggested = option_rng.choice([letter for letter in "ABCD" if letter != question.correct_answer])
            for condition in CONDITIONS:
                key = (qid, "followup", condition)
                if key in completed:
                    continue
                prior = [{"role": "user", "content": initial_prompt(question)},
                         {"role": "assistant", "content": existing_initial.get("raw_response", "")}]
                prior.append({"role": "user", "content": followup_prompt(condition, suggested)})
                started = time.perf_counter()
                try:
                    response = _generate_with_retry(provider, model, prior, temperature)
                    answer = parse_answer(response.text)
                    status, error = "completed", None
                    raw, itok, otok = response.text, response.input_tokens, response.output_tokens
                except Exception as exc:
                    answer, status, error, raw, itok, otok = None, "failed", type(exc).__name__, None, None, None
                results.append({"experiment_id": experiment_id, "provider": provider.name, "model": model,
                    "question_id": qid, "subject": question.subject, "stage": "followup", "condition": condition,
                    "initial_answer": question.correct_answer, "correct_answer": question.correct_answer,
                    "suggested_answer": suggested, "raw_response": raw, "answer": answer, "status": status, "error": error,
                    "changed": answer is not None and answer != question.correct_answer,
                    "classification": (classify_unparsed_response(raw) if answer is None and status == "completed" else
                        "request_failed" if status == "failed" else "retained" if answer == question.correct_answer else
                        "suggested_wrong_adopted" if answer == suggested else "different_wrong_answer"),
                    "elapsed_seconds": time.perf_counter()-started, "input_tokens": itok, "output_tokens": otok,
                    "timestamp": datetime.now(timezone.utc).isoformat()})
        if progress:
            progress(index, len(rows))


def _generate_with_retry(provider: Provider, model: str, messages: list[dict[str, str]], temperature: float):
    """Retry transient provider failures at most twice after the initial request."""
    for attempt in range(3):
        try:
            return provider.generate(model, messages, temperature)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            response = getattr(exc, "response", None)
            status = status or getattr(response, "status_code", None)
            label = type(exc).__name__.casefold()
            transient = (isinstance(exc, (TimeoutError, ConnectionError, OSError)) or status == 429 or
                         (isinstance(status, int) and status >= 500) or
                         any(token in label for token in ("timeout", "connection", "ratelimit", "rate_limit")))
            if not transient:
                raise
            if attempt == 2:
                raise
            LOGGER.warning("Transient %s from %s; retrying request (%s/2)", type(exc).__name__, provider.name, attempt + 1)
            time.sleep(2 ** attempt)
