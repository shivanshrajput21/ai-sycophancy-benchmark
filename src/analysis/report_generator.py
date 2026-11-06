from src.analysis.metrics import calculate_metrics


def generate_report(records: list[dict], dataset_count: int, experiment_id: str | None = None) -> str:
    metrics = calculate_metrics(records)
    if not records:
        return "# Research report\n\nNo experiment results are available. No findings have been generated.\n"
    provider_models = sorted({f"{row.get('provider')} / {row.get('model')}" for row in records})
    experiment_ids = sorted({row.get("experiment_id") for row in records if row.get("experiment_id")})
    displayed_experiment_id = experiment_id or (experiment_ids[0] if len(experiment_ids) == 1 else "multiple")
    simulated = any(row.get("provider") == "mock" for row in records)
    simulation_note = "**SIMULATED MOCK DATA: these are not model findings.**\n\n" if simulated else ""
    def pct(value):
        return "n/a" if value is None else f"{value:.1%}"
    def interval(value):
        return "n/a" if value is None else f"[{value[0]:.1%}, {value[1]:.1%}]"
    conditions = metrics.get("by_condition_flip_rate", {})
    condition_lines = "\n".join(
        f"- {name}: {pct(value['rate'])} ({value['numerator']}/{value['denominator']}); 95% Wilson CI {interval(value['ci_95'])}"
        for name, value in sorted(conditions.items())) or "- No completed follow-ups are available."
    return ("# Research report\n\n## Methodology\n\n"
            "Each question was presented as a multiple-choice prompt. Follow-up challenges were run only when the initial answer matched the supplied answer key. One seeded incorrect option was recorded for each question; polite, confident, and expert-authority prompts used that same wrong option. The neutral condition asked the model to double-check without presenting a suggested answer. Responses were stored incrementally as JSONL.\n\n"
            f"Provider: {', '.join(sorted({str(row.get('provider')) for row in records}))}.\n\n"
            f"Model: {', '.join(sorted({str(row.get('model')) for row in records}))}.\n\n"
            f"Experiment ID: `{displayed_experiment_id}`.\n\n"
            f"Question count: {dataset_count}.\n\nModels: {', '.join(provider_models)}.\n\n" +
            simulation_note +
            "## Recorded findings\n\n"
            f"Initial accuracy: {pct(metrics['initial_accuracy'])} ({metrics['initial_correct']}/{metrics.get('initial_denominator', 0)} completed initial answers).\n\n"
            f"Conditional answer-flip rate: {pct(metrics['conditional_flip_rate'])} (denominator: {metrics.get('conditional_followup_denominator', 0)} completed initially-correct follow-ups); 95% Wilson CI {interval(metrics.get('conditional_flip_ci_95'))}.\n\n"
            f"Suggested-wrong-answer adoption rate: {pct(metrics['suggested_adoption_rate'])} (denominator: {metrics.get('suggested_adoption_denominator', 0)}); 95% Wilson CI {interval(metrics.get('suggested_adoption_ci_95'))}.\n\n"
            f"Correct-answer retention after challenge: {pct(metrics['correct_answer_retention_rate'])} (denominator: {metrics.get('conditional_followup_denominator', 0)}). Other answer-change rate: {pct(metrics['other_change_rate'])} (denominator: {metrics.get('conditional_followup_denominator', 0)}).\n\n"
            f"Invalid response rate: {pct(metrics['invalid_rate'])} (denominator: {metrics.get('invalid_denominator', 0)}). Refusal rate: {pct(metrics['refusal_rate'])} (denominator: {metrics.get('refusal_denominator', 0)}). Missing response rate: {pct(metrics['missing_rate'])} (denominator: {metrics.get('missing_denominator', 0)}). Average response time: {metrics['average_response_time']} seconds (n={metrics.get('response_time_denominator', 0)}).\n\n"
            "Condition-specific flip rates (denominator is completed follow-ups in that condition):\n\n"
            f"{condition_lines}\n\n"
            "No paired hypothesis test is performed, so no statistical significance is claimed.\n\n"
            "## Limitations and conclusion\n\n"
            "The dataset now includes reference URLs, but wording questions remain for g02, g10, s06, and m03. This small multiple-choice benchmark does not establish general AI safety or overall model reliability. Treat findings as exploratory and do not interpret an answer change by itself as sycophancy.\n")
