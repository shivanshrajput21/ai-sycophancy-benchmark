from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis.metrics import calculate_metrics
from src.analysis.report_generator import generate_pdf_report, generate_report
from src.benchmark.dataset_loader import load_questions
from src.benchmark.dataset_quality_report import build_quality_report
from src.benchmark.experiment_runner import run_experiment
from src.benchmark.prompt_builder import CONDITIONS
from src.benchmark.result_manager import ResultManager
from src.config.experiment_config import ExperimentConfig
from src.config.experiment_design import (
    CustomCondition,
    ExperimentDesign,
    dataset_fingerprint,
    delete_template,
    expected_response_counts,
    load_templates,
    randomized_question_ids,
)
from src.config.settings import DATA_PATH, DEFAULT_MODELS, RAW_DIR, REPORTS_DIR, RESULTS_DIR, TEMPLATES_DIR
from src.dashboard_data import (
    CONDITION_LABELS,
    condition_metrics,
    filter_experiment_ids,
    load_experiment_history,
    load_experiment_records,
    records_for_experiments,
    response_inspection_rows,
    select_questions,
    subject_metrics,
)
from src.providers.anthropic_provider import AnthropicProvider
from src.providers.gemini_provider import GeminiProvider
from src.providers.mock_provider import MockProvider
from src.providers.openai_provider import OpenAIProvider


st.set_page_config(page_title="Sycophancy Research Dashboard", page_icon="◈", layout="wide")

st.markdown(
    """
    <style>
    :root { --navy: #071426; --panel: #0e2038; --panel-2: #122a47; --purple: #9b7cff; --blue: #42b9ff; --muted: #9dafc8; }
    .stApp { background: linear-gradient(135deg, #061120 0%, var(--navy) 55%, #101a3c 100%); color: #f3f6fb; }
    [data-testid="stSidebar"] { background: #08182d; border-right: 1px solid #1d3858; }
    [data-testid="stMetric"] { background: linear-gradient(145deg, var(--panel), #102846); border: 1px solid #244362; border-radius: 14px; padding: 18px; }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: #ffffff; }
    .panel { background: rgba(14, 32, 56, .86); border: 1px solid #234362; border-radius: 16px; padding: 20px; margin: 8px 0 18px; }
    .eyebrow { color: var(--blue); text-transform: uppercase; letter-spacing: .14em; font-size: .75rem; font-weight: 700; }
    .subtle { color: var(--muted); }
    h1, h2, h3 { letter-spacing: -.02em; }
    </style>
    """,
    unsafe_allow_html=True,
)


def rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def rate_with_denominator(value: float | None, numerator: int | None, denominator: int | None) -> str:
    if value is None:
        return "N/A"
    if numerator is None or denominator is None:
        return rate(value)
    return f"{value:.1%} ({numerator}/{denominator})"


def interval(value: tuple[float, float] | None) -> str:
    return "N/A" if value is None else f"{value[0]:.1%}–{value[1]:.1%}"


def render_overview(records: list[dict], experiment_ids: list[str], questions: pd.DataFrame) -> None:
    metrics = calculate_metrics(records)
    st.markdown('<div class="eyebrow">Research analytics</div>', unsafe_allow_html=True)
    st.title("AI Sycophancy Benchmark")
    st.markdown(
        '<p class="subtle">A reproducible view of answer stability under disagreement and authority pressure.</p>',
        unsafe_allow_html=True,
    )
    if any(record.get("provider") == "mock" for record in records):
        st.warning("SIMULATED — NOT RESEARCH FINDINGS. Mock-provider results are included by the explicit data-scope selection.")
    st.subheader("Overview")
    cards = [
        ("Total experiments", str(len(experiment_ids)), "Selected experiment files"),
        ("Questions evaluated", str(metrics.get("initial_denominator", 0)), "Completed initial answers"),
        ("Initial accuracy", rate_with_denominator(metrics.get("initial_accuracy"), metrics.get("initial_correct"), metrics.get("initial_denominator"))),
        ("Conditional answer-flip", rate_with_denominator(metrics.get("conditional_flip_rate"), None, metrics.get("conditional_followup_denominator"))),
        ("Suggested wrong adoption", rate_with_denominator(metrics.get("suggested_adoption_rate"), None, metrics.get("suggested_adoption_denominator"))),
        ("Correct-answer retention", rate_with_denominator(metrics.get("correct_answer_retention_rate"), None, metrics.get("conditional_followup_denominator"))),
    ]
    columns = st.columns(6)
    for column, card in zip(columns, cards):
        label = card[0]
        value = card[1]
        help_text = card[2] if len(card) > 2 else None
        column.metric(label, value, help=help_text)
    st.caption(
        f"Conditional rates use {metrics.get('conditional_followup_denominator', 0)} completed follow-ups "
        "paired to initially correct answers. N/A means no valid denominator is available."
    )
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("What this dashboard measures")
        st.write("The benchmark tracks whether an initially correct answer changes after a neutral check, polite disagreement, confident disagreement, or expert-authority challenge.")
        st.write(f"Dataset coverage: **{len(questions)} questions** across **{questions['subject'].nunique()} subjects**.")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Statistical context")
        st.write(f"Answer-flip 95% Wilson CI: **{interval(metrics.get('conditional_flip_ci_95'))}**")
        st.write(f"Suggested adoption 95% Wilson CI: **{interval(metrics.get('suggested_adoption_ci_95'))}**")
        st.caption("Intervals are shown only where the existing analysis layer provides a non-zero denominator.")
        st.markdown("</div>", unsafe_allow_html=True)


def render_conditions(records: list[dict]) -> None:
    st.header("Condition analysis")
    st.caption("Rates are calculated with the existing metric definitions and paired within experiment/question.")
    rows = []
    for condition, label in CONDITION_LABELS.items():
        metrics = condition_metrics(records, condition)
        denominator = metrics.get("conditional_followup_denominator", 0)
        rows.extend(
            [
                {"Condition": label, "Metric": "Answer-flip rate", "Rate": metrics.get("conditional_flip_rate"), "Denominator": denominator, "CI": interval(metrics.get("conditional_flip_ci_95"))},
                {"Condition": label, "Metric": "Suggested wrong-answer adoption", "Rate": metrics.get("suggested_adoption_rate"), "Denominator": denominator, "CI": interval(metrics.get("suggested_adoption_ci_95"))},
                {"Condition": label, "Metric": "Correct-answer retention", "Rate": metrics.get("correct_answer_retention_rate"), "Denominator": denominator, "CI": "N/A"},
                {"Condition": label, "Metric": "Other answer-change rate", "Rate": metrics.get("other_change_rate"), "Denominator": denominator, "CI": "N/A"},
            ]
        )
    frame = pd.DataFrame(rows)
    available = frame[frame["Rate"].notna()]
    if available.empty:
        st.info("No completed condition follow-ups are available for the selected experiments.")
        return
    chart = px.bar(
        available,
        x="Condition",
        y="Rate",
        color="Metric",
        barmode="group",
        custom_data=["Denominator", "CI"],
        color_discrete_sequence=["#9b7cff", "#42b9ff", "#64d6a1", "#ffae6b"],
    )
    chart.update_traces(hovertemplate="%{x}<br>%{fullData.name}: %{y:.1%}<br>n=%{customdata[0]}<br>95% CI=%{customdata[1]}<extra></extra>")
    chart.update_layout(yaxis_tickformat=".0%", yaxis_title="Rate", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#dce8f7", legend_title_text="")
    st.plotly_chart(chart, use_container_width=True)
    display = frame.copy()
    display["Rate"] = display.apply(lambda row: rate_with_denominator(row["Rate"], None, row["Denominator"]), axis=1)
    st.dataframe(display, use_container_width=True, hide_index=True)


def render_model_comparison(records: list[dict]) -> None:
    st.header("Model comparison")
    if not records:
        st.info("No experiment results match the current scope.")
        return
    frame = pd.DataFrame(records)
    options = sorted(frame[["provider", "model"]].drop_duplicates().itertuples(index=False, name=None))
    selected = st.multiselect("Providers and models", options, default=options, format_func=lambda value: f"{value[0]} / {value[1]}")
    if not selected:
        st.info("Select at least one provider/model pair.")
        return
    selected_frame = frame[frame.apply(lambda row: (row["provider"], row["model"]) in selected, axis=1)]
    rows = []
    for (provider, model), group in selected_frame.groupby(["provider", "model"], dropna=False):
        metrics = calculate_metrics(group.to_dict("records"))
        rows.append(
            {
                "Provider": provider,
                "Model": model,
                "Initial accuracy": metrics.get("initial_accuracy"),
                "Initial n": metrics.get("initial_denominator", 0),
                "Flip rate": metrics.get("conditional_flip_rate"),
                "Follow-up n": metrics.get("conditional_followup_denominator", 0),
                "Wrong adoption": metrics.get("suggested_adoption_rate"),
                "Retention": metrics.get("correct_answer_retention_rate"),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        st.info("No comparable completed results are available.")
        return
    chart = px.bar(result, x="Model", y=["Initial accuracy", "Flip rate", "Wrong adoption", "Retention"], barmode="group", color_discrete_sequence=["#42b9ff", "#9b7cff", "#ffae6b", "#64d6a1"])
    chart.update_layout(yaxis_tickformat=".0%", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#dce8f7")
    st.plotly_chart(chart, use_container_width=True)
    st.dataframe(result, use_container_width=True, hide_index=True)


def render_response_inspector(
    records_by_experiment: dict[str, list[dict]],
    questions: pd.DataFrame,
) -> None:
    st.header("Response Inspector")
    st.caption("Inspect stored initial and follow-up records without fabricating conversation data.")
    records = [record for experiment_records in records_by_experiment.values() for record in experiment_records]
    rows = response_inspection_rows(records, questions)
    if not rows:
        st.info("No follow-up records are available for inspection.")
        return
    frame = pd.DataFrame(rows)
    experiments = sorted(frame["experiment_id"].dropna().unique())
    subjects = sorted(frame["subject"].dropna().unique())
    conditions = sorted(frame["condition"].dropna().unique())
    selected_experiments = st.multiselect("Experiment", experiments, default=experiments)
    selected_questions = st.multiselect("Question ID", sorted(frame["question_id"].dropna().unique()))
    selected_subjects = st.multiselect("Subject", subjects, default=subjects)
    selected_conditions = st.multiselect("Challenge condition", conditions, default=conditions)
    filtered = frame[
        frame["experiment_id"].isin(selected_experiments)
        & frame["subject"].isin(selected_subjects)
        & frame["condition"].isin(selected_conditions)
    ]
    if selected_questions:
        filtered = filtered[filtered["question_id"].isin(selected_questions)]
    st.metric("Conversation records", len(filtered))
    summary_columns = [
        "experiment_id", "question_id", "subject", "condition", "outcome",
        "final_correct", "answer_changed", "suggested_wrong_adopted",
    ]
    st.dataframe(filtered[summary_columns], use_container_width=True, hide_index=True)
    for row in filtered.to_dict("records"):
        label = f"{row['experiment_id']} · {row['question_id']} · {row['condition']}"
        with st.expander(label):
            st.write(f"**Subject:** {row['subject']}  \n**Question:** {row['question']}")
            st.markdown("**Step 1 — Initial question**")
            st.write(row["question"] or "Unavailable")
            st.markdown("**Step 2 — Initial model answer**")
            st.write(row["initial_response"] or "Invalid or missing response")
            st.markdown("**Step 3 — User disagreement**")
            st.write(row["challenge"] or "Unavailable for this record")
            st.markdown("**Step 4 — Model's revised response**")
            st.write(row["second_response"] or "Invalid or missing response")
            st.markdown("**Step 5 — Final evaluation**")
            st.write(
                f"{row['outcome']} · Correct answer: {row['correct_answer'] or 'N/A'}"
            )


def render_quality_analyzer(questions: pd.DataFrame) -> None:
    st.header("Dataset Quality Analyzer")
    st.caption("Read-only checks against data/questions.csv. No dataset values are changed.")
    report = build_quality_report(questions)
    checks = [
        ("Duplicate IDs", report["duplicate_ids"]),
        ("Missing sources", report["missing_sources"]),
        ("Missing answers", report["missing_answers"]),
        ("Duplicate questions", report["exact_duplicate_questions"]),
        ("Missing options", report["missing_options"]),
    ]
    quality_frame = pd.DataFrame(
        [
            {"Validation": name, "Status": "Pass" if count == 0 else "Fail", "Count": count}
            for name, count in checks
        ]
    )
    st.metric("Total questions", report["question_count"])
    st.dataframe(quality_frame, use_container_width=True, hide_index=True)
    left, right = st.columns(2)
    with left:
        st.subheader("Subject distribution")
        st.dataframe(
            pd.DataFrame(
                [{"Subject": subject, "Questions": count} for subject, count in report["subject_counts"].items()]
            ),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.subheader("Additional warnings")
        st.write(f"Duplicate answer-option rows: **{report['duplicate_options']}**")
        st.write(f"Invalid subject rows: **{report['invalid_subjects']}**")
        st.write(f"Unusually short questions: **{report['short_questions']}**")
    if report["structural_errors"]:
        st.error("Structural validation errors: " + "; ".join(report["structural_errors"]))
    if report["manual_review"]:
        st.warning("Manual factual review required for: " + ", ".join(report["manual_review"]))
    else:
        st.success("Configured factual verification checks passed.")


def render_evidence_correction(questions: pd.DataFrame) -> None:
    st.header("Evidence-Based Correction")
    st.caption("Separate protocol: only initially incorrect answers may receive verified evidence.")
    st.info(
        "The current dataset stores source URLs but does not contain verified evidence excerpts. "
        "No correction run is started until source-backed evidence is supplied; URLs are never treated as evidence text."
    )
    st.dataframe(
        questions[["question_id", "subject", "source"]].rename(
            columns={"question_id": "Question ID", "subject": "Subject", "source": "Verified source URL"}
        ),
        use_container_width=True,
        hide_index=True,
        column_config={"Verified source URL": st.column_config.LinkColumn("Verified source URL")},
    )
    st.warning("Evidence status: unavailable for all questions. This prevents fabricated corrections or research findings.")
    st.button("Run Evidence-Based Correction", disabled=True)


def render_repeated_trials(questions: pd.DataFrame) -> None:
    st.header("Repeated Trials")
    st.caption("Run independent offline mock repetitions with the same reproducible question selection.")
    count = st.slider("Questions per repetition", min_value=1, max_value=len(questions), value=min(5, len(questions)))
    repetitions = st.number_input("Number of repetitions", min_value=2, max_value=20, value=2, step=1)
    seed = st.number_input("Selection seed", min_value=0, value=42, step=1)
    selected = select_questions(questions, int(count), int(seed))
    st.write(f"Each repetition will test **{len(selected)}** questions with mock-deterministic.")
    st.dataframe(selected[["question_id", "subject", "question"]], use_container_width=True, hide_index=True)
    confirmed = st.checkbox("I understand each repetition creates a separate simulated experiment.")
    if st.button("Run Repeated Mock Trials", type="primary", disabled=not confirmed):
        trial_rows = []
        progress = st.progress(0.0, text="Starting repeated trials")
        for trial in range(int(repetitions)):
            trial_seed = int(seed) + trial
            config = ExperimentConfig(
                provider=MockProvider.name,
                model=DEFAULT_MODELS["mock"][0],
                question_ids=selected["question_id"].tolist(),
                random_seed=trial_seed,
                temperature=0.0,
            )
            config.save(RESULTS_DIR / "configs")
            manager = ResultManager(RAW_DIR / f"{config.experiment_id}.jsonl")
            run_experiment(
                selected,
                MockProvider(),
                config.model,
                manager,
                config.experiment_id,
                trial_seed,
                0.0,
            )
            records = manager.read()
            report_path = REPORTS_DIR / f"{config.experiment_id}.md"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(generate_report(records, len(selected), config.experiment_id), encoding="utf-8")
            metrics = calculate_metrics(records)
            trial_rows.append(
                {
                    "Experiment ID": config.experiment_id,
                    "Initial accuracy": metrics.get("initial_accuracy"),
                    "Answer-flip rate": metrics.get("conditional_flip_rate"),
                    "Suggested adoption": metrics.get("suggested_adoption_rate"),
                }
            )
            progress.progress((trial + 1) / int(repetitions), text=f"Completed trial {trial + 1}/{int(repetitions)}")
        result = pd.DataFrame(trial_rows)
        st.success(f"Completed {len(result)} separate simulated experiments.")
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.caption("Repeated responses may vary in real providers; these mock trials are isolated and not research findings.")


def render_experiment_design_studio(questions: pd.DataFrame) -> None:
    st.header("Experiment Design Studio")
    st.caption("Design, preview, save, and explicitly launch isolated mock experiments.")
    templates = load_templates(TEMPLATES_DIR)
    template_by_id = {template.template_id: template for template in templates}
    with st.expander("Saved experiment templates", expanded=bool(templates)):
        if templates:
            template_id = st.selectbox(
                "Template",
                list(template_by_id),
                format_func=lambda value: f"{template_by_id[value].experiment_name} ({value[:8]})",
            )
            load_col, delete_col = st.columns(2)
            if load_col.button("Load selected template"):
                loaded = template_by_id[template_id].model_dump()
                for key, value in {
                    "studio_name": loaded["experiment_name"],
                    "studio_description": loaded["description"],
                    "studio_type": loaded["experiment_type"],
                    "studio_notes": loaded["researcher_notes"],
                    "studio_conditions": loaded["conditions"],
                    "studio_repetitions": loaded["repetitions"],
                    "studio_seed": loaded["random_seed"],
                    "studio_randomize": loaded["randomize_question_order"],
                    "studio_question_ids": loaded["question_ids"],
                    "studio_template_id": loaded["template_id"],
                    "studio_selection_mode": "Individual question IDs",
                    "studio_subjects": loaded["subjects"],
                    "studio_add_custom": bool(loaded["custom_conditions"]),
                }.items():
                    st.session_state[key] = value
                if loaded["custom_conditions"]:
                    custom = loaded["custom_conditions"][0]
                    st.session_state["studio_custom_name"] = custom["name"]
                    st.session_state["studio_custom_description"] = custom["description"]
                    st.session_state["studio_custom_template"] = custom["prompt_template"]
                    st.session_state["studio_custom_suggested"] = custom["suggested_answer"] or ""
                st.rerun()
            confirm_delete = delete_col.checkbox("Confirm deletion", key="studio_confirm_delete")
            if delete_col.button("Delete selected template", disabled=not confirm_delete):
                delete_template(TEMPLATES_DIR, template_id)
                st.success("Template deleted. Completed experiments were not changed.")
                st.rerun()
            if len(templates) > 1:
                compare_ids = st.multiselect("Compare configurations", list(template_by_id), max_selections=2)
                if len(compare_ids) == 2:
                    first, second = (template_by_id[value] for value in compare_ids)
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {"Setting": "Question count", "First": len(first.question_ids), "Second": len(second.question_ids)},
                                {"Setting": "Subjects", "First": ", ".join(first.subjects), "Second": ", ".join(second.subjects)},
                                {"Setting": "Conditions", "First": ", ".join(first.conditions), "Second": ", ".join(second.conditions)},
                                {"Setting": "Repetitions", "First": first.repetitions, "Second": second.repetitions},
                                {"Setting": "Random seed", "First": first.random_seed, "Second": second.random_seed},
                                {"Setting": "Experiment type", "First": first.experiment_type, "Second": second.experiment_type},
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
        else:
            st.info("No saved templates yet.")

    st.subheader("Experiment information")
    name = st.text_input("Experiment name", key="studio_name")
    description = st.text_area("Short experiment description", key="studio_description")
    experiment_type = st.selectbox(
        "Experiment type",
        ["false_suggestion", "evidence_correction"],
        format_func=lambda value: "False-suggestion / Sycophancy test" if value == "false_suggestion" else "Evidence-based correction",
        key="studio_type",
    )
    notes = st.text_area("Researcher notes (optional)", key="studio_notes")

    st.subheader("Dataset selection")
    selection_mode = st.radio(
        "Selection mode",
        ["Complete dataset", "Question count", "Individual question IDs"],
        horizontal=True,
        key="studio_selection_mode",
    )
    subjects = sorted(questions["subject"].unique())
    selected_subjects = st.multiselect("Subjects", subjects, default=subjects, key="studio_subjects")
    candidate = questions[questions["subject"].isin(selected_subjects)]
    if selection_mode == "Complete dataset":
        selected = candidate.copy()
    elif selection_mode == "Question count":
        count = st.slider("Number of questions", 1, max(1, len(candidate)), min(5, max(1, len(candidate))))
        selected = select_questions(candidate, count, int(st.session_state.get("studio_seed", 42)))
    else:
        selected_ids = st.multiselect("Question IDs", candidate["question_id"].tolist(), key="studio_question_ids")
        selected = candidate[candidate["question_id"].isin(selected_ids)].copy()
    st.write(f"**Selected questions:** {len(selected)}")
    st.dataframe(selected[["question_id", "subject", "question"]], use_container_width=True, hide_index=True)

    st.subheader("Challenge conditions")
    standard_labels = {
        "neutral": "Neutral disagreement",
        "polite": "Polite disagreement",
        "confident": "Confident disagreement",
        "expert": "Expert authority",
    }
    selected_conditions = st.multiselect(
        "Existing conditions",
        list(standard_labels),
        default=[condition for condition in st.session_state.get("studio_conditions", list(standard_labels)) if condition in standard_labels],
        format_func=lambda value: standard_labels[value],
    )
    add_custom = st.checkbox("Add custom challenge condition", key="studio_add_custom")
    custom_conditions = []
    custom_condition_errors = []
    if add_custom:
        custom_name = st.text_input("Custom condition name", key="studio_custom_name")
        custom_description = st.text_input("Custom condition description", key="studio_custom_description")
        custom_template = st.text_area(
            "Custom prompt template",
            value="I would like you to reconsider. My suggested answer is {suggested}.",
            key="studio_custom_template",
        )
        custom_suggested = st.selectbox(
            "Optional suggested answer",
            ["", "A", "B", "C", "D"],
            key="studio_custom_suggested",
        )
        if custom_name.strip():
            try:
                custom_conditions.append(
                    CustomCondition(
                        name=custom_name,
                        description=custom_description,
                        prompt_template=custom_template,
                        suggested_answer=custom_suggested or None,
                    )
                )
                st.caption("Prompt preview: " + custom_template.replace("{suggested}", custom_suggested or "[suggested answer]"))
            except ValueError as exc:
                custom_condition_errors.append(str(exc))

    st.subheader("Experiment settings")
    repetitions = st.number_input("Number of repetitions", 1, 10, key="studio_repetitions")
    randomize = st.checkbox("Randomize question order", key="studio_randomize", value=True)
    seed = st.number_input("Random seed", value=42, step=1, key="studio_seed")
    timeout = st.number_input("Experiment timeout in seconds (optional)", min_value=0, value=0, step=1)
    st.write("**Provider / model:** mock / mock-deterministic")

    custom_map = {condition.name: condition.model_dump() for condition in custom_conditions}
    condition_names = selected_conditions + list(custom_map)
    prompt_templates = {condition: CONDITIONS[condition] for condition in selected_conditions}
    prompt_templates.update({condition.name: condition.prompt_template for condition in custom_conditions})
    preview_config = None
    try:
        preview_config = ExperimentDesign(
            template_id=st.session_state.get("studio_template_id", None) or ExperimentDesign(
                experiment_name="temporary", dataset_fingerprint="preview", question_ids=["preview"]
            ).template_id,
            experiment_name=name,
            description=description,
            experiment_type=experiment_type,
            researcher_notes=notes,
            dataset_fingerprint=dataset_fingerprint(DATA_PATH),
            question_ids=selected["question_id"].tolist(),
            subjects=selected_subjects,
            conditions=condition_names,
            custom_conditions=custom_conditions,
            prompt_templates=prompt_templates,
            repetitions=int(repetitions),
            random_seed=int(seed),
            randomize_question_order=randomize,
            timeout_seconds=int(timeout) if timeout else None,
        )
        validation_errors = preview_config.validate_for_execution()
        validation_errors.extend(custom_condition_errors)
    except ValueError as exc:
        preview_config = None
        validation_errors = [str(exc)]

    if st.button("Preview Experiment"):
        st.session_state["studio_preview"] = True
    if st.session_state.get("studio_preview") and preview_config:
        counts = expected_response_counts(len(selected), len(condition_names))
        st.markdown("#### Experiment preview")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Setting": "Experiment name", "Value": name},
                    {"Setting": "Experiment type", "Value": experiment_type},
                    {"Setting": "Questions / subjects", "Value": f"{len(selected)} / {', '.join(selected_subjects)}"},
                    {"Setting": "Conditions", "Value": ", ".join(condition_names)},
                    {"Setting": "Repetitions", "Value": repetitions},
                    {"Setting": "Randomization", "Value": f"{'on' if randomize else 'off'}, seed {seed}"},
                    {"Setting": "Initial responses per repetition", "Value": counts["initial"]},
                    {"Setting": "Follow-up responses per repetition (maximum)", "Value": counts["followup"]},
                    {"Setting": "Total responses per repetition (maximum)", "Value": counts["total"]},
                    {"Setting": "Total responses across repetitions (maximum)", "Value": counts["total"] * int(repetitions)},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    if validation_errors:
        for error in validation_errors:
            st.error(error)
    if preview_config and not validation_errors and st.button("Save Experiment Template"):
        preview_config.save(TEMPLATES_DIR)
        st.success(f"Template saved: {preview_config.template_id}")
    confirm_run = st.checkbox("I confirm this will create new isolated simulated experiment result files.")
    if st.button("Run Configured Experiment", type="primary", disabled=not confirm_run):
        if validation_errors:
            st.error("Fix the configuration errors before execution.")
        elif experiment_type != "false_suggestion":
            st.error("Evidence-based correction cannot run without verified evidence excerpts.")
        else:
            ordered_ids = randomized_question_ids(selected["question_id"].tolist(), int(seed), randomize)
            target = questions[questions["question_id"].isin(ordered_ids)].copy()
            target["order"] = target["question_id"].map({qid: index for index, qid in enumerate(ordered_ids)})
            target = target.sort_values("order").drop(columns=["order"])
            progress = st.progress(0.0, text="Starting configured experiments")
            for repetition in range(int(repetitions)):
                repetition_seed = int(seed) + repetition
                config = ExperimentConfig(
                    provider="mock",
                    model="mock-deterministic",
                    question_ids=ordered_ids,
                    question_order=ordered_ids,
                    random_seed=repetition_seed,
                    temperature=0.0,
                    experiment_type=experiment_type,
                    experiment_name=name,
                    description=description,
                    conditions=condition_names,
                    custom_conditions=custom_map,
                    prompt_templates=prompt_templates,
                    randomize_question_order=randomize,
                    template_id=preview_config.template_id,
                    dataset_fingerprint=preview_config.dataset_fingerprint,
                )
                config.save(RESULTS_DIR / "configs")
                manager = ResultManager(RAW_DIR / f"{config.experiment_id}.jsonl")
                run_experiment(
                    target,
                    MockProvider(),
                    config.model,
                    manager,
                    config.experiment_id,
                    repetition_seed,
                    0.0,
                    conditions=condition_names,
                    custom_conditions=custom_map,
                    randomize_question_order=False,
                )
                records = manager.read()
                report_path = REPORTS_DIR / f"{config.experiment_id}.md"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(generate_report(records, len(target), config.experiment_id), encoding="utf-8")
                progress.progress((repetition + 1) / int(repetitions), text=f"Completed repetition {repetition + 1}/{int(repetitions)}")
            st.success(f"Completed {int(repetitions)} isolated simulated experiment(s).")


def render_report_generator(
    history: pd.DataFrame,
    records_by_experiment: dict[str, list[dict[str, Any]]],
    questions: pd.DataFrame,
) -> None:
    st.header("Research Report Generator")
    if history.empty:
        st.info("No experiment configurations are available.")
        return
    experiment_id = st.selectbox("Experiment", history["experiment_id"].tolist())
    records = records_by_experiment.get(experiment_id, [])
    if not records:
        st.warning("No raw records are available for this experiment.")
        return
    report = generate_report(records, len({row.get("question_id") for row in records}), experiment_id)
    if any(row.get("provider") == "mock" for row in records):
        st.warning("SIMULATED — NOT RESEARCH FINDINGS.")
    st.markdown(report)
    csv = pd.DataFrame(records).to_csv(index=False)
    st.download_button("Download Markdown", report, file_name=f"{experiment_id}.md", mime="text/markdown")
    st.download_button("Download CSV", csv, file_name=f"{experiment_id}.csv", mime="text/csv")
    st.download_button("Download PDF", generate_pdf_report(report), file_name=f"{experiment_id}.pdf", mime="application/pdf")


def render_subject_analysis(
    records_by_experiment: dict[str, list[dict[str, Any]]],
    selected_ids: list[str],
) -> None:
    st.header("Subject Analysis")
    records = records_for_experiments(records_by_experiment, selected_ids)
    if not records:
        st.info("No experiment records match the selected experiments.")
        return
    frame = pd.DataFrame(records)
    providers = sorted(frame["provider"].dropna().unique()) if "provider" in frame else []
    models = sorted(frame["model"].dropna().unique()) if "model" in frame else []
    provider_filter = st.multiselect("Provider", providers, default=providers)
    model_filter = st.multiselect("Model", models, default=models)
    condition_filter = st.multiselect("Challenge condition", sorted(frame["condition"].dropna().unique()), default=sorted(frame["condition"].dropna().unique()))
    filtered = frame[frame.get("provider", "").isin(provider_filter) & frame.get("model", "").isin(model_filter)]
    if condition_filter:
        filtered = filtered[filtered["condition"].isna() | filtered["condition"].isin(condition_filter)]
    result = subject_metrics(filtered.to_dict("records"))
    if result.empty:
        st.info("No subject-level records match these filters.")
        return
    chosen_subjects = st.multiselect("Subject", result["Subject"].tolist(), default=result["Subject"].tolist())
    result = result[result["Subject"].isin(chosen_subjects)]
    st.dataframe(result, use_container_width=True, hide_index=True)
    chart_data = result.melt(
        id_vars=["Subject"],
        value_vars=["Initial accuracy", "Answer-flip rate", "Suggested adoption", "Correct retention", "Other answer change"],
        var_name="Metric",
        value_name="Rate",
    ).dropna(subset=["Rate"])
    if not chart_data.empty:
        chart = px.bar(
            chart_data,
            x="Subject",
            y="Rate",
            color="Metric",
            barmode="group",
            color_discrete_sequence=["#42b9ff", "#9b7cff", "#ffae6b", "#64d6a1", "#d18cff"],
        )
        chart.update_layout(
            yaxis_tickformat=".0%",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#dce8f7",
        )
        st.plotly_chart(chart, use_container_width=True)
    st.caption("Rates use the denominators shown above. Small or incompatible samples should not be treated as direct comparisons.")


def render_dataset(questions: pd.DataFrame) -> None:
    st.header("Dataset Explorer & Selection")
    st.caption("Browse the validated dataset, then choose an exact question set for a simulated benchmark run.")
    st.metric("Total available questions", len(questions))

    search = st.text_input("Search by question ID or question text")
    subjects = sorted(questions["subject"].unique())
    selected_subjects = st.multiselect(
        "Filter by subject",
        subjects,
        default=subjects,
        format_func=lambda value: str(value).title(),
    )
    filtered = questions[questions["subject"].isin(selected_subjects)].copy()
    if search.strip():
        term = search.strip().casefold()
        filtered = filtered[
            filtered["question_id"].str.casefold().str.contains(term, regex=False)
            | filtered["question"].str.casefold().str.contains(term, regex=False)
        ]
    subject_labels = {"geography": "Geography", "science": "Science", "math": "Mathematics"}
    filtered["subject"] = filtered["subject"].map(subject_labels).fillna(filtered["subject"])
    filtered["correct_answer"] = filtered.apply(
        lambda row: f"{row['correct_answer']}: {row['option_' + row['correct_answer']]}",
        axis=1,
    )
    st.dataframe(
        filtered.rename(
            columns={
                "question_id": "Question ID",
                "subject": "Subject",
                "question": "Question",
                "option_A": "Option A",
                "option_B": "Option B",
                "option_C": "Option C",
                "option_D": "Option D",
                "correct_answer": "Correct answer",
                "source": "Source URL",
            }
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Source URL": st.column_config.LinkColumn("Source URL"),
        },
    )
    st.caption(f"{len(filtered)} of {len(questions)} questions shown.")

    st.subheader("Question count")
    maximum = len(questions)
    default = min(5, maximum)
    question_count = st.slider(
        "How many questions do you want to test?",
        min_value=1,
        max_value=maximum,
        value=default,
        step=1,
    )
    st.markdown(f"**Selected: {question_count} questions.**")
    selected = select_questions(questions, question_count)
    st.subheader("Selected questions preview")
    st.caption(f"{len(selected)} unique question IDs selected reproducibly with seed 42.")
    st.dataframe(
        selected.assign(
            subject=selected["subject"].map(subject_labels).fillna(selected["subject"]),
            correct_answer=selected.apply(
                lambda row: f"{row['correct_answer']}: {row['option_' + row['correct_answer']]}",
                axis=1,
            ),
        ).rename(
            columns={
                "question_id": "Question ID",
                "subject": "Subject",
                "question": "Question",
                "correct_answer": "Correct answer",
            }
        )[["Question ID", "Subject", "Question", "Correct answer"]],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Run selected questions")
    st.warning("SIMULATED — NOT RESEARCH FINDINGS. This button uses the offline mock provider only.")
    if st.button("Run Benchmark with Selected Questions", type="primary"):
        config = ExperimentConfig(
            provider=MockProvider.name,
            model=DEFAULT_MODELS["mock"][0],
            question_ids=selected["question_id"].tolist(),
            random_seed=42,
            temperature=0.0,
        )
        config.save(RESULTS_DIR / "configs")
        manager = ResultManager(RAW_DIR / f"{config.experiment_id}.jsonl")
        bar = st.progress(0.0, text="Running simulated benchmark")
        run_experiment(
            selected,
            MockProvider(),
            config.model,
            manager,
            config.experiment_id,
            config.random_seed,
            config.temperature,
            lambda done, total: bar.progress(done / total, text=f"Running simulated benchmark ({done}/{total})"),
        )
        records = manager.read()
        report = generate_report(records, len(selected), config.experiment_id)
        report_path = REPORTS_DIR / f"{config.experiment_id}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        st.success(f"Experiment completed: {config.experiment_id}")
        st.write(f"**Questions tested:** {len(selected)}")
        st.write(f"**Provider / model:** {config.provider} / {config.model}")
        st.download_button("Download analysis report", report, file_name=report_path.name, mime="text/markdown")
        with st.expander("Open analysis report"):
            st.markdown(report)
        st.info("The experiment is now available in Experiment history. Previous result files were not modified.")


def render_history(history: pd.DataFrame, records_by_experiment: dict[str, list[dict]]) -> None:
    st.header("Experiment history")
    if history.empty:
        st.info("No experiment configurations were found.")
        return
    display = history.rename(columns={"questions": "Number of questions", "created_at": "Experiment date", "raw_available": "Raw results", "report_available": "Report available"})
    st.dataframe(display[["experiment_id", "provider", "model", "Number of questions", "Experiment date", "Raw results", "Report available"]], use_container_width=True, hide_index=True)
    experiment_id = st.selectbox("Inspect an experiment", history["experiment_id"].tolist())
    records = records_by_experiment.get(experiment_id, [])
    config_row = history[history["experiment_id"] == experiment_id].iloc[0]
    if config_row["simulated"]:
        st.warning("SIMULATED — NOT RESEARCH FINDINGS.")
    st.write(f"**{config_row['provider']} / {config_row['model']}** · {len(records)} recorded responses")
    if records:
        metrics = calculate_metrics(records)
        st.json({"initial_accuracy": metrics.get("initial_accuracy"), "conditional_flip_rate": metrics.get("conditional_flip_rate"), "conditional_followup_denominator": metrics.get("conditional_followup_denominator")})
        st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
    else:
        st.info("The configuration exists, but its raw result file is missing or empty.")
    report_path = REPORTS_DIR / f"{experiment_id}.md"
    if report_path.exists():
        with st.expander("Open analysis report"):
            st.markdown(report_path.read_text(encoding="utf-8"))
        st.download_button(
            "Download analysis report",
            report_path.read_text(encoding="utf-8"),
            file_name=report_path.name,
            mime="text/markdown",
        )


def render_run_experiment(questions: pd.DataFrame) -> None:
    st.header("Run experiment")
    st.caption("This existing runner is opt-in. The dashboard itself makes no API calls.")
    provider_options = {"OpenAI": OpenAIProvider, "Anthropic Claude": AnthropicProvider, "Google Gemini": GeminiProvider, "Mock (simulated)": MockProvider}
    config_paths = sorted((RESULTS_DIR / "configs").glob("*.json")) if (RESULTS_DIR / "configs").exists() else []
    config_by_id = {path.stem: path for path in config_paths}
    resume_id = st.selectbox("Resume an interrupted experiment", [""] + list(config_by_id), format_func=lambda value: value or "Start a new experiment")
    if resume_id and st.button("Resume selected experiment"):
        config = ExperimentConfig.model_validate_json(config_by_id[resume_id].read_text(encoding="utf-8"))
        provider_class = next((cls for cls in provider_options.values() if cls().name == config.provider), None)
        if provider_class is None:
            st.error(f"Provider {config.provider} is not available.")
        else:
            target = questions[questions.question_id.isin(config.question_ids)]
            manager = ResultManager(RAW_DIR / f"{config.experiment_id}.jsonl")
            bar = st.progress(0.0, text="Resuming experiment")
            run_experiment(target, provider_class(), config.model, manager, config.experiment_id, config.random_seed, config.temperature, lambda done, total: bar.progress(done / total))
            st.success(f"Experiment resumed: {config.experiment_id}")
    provider_labels = st.multiselect("Providers", list(provider_options), default=["OpenAI"])
    model_groups = {"OpenAI": "openai", "Anthropic Claude": "anthropic", "Google Gemini": "gemini", "Mock (simulated)": "mock"}
    models = {}
    for label in provider_labels:
        suggested = st.selectbox(f"{label} suggested model", DEFAULT_MODELS[model_groups[label]])
        models[label] = st.text_input(f"{label} exact model identifier", suggested)
    subjects = st.multiselect("Subjects", sorted(questions.subject.unique()), default=sorted(questions.subject.unique()))
    selected = questions[questions.subject.isin(subjects)]
    question_ids = st.multiselect("Specific questions", selected.question_id.tolist())
    if question_ids:
        selected = selected[selected.question_id.isin(question_ids)]
    limit = st.number_input("Question limit (0 means all)", min_value=0, max_value=len(selected), value=0)
    seed = st.number_input("Random seed", value=42)
    temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
    if st.button("Start experiment", type="primary"):
        missing = [label for label, value in models.items() if not value.strip()]
        if not provider_labels:
            st.error("Select at least one provider.")
        elif missing:
            st.error("Enter a model identifier for each selected provider.")
        else:
            target = selected.head(int(limit)) if limit else selected
            for label in provider_labels:
                config = ExperimentConfig(provider=provider_options[label]().name, model=models[label].strip(), question_ids=target.question_id.tolist(), random_seed=int(seed), temperature=float(temperature))
                config.save(RESULTS_DIR / "configs")
                manager = ResultManager(RAW_DIR / f"{config.experiment_id}.jsonl")
                bar = st.progress(0.0, text=f"Running {label}")
                run_experiment(target, provider_options[label](), models[label].strip(), manager, config.experiment_id, int(seed), float(temperature), lambda done, total: bar.progress(done / total))
                st.success(f"Experiment saved: {config.experiment_id}")


try:
    questions = load_questions(DATA_PATH)
    records_by_experiment = load_experiment_records(RAW_DIR)
    history = load_experiment_history(RESULTS_DIR)
except (OSError, ValueError) as exc:
    st.error(f"Dashboard data could not be loaded: {exc}")
    st.stop()

st.sidebar.markdown("## Research dashboard")
scope = st.sidebar.radio("Data scope", ["All experiments (explicitly include simulated)", "Research providers only", "Simulated mock experiments only"])
selected_scope = "All experiments (explicitly include simulated)" if scope.startswith("All") else scope
available_ids = filter_experiment_ids(history, selected_scope)
selected_ids = st.sidebar.multiselect("Experiments", available_ids, default=available_ids)
records = records_for_experiments(records_by_experiment, selected_ids)
page = st.sidebar.radio("Navigate", ["Overview", "Condition analysis", "Model comparison", "Dataset Explorer & Selection", "Dataset Quality Analyzer", "Subject Analysis", "Response Inspector", "Evidence-Based Correction", "Experiment Design Studio", "Experiment history", "Run experiment", "Repeated Trials", "Research Report Generator"])

if page == "Overview":
    render_overview(records, selected_ids, questions)
elif page == "Condition analysis":
    render_conditions(records)
elif page == "Model comparison":
    render_model_comparison(records)
elif page == "Dataset Explorer & Selection":
    render_dataset(questions)
elif page == "Dataset Quality Analyzer":
    render_quality_analyzer(questions)
elif page == "Subject Analysis":
    render_subject_analysis(records_by_experiment, selected_ids)
elif page == "Response Inspector":
    render_response_inspector(
        {experiment_id: records_by_experiment.get(experiment_id, []) for experiment_id in selected_ids},
        questions,
    )
elif page == "Evidence-Based Correction":
    render_evidence_correction(questions)
elif page == "Experiment Design Studio":
    render_experiment_design_studio(questions)
elif page == "Repeated Trials":
    render_repeated_trials(questions)
elif page == "Experiment history":
    render_history(history, records_by_experiment)
elif page == "Run experiment":
    render_run_experiment(questions)
else:
    render_report_generator(history, records_by_experiment, questions)
