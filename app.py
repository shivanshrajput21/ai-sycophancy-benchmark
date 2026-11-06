from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis.metrics import calculate_metrics
from src.analysis.report_generator import generate_report
from src.benchmark.dataset_loader import load_questions
from src.benchmark.dataset_quality_report import build_quality_report
from src.benchmark.experiment_runner import run_experiment
from src.benchmark.result_manager import ResultManager
from src.config.experiment_config import ExperimentConfig
from src.config.settings import DATA_PATH, DEFAULT_MODELS, RAW_DIR, RESULTS_DIR
from src.dashboard_data import (
    CONDITION_LABELS,
    condition_metrics,
    filter_experiment_ids,
    load_experiment_history,
    load_experiment_records,
    records_for_experiments,
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
    for column, (label, value, help_text) in zip(columns, cards):
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


def render_dataset(questions: pd.DataFrame) -> None:
    st.header("Dataset explorer")
    subjects = st.multiselect("Subject", sorted(questions["subject"].unique()), default=sorted(questions["subject"].unique()))
    ids = st.multiselect("Question ID", sorted(questions["question_id"].unique()))
    filtered = questions[questions["subject"].isin(subjects)]
    if ids:
        filtered = filtered[filtered["question_id"].isin(ids)]
    st.dataframe(filtered, use_container_width=True, hide_index=True, column_config={"source": st.column_config.LinkColumn("Source URL")})
    st.caption(f"{len(filtered)} of {len(questions)} questions shown.")


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
page = st.sidebar.radio("Navigate", ["Overview", "Condition analysis", "Model comparison", "Dataset explorer", "Experiment history", "Run experiment", "Research report"])

if page == "Overview":
    render_overview(records, selected_ids, questions)
elif page == "Condition analysis":
    render_conditions(records)
elif page == "Model comparison":
    render_model_comparison(records)
elif page == "Dataset explorer":
    render_dataset(questions)
elif page == "Experiment history":
    render_history(history, records_by_experiment)
elif page == "Run experiment":
    render_run_experiment(questions)
else:
    st.header("Research report")
    st.markdown(generate_report(records, len(questions)))
