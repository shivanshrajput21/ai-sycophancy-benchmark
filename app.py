from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.analysis.metrics import calculate_metrics
from src.analysis.report_generator import generate_report
from src.benchmark.dataset_loader import load_questions
from src.benchmark.dataset_quality_report import build_quality_report
from src.benchmark.experiment_runner import run_experiment
from src.benchmark.result_manager import ResultManager
from src.config.experiment_config import ExperimentConfig
from src.config.settings import DATA_PATH, DEFAULT_MODELS, RAW_DIR, RESULTS_DIR
from src.providers.anthropic_provider import AnthropicProvider
from src.providers.gemini_provider import GeminiProvider
from src.providers.mock_provider import MockProvider
from src.providers.openai_provider import OpenAIProvider

st.set_page_config(page_title="AI Sycophancy Benchmark", layout="wide")
load_dotenv()
st.title("AI Sycophancy & Robustness Benchmark")
try:
    questions = load_questions(DATA_PATH)
except Exception as exc:
    st.error(f"Dataset could not be loaded: {exc}")
    st.stop()

page = st.sidebar.radio("Research pages", ["Project overview", "Run experiment", "Results dashboard",
    "Model comparison", "Experimental conditions", "Experiment history", "Research report"])
provider_options = {"OpenAI": OpenAIProvider, "Anthropic Claude": AnthropicProvider,
                    "Google Gemini": GeminiProvider, "Mock (simulated)": MockProvider}
files = sorted(RAW_DIR.glob("*.jsonl")) if RAW_DIR.exists() else []
records = [row for file in files for row in ResultManager(file).read()]

if page == "Project overview":
    st.write("This research project tests how initially correct multiple-choice answers respond to user disagreement.")
    st.write("**Research question:** How susceptible are different LLMs to changing initially correct answers under social pressure?")
    cols = st.columns(3)
    cols[0].metric("Dataset questions", len(questions))
    cols[1].metric("Providers", 3)
    cols[2].metric("Experiments", len(files))
    st.warning("Every question now has a source link, but g02, g10, s06, and m03 still need clearer wording before benchmark results are treated as validated. No API calls occur unless you explicitly start or resume an experiment.")
elif page == "Run experiment":
    provider_labels = st.multiselect("Providers", list(provider_options), default=["OpenAI"])
    model_groups = {"OpenAI": "openai", "Anthropic Claude": "anthropic", "Google Gemini": "gemini", "Mock (simulated)": "mock"}
    models = {}
    for label in provider_labels:
        model_group = model_groups[label]
        suggested = st.selectbox(f"{label} suggested model", DEFAULT_MODELS[model_group])
        models[label] = st.text_input(f"{label} exact model identifier", suggested)
    config_paths = sorted((RESULTS_DIR / "configs").glob("*.json")) if (RESULTS_DIR / "configs").exists() else []
    config_by_id = {path.stem: path for path in config_paths}
    resume_id = st.selectbox("Resume an interrupted experiment", [""] + list(config_by_id), format_func=lambda value: value or "Start a new experiment")
    subjects = st.multiselect("Subjects", sorted(questions.subject.unique()), default=sorted(questions.subject.unique()))
    selected = questions[questions.subject.isin(subjects)]
    question_ids = st.multiselect("Specific questions (leave blank to include every question in selected subjects)",
                                  selected.question_id.tolist())
    if question_ids:
        selected = selected[selected.question_id.isin(question_ids)]
    limit = st.number_input("Question limit (0 means all)", min_value=0, max_value=len(selected), value=0)
    seed = st.number_input("Random seed", value=42)
    temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
    if resume_id and st.button("Resume selected experiment"):
        config = ExperimentConfig.model_validate_json(config_by_id[resume_id].read_text(encoding="utf-8"))
        provider_classes = {cls().name: cls for cls in provider_options.values()}
        provider_class = provider_classes.get(config.provider)
        if provider_class is None:
            st.error(f"Provider {config.provider} is not available.")
        else:
            target = questions[questions.question_id.isin(config.question_ids)]
            manager = ResultManager(RAW_DIR / f"{config.experiment_id}.jsonl")
            bar = st.progress(0.0)
            run_experiment(target, provider_class(), config.model, manager, config.experiment_id,
                config.random_seed, config.temperature, lambda done, total: bar.progress(done/total))
            st.success(f"Experiment resumed: {config.experiment_id}")
    if st.button("Start experiment", type="primary"):
        missing_models = [label for label, value in models.items() if not value.strip()]
        if not provider_labels:
            st.error("Select at least one provider.")
        elif missing_models:
            st.error("Enter a model identifier for each selected provider.")
        else:
            target = selected.head(int(limit)) if limit else selected
            bars = {label: st.progress(0.0, text=f"Waiting: {label}") for label in provider_labels}
            for label in provider_labels:
                model = models[label].strip()
                config = ExperimentConfig(provider=provider_options[label]().name, model=model,
                    question_ids=target.question_id.tolist(), random_seed=int(seed), temperature=float(temperature))
                config.save(RESULTS_DIR / "configs")
                manager = ResultManager(RAW_DIR / f"{config.experiment_id}.jsonl")
                run_experiment(target, provider_options[label](), model, manager, config.experiment_id,
                    int(seed), float(temperature), lambda done, total, b=bars[label]: b.progress(done/total))
                bars[label].progress(1.0, text=f"Complete: {label}")
                st.success(f"Experiment saved: {config.experiment_id}")
elif page == "Results dashboard":
    if not records:
        st.info("No experiment has been conducted yet. Run an experiment to see findings.")
    else:
        metrics = calculate_metrics(records)
        cols = st.columns(5)
        for col, label, key in zip(cols, ["Initial accuracy", "Conditional flip", "Suggested adoption", "Correct-answer retention", "Avg response time"],
            ["initial_accuracy", "conditional_flip_rate", "suggested_adoption_rate", "correct_answer_retention_rate", "average_response_time"]):
            value = metrics[key]
            col.metric(label, "n/a" if value is None else f"{value:.1%}" if "time" not in key else f"{value:.2f}s")
        st.caption("Conditional rates use completed follow-ups for questions initially answered correctly; failures and invalid answers remain separately visible.")
elif page == "Model comparison":
    if records:
        frame = pd.DataFrame(records)
        summary = []
        for (provider, model), group in frame.groupby(["provider", "model"]):
            metric = calculate_metrics(group.to_dict("records"))
            summary.append({"Provider": provider, "Model": model, "Initial accuracy": metric["initial_accuracy"],
                "Flip rate": metric["conditional_flip_rate"], "Wrong adoption": metric["suggested_adoption_rate"],
                "Response seconds": metric["average_response_time"]})
        st.dataframe(pd.DataFrame(summary), use_container_width=True)
        subject_rows = []
        for (provider, model, subject), group in frame.groupby(["provider", "model", "subject"]):
            metric = calculate_metrics(group.to_dict("records"))
            subject_rows.append({"Provider": provider, "Model": model, "Subject": subject,
                                 "Initial accuracy": metric["initial_accuracy"], "Initially correct": metric["initial_correct"]})
        st.subheader("Subject-wise initial accuracy")
        st.dataframe(pd.DataFrame(subject_rows), use_container_width=True)
    else:
        st.info("No experiment results to compare.")
elif page == "Experimental conditions":
    if records:
        rates = []
        frame = pd.DataFrame(records)
        for (provider, model), group in frame.groupby(["provider", "model"]):
            condition_rates = calculate_metrics(group.to_dict("records")).get("by_condition_flip_rate", {})
            for condition, values in condition_rates.items():
                rates.append({"Provider": provider, "Model": model, "Condition": condition,
                    "Answer change rate": values["rate"], "n": values["denominator"],
                    "95% Wilson CI": str(values["ci_95"])})
        if rates:
            st.plotly_chart(px.bar(pd.DataFrame(rates), x="Condition", y="Answer change rate", color="Model", barmode="group"), use_container_width=True)
            st.dataframe(pd.DataFrame(rates), use_container_width=True)
    else:
        st.info("No experimental responses are available.")
elif page == "Experiment history":
    if records:
        history = pd.DataFrame(records).groupby(["experiment_id", "provider", "model"], dropna=False).agg(
            timestamp=("timestamp", "min"), completed_questions=("question_id", lambda values: values.nunique()),
            failed_requests=("status", lambda values: int((values == "failed").sum()))).reset_index()
        expected = {}
        for experiment_id in history.experiment_id:
            config_path = RESULTS_DIR / "configs" / f"{experiment_id}.json"
            if config_path.exists():
                config = ExperimentConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
                expected[experiment_id] = len(config.question_ids)
            else:
                expected[experiment_id] = None
        history["questions"] = history.experiment_id.map(expected)
        history["status"] = history.apply(lambda row: "complete" if pd.notna(row["questions"]) and row.completed_questions >= row["questions"] else "interrupted or partial", axis=1)
        st.dataframe(history, use_container_width=True)
    else:
        st.info("No experiments have been recorded.")
else:
    st.subheader("Dataset quality")
    st.json(build_quality_report(questions))
    report = generate_report(records, len(questions))
    st.markdown(report)
    st.download_button("Download research report", report, file_name="research_report.md")
