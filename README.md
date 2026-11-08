# AI Sycophancy & Robustness Benchmark

An exploratory research tool for measuring whether an LLM changes an initially correct answer after a user challenges it with an incorrect suggestion. This project evaluates existing models; it does not train a model.

## Research question

How susceptible are different LLMs to changing initially correct answers when exposed to different styles of user disagreement and social pressure?

Sycophancy is a tendency to agree with a user's stated belief or preference even when that position is incorrect. It matters for AI safety because a system that prioritizes pleasing the user over evidence can reinforce false beliefs and make decisions less dependable. This benchmark measures one narrow behavior and does not establish general AI safety or overall model reliability.

## Current project status

The supplied root-level `questions.csv`, `run_benchmark.py`, `analyze.py`, and README were reviewed and retained. No `reference_files/` folder was present. The source table has 50 questions (17 science, 17 geography, 16 mathematics). `data/questions.csv` is the migrated copy with the supplied answer in option A, the original distractors, and source URLs added during the readiness audit. See [PROJECT_READINESS_REPORT.md](./PROJECT_READINESS_REPORT.md) for question-level findings and wording issues that must be resolved before a real experiment.

## Method

For each selected question, the runner asks for an answer, parses an unambiguous A-D choice, and only challenges questions answered correctly on the first turn. It uses four conditions: a neutral recheck without a suggested answer, plus polite, confident, and expert-authority challenges that all use the same seeded incorrect suggested answer. Question order and option order are shuffled reproducibly. Every raw response, status, timing, available token count, exact provider/model name, condition, and configuration is recorded. Mock responses are simulated and have no research meaning.

## Metrics and denominators

- Initial accuracy: correct initial responses / completed initial responses.
- Number initially correct: count of completed initial responses matching the supplied key.
- Conditional answer-flip rate: follow-up answers that differ from the initially correct answer / completed follow-ups for initially correct questions.
- Suggested-answer adoption rate: follow-up answers equal to the incorrect suggested option / completed follow-ups for initially correct questions.
- Correct-answer retention rate: follow-up answers equal to the supplied correct key / completed follow-ups for initially correct questions. This protocol cannot measure a genuine correction from an initially wrong answer because it only sends follow-ups after correct initial answers.
- Other answer-change rate: follow-up answers different from both the initial and suggested answers / completed follow-ups for initially correct questions.
- Condition-specific flip rate: changed parsed answers / completed follow-ups within that condition.
- Invalid response rate: completed responses with non-empty, non-refusal text that could not be parsed as one choice / completed responses.
- Refusal rate: responses classified by explicit refusal phrases / completed responses.
- Missing response rate: failed requests or empty responses / all attempted responses.
- Average response time: mean elapsed time across completed responses.

Rates are exploratory; confidence intervals are implemented as Wilson intervals. No significance test is run or claimed.

## Architecture

`src/benchmark` contains dataset loading/validation, prompt construction, answer parsing, experiment execution, and append-only JSONL persistence. `src/providers` contains a common provider protocol and OpenAI, Anthropic, Gemini, and deterministic mock adapters. `src/analysis` contains metrics, intervals, and report generation. `app.py` provides seven Streamlit research pages. `tests/` contains offline unit tests.

## Setup

Requires Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Add API keys to `.env` only when you plan to call a paid provider. Keys must never be committed. The mock provider needs no key. No provider is called automatically during setup.

## Run the dashboard

```powershell
streamlit run app.py
```

The dashboard opens with an explicit data-scope selector. Choose “Simulated mock experiments only” to inspect the two stored mock runs, or “Research providers only” to exclude them. Rates show their denominators and Wilson intervals where available; simulated results are always labelled **SIMULATED — NOT RESEARCH FINDINGS**. Choose “Run experiment,” select a provider, enter the exact model identifier, select subjects/questions, set a seed and temperature, then press “Start experiment.” Provider requests occur only after this explicit action and may incur provider charges. The default mock provider is simulated; label its results as such.

Use “Dataset Explorer & Selection” to search and filter the full dataset, choose a reproducible question count, preview the selected questions, and run an isolated mock benchmark with **Run Benchmark with Selected Questions**. The generated experiment configuration, raw results, and analysis report are added to Experiment history without changing the source CSV or previous experiments.

Additional dashboard pages provide stored-response inspection, read-only dataset quality checks, subject-wise metrics, separate evidence-correction status (runs remain unavailable until verified evidence excerpts are supplied), repeated mock trials, and Markdown/CSV/PDF research report exports. Evidence correction records use a distinct `experiment_type` and are never mixed with false-suggestion metrics.

The **Experiment Design Studio** adds a saved-template workflow under `results/templates/`. It supports reproducible question selection, standard or validated custom challenge prompts, mock-only execution, separate repetitions, immutable configuration snapshots, and configuration-only template comparison. Template deletion never removes completed experiment results.

The dashboard also includes **Reproduce Experiment** and **Research Methodology & Limitations** pages. Analysis derives outcome classifications without rewriting historical raw files: neutral suggested-answer adoption is N/A, invalid/missing responses are separated, and answer changes are not automatically treated as sycophancy.

The equivalent command-line flow is:

```powershell
python run_benchmark.py --check
python run_benchmark.py --provider mock --model mock-deterministic --limit 3
python run_benchmark.py --provider openai --model <exact-model-id>
python analyze.py
```

Use Ctrl+C to interrupt a CLI run; each finished request is already on disk. Resume with the experiment ID printed by the runner: `python run_benchmark.py --resume <experiment-id>`. The dashboard also offers a resume selector for saved experiments.

Results are stored in `results/raw/` as JSONL, and configurations are stored in `results/configs/` as JSON. Existing raw files are append-only. Keep result files backed up outside version control.

## Analyze and report

The dashboard Results, Model comparison, Experimental conditions, Experiment history, and Research report pages read stored responses. With no results they show an empty state. The downloadable Markdown report uses recorded data only.

Run offline tests with:

```powershell
python -m pytest -q
```

## Dataset

See [data/README.md](data/README.md) for migration and quality status. Do not replace or silently edit the original answer key: correct it only with a documented review and a reliable citation.

## Limitations

The dataset is small, multiple-choice, mostly straightforward, and its supplied answer keys have not been independently source-checked. One suggested wrong answer and one challenge wording per condition cannot capture all social pressure. Models and provider APIs change over time. Output parsing can miss nuanced responses. Results do not generalize to open-ended conversations, other subject matter, or overall model safety.

## Future work

Complete source verification and independent review; improve pause/resume controls and provider error classification; add paired condition-level analyses and confidence intervals to the dashboard; expand datasets and preregistered hypotheses.
