"""Run an experiment from the command line. No paid calls happen without this command."""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from src.benchmark.dataset_loader import load_questions
from src.benchmark.dataset_quality_report import build_quality_report
from src.benchmark.experiment_runner import run_experiment
from src.benchmark.result_manager import ResultManager
from src.config.experiment_config import ExperimentConfig
from src.config.settings import DATA_PATH, RAW_DIR, RESULTS_DIR
from src.providers.anthropic_provider import AnthropicProvider
from src.providers.gemini_provider import GeminiProvider
from src.providers.mock_provider import MockProvider
from src.providers.openai_provider import OpenAIProvider

PROVIDERS = {"openai": OpenAIProvider, "anthropic": AnthropicProvider, "gemini": GeminiProvider, "mock": MockProvider}


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate dataset without making API calls")
    parser.add_argument("--provider", choices=PROVIDERS)
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--resume", help="resume an experiment by its saved experiment ID")
    args = parser.parse_args()
    questions = load_questions(DATA_PATH)
    if args.check:
        print(json.dumps(build_quality_report(questions), indent=2))
        return
    if args.resume:
        config_path = RESULTS_DIR / "configs" / f"{args.resume}.json"
        if not config_path.exists():
            parser.error(f"No saved configuration for experiment {args.resume}")
        config = ExperimentConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
        provider_class = PROVIDERS[config.provider]
    else:
        if not args.provider or not args.model:
            parser.error("--provider and --model are required unless --check or --resume is used")
        provider_class = PROVIDERS[args.provider]
        selected = questions.head(args.limit) if args.limit else questions
        config = ExperimentConfig(provider=args.provider, model=args.model, question_ids=selected.question_id.tolist(),
                                  random_seed=args.seed, temperature=args.temperature)
        config.save(RESULTS_DIR / "configs")
    target = questions[questions.question_id.isin(config.question_ids)]
    manager = ResultManager(RAW_DIR / f"{config.experiment_id}.jsonl")
    try:
        run_experiment(target, provider_class(), config.model, manager, config.experiment_id,
                       config.random_seed, config.temperature,
                       lambda done, total: print(f"Progress: {done}/{total}", end="\r"))
    except KeyboardInterrupt:
        print(f"\nInterrupted. Resume with --resume {config.experiment_id}")
        return
    print(f"\nExperiment {config.experiment_id} saved to {manager.path}")


if __name__ == "__main__":
    main()
