from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "questions.csv"
RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
PROCESSED_DIR = RESULTS_DIR / "processed"
REPORTS_DIR = RESULTS_DIR / "reports"
DEFAULT_MODELS = {
    "openai": ["gpt-6-luna", "gpt-6.1-sol", "gpt-6.1-terra"],
    "anthropic": ["claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5-20251001"],
    "gemini": ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash"],
    "mock": ["mock-deterministic"],
}
