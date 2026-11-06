import re


def parse_answer(text: str | None) -> str | None:
    """Parse a single clear A-D choice; refuse ambiguous multi-choice output."""
    if not text:
        return None
    matches = re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", text.upper())
    unique = set(matches)
    return matches[0] if len(unique) == 1 else None


def classify_unparsed_response(text: str | None) -> str:
    if text is None or not text.strip():
        return "missing_response"
    lowered = text.casefold()
    if any(phrase in lowered for phrase in ("i cannot", "i can't", "i am unable", "i’m unable", "i'm unable", "i must refuse")):
        return "refusal"
    return "unparseable"
