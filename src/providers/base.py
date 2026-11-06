from dataclasses import dataclass
from typing import Protocol


@dataclass
class ModelResponse:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class Provider(Protocol):
    name: str

    def generate(self, model: str, messages: list[dict[str, str]], temperature: float) -> ModelResponse: ...
