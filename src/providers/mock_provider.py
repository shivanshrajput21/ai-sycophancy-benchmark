import hashlib
import random
from .base import ModelResponse


class MockProvider:
    name = "mock"

    def generate(self, model: str, messages: list[dict[str, str]], temperature: float) -> ModelResponse:
        seed = int(hashlib.sha256(repr(messages).encode()).hexdigest()[:8], 16)
        return ModelResponse(random.Random(seed).choice("ABCD"))
