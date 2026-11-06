from .base import ModelResponse


class AnthropicProvider:
    name = "anthropic"

    def generate(self, model: str, messages: list[dict[str, str]], temperature: float) -> ModelResponse:
        from anthropic import Anthropic
        response = Anthropic(timeout=45, max_retries=0).messages.create(
            model=model, max_tokens=40, temperature=temperature,
            messages=[m for m in messages if m["role"] != "system"])
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        return ModelResponse(text, response.usage.input_tokens, response.usage.output_tokens)
