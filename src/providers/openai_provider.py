from .base import ModelResponse


class OpenAIProvider:
    name = "openai"

    def generate(self, model: str, messages: list[dict[str, str]], temperature: float) -> ModelResponse:
        from openai import OpenAI
        response = OpenAI(timeout=45, max_retries=0).chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_completion_tokens=40)
        usage = response.usage
        message = response.choices[0].message
        return ModelResponse(message.content or getattr(message, "refusal", None) or "",
                             usage.prompt_tokens if usage else None,
                             usage.completion_tokens if usage else None)
