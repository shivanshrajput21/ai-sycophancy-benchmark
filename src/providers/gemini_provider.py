from .base import ModelResponse


class GeminiProvider:
    name = "gemini"

    def generate(self, model: str, messages: list[dict[str, str]], temperature: float) -> ModelResponse:
        from google import genai
        from google.genai import types
        client = genai.Client()
        contents = [types.Content(
            role="model" if message["role"] == "assistant" else "user",
            parts=[types.Part.from_text(text=message["content"])],
        ) for message in messages]
        response = client.models.generate_content(model=model, contents=contents,
            config=types.GenerateContentConfig(temperature=temperature, max_output_tokens=40))
        usage = response.usage_metadata
        return ModelResponse(response.text or "",
            getattr(usage, "prompt_token_count", None), getattr(usage, "candidates_token_count", None))
