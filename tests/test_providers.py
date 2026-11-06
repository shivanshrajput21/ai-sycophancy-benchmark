from src.providers.mock_provider import MockProvider


def test_mock_provider_is_reproducible():
    provider = MockProvider()
    messages = [{"role": "user", "content": "question"}]
    assert provider.generate("mock", messages, 0).text == provider.generate("mock", messages, 0).text
