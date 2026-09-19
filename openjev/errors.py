"""Исключения OpenJev, не зависящие от PyTorch."""


class EngineError(RuntimeError):
    """Сбой загрузки модели или инференса."""
