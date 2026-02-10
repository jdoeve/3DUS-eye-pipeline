"""UC-Eye package public API."""


def run_pipeline(*args, **kwargs):
    from .pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)


__all__ = ["run_pipeline"]
