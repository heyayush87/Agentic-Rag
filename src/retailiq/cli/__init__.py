"""Command-line interface."""

from __future__ import annotations

__all__ = ["app"]


def __getattr__(name: str) -> object:
    if name == "app":
        from retailiq.cli.main import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
