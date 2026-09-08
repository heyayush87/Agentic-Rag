"""HTTP transport. A thin shell over `services` — no business logic here."""

from __future__ import annotations

__all__ = ["create_app"]


def __getattr__(name: str) -> object:
    # Lazy so importing `retailiq.api` does not require FastAPI to be present
    # in deployments that only use the CLI.
    if name == "create_app":
        from retailiq.api.main import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
