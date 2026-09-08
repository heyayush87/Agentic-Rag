"""Enable `python -m retailiq` alongside the installed `retailiq` script."""

from __future__ import annotations

from retailiq.cli.main import app

if __name__ == "__main__":
    app()
