"""API routers, one module per resource."""

from __future__ import annotations

from retailiq.api.routers import admin, health, query

__all__ = ["admin", "health", "query"]
