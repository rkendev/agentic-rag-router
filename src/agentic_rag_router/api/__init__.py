"""HTTP surface for the router --- the FastAPI ``POST /ask`` app (D5).

`app.create_app` builds the application; `app.app` is the module-level instance
uvicorn serves. See `app.py` for the lifespan-loading and threadpool details.
"""

from __future__ import annotations

from agentic_rag_router.api.app import app, create_app

__all__ = ["app", "create_app"]
