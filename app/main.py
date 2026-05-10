from __future__ import annotations

from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from .db import init_db
from .routers.auth_router import router as auth_router
from .routers.protected_router import router as protected_router
from .worker import background_worker_loop
from .ws_manager import manager
from .auth import SECRET_KEY, ALGORITHM
from .mistral_client import MistralLLM
import jwt
import asyncio
import logging

logger = logging.getLogger("app")

app = FastAPI(title="Opply AI", version="0.2.0")

# ── CORS must be registered immediately after app creation, BEFORE routers ──
# In Starlette, middleware wraps the app in registration order. If routers are
# included first, the CORS middleware won't wrap those handlers, so preflight
# OPTIONS requests will get no Access-Control-Allow-Origin header.
import os as _os

_frontend_origin = _os.getenv("FRONTEND_ORIGIN", "").rstrip("/") or "https://opply-ai.vercel.app"

origins = list({
    _frontend_origin,
    "https://opply-ai.vercel.app",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
})

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()
    asyncio.create_task(background_worker_loop())


app.include_router(auth_router)
app.include_router(protected_router)

_llm = MistralLLM()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "llm_available": _llm.available,
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    origin = request.headers.get("origin", "")
    
    response = JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": str(exc),
            "hint": "Check backend logs; if using Mistral, confirm MISTRAL_API_KEY is set.",
        },
    )
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("Validation error: %s", exc)
    origin = request.headers.get("origin", "")
    
    response = JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "detail": str(exc),
        },
    )
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
    return response


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str):
    user_id = None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        sub = payload.get("sub")
        if sub is None:
            await websocket.close(code=1008)
            return
        user_id = int(sub)
    except Exception:
        await websocket.close(code=1008)
        return

    await manager.connect(websocket, user_id)
    try:
        while True:
            # Keep the connection open
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
