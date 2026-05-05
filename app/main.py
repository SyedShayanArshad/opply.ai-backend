from __future__ import annotations

from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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


@app.on_event("startup")
def on_startup():
    init_db()
    asyncio.create_task(background_worker_loop())


app.include_router(auth_router)
app.include_router(protected_router)

_llm = MistralLLM()


def _cors_origin() -> str:
    import os
    frontend_origin = os.getenv("FRONTEND_ORIGIN")
    if not frontend_origin:
        if os.getenv("RENDER") or os.getenv("RENDER_EXTERNAL_URL"):
            frontend_origin = "https://opply-ai.vercel.app"
        else:
            frontend_origin = "http://localhost:5173"
            
    # Clean the origin: remove trailing slash if present
    if frontend_origin.endswith("/"):
        frontend_origin = frontend_origin[:-1]
    return frontend_origin


origins = [
    _cors_origin(),
    "https://opply-ai.vercel.app",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "llm_available": _llm.available,
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": str(exc),
            "hint": "Check backend logs; if using Mistral, confirm MISTRAL_API_KEY is set.",
        },
    )


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
