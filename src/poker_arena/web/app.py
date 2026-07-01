from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from poker_arena.actions import Action
from poker_arena.web.bot_loader import bot_policy_factory_from_env
from poker_arena.web.room import PokerRoom

try:
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
except ModuleNotFoundError as exc:  # pragma: no cover - exercised when dependencies are missing
    raise ModuleNotFoundError(
        "FastAPI and uvicorn are required for the web GUI. Install with: python3.13 -m pip install -e ."
    ) from exc


room = PokerRoom(bot_policy_factory=bot_policy_factory_from_env())
app = FastAPI(title="Poker Arena")
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: list[tuple[WebSocket, str | None]] = []

    async def connect(self, websocket: WebSocket, seat_token: str | None) -> None:
        await websocket.accept()
        self._connections.append((websocket, seat_token))
        await websocket.send_json(room.snapshot_for(seat_token=seat_token) if seat_token else room.snapshot_for())

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections = [(ws, token) for ws, token in self._connections if ws is not websocket]

    async def broadcast(self) -> None:
        live: list[tuple[WebSocket, str | None]] = []
        for websocket, seat_token in self._connections:
            try:
                await websocket.send_json(room.snapshot_for(seat_token=seat_token) if seat_token else room.snapshot_for())
                live.append((websocket, seat_token))
            except Exception:
                continue
        self._connections = live


manager = ConnectionManager()


def _error_response(exc: Exception) -> JSONResponse:
    status = 403 if isinstance(exc, PermissionError) else 400
    return JSONResponse({"error": str(exc)}, status_code=status)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (STATIC_DIR / "index.html").read_text()


@app.post("/api/join")
async def join(request: Request) -> JSONResponse:
    data = await request.json()
    try:
        result = room.join(
            room_code=str(data.get("room_code", "")),
            seat_id=int(data.get("seat_id")),
            nickname=str(data.get("nickname", "")),
        )
    except Exception as exc:
        return _error_response(exc)
    await manager.broadcast()
    return JSONResponse(result)


@app.post("/api/action")
async def action(request: Request) -> JSONResponse:
    data = await request.json()
    try:
        raw_action: dict[str, Any] = data.get("action", {})
        result = room.submit_action(seat_token=str(data.get("seat_token", "")), action=Action.from_dict(raw_action))
    except Exception as exc:
        return _error_response(exc)
    await manager.broadcast()
    return JSONResponse(result)


@app.post("/api/host/bots")
async def host_bots(request: Request) -> JSONResponse:
    data = await request.json()
    try:
        result = room.reserve_bot(host_token=str(data.get("host_token", "")), seat_id=int(data.get("seat_id")))
    except Exception as exc:
        return _error_response(exc)
    await manager.broadcast()
    return JSONResponse(result)


@app.get("/api/logs/session.json")
async def session_json(host_token: str) -> JSONResponse:
    try:
        payload = room.session_log(host_token=host_token)
    except Exception as exc:
        return _error_response(exc)
    return JSONResponse(
        payload,
        headers={"Content-Disposition": "attachment; filename=poker-arena-session.json"},
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    seat_token = websocket.query_params.get("seat_token")
    await manager.connect(websocket, seat_token)
    try:
        while True:
            message = await websocket.receive_text()
            if message == "ping":
                await websocket.send_text("pong")
            else:
                json.loads(message)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
