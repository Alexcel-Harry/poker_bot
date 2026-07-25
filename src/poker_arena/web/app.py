from __future__ import annotations

import hmac
import ipaddress
import json
import os
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


REVEAL_ALL_HOLE_CARDS = os.environ.get("POKER_ARENA_REVEAL_CARDS", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
room = PokerRoom(
    bot_policy_factory=bot_policy_factory_from_env(),
    reveal_all_hole_cards=REVEAL_ALL_HOLE_CARDS,
)
app = FastAPI(title="Poker Arena")
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

INVALID_SEAT_TOKEN_CLOSE_CODE = 4401
HOST_SESSION_COOKIE = "poker_arena_host_session"
HOST_SESSION_MAX_AGE_SECONDS = 12 * 60 * 60


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: list[tuple[WebSocket, str | None]] = []

    async def connect(self, websocket: WebSocket, seat_token: str | None) -> bool:
        await websocket.accept()
        try:
            snapshot = room.snapshot_for(seat_token=seat_token) if seat_token else room.snapshot_for()
        except PermissionError:
            await websocket.close(
                code=INVALID_SEAT_TOKEN_CLOSE_CODE,
                reason="Seat session expired; reconnect without the stored token",
            )
            return False
        self._connections.append((websocket, seat_token))
        await websocket.send_json(snapshot)
        return True

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


def _is_loopback_request(request: Request) -> bool:
    client = getattr(request, "client", None)
    host = getattr(client, "host", "") if client is not None else ""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _host_token_for_request(request: Request, supplied_token: str = "") -> str:
    """Prefer any credential that matches the current server session.

    A browser tab can retain an expired token after the server restarts.  The
    cookie is refreshed by the newly printed host URL and is shared by tabs on
    the same origin, so a stale form field cannot mask a valid host session.
    """

    if _is_loopback_request(request):
        return room.host_token

    cookie_token = request.cookies.get(HOST_SESSION_COOKIE, "")
    for candidate in (cookie_token, supplied_token):
        if candidate and hmac.compare_digest(candidate, room.host_token):
            return candidate
    return supplied_token or cookie_token


def _set_host_session_cookie(response: JSONResponse | HTMLResponse) -> None:
    response.set_cookie(
        key=HOST_SESSION_COOKIE,
        value=room.host_token,
        max_age=HOST_SESSION_MAX_AGE_SECONDS,
        httponly=True,
        samesite="strict",
        path="/",
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    response = HTMLResponse((STATIC_DIR / "index.html").read_text())
    supplied_token = request.query_params.get("host_token", "")
    if _is_loopback_request(request) or (
        supplied_token and hmac.compare_digest(supplied_token, room.host_token)
    ):
        _set_host_session_cookie(response)
    elif supplied_token:
        response.delete_cookie(HOST_SESSION_COOKIE, path="/")
    return response


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


@app.post("/api/next-hand")
async def next_hand(request: Request) -> JSONResponse:
    data = await request.json()
    try:
        result = room.start_next_hand(seat_token=str(data.get("seat_token", "")))
    except Exception as exc:
        return _error_response(exc)
    await manager.broadcast()
    return JSONResponse(result)


@app.post("/api/host/bots")
async def host_bots(request: Request) -> JSONResponse:
    data = await request.json()
    try:
        host_token = _host_token_for_request(request, str(data.get("host_token", "")))
        result = room.reserve_bot(host_token=host_token, seat_id=int(data.get("seat_id")))
    except Exception as exc:
        return _error_response(exc)
    await manager.broadcast()
    response = JSONResponse(result)
    _set_host_session_cookie(response)
    return response


@app.get("/api/logs/session.json")
async def session_json(request: Request, host_token: str = "") -> JSONResponse:
    try:
        payload = room.session_log(host_token=_host_token_for_request(request, host_token))
    except Exception as exc:
        return _error_response(exc)
    response = JSONResponse(
        payload,
        headers={"Content-Disposition": "attachment; filename=poker-arena-session.json"},
    )
    _set_host_session_cookie(response)
    return response


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    seat_token = websocket.query_params.get("seat_token")
    if not await manager.connect(websocket, seat_token):
        return
    try:
        while True:
            message = await websocket.receive_text()
            if message == "ping":
                await websocket.send_text("pong")
            else:
                json.loads(message)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
