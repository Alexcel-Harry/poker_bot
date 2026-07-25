import unittest
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from poker_arena import Action, CheckCallBot
from poker_arena.web.bot_loader import bot_policy_factory_from_env
from poker_arena.web.__main__ import build_parser
from poker_arena.web.room import PokerRoom


class FakeWebSocket:
    def __init__(self) -> None:
        self.accepted = False
        self.sent: list[dict[str, object]] = []
        self.closed: tuple[int, str] | None = None

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, payload: dict[str, object]) -> None:
        self.sent.append(payload)

    async def close(self, code: int, reason: str) -> None:
        self.closed = (code, reason)


class FakeRequest:
    def __init__(self, *, client_host=None, cookies=None, query_params=None, payload=None) -> None:
        self.client = SimpleNamespace(host=client_host) if client_host is not None else None
        self.cookies = cookies or {}
        self.query_params = query_params or {}
        self.payload = payload or {}

    async def json(self) -> dict[str, object]:
        return self.payload


class PokerRoomSecurityTests(unittest.TestCase):
    def test_room_has_distinct_host_token_and_guest_code(self):
        room = PokerRoom(seed=11)

        self.assertNotEqual(room.host_token, room.room_code)
        self.assertGreaterEqual(len(room.host_token), 20)
        self.assertGreaterEqual(len(room.room_code), 6)

    def test_guest_claims_seat_with_room_code_and_secret_token(self):
        room = PokerRoom(seed=11)

        claim = room.join(room_code=room.room_code, seat_id=0, nickname="Ada")

        self.assertEqual(claim["seat_id"], 0)
        self.assertEqual(claim["nickname"], "Ada")
        self.assertGreaterEqual(len(claim["seat_token"]), 20)
        with self.assertRaises(PermissionError):
            room.join(room_code="bad-code", seat_id=1, nickname="Grace")
        with self.assertRaises(ValueError):
            room.join(room_code=room.room_code, seat_id=0, nickname="Grace")

    def test_host_can_reserve_bot_but_invalid_host_cannot(self):
        room = PokerRoom(seed=11)

        with self.assertRaises(PermissionError):
            room.reserve_bot(host_token="bad-token", seat_id=2)

        bot = room.reserve_bot(host_token=room.host_token, seat_id=2)

        self.assertEqual(bot["seat_id"], 2)
        self.assertEqual(bot["kind"], "bot")
        self.assertIn(".pt pending", bot["nickname"])

    def test_host_cannot_modify_human_seat_with_bot_reservation(self):
        room = PokerRoom(seed=11)
        room.join(room_code=room.room_code, seat_id=0, nickname="Ada")

        with self.assertRaises(ValueError):
            room.reserve_bot(host_token=room.host_token, seat_id=0)


class PokerRoomPlayTests(unittest.TestCase):
    def test_player_payload_only_contains_own_private_cards(self):
        room = PokerRoom(seed=7)
        ada = room.join(room_code=room.room_code, seat_id=0, nickname="Ada")
        grace = room.join(room_code=room.room_code, seat_id=1, nickname="Grace")

        ada_payload = room.snapshot_for(seat_token=ada["seat_token"])
        grace_payload = room.snapshot_for(seat_token=grace["seat_token"])
        public_payload = room.snapshot_for()

        self.assertEqual(len(ada_payload["private_hole_cards"]), 2)
        self.assertEqual(len(grace_payload["private_hole_cards"]), 2)
        self.assertNotEqual(ada_payload["private_hole_cards"], grace_payload["private_hole_cards"])
        self.assertIsNone(public_payload["private_hole_cards"])
        self.assertNotIn("deck_cards", str(public_payload))

    def test_wrong_seat_cannot_submit_current_player_action(self):
        room = PokerRoom(seed=7)
        ada = room.join(room_code=room.room_code, seat_id=0, nickname="Ada")
        grace = room.join(room_code=room.room_code, seat_id=1, nickname="Grace")

        self.assertEqual(room.snapshot_for()["current_actor"], 0)
        with self.assertRaises(PermissionError):
            room.submit_action(seat_token=grace["seat_token"], action=Action.call())

        room.submit_action(seat_token=ada["seat_token"], action=Action.call())

    def test_inactive_bot_turn_pauses_table(self):
        room = PokerRoom(seed=7)
        ada = room.join(room_code=room.room_code, seat_id=0, nickname="Ada")
        room.reserve_bot(host_token=room.host_token, seat_id=1)

        room.submit_action(seat_token=ada["seat_token"], action=Action.call())
        payload = room.snapshot_for(seat_token=ada["seat_token"])

        self.assertEqual(payload["current_actor"], 1)
        self.assertEqual(payload["status"], "paused")
        self.assertEqual(payload["paused_reason"], "waiting for unavailable bot")

    def test_policy_backed_bot_auto_advances_after_human_action(self):
        room = PokerRoom(seed=7, bot_policy_factory=lambda _seat_id: CheckCallBot())
        ada = room.join(room_code=room.room_code, seat_id=0, nickname="Ada")
        room.reserve_bot(host_token=room.host_token, seat_id=1)

        room.submit_action(seat_token=ada["seat_token"], action=Action.call())
        payload = room.snapshot_for(seat_token=ada["seat_token"])

        self.assertEqual(payload["status"], "playing")
        self.assertEqual(payload["current_actor"], 0)
        self.assertTrue(
            any(
                event["event_type"] == "action"
                and event["data"]["seat_id"] == 1
                and event["data"]["action"]["type"] == "check"
                for event in payload["log"]
            )
        )

    def test_human_snapshot_explicitly_reports_raise_availability(self):
        room = PokerRoom(seed=7)
        ada = room.join(room_code=room.room_code, seat_id=0, nickname="Ada")
        room.join(room_code=room.room_code, seat_id=1, nickname="Grace")

        legal = room.snapshot_for(seat_token=ada["seat_token"])["legal_actions"]

        self.assertTrue(legal["can_raise"])
        self.assertEqual(legal["min_raise_to"], 40)

    def test_policy_backed_bot_auto_advances_when_human_join_starts_hand(self):
        room = PokerRoom(seed=7, bot_policy_factory=lambda _seat_id: CheckCallBot())
        room.reserve_bot(host_token=room.host_token, seat_id=0)

        ada = room.join(room_code=room.room_code, seat_id=1, nickname="Ada")
        payload = room.snapshot_for(seat_token=ada["seat_token"])

        self.assertEqual(payload["status"], "playing")
        self.assertEqual(payload["current_actor"], 1)

    def test_bot_advance_guard_stops_bot_only_loops(self):
        room = PokerRoom(seed=7, bot_policy_factory=lambda _seat_id: CheckCallBot())
        room.reserve_bot(host_token=room.host_token, seat_id=0)
        room.reserve_bot(host_token=room.host_token, seat_id=1)

        with self.assertRaises(RuntimeError):
            room.advance_bots(max_actions=1)

    def test_session_log_excludes_private_cards_and_snapshots(self):
        room = PokerRoom(seed=7)
        ada = room.join(room_code=room.room_code, seat_id=0, nickname="Ada")
        room.join(room_code=room.room_code, seat_id=1, nickname="Grace")
        room.submit_action(seat_token=ada["seat_token"], action=Action.call())

        log = room.session_log(host_token=room.host_token)

        self.assertEqual(log["room_code"], room.room_code)
        self.assertTrue(log["hands"])
        self.assertNotIn("snapshot", str(log))
        self.assertEqual(len(log["hands"][0]["hole_cards"]), 2)
        self.assertTrue(all(len(player["cards"]) == 2 for player in log["hands"][0]["hole_cards"]))
        with self.assertRaises(PermissionError):
            room.session_log(host_token="bad-token")

    def test_log_events_keep_display_seat_ids_when_an_earlier_seat_is_inactive(self):
        room = PokerRoom(seed=7)
        ada = room.join(room_code=room.room_code, seat_id=0, nickname="Ada")
        room.join(room_code=room.room_code, seat_id=2, nickname="Grace")

        room.submit_action(seat_token=ada["seat_token"], action=Action.fold())
        log = room.session_log(host_token=room.host_token)
        events = log["hands"][0]["events"]
        hand_started = next(event for event in events if event["event_type"] == "hand_started")
        big_blind = next(event for event in events if event["event_type"] == "big_blind")
        award = next(event for event in events if event["event_type"] == "pot_awarded")
        finished = next(event for event in events if event["event_type"] == "hand_finished")

        self.assertEqual(log["seat_id_space"], "display")
        self.assertEqual(hand_started["data"]["big_blind_seat"], 2)
        self.assertEqual(big_blind["data"]["seat_id"], 2)
        self.assertEqual(award["data"]["seat_id"], 2)
        self.assertEqual([item["seat_id"] for item in finished["data"]["stacks_by_seat"]], [0, 2])
        self.assertEqual([player["seat_id"] for player in log["hands"][0]["hole_cards"]], [0, 2])

    def test_debug_reveal_exposes_all_hole_cards_in_snapshot(self):
        room = PokerRoom(seed=7, reveal_all_hole_cards=True)
        ada = room.join(room_code=room.room_code, seat_id=0, nickname="Ada")
        room.join(room_code=room.room_code, seat_id=1, nickname="Grace")

        snapshot = room.snapshot_for(seat_token=ada["seat_token"])

        self.assertTrue(snapshot["debug_reveal"])
        self.assertEqual(set(snapshot["revealed_hole_cards"]), {0, 1})
        self.assertTrue(all(len(cards) == 2 for cards in snapshot["revealed_hole_cards"].values()))

    def test_finished_hand_remains_visible_until_human_starts_next_hand(self):
        room = PokerRoom(seed=7, bot_policy_factory=lambda _seat_id: CheckCallBot())
        ada = room.join(room_code=room.room_code, seat_id=0, nickname="Ada")
        room.reserve_bot(host_token=room.host_token, seat_id=1)

        while room.table is not None and room.table.current_hand is not None and not room.table.current_hand.is_terminal:
            state = room.table.current_hand
            if room.engine_to_display[state.current_actor] == 0:
                legal = state.legal_actions(state.current_actor)
                action = Action.check() if legal.can_check else Action.call()
                room.submit_action(seat_token=ada["seat_token"], action=action)
            else:
                room.advance_bots()

        finished = room.snapshot_for(seat_token=ada["seat_token"])
        self.assertEqual(finished["status"], "finished")
        self.assertEqual(finished["street"], "showdown")
        self.assertEqual(len(finished["board"]), 5)
        self.assertEqual(len(room.completed_hands), 1)

        room.start_next_hand(seat_token=ada["seat_token"])
        next_snapshot = room.snapshot_for(seat_token=ada["seat_token"])
        self.assertNotEqual(next_snapshot["status"], "finished")
        next_hand_started = next(event for event in next_snapshot["log"] if event["event_type"] == "hand_started")
        self.assertEqual(next_hand_started["data"]["hand_number"], 2)


class StaticAppTests(unittest.TestCase):
    def test_bot_policy_factory_from_env_loads_one_shared_checkpoint(self):
        loaded_policy = CheckCallBot()

        with patch("poker_arena.bots.TorchPolicyBot.from_checkpoint", return_value=loaded_policy) as load:
            factory = bot_policy_factory_from_env(
                {"POKER_BOT_MODEL": "runs/poker_policy.pt", "POKER_BOT_DEVICE": "cpu"}
            )

            self.assertIsNotNone(factory)
            assert factory is not None
            self.assertIs(factory(1), loaded_policy)
            self.assertIs(factory(2), loaded_policy)
            load.assert_called_once_with("runs/poker_policy.pt", device="cpu")

    def test_bot_policy_factory_from_env_returns_none_without_model_path(self):
        self.assertIsNone(bot_policy_factory_from_env({}))

    def test_static_frontend_contains_required_controls(self):
        static_dir = Path(__file__).resolve().parents[1] / "src" / "poker_arena" / "web" / "static"

        index = (static_dir / "index.html").read_text()
        app_js = (static_dir / "app.js").read_text()
        styles = (static_dir / "styles.css").read_text()

        self.assertIn("seat-grid", index)
        self.assertIn("raiseByInput", index)
        self.assertIn("Raise By", index)
        self.assertIn("nextHandButton", index)
        self.assertIn("raiseForm", index)
        self.assertIn("seat-cards", index)
        self.assertIn("downloadLog", app_js)
        self.assertIn("Automatic on this PC; required over LAN", index)
        self.assertIn("raise_by", app_js)
        self.assertIn("raiseByToRaiseTo", app_js)
        self.assertIn("localStorage", app_js)
        self.assertIn('localStorage.removeItem("pokerArenaSeatToken")', app_js)
        self.assertIn("INVALID_SEAT_TOKEN_CLOSE_CODE", app_js)
        self.assertIn('event.key !== "pokerArenaHostToken"', app_js)
        self.assertIn('document.getElementById("roomCodeInput").value = snap.room_code', app_js)
        self.assertNotIn('document.getElementById("roomCodeInput").value ||= snap.room_code', app_js)
        self.assertNotIn('.seat[data-seat="8"] { left: calc(50% - 67px); top: 42%; }', styles)
        self.assertIn('.seat[data-seat="8"] { left: 18%; top: 72px; }', styles)
        self.assertIn("styles.css?v=20260725-host-session-1", index)
        self.assertIn("submitQuickRaise", app_js)
        self.assertIn("submitCustomRaise", app_js)
        self.assertIn("debug cards: ON", app_js)
        self.assertIn("uncalled bet returned", app_js)
        self.assertNotIn("if (!hostToken) return", app_js)
        self.assertIn("app.js?v=20260725-pot-audit-1", index)

    def test_web_cli_accepts_explicit_model_device_and_reveal_settings(self):
        args = build_parser().parse_args(
            ["--model", "runs/custom.pt", "--device", "cuda", "--no-reveal-cards", "--port", "8123"]
        )

        self.assertEqual(args.model, Path("runs/custom.pt"))
        self.assertEqual(args.device, "cuda")
        self.assertFalse(args.reveal_cards)
        self.assertEqual(args.port, 8123)

    def test_windows_launcher_does_not_modify_powershell_environment(self):
        launcher = Path(__file__).resolve().parents[1] / "scripts" / "run_poker_arena.ps1"
        script = launcher.read_text()

        self.assertIn('"--model", $resolvedModel', script)
        self.assertIn('"--device", $Device', script)
        self.assertIn('"--reveal-cards"', script)
        self.assertNotIn("$env:", script.lower())
        self.assertNotIn("POKER_BOT_MODEL", script)


class HostSessionRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_loopback_request_can_reserve_bot_without_host_token(self):
        web_app = importlib.import_module("poker_arena.web.app")
        fresh_room = PokerRoom(seed=23)
        request = FakeRequest(client_host="127.0.0.1", payload={"host_token": "", "seat_id": 2})

        with patch.object(web_app, "room", fresh_room):
            response = await web_app.host_bots(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(fresh_room.seats[2].kind, "bot")

    async def test_remote_request_without_host_token_is_forbidden(self):
        web_app = importlib.import_module("poker_arena.web.app")
        fresh_room = PokerRoom(seed=23)
        request = FakeRequest(client_host="192.0.2.10", payload={"host_token": "", "seat_id": 2})

        with patch.object(web_app, "room", fresh_room):
            response = await web_app.host_bots(request)

        self.assertEqual(response.status_code, 403)
        self.assertEqual(fresh_room.seats[2].kind, "empty")

    async def test_current_host_cookie_overrides_stale_form_token(self):
        web_app = importlib.import_module("poker_arena.web.app")
        fresh_room = PokerRoom(seed=23)
        request = FakeRequest(
            cookies={web_app.HOST_SESSION_COOKIE: fresh_room.host_token},
            payload={"host_token": "expired-token-from-old-tab", "seat_id": 2},
        )

        with patch.object(web_app, "room", fresh_room):
            response = await web_app.host_bots(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(fresh_room.seats[2].kind, "bot")
        self.assertIn(web_app.HOST_SESSION_COOKIE, response.headers["set-cookie"])

    async def test_expired_form_token_without_current_host_cookie_is_forbidden(self):
        web_app = importlib.import_module("poker_arena.web.app")
        fresh_room = PokerRoom(seed=23)
        request = FakeRequest(payload={"host_token": "expired-token-from-old-tab", "seat_id": 2})

        with patch.object(web_app, "room", fresh_room):
            response = await web_app.host_bots(request)

        self.assertEqual(response.status_code, 403)
        self.assertEqual(fresh_room.seats[2].kind, "empty")

    async def test_current_host_url_sets_session_cookie(self):
        web_app = importlib.import_module("poker_arena.web.app")
        fresh_room = PokerRoom(seed=23)
        request = FakeRequest(query_params={"host_token": fresh_room.host_token})

        with patch.object(web_app, "room", fresh_room):
            response = await web_app.index(request)

        self.assertEqual(response.status_code, 200)
        self.assertIn(web_app.HOST_SESSION_COOKIE, response.headers["set-cookie"])
        self.assertIn("HttpOnly", response.headers["set-cookie"])

    async def test_bare_loopback_root_sets_session_cookie(self):
        web_app = importlib.import_module("poker_arena.web.app")
        fresh_room = PokerRoom(seed=23)
        request = FakeRequest(client_host="::1")

        with patch.object(web_app, "room", fresh_room):
            response = await web_app.index(request)

        self.assertEqual(response.status_code, 200)
        self.assertIn(web_app.HOST_SESSION_COOKIE, response.headers["set-cookie"])


class WebSocketSessionRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_stored_seat_token_closes_cleanly_without_registering_connection(self):
        web_app = importlib.import_module("poker_arena.web.app")
        connection_manager = web_app.ConnectionManager()
        websocket = FakeWebSocket()

        with patch.object(web_app, "room", PokerRoom(seed=17)):
            connected = await connection_manager.connect(websocket, "expired-seat-token")

        self.assertFalse(connected)
        self.assertTrue(websocket.accepted)
        self.assertEqual(websocket.closed[0], web_app.INVALID_SEAT_TOKEN_CLOSE_CODE)
        self.assertIn("expired", websocket.closed[1].lower())
        self.assertEqual(connection_manager._connections, [])
        self.assertEqual(websocket.sent, [])

    async def test_anonymous_reconnect_receives_public_snapshot(self):
        web_app = importlib.import_module("poker_arena.web.app")
        connection_manager = web_app.ConnectionManager()
        websocket = FakeWebSocket()
        fresh_room = PokerRoom(seed=17)

        with patch.object(web_app, "room", fresh_room):
            connected = await connection_manager.connect(websocket, None)

        self.assertTrue(connected)
        self.assertIsNone(websocket.closed)
        self.assertEqual(websocket.sent[0]["room_code"], fresh_room.room_code)
        self.assertEqual(len(connection_manager._connections), 1)


if __name__ == "__main__":
    unittest.main()
