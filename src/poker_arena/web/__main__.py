from __future__ import annotations

import argparse
from pathlib import Path
import socket
import sys


DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "runs" / "poker_policy_gpu.pt"


def _lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"
    finally:
        sock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Poker Arena web table.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--lan", action="store_true", help="Bind to 0.0.0.0 for LAN access.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Bot checkpoint path (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default="cuda",
        help="Torch inference device (default: cuda).",
    )
    parser.add_argument(
        "--reveal-cards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reveal every active player's hole cards for debugging (default: on).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        parser.error(f"Bot checkpoint does not exist: {model_path}")

    try:
        import uvicorn
        import poker_arena.web.app as app_module
    except ModuleNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    room = app_module.configure_room(
        model_path=model_path,
        device=args.device,
        reveal_all_hole_cards=args.reveal_cards,
    )

    host = "0.0.0.0" if args.lan else "127.0.0.1"
    public_host = _lan_ip() if args.lan else host
    base_url = f"http://{public_host}:{args.port}"
    print(f"Poker Arena host URL: {base_url}/?host_token={room.host_token}")
    print(f"Guest URL: {base_url}/?room_code={room.room_code}")
    print(f"Bot checkpoint: {model_path}")
    print(f"Bot device: {args.device}")
    print(f"Debug card reveal: {'ON' if room.reveal_all_hole_cards else 'OFF'}")
    print("LAN HTTP is for trusted local networks only.")
    uvicorn.run(app_module.app, host=host, port=args.port)


if __name__ == "__main__":
    main()
