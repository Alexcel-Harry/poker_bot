from __future__ import annotations

import argparse
import socket
import sys


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Poker Arena web table.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--lan", action="store_true", help="Bind to 0.0.0.0 for LAN access.")
    args = parser.parse_args()

    try:
        import uvicorn
        from poker_arena.web.app import app, room
    except ModuleNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    host = "0.0.0.0" if args.lan else "127.0.0.1"
    public_host = _lan_ip() if args.lan else host
    base_url = f"http://{public_host}:{args.port}"
    print(f"Poker Arena host URL: {base_url}/?host_token={room.host_token}")
    print(f"Guest URL: {base_url}/?room_code={room.room_code}")
    print("LAN HTTP is for trusted local networks only.")
    uvicorn.run(app, host=host, port=args.port)


if __name__ == "__main__":
    main()
