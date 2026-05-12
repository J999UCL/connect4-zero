"""Minimal web server for playing 4x4x4 Connect Four against MCTS."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from connect4_zero.game import Connect4x4x4Batch  # noqa: E402
from connect4_zero.game.constants import ACTION_SIZE, BOARD_SIZE  # noqa: E402
from connect4_zero.search import MCTS, MCTSConfig  # noqa: E402

BotActionProvider = Callable[[Connect4x4x4Batch], int]


class GameSession:
    """One human-vs-MCTS game session.

    The core engine stores canonical state from the current player perspective.
    ``engine_player`` records whose perspective the engine currently represents
    so the web payload can always display the board from the human perspective.
    """

    def __init__(
        self,
        simulations: int = 48,
        rollout_batch_size: int = 32,
        device: str = "cpu",
        seed: Optional[int] = None,
        bot_action_provider: Optional[BotActionProvider] = None,
    ) -> None:
        self.simulations = int(simulations)
        self.rollout_batch_size = int(rollout_batch_size)
        self.device = self._resolve_device(device)
        self.seed = seed
        self.bot_action_provider = bot_action_provider
        self._lock = threading.RLock()
        self.reset()

    def reset(self) -> Dict[str, Any]:
        with self._lock:
            self.game = Connect4x4x4Batch(batch_size=1, device=self.device)
            self.engine_player = "human"
            self.winner: Optional[str] = None
            self.last_human_action: Optional[int] = None
            self.last_bot_action: Optional[int] = None
            self.last_bot_policy = [0.0 for _ in range(ACTION_SIZE)]
            self.move_count = 0
            self.message = "Your move."
            return self.state_payload()

    def human_move(self, action: int) -> Dict[str, Any]:
        with self._lock:
            if self.is_terminal:
                raise ValueError("The game is already over. Reset to play again.")
            if self.engine_player != "human":
                raise ValueError("It is not the human player's turn.")
            if not isinstance(action, int):
                raise ValueError("Action must be an integer from 0 to 15.")

            result = self.game.step(torch.tensor([action], dtype=torch.long, device=self.game.device))
            if not bool(result.legal[0].item()):
                raise ValueError(f"Action {action} is not legal.")

            self.last_human_action = action
            self.move_count += 1

            if bool(result.won[0].item()):
                self.winner = "human"
                self.message = "You won."
                return self.state_payload()
            if bool(result.draw[0].item()):
                self.winner = "draw"
                self.message = "Draw."
                return self.state_payload()

            self.engine_player = "bot"
            self.message = "Bot thinking."
            return self._bot_move()

    @property
    def is_terminal(self) -> bool:
        return bool(self.game.done[0].item())

    def state_payload(self) -> Dict[str, Any]:
        with self._lock:
            legal_mask = self._human_legal_mask()
            board = self._board_for_human().cpu().tolist()
            heights = self.game.heights[0].cpu().tolist()

            return {
                "board": board,
                "heights": heights,
                "legalActions": legal_mask,
                "turn": "game_over" if self.is_terminal else self.engine_player,
                "winner": self.winner,
                "message": self.message,
                "lastHumanAction": self.last_human_action,
                "lastBotAction": self.last_bot_action,
                "botPolicy": self.last_bot_policy,
                "moveCount": self.move_count,
                "device": str(self.device),
                "simulations": self.simulations,
                "rolloutBatchSize": self.rollout_batch_size,
            }

    def _bot_move(self) -> Dict[str, Any]:
        action = self._choose_bot_action()
        result = self.game.step(torch.tensor([action], dtype=torch.long, device=self.game.device))
        if not bool(result.legal[0].item()):
            raise RuntimeError(f"Bot selected illegal action {action}.")

        self.last_bot_action = action
        self.move_count += 1

        if bool(result.won[0].item()):
            self.winner = "bot"
            self.message = f"Bot played {action} and won."
            return self.state_payload()
        if bool(result.draw[0].item()):
            self.winner = "draw"
            self.message = f"Bot played {action}. Draw."
            return self.state_payload()

        self.engine_player = "human"
        self.message = f"Bot played {action}. Your move."
        return self.state_payload()

    def _choose_bot_action(self) -> int:
        if self.bot_action_provider is not None:
            return int(self.bot_action_provider(self.game.clone()))

        config = MCTSConfig(
            num_simulations=self.simulations,
            rollout_batch_size=self.rollout_batch_size,
            rollout_device=str(self.device),
            seed=None if self.seed is None else self.seed + self.move_count,
        )
        result = MCTS(config=config).search(self.game)
        self.last_bot_policy = [float(value) for value in result.policy.detach().cpu().tolist()]
        return int(result.policy.argmax().detach().cpu().item())

    def _board_for_human(self) -> torch.Tensor:
        board = self.game.board[0]
        if self.engine_player == "bot":
            return -board
        return board

    def _human_legal_mask(self) -> List[bool]:
        if self.is_terminal or self.engine_player != "human":
            return [False for _ in range(ACTION_SIZE)]
        return [bool(value) for value in self.game.legal_mask()[0].cpu().tolist()]

    def _resolve_device(self, requested: str) -> torch.device:
        if requested == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(requested)


class PlayRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler serving the static app and JSON API."""

    session: GameSession
    static_dir: Path

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            self._send_json(self.session.state_payload())
            return
        if parsed.path in ("", "/"):
            self._send_file(self.static_dir / "index.html")
            return

        path = (self.static_dir / parsed.path.lstrip("/")).resolve()
        if self.static_dir not in path.parents and path != self.static_dir:
            self._send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        self._send_file(path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/reset":
                self._send_json(self.session.reset())
                return
            if parsed.path == "/api/move":
                payload = self._read_json()
                self._send_json(self.session.human_move(int(payload["action"])))
                return
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")
        except (KeyError, TypeError, ValueError) as exc:
            self._send_json({"error": str(exc), "state": self.session.state_payload()}, status=HTTPStatus.BAD_REQUEST)
        except RuntimeError as exc:
            self._send_json({"error": str(exc), "state": self.session.state_payload()}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        data = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status=status)


def make_server(
    host: str,
    port: int,
    session: GameSession,
    static_dir: Optional[Path] = None,
) -> ThreadingHTTPServer:
    static_path = static_dir if static_dir is not None else Path(__file__).resolve().parent / "static"

    class Handler(PlayRequestHandler):
        pass

    Handler.session = session
    Handler.static_dir = static_path.resolve()
    return ThreadingHTTPServer((host, port), Handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play 4x4x4 Connect Four against MCTS.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--simulations", type=int, default=48)
    parser.add_argument("--rollout-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu", help="cpu, mps, cuda, or auto")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = GameSession(
        simulations=args.simulations,
        rollout_batch_size=args.rollout_batch_size,
        device=args.device,
        seed=args.seed,
    )
    server = make_server(host=args.host, port=args.port, session=session)
    print(f"Serving Connect4 Zero play app at http://{args.host}:{args.port}", flush=True)
    print(
        f"Bot: {args.simulations} simulations, rollout batch {args.rollout_batch_size}, device {session.device}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
