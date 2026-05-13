"""Terminal play loop for human-vs-deep-MCTS 4x4x4 Connect Four."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional, TextIO

import torch

from connect4_zero.game import Connect4x4x4Batch
from connect4_zero.game.constants import ACTION_SIZE, BOARD_SIZE, CURRENT_PLAYER, OPPONENT_PLAYER
from connect4_zero.scripts._common import format_seconds, resolve_device, sync_if_cuda
from connect4_zero.search import BatchedSearchResult, BatchedTreeMCTS, SearchTree, TreeMCTSConfig

InputFn = Callable[[str], str]


@dataclass(frozen=True)
class MoveSummary:
    """Human-readable result of one move."""

    actor: str
    action: int
    won: bool
    draw: bool
    tree_reused: bool


@dataclass(frozen=True)
class BotSearchSummary:
    """Diagnostics from the most recent bot search."""

    action: int
    duration_seconds: float
    root_value: float
    tree_reused_before_search: bool
    tree_reused_by_search: bool
    tree_reused_after_move: bool
    nodes: int
    max_depth: int
    leaf_evaluations: int
    terminal_evaluations: int
    max_leaf_batch: int
    max_expansion_batch: int
    timing: dict[str, float]
    policy: torch.Tensor
    visit_counts: torch.Tensor
    q_values: torch.Tensor


class TerminalPlaySession:
    """One terminal game against production batched tree MCTS."""

    def __init__(
        self,
        search: BatchedTreeMCTS,
        game: Optional[Connect4x4x4Batch] = None,
        human_starts: bool = True,
    ) -> None:
        self.search = search
        self.root_device = _root_device_for(search.config.rollout_device)
        self.game = game if game is not None else Connect4x4x4Batch(1, device=self.root_device)
        if self.game.batch_size != 1:
            raise ValueError("terminal play requires a single-game batch")

        self.engine_player = "human" if human_starts else "bot"
        self.current_tree: Optional[SearchTree] = None
        self.winner: Optional[str] = None
        self.move_count = 0
        self.last_human_action: Optional[int] = None
        self.last_bot_action: Optional[int] = None
        self.last_bot_summary: Optional[BotSearchSummary] = None

    @property
    def is_terminal(self) -> bool:
        return bool(self.game.done[0].item())

    def reset(self, human_starts: bool = True) -> None:
        self.game = Connect4x4x4Batch(1, device=self.root_device)
        self.engine_player = "human" if human_starts else "bot"
        self.current_tree = None
        self.winner = None
        self.move_count = 0
        self.last_human_action = None
        self.last_bot_action = None
        self.last_bot_summary = None

    def legal_actions(self) -> list[int]:
        mask = self.game.legal_mask()[0].detach().cpu().tolist()
        return [index for index, legal in enumerate(mask) if bool(legal)]

    def human_move(self, action: int) -> MoveSummary:
        if self.is_terminal:
            raise ValueError("The game is already over. Use reset to start again.")
        if self.engine_player != "human":
            raise ValueError("It is not the human turn.")

        action = int(action)
        if action < 0 or action >= ACTION_SIZE:
            raise ValueError(f"Action must be between 0 and {ACTION_SIZE - 1}.")

        previous_tree = self.current_tree
        result = self.game.step(torch.tensor([action], dtype=torch.long, device=self.game.device))
        if not bool(result.legal[0].item()):
            self.current_tree = previous_tree
            raise ValueError(f"Action {action} is not legal.")

        tree_reused = self._advance_current_tree(action)
        self.last_human_action = action
        self.move_count += 1

        if bool(result.won[0].item()):
            self.winner = "human"
        elif bool(result.draw[0].item()):
            self.winner = "draw"
        else:
            self.engine_player = "bot"

        return MoveSummary(
            actor="human",
            action=action,
            won=bool(result.won[0].item()),
            draw=bool(result.draw[0].item()),
            tree_reused=tree_reused,
        )

    def bot_move(self) -> MoveSummary:
        if self.is_terminal:
            raise ValueError("The game is already over. Use reset to start again.")
        if self.engine_player != "bot":
            raise ValueError("It is not the bot turn.")

        tree_before = self.current_tree
        started_at = time.perf_counter()
        result = self.search.search_batch_with_trees(self.game, [self.current_tree])
        sync_if_cuda(torch.device(self.search.config.rollout_device or "cpu"))
        duration = time.perf_counter() - started_at

        searched_tree = self.search.last_trees[0]
        action = self._choose_bot_action(result)
        step = self.game.step(torch.tensor([action], dtype=torch.long, device=self.game.device))
        if not bool(step.legal[0].item()):
            raise RuntimeError(f"Bot selected illegal action {action}.")

        advanced_tree = self.search.advance_tree(searched_tree, action)
        self.current_tree = None if bool(step.done[0].item()) else advanced_tree
        self.last_bot_action = action
        self.move_count += 1

        if bool(step.won[0].item()):
            self.winner = "bot"
        elif bool(step.draw[0].item()):
            self.winner = "draw"
        else:
            self.engine_player = "human"

        self.last_bot_summary = self._make_bot_summary(
            result=result,
            action=action,
            duration_seconds=duration,
            tree_before=tree_before,
            searched_tree=searched_tree,
            advanced_tree=advanced_tree,
        )
        return MoveSummary(
            actor="bot",
            action=action,
            won=bool(step.won[0].item()),
            draw=bool(step.draw[0].item()),
            tree_reused=advanced_tree is not None and not bool(step.done[0].item()),
        )

    def board_for_human(self) -> torch.Tensor:
        board = self.game.board[0]
        if self.engine_player == "bot":
            return -board
        return board

    def render_board(self) -> str:
        return render_board(self.board_for_human())

    def render_status(self) -> str:
        lines = [
            f"Turn: {self.turn_label()} | moves: {self.move_count}",
            f"Legal actions: {' '.join(str(action) for action in self.legal_actions()) or 'none'}",
        ]
        if self.last_human_action is not None:
            lines.append(f"Last human action: {self.last_human_action}")
        if self.last_bot_action is not None:
            lines.append(f"Last bot action: {self.last_bot_action}")
        if self.current_tree is not None:
            lines.append(f"Retained tree: nodes={self.current_tree.num_nodes} max_depth={self.current_tree.max_depth}")
        else:
            lines.append("Retained tree: none")
        if self.winner is not None:
            lines.append(f"Result: {self.winner}" if self.winner == "draw" else f"Result: {self.winner} won")
        return "\n".join(lines)

    def render_policy(self, top_k: int = 8) -> str:
        if self.last_bot_summary is None:
            return "No bot search has run yet."
        summary = self.last_bot_summary
        rows = _top_policy_rows(
            policy=summary.policy,
            visit_counts=summary.visit_counts,
            q_values=summary.q_values,
            top_k=top_k,
        )
        lines = [
            "Last bot policy:",
            "action  policy   visits  q",
        ]
        lines.extend(rows)
        return "\n".join(lines)

    def render_tree(self) -> str:
        if self.last_bot_summary is None:
            return "No bot search has run yet."
        summary = self.last_bot_summary
        timing = summary.timing
        return "\n".join(
            [
                "Last bot search:",
                f"action={summary.action} duration={format_seconds(summary.duration_seconds)} root_value={summary.root_value:.4f}",
                (
                    "reuse="
                    f"before:{summary.tree_reused_before_search} "
                    f"search:{summary.tree_reused_by_search} "
                    f"after:{summary.tree_reused_after_move}"
                ),
                (
                    f"tree nodes={summary.nodes} max_depth={summary.max_depth} "
                    f"leaf_evals={summary.leaf_evaluations} terminal_evals={summary.terminal_evaluations}"
                ),
                (
                    f"max_leaf_batch={summary.max_leaf_batch} "
                    f"max_expansion_batch={summary.max_expansion_batch}"
                ),
                (
                    "timing "
                    f"prepare={timing.get('prepare_trees', 0.0):.3f}s "
                    f"select={timing.get('select', 0.0):.3f}s "
                    f"expand={timing.get('expand', 0.0):.3f}s "
                    f"rollout={timing.get('rollout_eval', 0.0):.3f}s "
                    f"backprop={timing.get('backprop', 0.0):.3f}s "
                    f"build={timing.get('build_result', 0.0):.3f}s"
                ),
            ]
        )

    def turn_label(self) -> str:
        if self.is_terminal:
            return "game over"
        return self.engine_player

    def _advance_current_tree(self, action: int) -> bool:
        if self.current_tree is None:
            return False
        advanced = self.search.advance_tree(self.current_tree, action)
        self.current_tree = None if self.is_terminal else advanced
        return advanced is not None and not self.is_terminal

    def _choose_bot_action(self, result: BatchedSearchResult) -> int:
        legal_mask = self.game.legal_mask()[0].detach().cpu()
        legal_actions = legal_mask.nonzero(as_tuple=False).flatten()
        if legal_actions.numel() == 0:
            raise RuntimeError("Bot has no legal actions.")

        visits = result.visit_counts[0].detach().cpu().masked_fill(~legal_mask, -1.0)
        best_visit_count = visits[legal_actions].max()
        visit_tied_actions = legal_actions[visits[legal_actions].eq(best_visit_count)]

        q_values = result.q_values[0].detach().cpu().masked_fill(~legal_mask, -float("inf"))
        best_q_value = q_values[visit_tied_actions].max()
        q_tied_actions = visit_tied_actions[
            torch.isclose(q_values[visit_tied_actions], best_q_value, rtol=0.0, atol=1e-6)
        ]
        return _center_preferred_action(q_tied_actions)

    def _make_bot_summary(
        self,
        result: BatchedSearchResult,
        action: int,
        duration_seconds: float,
        tree_before: Optional[SearchTree],
        searched_tree: SearchTree,
        advanced_tree: Optional[SearchTree],
    ) -> BotSearchSummary:
        leaf_batches = self.search.last_leaf_batch_sizes
        expansion_batches = self.search.last_expansion_batch_sizes
        return BotSearchSummary(
            action=action,
            duration_seconds=duration_seconds,
            root_value=float(result.root_values[0].detach().cpu().item()),
            tree_reused_before_search=tree_before is not None,
            tree_reused_by_search=self.search.last_reused_roots == 1,
            tree_reused_after_move=advanced_tree is not None,
            nodes=searched_tree.num_nodes,
            max_depth=searched_tree.max_depth,
            leaf_evaluations=self.search.last_leaf_evaluations,
            terminal_evaluations=self.search.last_terminal_evaluations,
            max_leaf_batch=max(leaf_batches) if leaf_batches else 0,
            max_expansion_batch=max(expansion_batches) if expansion_batches else 0,
            timing=dict(self.search.last_timing_seconds),
            policy=result.policy[0].detach().cpu().clone(),
            visit_counts=result.visit_counts[0].detach().cpu().clone(),
            q_values=result.q_values[0].detach().cpu().clone(),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play 4x4x4 Connect Four against production deep MCTS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default="auto", help="Rollout device: auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--simulations-per-root", type=int, default=128, help="Full MCTS simulations per bot move.")
    parser.add_argument("--max-leaf-batch-size", type=int, default=16384, help="Selected leaves evaluated per batch.")
    parser.add_argument("--rollouts-per-leaf", type=int, default=128, help="Random continuations per selected leaf.")
    parser.add_argument("--max-rollouts-per-chunk", type=int, default=2097152, help="Largest rollout batch inside evaluator.")
    parser.add_argument("--exploration-constant", type=float, default=1.4, help="UCB exploration constant.")
    parser.add_argument("--virtual-loss", type=float, default=1.0, help="Virtual loss while collecting leaf batches.")
    parser.add_argument("--policy-temperature", type=float, default=1.0, help="Visit-count policy temperature.")
    parser.add_argument("--max-plies", type=int, default=64, help="Safety cap for random rollout plies.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for rollout evaluator.")
    parser.add_argument("--bot-first", action="store_true", help="Let the bot make the first move.")
    return parser


def make_session(args: argparse.Namespace) -> TerminalPlaySession:
    _validate_args(args)
    rollout_device = resolve_device(args.device)
    config = TreeMCTSConfig(
        simulations_per_root=args.simulations_per_root,
        max_leaf_batch_size=args.max_leaf_batch_size,
        rollouts_per_leaf=args.rollouts_per_leaf,
        exploration_constant=args.exploration_constant,
        virtual_loss=args.virtual_loss,
        policy_temperature=args.policy_temperature,
        rollout_device=rollout_device,
        seed=args.seed,
        max_rollout_steps=args.max_plies,
        max_rollouts_per_chunk=args.max_rollouts_per_chunk,
    )
    return TerminalPlaySession(search=BatchedTreeMCTS(config), human_starts=not args.bot_first)


def run_repl(
    session: TerminalPlaySession,
    input_fn: InputFn = input,
    output: Optional[TextIO] = None,
) -> int:
    out = sys.stdout if output is None else output
    _print_intro(session, out)

    while True:
        if session.engine_player == "bot" and not session.is_terminal:
            print("Bot thinking...", file=out, flush=True)
            move = session.bot_move()
            print(_move_message(move), file=out)
            print(session.render_tree(), file=out)
            print(session.render_board(), file=out)
            if session.is_terminal:
                print(_terminal_message(session), file=out)
                continue

        try:
            raw = input_fn("connect4> ").strip().lower()
        except EOFError:
            print("bye", file=out)
            return 0

        if raw in ("q", "quit", "exit"):
            print("bye", file=out)
            return 0
        if raw in ("h", "help", "?"):
            print(help_text(), file=out)
            continue
        if raw == "policy":
            print(session.render_policy(), file=out)
            continue
        if raw == "tree":
            print(session.render_tree(), file=out)
            continue
        if raw == "board":
            print(session.render_board(), file=out)
            continue
        if raw == "reset":
            session.reset(human_starts=True)
            print("Game reset. You move first.", file=out)
            print(session.render_board(), file=out)
            continue
        if raw == "bot":
            session.reset(human_starts=False)
            print("Game reset. Bot moves first.", file=out)
            continue
        if session.is_terminal:
            print("The game is over. Type reset, bot, or quit.", file=out)
            continue

        try:
            action = int(raw)
            move = session.human_move(action)
            print(_move_message(move), file=out)
            print(session.render_board(), file=out)
            if session.is_terminal:
                print(_terminal_message(session), file=out)
        except ValueError as exc:
            print(f"Invalid input: {exc}", file=out)


def main(
    argv: Optional[list[str]] = None,
    input_fn: InputFn = input,
    output: Optional[TextIO] = None,
) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = build_parser().parse_args(argv)
    session = make_session(args)
    return run_repl(session=session, input_fn=input_fn, output=output)


def render_board(board: torch.Tensor) -> str:
    """Render a human-perspective board tensor as z-layers."""
    if tuple(board.shape) != (BOARD_SIZE, BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"board must have shape ({BOARD_SIZE}, {BOARD_SIZE}, {BOARD_SIZE})")

    lines = [
        "Action map:",
        render_action_map(),
        "",
        "Board: H=human, B=bot, .=empty",
    ]
    cpu_board = board.detach().cpu()
    for z in range(BOARD_SIZE):
        lines.append(f"z={z}")
        for y in reversed(range(BOARD_SIZE)):
            cells = []
            for x in range(BOARD_SIZE):
                value = int(cpu_board[x, y, z].item())
                cells.append(_piece_symbol(value))
            lines.append(" ".join(cells))
        lines.append("")
    return "\n".join(lines).rstrip()


def render_action_map() -> str:
    lines = []
    for y in reversed(range(BOARD_SIZE)):
        lines.append(" ".join(f"{x * BOARD_SIZE + y:2d}" for x in range(BOARD_SIZE)))
    return "\n".join(lines)


def help_text() -> str:
    return "\n".join(
        [
            "Commands:",
            "  0-15    play a column action",
            "  board   print the board",
            "  policy  print the last bot policy table",
            "  tree    print last bot tree/search diagnostics",
            "  reset   reset with human first",
            "  bot     reset with bot first",
            "  help    show this help",
            "  quit    exit",
        ]
    )


def _root_device_for(rollout_device: object) -> torch.device:
    device = torch.device("cpu" if rollout_device is None else rollout_device)
    if device.type in ("cuda", "mps"):
        return torch.device("cpu")
    return device


def _piece_symbol(value: int) -> str:
    if value == CURRENT_PLAYER:
        return "H"
    if value == OPPONENT_PLAYER:
        return "B"
    return "."


def _center_preferred_action(actions: torch.Tensor) -> int:
    """Break exact action ties toward the middle of the 4x4 base."""
    if actions.numel() == 0:
        raise RuntimeError("cannot choose from no actions")
    best_action = int(actions[0].item())
    best_distance = float("inf")
    center = (BOARD_SIZE - 1) / 2.0
    for action_tensor in actions:
        action = int(action_tensor.item())
        x = action // BOARD_SIZE
        y = action % BOARD_SIZE
        distance = (x - center) ** 2 + (y - center) ** 2
        if distance < best_distance or (distance == best_distance and action < best_action):
            best_action = action
            best_distance = distance
    return best_action


def _top_policy_rows(
    policy: torch.Tensor,
    visit_counts: torch.Tensor,
    q_values: torch.Tensor,
    top_k: int,
) -> list[str]:
    rows = []
    order = torch.argsort(policy, descending=True)
    shown = 0
    for action_tensor in order:
        action = int(action_tensor.item())
        probability = float(policy[action].item())
        visits = int(visit_counts[action].item())
        q_value = float(q_values[action].item())
        if probability <= 0.0 and visits <= 0:
            continue
        rows.append(f"{action:>6d}  {probability:>6.3f}  {visits:>6d}  {q_value:> .3f}")
        shown += 1
        if shown >= top_k:
            break
    return rows or ["  none"]


def _move_message(move: MoveSummary) -> str:
    message = f"{move.actor} played {move.action}"
    if move.won:
        message += " and won."
    elif move.draw:
        message += ". Draw."
    else:
        message += f". tree_reused={move.tree_reused}"
    return message


def _terminal_message(session: TerminalPlaySession) -> str:
    if session.winner == "draw":
        return "Game over: draw."
    return f"Game over: {session.winner} won."


def _print_intro(session: TerminalPlaySession, output: TextIO) -> None:
    print("4x4x4 Connect Four vs deep MCTS", file=output)
    print(
        "Config: "
        f"simulations_per_root={session.search.config.simulations_per_root}, "
        f"rollouts_per_leaf={session.search.config.rollouts_per_leaf}, "
        f"max_leaf_batch_size={session.search.config.max_leaf_batch_size}, "
        f"rollout_device={session.search.config.rollout_device}",
        file=output,
    )
    print(help_text(), file=output)
    print(session.render_board(), file=output)
    print(session.render_status(), file=output)


def _validate_args(args: argparse.Namespace) -> None:
    if args.simulations_per_root <= 0:
        raise ValueError("--simulations-per-root must be positive")
    if args.max_leaf_batch_size <= 0:
        raise ValueError("--max-leaf-batch-size must be positive")
    if args.rollouts_per_leaf <= 0:
        raise ValueError("--rollouts-per-leaf must be positive")
    if args.max_rollouts_per_chunk <= 0:
        raise ValueError("--max-rollouts-per-chunk must be positive")
    if args.max_plies <= 0:
        raise ValueError("--max-plies must be positive")


if __name__ == "__main__":
    raise SystemExit(main())
