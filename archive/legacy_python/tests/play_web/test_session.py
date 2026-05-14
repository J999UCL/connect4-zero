import pytest

from play_web.server import GameSession


def fixed_bot(action: int):
    def choose_action(_state):
        return action

    return choose_action


def test_reset_payload_starts_human_turn() -> None:
    session = GameSession(bot_action_provider=fixed_bot(1))

    payload = session.reset()

    assert payload["turn"] == "human"
    assert payload["winner"] is None
    assert payload["moveCount"] == 0
    assert sum(payload["legalActions"]) == 16
    assert payload["board"][0][0][0] == 0


def test_human_move_triggers_bot_reply_and_restores_human_perspective() -> None:
    session = GameSession(bot_action_provider=fixed_bot(1))

    payload = session.human_move(0)

    assert payload["turn"] == "human"
    assert payload["winner"] is None
    assert payload["lastHumanAction"] == 0
    assert payload["lastBotAction"] == 1
    assert payload["moveCount"] == 2
    assert payload["board"][0][0][0] == 1
    assert payload["board"][0][1][0] == -1


def test_illegal_human_move_does_not_advance_game() -> None:
    session = GameSession(bot_action_provider=fixed_bot(1))

    with pytest.raises(ValueError, match="not legal"):
        session.human_move(99)

    payload = session.state_payload()
    assert payload["turn"] == "human"
    assert payload["moveCount"] == 0
    assert payload["lastHumanAction"] is None
    assert payload["lastBotAction"] is None


def test_bot_win_displays_board_from_human_perspective() -> None:
    session = GameSession(bot_action_provider=fixed_bot(0))
    session.engine_player = "bot"
    session.game.board[0, 0, 0, 0:3] = 1
    session.game.heights[0, 0, 0] = 3

    payload = session._bot_move()

    assert payload["turn"] == "game_over"
    assert payload["winner"] == "bot"
    assert payload["board"][0][0] == [-1, -1, -1, -1]
