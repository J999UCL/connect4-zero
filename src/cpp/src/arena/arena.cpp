#include "c4zero/arena/arena.hpp"

#include <sstream>
#include <stdexcept>

namespace c4zero::arena {

double ArenaResult::first_score_rate() const {
  if (games == 0) {
    return 0.0;
  }
  return (static_cast<double>(first_wins) + 0.5 * static_cast<double>(draws)) / static_cast<double>(games);
}

std::string ArenaResult::summary() const {
  std::ostringstream out;
  out << "games=" << games
      << " first_wins=" << first_wins
      << " second_wins=" << second_wins
      << " draws=" << draws
      << " score_rate=" << first_score_rate()
      << " avg_plies=" << (games == 0 ? 0.0 : static_cast<double>(total_plies) / games);
  return out.str();
}

ArenaResult play_bot_match(
    const bots::Bot& first,
    const bots::Bot& second,
    int games,
    bool alternate_starts) {
  ArenaResult result;
  result.games = games;
  for (int game = 0; game < games; ++game) {
    core::Position position = core::Position::empty();
    const bool first_controls_initial = !alternate_starts || game % 2 == 0;
    while (!position.is_terminal()) {
      const bool initial_player_to_move = (position.ply % 2 == 0);
      const bool first_to_move = first_controls_initial == initial_player_to_move;
      const bots::Bot& bot = first_to_move ? first : second;
      const core::Action action = bot.select_move(position);
      if (!position.is_legal(action)) {
        throw std::runtime_error(bot.name() + " selected illegal action");
      }
      position = position.play(action);
    }
    result.total_plies += position.ply;
    const float terminal = *position.terminal_value();
    if (terminal == 0.0f) {
      result.draws += 1;
      continue;
    }
    const bool initial_player_won = (position.ply % 2 == 1);
    const bool first_won = first_controls_initial == initial_player_won;
    if (first_won) {
      result.first_wins += 1;
    } else {
      result.second_wins += 1;
    }
  }
  return result;
}

}  // namespace c4zero::arena
