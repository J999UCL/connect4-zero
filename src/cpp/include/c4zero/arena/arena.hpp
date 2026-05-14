#pragma once

#include "c4zero/bots/heuristic.hpp"
#include "c4zero/core/position.hpp"

#include <string>

namespace c4zero::arena {

struct ArenaResult {
  int games = 0;
  int first_wins = 0;
  int second_wins = 0;
  int draws = 0;
  int total_plies = 0;

  [[nodiscard]] double first_score_rate() const;
  [[nodiscard]] std::string summary() const;
};

[[nodiscard]] ArenaResult play_bot_match(
    const bots::Bot& first,
    const bots::Bot& second,
    int games,
    bool alternate_starts);

}  // namespace c4zero::arena
