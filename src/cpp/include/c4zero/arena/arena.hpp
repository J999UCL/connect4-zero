#pragma once

#include <cstdint>
#include <string>

namespace c4zero::arena {

struct ArenaConfig {
  std::string model_a;
  std::string model_b;
  std::string device = "cpu";
  int games = 2;
  int simulations = 800;
  int search_threads = 1;
  std::uint64_t seed = 1;
};

struct ArenaResult {
  int games = 0;
  int model_a_wins = 0;
  int model_b_wins = 0;
  int draws = 0;
  int total_plies = 0;

  [[nodiscard]] double model_a_score_rate() const;
  [[nodiscard]] std::string summary() const;
};

[[nodiscard]] ArenaResult play_checkpoint_match(const ArenaConfig& config);

}  // namespace c4zero::arena
