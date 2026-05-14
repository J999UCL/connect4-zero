#pragma once

#include "c4zero/data/shard.hpp"
#include "c4zero/search/puct.hpp"

#include <cstdint>
#include <vector>

namespace c4zero::selfplay {

struct SelfPlayConfig {
  search::PuctConfig mcts;
  int games = 1;
  int temperature_sampling_plies = 30;
  bool add_root_noise = true;
  std::uint64_t seed = 1;
};

struct GeneratedGame {
  std::vector<data::SelfPlaySample> samples;
  float terminal_value = 0.0f;
  int plies = 0;
  int completed_simulations = 0;
  int leaf_evaluations = 0;
  int terminal_evaluations = 0;
  int max_depth = 0;
  double search_time_ms = 0.0;
};

[[nodiscard]] GeneratedGame generate_game(
    search::Evaluator& evaluator,
    const SelfPlayConfig& config,
    std::uint64_t game_id);

}  // namespace c4zero::selfplay
