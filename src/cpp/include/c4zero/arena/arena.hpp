#pragma once

#include <cstdint>
#include <string>

namespace c4zero::arena {

constexpr double kDefaultPromotionThreshold = 0.55;

struct ArenaConfig {
  std::string model_a;
  std::string model_b;
  std::string bot_a;
  std::string bot_b;
  std::string device = "cpu";
  int games = 2;
  int simulations = 800;
  int search_threads = 1;
  int arena_workers = 4;
  int opening_count = 0;
  int opening_plies = 4;
  int games_per_opening = 4;
  bool add_root_noise = false;
  double root_dirichlet_alpha = 0.625;
  double root_exploration_fraction = 0.25;
  double promotion_threshold = kDefaultPromotionThreshold;
  std::uint64_t seed = 1;
};

struct ArenaResult {
  int games = 0;
  int model_a_wins = 0;
  int model_b_wins = 0;
  int draws = 0;
  int total_plies = 0;
  bool root_noise = false;
  int arena_workers = 1;
  int opening_count = 0;
  int opening_plies = 0;
  int games_per_opening = 1;
  double promotion_threshold = kDefaultPromotionThreshold;
  std::string player_a;
  std::string player_b;

  [[nodiscard]] double model_a_score_rate() const;
  [[nodiscard]] bool model_a_promoted() const;
  [[nodiscard]] std::string summary() const;
};

[[nodiscard]] ArenaResult play_checkpoint_match(const ArenaConfig& config);

}  // namespace c4zero::arena
