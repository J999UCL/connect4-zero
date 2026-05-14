#include "c4zero/selfplay/selfplay.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <string>

int main() {
  c4zero::search::UniformEvaluator evaluator;
  c4zero::selfplay::SelfPlayConfig config;
  config.mcts.simulations_per_move = 16;
  config.mcts.search_threads = 4;
  config.seed = 101;

  auto game = c4zero::selfplay::generate_game(evaluator, config, 7);
  C4ZERO_CHECK(!game.samples.empty());
  C4ZERO_CHECK(game.plies > 0);
  C4ZERO_CHECK(game.plies <= c4zero::core::kMaxPlies);
  C4ZERO_CHECK_EQ(game.completed_simulations, game.plies * config.mcts.simulations_per_move);
  C4ZERO_CHECK(game.max_depth >= 1);
  C4ZERO_CHECK(game.search_time_ms >= 0.0);

  for (const auto& sample : game.samples) {
    C4ZERO_CHECK(sample.action < c4zero::core::kNumActions);
    C4ZERO_CHECK((sample.legal_mask & (1u << sample.action)) != 0);
    C4ZERO_CHECK(sample.value >= -1.0f && sample.value <= 1.0f);
    float policy_sum = 0.0f;
    std::uint32_t visit_sum = 0;
    for (int action = 0; action < c4zero::core::kNumActions; ++action) {
      if ((sample.legal_mask & (1u << action)) == 0) {
        C4ZERO_CHECK_EQ(sample.policy[action], 0.0f);
        C4ZERO_CHECK_EQ(sample.visit_counts[action], 0);
      }
      policy_sum += sample.policy[action];
      visit_sum += sample.visit_counts[action];
    }
    C4ZERO_CHECK(policy_sum > 0.999f && policy_sum < 1.001f);
    C4ZERO_CHECK_EQ(visit_sum, static_cast<std::uint32_t>(config.mcts.simulations_per_move));
  }

  std::set<std::string> openings;
  for (std::uint64_t game_id = 0; game_id < 12; ++game_id) {
    auto noisy = c4zero::selfplay::generate_game(evaluator, config, game_id);
    std::string opening;
    const int limit = std::min<int>(4, noisy.samples.size());
    for (int index = 0; index < limit; ++index) {
      opening += std::to_string(noisy.samples[index].action);
      opening += ",";
    }
    openings.insert(opening);
  }
  C4ZERO_CHECK(openings.size() > 1);
  return 0;
}
