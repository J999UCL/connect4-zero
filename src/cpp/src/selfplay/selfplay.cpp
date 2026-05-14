#include "c4zero/selfplay/selfplay.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace c4zero::selfplay {
namespace {

struct PendingSample {
  core::Position position;
  std::array<float, core::kNumActions> policy{};
  std::array<std::uint32_t, core::kNumActions> visits{};
  core::Action action = -1;
};

}  // namespace

GeneratedGame generate_game(
    search::Evaluator& evaluator,
    const SelfPlayConfig& config,
    std::uint64_t game_id) {
  search::PuctConfig mcts_config = config.mcts;
  mcts_config.seed = config.seed ^ (0x9E3779B97F4A7C15ULL + (game_id << 6) + (game_id >> 2));
  search::PuctMcts mcts(mcts_config);
  core::Position position = core::Position::empty();
  search::SearchTree tree(position);
  std::vector<PendingSample> pending;
  GeneratedGame game;

  while (!position.is_terminal()) {
    const double temperature = position.ply < config.temperature_sampling_plies ? 1.0 : 0.0;
    search::SearchResult result = mcts.search(tree, evaluator, config.add_root_noise, temperature);
    if (result.selected_action < 0 || !position.is_legal(result.selected_action)) {
      throw std::runtime_error("self-play selected illegal action");
    }
    std::uint32_t visits = 0;
    for (auto count : result.visit_counts) {
      visits += count;
    }
    if (visits != static_cast<std::uint32_t>(config.mcts.simulations_per_move)) {
      throw std::runtime_error("self-play MCTS visit counts do not match simulations");
    }
    pending.push_back(PendingSample{position, result.policy, result.visit_counts, result.selected_action});
    game.completed_simulations += result.completed_simulations;
    game.leaf_evaluations += result.leaf_evaluations;
    game.terminal_evaluations += result.terminal_evaluations;
    game.max_depth = std::max(game.max_depth, result.max_depth);
    game.search_time_ms += result.search_time_ms;
    position = position.play(result.selected_action);
    if (!tree.advance(result.selected_action)) {
      tree = search::SearchTree(position);
    }
  }

  const auto terminal = position.terminal_value();
  if (!terminal.has_value()) {
    throw std::runtime_error("self-play ended without terminal value");
  }

  game.terminal_value = *terminal;
  game.plies = position.ply;
  game.samples.reserve(pending.size());
  for (const PendingSample& sample : pending) {
    const int flips = static_cast<int>(position.ply) - static_cast<int>(sample.position.ply);
    const float value = *terminal * ((flips % 2 == 0) ? 1.0f : -1.0f);
    game.samples.push_back(data::SelfPlaySample::from_position(
        sample.position,
        sample.policy,
        sample.visits,
        value,
        sample.action,
        game_id));
  }
  return game;
}

}  // namespace c4zero::selfplay
