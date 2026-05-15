#include "c4zero/arena/arena.hpp"

#include "c4zero/core/position.hpp"
#include "c4zero/model/torchscript.hpp"
#include "c4zero/search/puct.hpp"

#include <sstream>
#include <stdexcept>
#include <torch/torch.h>

namespace c4zero::arena {
namespace {

torch::Device parse_device(const std::string& device) {
  if (device == "cuda") {
    return torch::Device(torch::kCUDA);
  }
  if (device == "cpu") {
    return torch::Device(torch::kCPU);
  }
  throw std::invalid_argument("unsupported arena device: " + device);
}

}  // namespace

double ArenaResult::model_a_score_rate() const {
  if (games == 0) {
    return 0.0;
  }
  return (static_cast<double>(model_a_wins) + 0.5 * static_cast<double>(draws)) / static_cast<double>(games);
}

std::string ArenaResult::summary() const {
  std::ostringstream out;
  out << "games=" << games
      << " model_a_wins=" << model_a_wins
      << " model_b_wins=" << model_b_wins
      << " draws=" << draws
      << " model_a_score_rate=" << model_a_score_rate()
      << " avg_plies=" << (games == 0 ? 0.0 : static_cast<double>(total_plies) / games)
      << " root_noise=" << (root_noise ? 1 : 0);
  return out.str();
}

ArenaResult play_checkpoint_match(const ArenaConfig& config) {
  if (config.model_a.empty() || config.model_b.empty()) {
    throw std::invalid_argument("arena requires --model-a and --model-b");
  }
  if (config.games < 0) {
    throw std::invalid_argument("arena games must be non-negative");
  }
  if (config.simulations <= 0) {
    throw std::invalid_argument("arena simulations must be positive");
  }
  if (config.search_threads <= 0) {
    throw std::invalid_argument("arena search threads must be positive");
  }
  if (config.root_dirichlet_alpha <= 0.0) {
    throw std::invalid_argument("arena root Dirichlet alpha must be positive");
  }
  if (config.root_exploration_fraction < 0.0 || config.root_exploration_fraction > 1.0) {
    throw std::invalid_argument("arena root exploration fraction must be in [0, 1]");
  }

  const torch::Device device = parse_device(config.device);
  model::TorchScriptEvaluator evaluator_a(config.model_a, device);
  model::TorchScriptEvaluator evaluator_b(config.model_b, device);

  search::PuctConfig mcts_config;
  mcts_config.simulations_per_move = config.simulations;
  mcts_config.search_threads = config.search_threads;
  mcts_config.root_dirichlet_alpha = config.root_dirichlet_alpha;
  mcts_config.root_exploration_fraction = config.root_exploration_fraction;
  mcts_config.seed = config.seed;
  search::PuctMcts mcts_a(mcts_config);
  mcts_config.seed = config.seed ^ 0x9E3779B97F4A7C15ULL;
  search::PuctMcts mcts_b(mcts_config);

  ArenaResult result;
  result.games = config.games;
  result.root_noise = config.add_root_noise;
  for (int game = 0; game < config.games; ++game) {
    core::Position position = core::Position::empty();
    search::SearchTree tree_a(position);
    search::SearchTree tree_b(position);
    const bool a_controls_initial = (game % 2 == 0);

    while (!position.is_terminal()) {
      const bool initial_player_to_move = (position.ply % 2 == 0);
      const bool a_to_move = a_controls_initial == initial_player_to_move;
      search::PuctMcts& mcts = a_to_move ? mcts_a : mcts_b;
      search::Evaluator& evaluator = a_to_move
          ? static_cast<search::Evaluator&>(evaluator_a)
          : static_cast<search::Evaluator&>(evaluator_b);
      search::SearchTree& active_tree = a_to_move ? tree_a : tree_b;

      search::SearchResult search = mcts.search(active_tree, evaluator, config.add_root_noise, 0.0);
      if (search.selected_action < 0 || !position.is_legal(search.selected_action)) {
        throw std::runtime_error("arena checkpoint selected illegal action");
      }
      position = position.play(search.selected_action);
      if (!tree_a.advance(search.selected_action)) {
        tree_a = search::SearchTree(position);
      }
      if (!tree_b.advance(search.selected_action)) {
        tree_b = search::SearchTree(position);
      }
    }

    result.total_plies += position.ply;
    const float terminal = *position.terminal_value();
    if (terminal == 0.0f) {
      result.draws += 1;
      continue;
    }
    const bool initial_player_won = (position.ply % 2 == 1);
    const bool model_a_won = a_controls_initial == initial_player_won;
    if (model_a_won) {
      result.model_a_wins += 1;
    } else {
      result.model_b_wins += 1;
    }
  }
  return result;
}

}  // namespace c4zero::arena
