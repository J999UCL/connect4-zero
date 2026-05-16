#include "c4zero/arena/arena.hpp"

#include "c4zero/bots/heuristic.hpp"
#include "c4zero/core/position.hpp"
#include "c4zero/model/torchscript.hpp"
#include "c4zero/search/puct.hpp"

#include <memory>
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

enum class SideKind {
  Model,
  Bot,
};

struct SideSpec {
  SideKind kind = SideKind::Model;
  std::string name;
};

SideSpec parse_side(const std::string& model, const std::string& bot, const std::string& label) {
  const bool has_model = !model.empty();
  const bool has_bot = !bot.empty();
  if (has_model == has_bot) {
    throw std::invalid_argument(
        "arena side " + label + " requires exactly one of --model-" + label + " or --bot-" + label);
  }
  if (has_model) {
    return SideSpec{SideKind::Model, model};
  }
  return SideSpec{SideKind::Bot, bot};
}

std::string describe_side(const SideSpec& side) {
  if (side.kind == SideKind::Model) {
    return "model:" + side.name;
  }
  return "bot:" + side.name;
}

struct ArenaSide {
  SideSpec spec;
  std::unique_ptr<model::TorchScriptEvaluator> evaluator;
  std::unique_ptr<search::PuctMcts> mcts;
  std::unique_ptr<search::SearchTree> tree;
  std::unique_ptr<bots::Bot> bot;

  [[nodiscard]] bool is_model() const {
    return spec.kind == SideKind::Model;
  }
};

search::PuctConfig make_mcts_config(const ArenaConfig& config, std::uint64_t seed) {
  search::PuctConfig mcts_config;
  mcts_config.simulations_per_move = config.simulations;
  mcts_config.search_threads = config.search_threads;
  mcts_config.root_dirichlet_alpha = config.root_dirichlet_alpha;
  mcts_config.root_exploration_fraction = config.root_exploration_fraction;
  mcts_config.seed = seed;
  return mcts_config;
}

ArenaSide make_side(const SideSpec& spec, const ArenaConfig& config, const torch::Device& device, std::uint64_t seed) {
  ArenaSide side;
  side.spec = spec;
  if (spec.kind == SideKind::Model) {
    side.evaluator = std::make_unique<model::TorchScriptEvaluator>(spec.name, device);
    side.mcts = std::make_unique<search::PuctMcts>(make_mcts_config(config, seed));
  } else {
    side.bot = bots::make_bot(spec.name);
  }
  return side;
}

void reset_tree(ArenaSide& side, const core::Position& position) {
  if (side.is_model()) {
    side.tree = std::make_unique<search::SearchTree>(position);
  }
}

core::Action select_move(ArenaSide& side, const core::Position& position, bool add_root_noise) {
  if (!side.is_model()) {
    const core::Action action = side.bot->select_move(position);
    if (!position.is_legal(action)) {
      throw std::runtime_error(side.bot->name() + " selected illegal action");
    }
    return action;
  }
  if (!side.tree) {
    side.tree = std::make_unique<search::SearchTree>(position);
  }
  search::SearchResult search = side.mcts->search(*side.tree, *side.evaluator, add_root_noise, 0.0);
  if (search.selected_action < 0 || !position.is_legal(search.selected_action)) {
    throw std::runtime_error("arena model selected illegal action");
  }
  return search.selected_action;
}

void advance_tree(ArenaSide& side, core::Action action, const core::Position& position) {
  if (!side.is_model()) {
    return;
  }
  if (!side.tree || !side.tree->advance(action)) {
    side.tree = std::make_unique<search::SearchTree>(position);
  }
}

}  // namespace

double ArenaResult::model_a_score_rate() const {
  if (games == 0) {
    return 0.0;
  }
  return (static_cast<double>(model_a_wins) + 0.5 * static_cast<double>(draws)) / static_cast<double>(games);
}

bool ArenaResult::model_a_promoted() const {
  return games > 0 && model_a_score_rate() >= promotion_threshold;
}

std::string ArenaResult::summary() const {
  std::ostringstream out;
  out << "games=" << games
      << " model_a_wins=" << model_a_wins
      << " model_b_wins=" << model_b_wins
      << " draws=" << draws
      << " model_a_score_rate=" << model_a_score_rate()
      << " avg_plies=" << (games == 0 ? 0.0 : static_cast<double>(total_plies) / games)
      << " root_noise=" << (root_noise ? 1 : 0)
      << " promotion_threshold=" << promotion_threshold
      << " promote_model_a=" << (model_a_promoted() ? 1 : 0);
  if (!player_a.empty()) {
    out << " player_a=" << player_a;
  }
  if (!player_b.empty()) {
    out << " player_b=" << player_b;
  }
  return out.str();
}

ArenaResult play_checkpoint_match(const ArenaConfig& config) {
  const SideSpec spec_a = parse_side(config.model_a, config.bot_a, "a");
  const SideSpec spec_b = parse_side(config.model_b, config.bot_b, "b");
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
  if (config.promotion_threshold < 0.0 || config.promotion_threshold > 1.0) {
    throw std::invalid_argument("arena promotion threshold must be in [0, 1]");
  }

  const torch::Device device = parse_device(config.device);
  ArenaSide side_a = make_side(spec_a, config, device, config.seed);
  ArenaSide side_b = make_side(spec_b, config, device, config.seed ^ 0x9E3779B97F4A7C15ULL);

  ArenaResult result;
  result.games = config.games;
  result.root_noise = config.add_root_noise;
  result.promotion_threshold = config.promotion_threshold;
  result.player_a = describe_side(spec_a);
  result.player_b = describe_side(spec_b);
  for (int game = 0; game < config.games; ++game) {
    core::Position position = core::Position::empty();
    reset_tree(side_a, position);
    reset_tree(side_b, position);
    const bool a_controls_initial = (game % 2 == 0);

    while (!position.is_terminal()) {
      const bool initial_player_to_move = (position.ply % 2 == 0);
      const bool a_to_move = a_controls_initial == initial_player_to_move;
      ArenaSide& active_side = a_to_move ? side_a : side_b;
      const core::Action action = select_move(active_side, position, config.add_root_noise);
      position = position.play(action);
      advance_tree(side_a, action, position);
      advance_tree(side_b, action, position);
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
