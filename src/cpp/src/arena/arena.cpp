#include "c4zero/arena/arena.hpp"

#include "c4zero/bots/heuristic.hpp"
#include "c4zero/core/position.hpp"
#include "c4zero/core/symmetry.hpp"
#include "c4zero/model/torchscript.hpp"
#include "c4zero/search/puct.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <torch/torch.h>
#include <vector>

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

struct ArenaGameSpec {
  core::Position opening;
  bool a_controls_initial = true;
  int game_index = 0;
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

core::Position random_opening(int opening_plies, std::mt19937_64& rng) {
  if (opening_plies <= 0) {
    return core::Position::empty();
  }
  for (int attempt = 0; attempt < 1024; ++attempt) {
    core::Position position = core::Position::empty();
    for (int ply = 0; ply < opening_plies && !position.is_terminal(); ++ply) {
      const auto legal = core::legal_actions(position);
      if (legal.empty()) {
        break;
      }
      std::uniform_int_distribution<std::size_t> dist(0, legal.size() - 1);
      position = position.play(legal.at(dist(rng)));
    }
    if (!position.is_terminal()) {
      return position;
    }
  }
  throw std::runtime_error("failed to generate a non-terminal random arena opening");
}

std::vector<ArenaGameSpec> make_game_specs(const ArenaConfig& config) {
  std::vector<ArenaGameSpec> specs;
  specs.reserve(static_cast<std::size_t>(std::max(0, config.games)));
  if (config.games == 0) {
    return specs;
  }

  const int games_per_opening = std::max(1, config.games_per_opening);
  const int opening_count = config.opening_count > 0
      ? config.opening_count
      : static_cast<int>(std::ceil(static_cast<double>(config.games) / static_cast<double>(games_per_opening)));

  std::mt19937_64 rng(config.seed ^ 0xA9D3B1C47F21D04BULL);
  for (int opening_index = 0; opening_index < opening_count && static_cast<int>(specs.size()) < config.games;
       ++opening_index) {
    const core::Position base = random_opening(config.opening_plies, rng);
    const auto symmetry = static_cast<core::Symmetry>(1 + (opening_index % 7));
    for (int repeat = 0; repeat < games_per_opening && static_cast<int>(specs.size()) < config.games; ++repeat) {
      ArenaGameSpec spec;
      spec.opening = repeat < 2 ? base : core::transform(base, symmetry);
      spec.a_controls_initial = (repeat % 2 == 0);
      spec.game_index = static_cast<int>(specs.size());
      specs.push_back(spec);
    }
  }
  return specs;
}

void add_result(ArenaResult& total, const ArenaResult& delta) {
  total.games += delta.games;
  total.model_a_wins += delta.model_a_wins;
  total.model_b_wins += delta.model_b_wins;
  total.draws += delta.draws;
  total.total_plies += delta.total_plies;
}

ArenaResult play_one_game(
    ArenaSide& side_a,
    ArenaSide& side_b,
    bool add_root_noise,
    const ArenaGameSpec& spec,
    int worker_id) {
  (void)worker_id;
  core::Position position = spec.opening;
  reset_tree(side_a, position);
  reset_tree(side_b, position);
  std::cerr << "arena_game_start"
            << " worker=" << worker_id
            << " game=" << spec.game_index
            << " opening_ply=" << static_cast<int>(position.ply)
            << " a_controls_initial=" << (spec.a_controls_initial ? 1 : 0)
            << "\n";

  while (!position.is_terminal()) {
    const bool initial_player_to_move = (position.ply % 2 == 0);
    const bool a_to_move = spec.a_controls_initial == initial_player_to_move;
    ArenaSide& active_side = a_to_move ? side_a : side_b;
    const core::Action action = select_move(active_side, position, add_root_noise);
    position = position.play(action);
    advance_tree(side_a, action, position);
    advance_tree(side_b, action, position);
  }

  ArenaResult result;
  result.games = 1;
  result.total_plies = position.ply;
  const float terminal = *position.terminal_value();
  if (terminal == 0.0f) {
    result.draws = 1;
    std::cerr << "arena_game_done"
              << " worker=" << worker_id
              << " game=" << spec.game_index
              << " result=draw"
              << " plies=" << result.total_plies
              << "\n";
    return result;
  }
  const bool initial_player_won = (position.ply % 2 == 1);
  const bool model_a_won = spec.a_controls_initial == initial_player_won;
  if (model_a_won) {
    result.model_a_wins = 1;
  } else {
    result.model_b_wins = 1;
  }
  std::cerr << "arena_game_done"
            << " worker=" << worker_id
            << " game=" << spec.game_index
            << " result=" << (model_a_won ? "model_a" : "model_b")
            << " plies=" << result.total_plies
            << "\n";
  return result;
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
      << " arena_workers=" << arena_workers
      << " opening_count=" << opening_count
      << " opening_plies=" << opening_plies
      << " games_per_opening=" << games_per_opening
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
  if (config.arena_workers <= 0) {
    throw std::invalid_argument("arena workers must be positive");
  }
  if (config.opening_count < 0) {
    throw std::invalid_argument("arena opening count must be non-negative");
  }
  if (config.opening_plies < 0) {
    throw std::invalid_argument("arena opening plies must be non-negative");
  }
  if (config.games_per_opening <= 0) {
    throw std::invalid_argument("arena games per opening must be positive");
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
  const auto game_specs = make_game_specs(config);
  const int effective_opening_count = config.opening_count > 0
      ? config.opening_count
      : (config.games == 0 ? 0 : static_cast<int>(
            std::ceil(static_cast<double>(config.games) / static_cast<double>(config.games_per_opening))));

  ArenaResult result;
  result.root_noise = config.add_root_noise;
  result.arena_workers = std::min(config.arena_workers, std::max(1, config.games));
  result.opening_count = effective_opening_count;
  result.opening_plies = config.opening_plies;
  result.games_per_opening = config.games_per_opening;
  result.promotion_threshold = config.promotion_threshold;
  result.player_a = describe_side(spec_a);
  result.player_b = describe_side(spec_b);

  const int worker_count = std::min(config.arena_workers, std::max(1, static_cast<int>(game_specs.size())));
  std::vector<ArenaResult> worker_results(static_cast<std::size_t>(worker_count));
  std::vector<std::thread> workers;
  std::exception_ptr error;
  std::mutex error_mutex;
  workers.reserve(static_cast<std::size_t>(worker_count));
  for (int worker_id = 0; worker_id < worker_count; ++worker_id) {
    workers.emplace_back([&, worker_id]() {
      try {
        ArenaSide side_a = make_side(spec_a, config, device, config.seed ^ (0xD1B54A32D192ED03ULL + worker_id));
        ArenaSide side_b = make_side(spec_b, config, device, config.seed ^ (0x9E3779B97F4A7C15ULL + worker_id));
        for (int game_index = worker_id; game_index < static_cast<int>(game_specs.size()); game_index += worker_count) {
          const auto delta = play_one_game(
              side_a,
              side_b,
              config.add_root_noise,
              game_specs.at(game_index),
              worker_id);
          add_result(worker_results.at(static_cast<std::size_t>(worker_id)), delta);
        }
      } catch (...) {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (!error) {
          error = std::current_exception();
        }
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  if (error) {
    std::rethrow_exception(error);
  }
  for (const ArenaResult& worker_result : worker_results) {
    add_result(result, worker_result);
  }
  if (result.games != config.games) {
    throw std::runtime_error("arena completed an unexpected number of games");
  }
  return result;
}

}  // namespace c4zero::arena
