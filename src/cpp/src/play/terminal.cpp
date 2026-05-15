#include "c4zero/play/terminal.hpp"

#include "c4zero/core/position.hpp"
#include "c4zero/model/torchscript.hpp"
#include "c4zero/search/puct.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
#include <vector>

namespace c4zero::play {
namespace {

torch::Device parse_device(const std::string& device) {
  if (device == "cuda") {
    return torch::Device(torch::kCUDA);
  }
  if (device == "cpu") {
    return torch::Device(torch::kCPU);
  }
  throw std::invalid_argument("unsupported play device: " + device);
}

std::string trim(std::string value) {
  while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())) != 0) {
    value.erase(value.begin());
  }
  while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())) != 0) {
    value.pop_back();
  }
  return value;
}

bool human_to_move(const core::Position& position, bool bot_first) {
  const bool human_is_first = !bot_first;
  const bool initial_player_to_move = (position.ply % 2 == 0);
  return human_is_first == initial_player_to_move;
}

bool bit_is_set(core::Bitboard bits, int x, int y, int z) {
  return (bits & core::cell_mask(x, y, z)) != 0;
}

char marker_at(const core::Position& position, bool bot_first, int x, int y, int z) {
  const bool human_turn = human_to_move(position, bot_first);
  const core::Bitboard human_bits = human_turn ? position.current : position.opponent;
  const core::Bitboard bot_bits = human_turn ? position.opponent : position.current;
  if (bit_is_set(human_bits, x, y, z)) {
    return 'X';
  }
  if (bit_is_set(bot_bits, x, y, z)) {
    return 'O';
  }
  return '.';
}

void print_action_grid(std::ostream& output) {
  output << "Actions (top view):\n";
  for (int y = core::kBoardSize - 1; y >= 0; --y) {
    output << "  ";
    for (int x = 0; x < core::kBoardSize; ++x) {
      output << std::setw(2) << core::xy_to_action(x, y) << ' ';
    }
    output << "\n";
  }
}

void print_board(std::ostream& output, const core::Position& position, bool bot_first) {
  output << "\nBoard: X=you O=bot .=empty  ply=" << static_cast<int>(position.ply) << "\n";
  for (int z = core::kBoardSize - 1; z >= 0; --z) {
    output << "z=" << z << (z == 0 ? " bottom" : (z == core::kBoardSize - 1 ? " top" : "")) << "\n";
    for (int y = core::kBoardSize - 1; y >= 0; --y) {
      output << "  ";
      for (int x = 0; x < core::kBoardSize; ++x) {
        output << marker_at(position, bot_first, x, y, z) << ' ';
      }
      output << "\n";
    }
  }
  output << "Heights: ";
  for (core::Action action = 0; action < core::kNumActions; ++action) {
    output << action << ':' << static_cast<int>(position.heights[action]);
    if (action + 1 < core::kNumActions) {
      output << ' ';
    }
  }
  output << "\n";
}

void print_legal_actions(std::ostream& output, const core::Position& position) {
  output << "Legal:";
  bool any = false;
  for (core::Action action = 0; action < core::kNumActions; ++action) {
    if (position.is_legal(action)) {
      output << ' ' << action;
      any = true;
    }
  }
  if (!any) {
    output << " none";
  }
  output << "\n";
}

struct MoveView {
  core::Action action = -1;
  float policy = 0.0f;
  std::uint32_t visits = 0;
  float q = 0.0f;
};

std::vector<MoveView> top_moves(const search::SearchResult& result, int limit) {
  std::vector<MoveView> moves;
  for (core::Action action = 0; action < core::kNumActions; ++action) {
    if (result.visit_counts[action] == 0 && result.policy[action] == 0.0f) {
      continue;
    }
    moves.push_back(MoveView{action, result.policy[action], result.visit_counts[action], result.q_values[action]});
  }
  std::sort(moves.begin(), moves.end(), [](const MoveView& left, const MoveView& right) {
    if (left.visits != right.visits) {
      return left.visits > right.visits;
    }
    return left.action < right.action;
  });
  if (static_cast<int>(moves.size()) > limit) {
    moves.resize(static_cast<std::size_t>(limit));
  }
  return moves;
}

void print_policy(std::ostream& output, const search::SearchResult& result) {
  output << "Top bot search moves:\n";
  output << " action  policy  visits      q\n";
  for (const MoveView& move : top_moves(result, 8)) {
    output << std::setw(7) << move.action
           << ' ' << std::fixed << std::setprecision(3) << std::setw(7) << move.policy
           << ' ' << std::setw(7) << move.visits
           << ' ' << std::setw(6) << move.q << "\n";
  }
  output.unsetf(std::ios::floatfield);
}

void print_search_stats(std::ostream& output, const search::SearchResult& result) {
  output << "Search stats:"
         << " simulations=" << result.completed_simulations
         << " root_visits=" << result.root_real_visits
         << " nodes=" << result.expanded_nodes
         << " max_depth=" << result.max_depth
         << " leaf_evals=" << result.leaf_evaluations
         << " terminal_evals=" << result.terminal_evaluations
         << " pending_waits=" << result.pending_eval_waits
         << " virtual_waits=" << result.virtual_loss_waits
         << " search_ms=" << std::fixed << std::setprecision(1) << result.search_time_ms
         << "\n";
  output.unsetf(std::ios::floatfield);
}

void print_help(std::ostream& output) {
  output
      << "Commands:\n"
      << "  0..15   play an action/column\n"
      << "  board   redraw board\n"
      << "  policy  show last bot policy\n"
      << "  stats   show last search stats\n"
      << "  help    show commands\n"
      << "  reset   start over\n"
      << "  quit    exit\n";
}

bool parse_action(const std::string& text, core::Action& action) {
  try {
    std::size_t parsed = 0;
    const int value = std::stoi(text, &parsed);
    if (parsed != text.size()) {
      return false;
    }
    action = value;
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

std::string terminal_message(const core::Position& position, bool bot_first) {
  const auto value = position.terminal_value();
  if (!value.has_value()) {
    return "";
  }
  if (*value == 0.0f) {
    return "Game over: draw.";
  }
  const bool next_turn_is_human = human_to_move(position, bot_first);
  return next_turn_is_human ? "Game over: bot wins." : "Game over: you win.";
}

class ValueOverrideEvaluator final : public search::Evaluator {
 public:
  ValueOverrideEvaluator(search::Evaluator& evaluator, ValueMode mode) : evaluator_(evaluator), mode_(mode) {}

  search::Evaluation evaluate(const core::Position& position) override {
    search::Evaluation evaluation = evaluator_.evaluate(position);
    if (mode_ == ValueMode::Zero) {
      evaluation.value = 0.0f;
    }
    return evaluation;
  }

 private:
  search::Evaluator& evaluator_;
  ValueMode mode_;
};

struct BotMove {
  search::SearchResult search;
  bool reused_before = false;
  bool reused_after = false;
};

BotMove play_bot_move(
    core::Position& position,
    search::SearchTree& tree,
    search::PuctMcts& mcts,
    search::Evaluator& evaluator) {
  BotMove move;
  move.reused_before = (tree.root().position.compact_string() == position.compact_string());
  if (!move.reused_before) {
    tree = search::SearchTree(position);
  }
  move.search = mcts.search(tree, evaluator, false, 0.0);
  if (move.search.selected_action < 0 || !position.is_legal(move.search.selected_action)) {
    throw std::runtime_error("play command selected an illegal bot action");
  }
  position = position.play(move.search.selected_action);
  move.reused_after = tree.advance(move.search.selected_action);
  if (!move.reused_after) {
    tree = search::SearchTree(position);
  }
  return move;
}

}  // namespace

ValueMode parse_value_mode(const std::string& value) {
  if (value == "model") {
    return ValueMode::Model;
  }
  if (value == "zero") {
    return ValueMode::Zero;
  }
  throw std::invalid_argument("value mode must be 'model' or 'zero'");
}

int run_terminal_game(std::istream& input, std::ostream& output, const TerminalPlayConfig& config) {
  if (config.simulations <= 0) {
    throw std::invalid_argument("play simulations must be positive");
  }
  if (config.search_threads <= 0) {
    throw std::invalid_argument("play search threads must be positive");
  }

  search::UniformEvaluator uniform_evaluator;
  std::unique_ptr<model::AsyncBatchedTorchScriptEvaluator> torchscript_evaluator;
  search::Evaluator* base_evaluator = &uniform_evaluator;
  if (!config.model_path.empty()) {
    model::AsyncBatchedTorchScriptConfig inference_config;
    inference_config.max_batch_size = config.inference_batch_size;
    inference_config.max_wait_us = config.inference_max_wait_us;
    torchscript_evaluator = std::make_unique<model::AsyncBatchedTorchScriptEvaluator>(
        config.model_path,
        parse_device(config.device),
        inference_config);
    base_evaluator = torchscript_evaluator.get();
  }
  ValueOverrideEvaluator evaluator(*base_evaluator, config.value_mode);

  search::PuctConfig mcts_config;
  mcts_config.simulations_per_move = config.simulations;
  mcts_config.search_threads = config.search_threads;
  mcts_config.virtual_loss = config.virtual_loss;
  mcts_config.seed = config.seed;
  search::PuctMcts mcts(mcts_config);

  core::Position position = core::Position::empty();
  search::SearchTree tree(position);
  search::SearchResult last_search;
  bool has_last_search = false;
  bool running = true;

  output << "4x4x4 Connect Four. X=you, O=bot.\n";
  if (config.value_mode == ValueMode::Zero) {
    output << "Using model priors with value forced to 0.0. Good for Stage 0 policy-only checkpoints.\n";
  }
  print_action_grid(output);
  print_help(output);

  while (running) {
    if (position.is_terminal()) {
      print_board(output, position, config.bot_first);
      output << terminal_message(position, config.bot_first) << "\n";
      break;
    }

    if (!human_to_move(position, config.bot_first)) {
      output << "\nBot thinking...\n";
      BotMove bot = play_bot_move(position, tree, mcts, evaluator);
      last_search = bot.search;
      has_last_search = true;
      output << "Bot played action " << bot.search.selected_action
             << " reuse_before=" << (bot.reused_before ? "yes" : "no")
             << " reuse_after=" << (bot.reused_after ? "yes" : "no") << "\n";
      print_search_stats(output, bot.search);
      print_policy(output, bot.search);
      continue;
    }

    print_board(output, position, config.bot_first);
    print_legal_actions(output, position);
    output << "connect4> " << std::flush;

    std::string line;
    if (!std::getline(input, line)) {
      break;
    }
    line = trim(line);
    if (line.empty()) {
      continue;
    }
    if (line == "quit" || line == "q" || line == "exit") {
      break;
    }
    if (line == "help" || line == "h") {
      print_help(output);
      continue;
    }
    if (line == "board" || line == "b") {
      print_board(output, position, config.bot_first);
      print_action_grid(output);
      continue;
    }
    if (line == "policy" || line == "p") {
      if (has_last_search) {
        print_policy(output, last_search);
      } else {
        output << "No bot search yet.\n";
      }
      continue;
    }
    if (line == "stats" || line == "s") {
      if (has_last_search) {
        print_search_stats(output, last_search);
      } else {
        output << "No bot search yet.\n";
      }
      continue;
    }
    if (line == "reset" || line == "r") {
      position = core::Position::empty();
      tree = search::SearchTree(position);
      has_last_search = false;
      output << "Reset.\n";
      continue;
    }

    core::Action action = -1;
    if (!parse_action(line, action)) {
      output << "Invalid input. Type a number 0..15 or 'help'.\n";
      continue;
    }
    if (!position.is_legal(action)) {
      output << "Illegal action " << action << ".\n";
      continue;
    }
    position = position.play(action);
    if (!tree.advance(action)) {
      tree = search::SearchTree(position);
    }
    output << "You played action " << action << ".\n";
  }

  if (torchscript_evaluator) {
    const auto stats = torchscript_evaluator->stats();
    output << "Inference stats:"
           << " requests=" << stats.requests
           << " batches=" << stats.batches
           << " mean_batch_size=" << stats.mean_batch_size()
           << " max_batch_size=" << stats.max_batch_size
           << " mean_wait_ms=" << stats.mean_wait_ms()
           << " total_inference_ms=" << stats.total_inference_ms << "\n";
  }
  return 0;
}

}  // namespace c4zero::play
