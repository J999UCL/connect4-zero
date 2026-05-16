#include "c4zero/bots/heuristic.hpp"

#include "c4zero/oracle/solver.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace c4zero::bots {
namespace {

int count_bits(core::Bitboard bits) {
#if defined(__GNUG__) || defined(__clang__)
  return __builtin_popcountll(bits);
#else
  int count = 0;
  while (bits != 0) {
    bits &= bits - 1;
    ++count;
  }
  return count;
#endif
}

core::Action first_by_center_order(const std::vector<core::Action>& actions) {
  for (core::Action ordered : center_order()) {
    if (std::find(actions.begin(), actions.end(), ordered) != actions.end()) {
      return ordered;
    }
  }
  throw std::runtime_error("no legal actions");
}

int line_score_from_counts(int mine, int theirs) {
  if (mine > 0 && theirs > 0) {
    return 0;
  }
  if (mine == 1) return 1;
  if (mine == 2) return 8;
  if (mine == 3) return 80;
  if (mine == 4) return 1000000;
  if (theirs == 1) return -1;
  if (theirs == 2) return -10;
  if (theirs == 3) return -120;
  if (theirs == 4) return -1000000;
  return 0;
}

int candidate_score(const core::Position& position, core::Action action) {
  const core::Position child = position.play(action);
  if (child.terminal_value().has_value() && *child.terminal_value() == -1.0f) {
    return 100000000;
  }
  int score = evaluate_lines_for_bits(child.opponent, child.current);
  if (std::find(center_order().begin(), center_order().begin() + 4, action) != center_order().begin() + 4) {
    score += 3;
  }
  score -= 300 * opponent_winning_replies_after(position, action);
  return score;
}

std::vector<core::Action> tactically_safe_actions(const core::Position& position) {
  std::vector<core::Action> safe;
  for (core::Action action : ordered_legal_actions(position)) {
    if (opponent_winning_replies_after(position, action) == 0) {
      safe.push_back(action);
    }
  }
  return safe.empty() ? ordered_legal_actions(position) : safe;
}

int negamax(const core::Position& position, int depth, int ply_distance, int alpha, int beta) {
  if (const auto terminal = position.terminal_value()) {
    if (*terminal > 0.0f) {
      return 1000000 - ply_distance;
    }
    if (*terminal < 0.0f) {
      return -1000000 + ply_distance;
    }
    return 0;
  }
  if (depth <= 0) {
    return evaluate_position_for_side_to_move(position);
  }

  int best = std::numeric_limits<int>::min() / 4;
  for (core::Action action : ordered_legal_actions(position)) {
    const int value = -negamax(position.play(action), depth - 1, ply_distance + 1, -beta, -alpha);
    best = std::max(best, value);
    alpha = std::max(alpha, value);
    if (alpha >= beta) {
      break;
    }
  }
  return best;
}

}  // namespace

double BotMatchResult::first_score_rate() const {
  if (games == 0) {
    return 0.0;
  }
  return (static_cast<double>(first_wins) + 0.5 * static_cast<double>(draws)) / static_cast<double>(games);
}

std::string BotMatchResult::summary() const {
  std::ostringstream out;
  out << "games=" << games
      << " first_wins=" << first_wins
      << " second_wins=" << second_wins
      << " draws=" << draws
      << " score_rate=" << first_score_rate()
      << " avg_plies=" << (games == 0 ? 0.0 : static_cast<double>(total_plies) / games);
  return out.str();
}

const std::array<core::Action, core::kNumActions>& center_order() {
  static const std::array<core::Action, core::kNumActions> order{5, 6, 9, 10, 1, 2, 4, 7, 8, 11, 13, 14, 0, 3, 12, 15};
  return order;
}

bool is_immediate_win(const core::Position& position, core::Action action) {
  if (!position.is_legal(action)) {
    return false;
  }
  const core::Position child = position.play(action);
  const auto terminal = child.terminal_value();
  return terminal.has_value() && *terminal == -1.0f;
}

std::vector<core::Action> immediate_winning_actions(const core::Position& position) {
  std::vector<core::Action> wins;
  for (core::Action action : ordered_legal_actions(position)) {
    if (is_immediate_win(position, action)) {
      wins.push_back(action);
    }
  }
  return wins;
}

int opponent_winning_replies_after(const core::Position& position, core::Action action) {
  if (!position.is_legal(action)) {
    return std::numeric_limits<int>::max() / 4;
  }
  return static_cast<int>(immediate_winning_actions(position.play(action)).size());
}

int playable_threat_count_after(const core::Position& position, core::Action action) {
  if (!position.is_legal(action)) {
    return 0;
  }
  const core::Position child = position.play(action);
  core::Position mover_to_move;
  mover_to_move.current = child.opponent;
  mover_to_move.opponent = child.current;
  mover_to_move.heights = child.heights;
  mover_to_move.ply = child.ply;
  return static_cast<int>(immediate_winning_actions(mover_to_move).size());
}

int evaluate_lines_for_bits(core::Bitboard mine, core::Bitboard theirs) {
  int score = 0;
  for (core::Bitboard mask : core::winning_masks()) {
    score += line_score_from_counts(count_bits(mine & mask), count_bits(theirs & mask));
  }
  return score;
}

int evaluate_position_for_side_to_move(const core::Position& position) {
  if (const auto terminal = position.terminal_value()) {
    if (*terminal > 0.0f) {
      return 1000000;
    }
    if (*terminal < 0.0f) {
      return -1000000;
    }
    return 0;
  }
  return evaluate_lines_for_bits(position.current, position.opponent);
}

std::vector<core::Action> ordered_legal_actions(const core::Position& position) {
  std::vector<core::Action> actions;
  const std::uint16_t mask = position.legal_mask();
  for (core::Action action : center_order()) {
    if ((mask & (1u << action)) != 0) {
      actions.push_back(action);
    }
  }
  return actions;
}

core::Action FirstLegalBot::select_move(const core::Position& position) const {
  const auto actions = core::legal_actions(position);
  if (actions.empty()) {
    throw std::runtime_error("FirstLegalBot called on terminal position");
  }
  return actions.front();
}

std::string FirstLegalBot::name() const {
  return "first-legal";
}

core::Action CenterOrderBot::select_move(const core::Position& position) const {
  return first_by_center_order(ordered_legal_actions(position));
}

std::string CenterOrderBot::name() const {
  return "center-order";
}

core::Action OnePlyTacticalBot::select_move(const core::Position& position) const {
  const auto wins = immediate_winning_actions(position);
  if (!wins.empty()) {
    return first_by_center_order(wins);
  }

  std::vector<core::Action> safe;
  int worst_reply_count = 0;
  for (core::Action action : ordered_legal_actions(position)) {
    const int replies = opponent_winning_replies_after(position, action);
    worst_reply_count = std::max(worst_reply_count, replies);
    if (replies == 0) {
      safe.push_back(action);
    }
  }
  if (worst_reply_count > 0 && !safe.empty()) {
    return first_by_center_order(safe);
  }
  return CenterOrderBot{}.select_move(position);
}

std::string OnePlyTacticalBot::name() const {
  return "one-ply-tactical";
}

core::Action LineScoreBot::select_move(const core::Position& position) const {
  const auto wins = immediate_winning_actions(position);
  if (!wins.empty()) {
    return first_by_center_order(wins);
  }

  int best_score = std::numeric_limits<int>::min();
  core::Action best_action = -1;
  for (core::Action action : tactically_safe_actions(position)) {
    const int score = candidate_score(position, action);
    if (score > best_score) {
      best_score = score;
      best_action = action;
    }
  }
  if (best_action < 0) {
    throw std::runtime_error("LineScoreBot called on terminal position");
  }
  return best_action;
}

std::string LineScoreBot::name() const {
  return "line-score";
}

core::Action ForkThreatBot::select_move(const core::Position& position) const {
  const auto wins = immediate_winning_actions(position);
  if (!wins.empty()) {
    return first_by_center_order(wins);
  }

  int best_score = std::numeric_limits<int>::min();
  core::Action best_action = -1;
  for (core::Action action : tactically_safe_actions(position)) {
    int score = candidate_score(position, action);
    const int own_forks = playable_threat_count_after(position, action);
    if (own_forks >= 2) {
      score += 10000 + 100 * own_forks;
    }
    const int opponent_replies = opponent_winning_replies_after(position, action);
    if (opponent_replies >= 2) {
      score -= 12000 + 100 * opponent_replies;
    }
    if (score > best_score) {
      best_score = score;
      best_action = action;
    }
  }
  if (best_action < 0) {
    throw std::runtime_error("ForkThreatBot called on terminal position");
  }
  return best_action;
}

std::string ForkThreatBot::name() const {
  return "fork-threat";
}

DepthLimitedMinimaxBot::DepthLimitedMinimaxBot(int depth) : depth_(depth) {}

core::Action DepthLimitedMinimaxBot::select_move(const core::Position& position) const {
  int best_score = std::numeric_limits<int>::min();
  core::Action best_action = -1;
  for (core::Action action : ordered_legal_actions(position)) {
    const int score = -negamax(position.play(action), depth_ - 1, 1, std::numeric_limits<int>::min() / 4, std::numeric_limits<int>::max() / 4);
    if (score > best_score) {
      best_score = score;
      best_action = action;
    }
  }
  if (best_action < 0) {
    throw std::runtime_error("DepthLimitedMinimaxBot called on terminal position");
  }
  return best_action;
}

std::string DepthLimitedMinimaxBot::name() const {
  return "minimax-depth-" + std::to_string(depth_);
}

OracleBot::OracleBot(int depth, int tt_size_mb) : depth_(depth), tt_size_mb_(tt_size_mb) {
  if (depth_ <= 0) {
    throw std::invalid_argument("oracle bot depth must be positive");
  }
}

core::Action OracleBot::select_move(const core::Position& position) const {
  oracle::Solver solver(tt_size_mb_);
  const auto result = solver.solve_with_move_values(position, depth_);
  if (result.best_action < 0 || !position.is_legal(result.best_action)) {
    throw std::runtime_error("OracleBot called on terminal position");
  }
  return result.best_action;
}

std::string OracleBot::name() const {
  return "oracle-d" + std::to_string(depth_);
}

int parse_depth_suffix(const std::string& name, const std::string& prefix) {
  if (name.rfind(prefix, 0) != 0) {
    return -1;
  }
  const std::string suffix = name.substr(prefix.size());
  if (suffix.empty()) {
    return -1;
  }
  for (char ch : suffix) {
    if (!std::isdigit(static_cast<unsigned char>(ch))) {
      return -1;
    }
  }
  return std::stoi(suffix);
}

std::unique_ptr<Bot> make_bot(const std::string& name) {
  if (name == "first" || name == "first-legal") return std::make_unique<FirstLegalBot>();
  if (name == "center" || name == "center-order") return std::make_unique<CenterOrderBot>();
  if (name == "tactical" || name == "one-ply-tactical") return std::make_unique<OnePlyTacticalBot>();
  if (name == "line" || name == "line-score") return std::make_unique<LineScoreBot>();
  if (name == "fork" || name == "fork-threat") return std::make_unique<ForkThreatBot>();
  if (name == "minimax3") return std::make_unique<DepthLimitedMinimaxBot>(3);
  if (name == "minimax5") return std::make_unique<DepthLimitedMinimaxBot>(5);
  if (const int depth = parse_depth_suffix(name, "oracle-d"); depth > 0) {
    return std::make_unique<OracleBot>(depth);
  }
  if (const int depth = parse_depth_suffix(name, "oracle_d"); depth > 0) {
    return std::make_unique<OracleBot>(depth);
  }
  throw std::invalid_argument("unknown bot: " + name);
}

std::vector<std::string> bot_names() {
  return {
      "first-legal",
      "center-order",
      "one-ply-tactical",
      "line-score",
      "fork-threat",
      "minimax3",
      "minimax5",
      "oracle-d2",
      "oracle-d4",
      "oracle-d6",
      "oracle-d8",
      "oracle-d16"};
}

BotMatchResult play_bot_match(
    const Bot& first,
    const Bot& second,
    int games,
    bool alternate_starts) {
  BotMatchResult result;
  result.games = games;
  for (int game = 0; game < games; ++game) {
    core::Position position = core::Position::empty();
    const bool first_controls_initial = !alternate_starts || game % 2 == 0;
    while (!position.is_terminal()) {
      const bool initial_player_to_move = (position.ply % 2 == 0);
      const bool first_to_move = first_controls_initial == initial_player_to_move;
      const Bot& bot = first_to_move ? first : second;
      const core::Action action = bot.select_move(position);
      if (!position.is_legal(action)) {
        throw std::runtime_error(bot.name() + " selected illegal action");
      }
      position = position.play(action);
    }
    result.total_plies += position.ply;
    const float terminal = *position.terminal_value();
    if (terminal == 0.0f) {
      result.draws += 1;
      continue;
    }
    const bool initial_player_won = (position.ply % 2 == 1);
    const bool first_won = first_controls_initial == initial_player_won;
    if (first_won) {
      result.first_wins += 1;
    } else {
      result.second_wins += 1;
    }
  }
  return result;
}

}  // namespace c4zero::bots
