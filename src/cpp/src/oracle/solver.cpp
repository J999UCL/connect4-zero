#include "c4zero/oracle/solver.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace c4zero::oracle {
namespace {

constexpr std::uint8_t kFlagExact = 0;
constexpr std::uint8_t kFlagLower = 1;
constexpr std::uint8_t kFlagUpper = 2;
constexpr std::uint8_t kFlagEmpty = 0xff;

constexpr std::array<core::Action, core::kNumActions> kGerardCenterOrder{
    5, 6, 9, 10, 4, 7, 8, 11, 1, 2, 13, 14, 0, 3, 12, 15};

std::uint64_t splitmix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ULL;
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31);
}

const std::array<std::uint64_t, core::kNumCells * 2>& zobrist_keys() {
  static const auto keys = [] {
    std::array<std::uint64_t, core::kNumCells * 2> out{};
    std::uint64_t state = 0xDEADC0DEDEADBEEFULL;
    for (std::uint64_t& key : out) {
      state = splitmix64(state);
      key = state;
    }
    return out;
  }();
  return keys;
}

std::uint64_t zobrist(const core::Position& position) {
  const auto& keys = zobrist_keys();
  std::uint64_t hash = 0;
  core::Bitboard bits = position.current;
  while (bits != 0) {
    const int cell = __builtin_ctzll(bits);
    hash ^= keys[static_cast<std::size_t>(cell)];
    bits &= bits - 1;
  }
  bits = position.opponent;
  while (bits != 0) {
    const int cell = __builtin_ctzll(bits);
    hash ^= keys[core::kNumCells + static_cast<std::size_t>(cell)];
    bits &= bits - 1;
  }
  return hash;
}

int line_score(int mine, int theirs) {
  if (mine > 0 && theirs > 0) {
    return 0;
  }
  if (mine == 1) return 1;
  if (mine == 2) return 4;
  if (mine == 3) return 32;
  if (theirs == 1) return -1;
  if (theirs == 2) return -4;
  if (theirs == 3) return -32;
  return 0;
}

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

std::array<core::Action, core::kNumActions> ordered_moves(std::uint16_t legal, int tt_best, int& count) {
  std::array<core::Action, core::kNumActions> out{};
  count = 0;
  if (tt_best >= 0 && tt_best < core::kNumActions && (legal & (1u << tt_best)) != 0) {
    out[static_cast<std::size_t>(count++)] = tt_best;
  }
  for (core::Action action : kGerardCenterOrder) {
    if ((legal & (1u << action)) == 0) {
      continue;
    }
    if (action == tt_best) {
      continue;
    }
    out[static_cast<std::size_t>(count++)] = action;
  }
  return out;
}

std::uint8_t as_depth(int depth) {
  return static_cast<std::uint8_t>(std::clamp(depth, 0, 255));
}

}  // namespace

TranspositionTable::TranspositionTable(int size_mb) {
  const std::size_t bytes = static_cast<std::size_t>(std::max(1, size_mb)) * 1024ULL * 1024ULL;
  const std::size_t raw = std::max<std::size_t>(1, bytes / sizeof(Entry));
  std::size_t size = 1;
  while (size * 2 <= raw) {
    size *= 2;
  }
  entries_.assign(size, Entry{});
  for (auto& entry : entries_) {
    entry.flag = kFlagEmpty;
  }
  mask_ = size - 1;
}

TranspositionTable::Entry* TranspositionTable::probe(std::uint64_t key) {
  Entry& entry = entries_[static_cast<std::size_t>(key) & mask_];
  if (entry.flag != kFlagEmpty && entry.key == key) {
    return &entry;
  }
  return nullptr;
}

void TranspositionTable::store(
    std::uint64_t key,
    int value,
    std::uint8_t depth,
    std::uint8_t flag,
    std::uint8_t best_move) {
  Entry& entry = entries_[static_cast<std::size_t>(key) & mask_];
  if (entry.flag == kFlagEmpty || entry.depth <= depth) {
    entry = Entry{key, value, depth, flag, best_move};
  }
}

void TranspositionTable::clear() {
  for (auto& entry : entries_) {
    entry.flag = kFlagEmpty;
  }
}

Solver::Solver(int tt_size_mb) : tt_(tt_size_mb) {}

void Solver::clear() {
  tt_.clear();
}

bool Solver::time_up() const {
  return time_budget_.count() > 0 && std::chrono::steady_clock::now() - started_at_ >= time_budget_;
}

bool is_winning_move(const core::Position& position, core::Action action) {
  if (!position.is_legal(action)) {
    return false;
  }
  const auto terminal = position.play(action).terminal_value();
  return terminal.has_value() && *terminal < 0.0f;
}

int evaluate_heuristic(const core::Position& position) {
  if (const auto terminal = position.terminal_value()) {
    if (*terminal < 0.0f) {
      return -kMateBase;
    }
    return 0;
  }
  int score = 0;
  for (core::Bitboard line : core::winning_masks()) {
    score += line_score(count_bits(position.current & line), count_bits(position.opponent & line));
  }
  return score;
}

int Solver::negamax(const core::Position& position, int depth, int alpha, int beta, int ply_distance) {
  ++nodes_;
  if ((nodes_ & 0xfffULL) == 0 && time_up()) {
    stopped_ = true;
  }
  if (stopped_) {
    return 0;
  }

  if (const auto terminal = position.terminal_value()) {
    if (*terminal < 0.0f) {
      return -kMateBase + ply_distance;
    }
    return 0;
  }

  const std::uint16_t legal = position.legal_mask();
  if (legal == 0) {
    return 0;
  }

  std::uint16_t remaining = legal;
  while (remaining != 0) {
    const core::Action action = __builtin_ctz(remaining);
    if (is_winning_move(position, action)) {
      const int value = kMateBase - ply_distance;
      tt_.store(zobrist(position), value, as_depth(depth), kFlagExact, static_cast<std::uint8_t>(action));
      return value;
    }
    remaining &= static_cast<std::uint16_t>(remaining - 1);
  }

  const std::uint64_t key = zobrist(position);
  int tt_best = -1;
  if (auto* entry = tt_.probe(key)) {
    if (entry->depth >= as_depth(depth)) {
      if (entry->flag == kFlagExact) {
        return entry->value;
      }
      if (entry->flag == kFlagLower && entry->value >= beta) {
        return entry->value;
      }
      if (entry->flag == kFlagUpper && entry->value <= alpha) {
        return entry->value;
      }
    }
    if (entry->best_move < core::kNumActions) {
      tt_best = entry->best_move;
    }
  }

  if (depth <= 0) {
    return evaluate_heuristic(position);
  }

  const int alpha_original = alpha;
  int count = 0;
  const auto order = ordered_moves(legal, tt_best, count);
  int best = -kInfinity;
  core::Action best_move = count > 0 ? order[0] : -1;

  for (int index = 0; index < count; ++index) {
    const core::Action action = order[static_cast<std::size_t>(index)];
    const int value = -negamax(position.play(action), depth - 1, -beta, -alpha, ply_distance + 1);
    if (value > best) {
      best = value;
      best_move = action;
      alpha = std::max(alpha, value);
      if (alpha >= beta) {
        break;
      }
    }
  }

  const std::uint8_t flag = best <= alpha_original ? kFlagUpper : (best >= beta ? kFlagLower : kFlagExact);
  tt_.store(key, best, as_depth(depth), flag, static_cast<std::uint8_t>(best_move));
  return best;
}

SolveResult Solver::solve(const core::Position& position, int max_depth, int time_ms) {
  if (max_depth <= 0) {
    throw std::invalid_argument("oracle max_depth must be positive");
  }
  SolveResult result;
  result.move_values.fill(kInvalidMoveValue);
  if (const auto terminal = position.terminal_value()) {
    result.value = *terminal < 0.0f ? -kMateBase : 0;
    return result;
  }
  if (position.legal_mask() == 0) {
    return result;
  }

  nodes_ = 0;
  stopped_ = false;
  started_at_ = std::chrono::steady_clock::now();
  time_budget_ = time_ms > 0 ? std::chrono::milliseconds(time_ms) : std::chrono::nanoseconds(0);

  for (int depth = 1; depth <= max_depth; ++depth) {
    const int value = negamax(position, depth, -kInfinity, kInfinity, 0);
    if (stopped_) {
      break;
    }
    const auto* entry = tt_.probe(zobrist(position));
    result.value = value;
    result.best_action = entry != nullptr && entry->best_move < core::kNumActions ? entry->best_move : -1;
    result.depth = depth;
    if (std::abs(value) > kMateBase - 1000) {
      break;
    }
    if (time_up()) {
      break;
    }
  }

  result.nodes = nodes_;
  result.stopped = stopped_;
  time_budget_ = std::chrono::nanoseconds(0);
  return result;
}

SolveResult Solver::solve_with_move_values(const core::Position& position, int max_depth, int time_ms) {
  SolveResult result = solve(position, max_depth, time_ms);
  if (position.is_terminal() || position.legal_mask() == 0) {
    return result;
  }

  const int probe_depth = std::max(0, result.depth - 1);
  started_at_ = std::chrono::steady_clock::now();
  time_budget_ = time_ms > 0 ? std::chrono::milliseconds(std::max(50, time_ms)) : std::chrono::nanoseconds(0);
  stopped_ = false;

  const std::uint16_t legal = position.legal_mask();
  for (core::Action action = 0; action < core::kNumActions; ++action) {
    if ((legal & (1u << action)) == 0) {
      result.move_values[static_cast<std::size_t>(action)] = kInvalidMoveValue;
      continue;
    }
    if (is_winning_move(position, action)) {
      result.move_values[static_cast<std::size_t>(action)] = kMateBase;
      continue;
    }
    if (time_up()) {
      stopped_ = true;
      break;
    }
    const int value = -negamax(position.play(action), probe_depth, -kInfinity, kInfinity, 1);
    if (!stopped_) {
      result.move_values[static_cast<std::size_t>(action)] = value;
    }
  }

  int best_value = kInvalidMoveValue;
  core::Action best_action = -1;
  for (core::Action action = 0; action < core::kNumActions; ++action) {
    const int value = result.move_values[static_cast<std::size_t>(action)];
    if (value > best_value) {
      best_value = value;
      best_action = action;
    }
  }
  if (best_action >= 0) {
    result.value = best_value;
    result.best_action = best_action;
  }
  result.nodes = nodes_;
  result.stopped = result.stopped || stopped_;
  time_budget_ = std::chrono::nanoseconds(0);
  return result;
}

SolveResult solve(const core::Position& position, const SolverConfig& config) {
  Solver solver(config.tt_size_mb);
  return solver.solve_with_move_values(position, config.max_depth, config.time_ms);
}

}  // namespace c4zero::oracle
