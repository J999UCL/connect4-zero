#pragma once

#include "c4zero/core/position.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <vector>

namespace c4zero::oracle {

constexpr int kMateBase = 100000;
constexpr int kInfinity = 1000000;
constexpr int kInvalidMoveValue = -2147483647 - 1;

struct SolveResult {
  int value = 0;
  core::Action best_action = -1;
  int depth = 0;
  std::uint64_t nodes = 0;
  bool stopped = false;
  std::array<int, core::kNumActions> move_values{};
};

struct SolverConfig {
  int tt_size_mb = 64;
  int max_depth = 4;
  int time_ms = 0;
};

class TranspositionTable {
 public:
  explicit TranspositionTable(int size_mb = 64);

  struct Entry {
    std::uint64_t key = 0;
    int value = 0;
    std::uint8_t depth = 0;
    std::uint8_t flag = 0xff;
    std::uint8_t best_move = 0xff;
  };

  [[nodiscard]] Entry* probe(std::uint64_t key);
  void store(std::uint64_t key, int value, std::uint8_t depth, std::uint8_t flag, std::uint8_t best_move);
  void clear();

 private:
  std::vector<Entry> entries_;
  std::size_t mask_ = 0;
};

class Solver {
 public:
  explicit Solver(int tt_size_mb = 64);

  void clear();
  [[nodiscard]] SolveResult solve(const core::Position& position, int max_depth, int time_ms = 0);
  [[nodiscard]] SolveResult solve_with_move_values(const core::Position& position, int max_depth, int time_ms = 0);

 private:
  [[nodiscard]] bool time_up() const;
  [[nodiscard]] int negamax(const core::Position& position, int depth, int alpha, int beta, int ply_distance);

  TranspositionTable tt_;
  std::uint64_t nodes_ = 0;
  bool stopped_ = false;
  std::chrono::steady_clock::time_point started_at_{};
  std::chrono::nanoseconds time_budget_{0};
};

[[nodiscard]] bool is_winning_move(const core::Position& position, core::Action action);
[[nodiscard]] int evaluate_heuristic(const core::Position& position);
[[nodiscard]] SolveResult solve(const core::Position& position, const SolverConfig& config);

}  // namespace c4zero::oracle
