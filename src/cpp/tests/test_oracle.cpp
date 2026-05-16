#include "c4zero/oracle/solver.hpp"
#include "test_support.hpp"

#include <array>

using namespace c4zero;

int main() {
  C4ZERO_CHECK_EQ(core::winning_masks().size(), 76);

  const auto immediate_win = core::from_actions({0, 4, 1, 5, 2, 6});
  C4ZERO_CHECK(oracle::is_winning_move(immediate_win, 3));
  {
    oracle::Solver solver(4);
    const auto result = solver.solve_with_move_values(immediate_win, 4);
    C4ZERO_CHECK_EQ(result.best_action, 3);
    C4ZERO_CHECK(result.value > oracle::kMateBase - 10);
    C4ZERO_CHECK(result.move_values[3] > oracle::kMateBase - 10);
    C4ZERO_CHECK(result.nodes > 0);
    C4ZERO_CHECK(!result.stopped);
  }

  const auto immediate_block = core::from_actions({2, 1, 2, 1, 4, 1});
  C4ZERO_CHECK(!oracle::is_winning_move(immediate_block, 3));
  {
    oracle::Solver solver(8);
    const auto result = solver.solve_with_move_values(immediate_block, 4);
    C4ZERO_CHECK_EQ(result.best_action, 1);
    C4ZERO_CHECK(result.move_values[1] > result.move_values[5]);
  }

  const auto full_column = core::from_actions({0, 0, 0, 0});
  {
    oracle::Solver solver(4);
    const auto result = solver.solve_with_move_values(full_column, 3);
    C4ZERO_CHECK_EQ(result.move_values[0], oracle::kInvalidMoveValue);
    C4ZERO_CHECK(result.best_action >= 0);
    C4ZERO_CHECK(full_column.is_legal(result.best_action));
  }

  const auto terminal = immediate_win.play(3);
  C4ZERO_CHECK(terminal.is_terminal());
  {
    oracle::Solver solver(4);
    const auto result = solver.solve_with_move_values(terminal, 4);
    C4ZERO_CHECK_EQ(result.best_action, -1);
    C4ZERO_CHECK(result.value < -oracle::kMateBase + 10);
    C4ZERO_CHECK_EQ(result.nodes, 0);
  }

  const auto empty = core::Position::empty();
  {
    const auto result = oracle::solve(empty, oracle::SolverConfig{4, 2, 0});
    C4ZERO_CHECK(result.best_action >= 0);
    C4ZERO_CHECK(empty.is_legal(result.best_action));
    C4ZERO_CHECK(result.depth >= 1);
  }

  return 0;
}
