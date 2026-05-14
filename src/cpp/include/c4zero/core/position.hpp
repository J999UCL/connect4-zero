#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace c4zero::core {

using Action = int;
using Bitboard = std::uint64_t;

constexpr int kBoardSize = 4;
constexpr int kNumActions = 16;
constexpr int kNumCells = 64;
constexpr int kMaxPlies = 64;

struct Coordinates {
  int x;
  int y;
  int z;
};

struct Position {
  Bitboard current = 0;
  Bitboard opponent = 0;
  std::array<std::uint8_t, kNumActions> heights{};
  std::uint8_t ply = 0;

  static Position empty();

  [[nodiscard]] Bitboard occupancy() const;
  [[nodiscard]] bool is_legal(Action action) const;
  [[nodiscard]] std::uint16_t legal_mask() const;
  [[nodiscard]] bool is_full() const;
  [[nodiscard]] std::optional<float> terminal_value() const;
  [[nodiscard]] bool is_terminal() const;
  [[nodiscard]] Position play(Action action) const;
  [[nodiscard]] std::string compact_string() const;
};

[[nodiscard]] constexpr Coordinates action_to_xy(Action action) {
  return Coordinates{action % kBoardSize, action / kBoardSize, 0};
}

[[nodiscard]] constexpr Action xy_to_action(int x, int y) {
  return y * kBoardSize + x;
}

[[nodiscard]] constexpr int cell_index(int x, int y, int z) {
  return z * 16 + y * 4 + x;
}

[[nodiscard]] constexpr int cell_index(Action action, int z) {
  return z * 16 + action;
}

[[nodiscard]] constexpr Bitboard cell_mask(int x, int y, int z) {
  return Bitboard{1} << cell_index(x, y, z);
}

[[nodiscard]] constexpr Bitboard cell_mask(Action action, int z) {
  return Bitboard{1} << cell_index(action, z);
}

[[nodiscard]] const std::vector<Bitboard>& winning_masks();
[[nodiscard]] bool has_winning_line(Bitboard bits);
[[nodiscard]] std::vector<Action> legal_actions(const Position& position);
[[nodiscard]] Position from_actions(const std::vector<Action>& actions);

}  // namespace c4zero::core
