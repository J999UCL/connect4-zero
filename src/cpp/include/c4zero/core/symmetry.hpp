#pragma once

#include "c4zero/core/position.hpp"

#include <array>

namespace c4zero::core {

enum class Symmetry : int {
  Identity = 0,
  Rot90 = 1,
  Rot180 = 2,
  Rot270 = 3,
  MirrorX = 4,
  MirrorY = 5,
  Diagonal = 6,
  AntiDiagonal = 7,
};

[[nodiscard]] std::array<Action, kNumActions> action_permutation(Symmetry symmetry);
[[nodiscard]] Position transform(const Position& position, Symmetry symmetry);
[[nodiscard]] std::array<float, kNumActions> transform_policy(const std::array<float, kNumActions>& policy, Symmetry symmetry);
[[nodiscard]] std::array<std::uint32_t, kNumActions> transform_visits(const std::array<std::uint32_t, kNumActions>& visits, Symmetry symmetry);
[[nodiscard]] std::uint16_t transform_legal_mask(std::uint16_t mask, Symmetry symmetry);

}  // namespace c4zero::core
