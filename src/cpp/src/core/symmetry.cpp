#include "c4zero/core/symmetry.hpp"

namespace c4zero::core {
namespace {

Coordinates map_xy(int x, int y, Symmetry symmetry) {
  switch (symmetry) {
    case Symmetry::Identity:
      return Coordinates{x, y, 0};
    case Symmetry::Rot90:
      return Coordinates{3 - y, x, 0};
    case Symmetry::Rot180:
      return Coordinates{3 - x, 3 - y, 0};
    case Symmetry::Rot270:
      return Coordinates{y, 3 - x, 0};
    case Symmetry::MirrorX:
      return Coordinates{3 - x, y, 0};
    case Symmetry::MirrorY:
      return Coordinates{x, 3 - y, 0};
    case Symmetry::Diagonal:
      return Coordinates{y, x, 0};
    case Symmetry::AntiDiagonal:
      return Coordinates{3 - y, 3 - x, 0};
  }
  return Coordinates{x, y, 0};
}

Bitboard transform_bits(Bitboard bits, Symmetry symmetry) {
  Bitboard out = 0;
  for (int z = 0; z < kBoardSize; ++z) {
    for (int y = 0; y < kBoardSize; ++y) {
      for (int x = 0; x < kBoardSize; ++x) {
        const Bitboard mask = cell_mask(x, y, z);
        if ((bits & mask) == 0) {
          continue;
        }
        const Coordinates mapped = map_xy(x, y, symmetry);
        out |= cell_mask(mapped.x, mapped.y, z);
      }
    }
  }
  return out;
}

}  // namespace

std::array<Action, kNumActions> action_permutation(Symmetry symmetry) {
  std::array<Action, kNumActions> permutation{};
  for (Action action = 0; action < kNumActions; ++action) {
    const Coordinates xy = action_to_xy(action);
    const Coordinates mapped = map_xy(xy.x, xy.y, symmetry);
    permutation[action] = xy_to_action(mapped.x, mapped.y);
  }
  return permutation;
}

Position transform(const Position& position, Symmetry symmetry) {
  Position out;
  out.current = transform_bits(position.current, symmetry);
  out.opponent = transform_bits(position.opponent, symmetry);
  out.ply = position.ply;
  const Bitboard occupancy = out.occupancy();
  for (Action action = 0; action < kNumActions; ++action) {
    std::uint8_t height = 0;
    while (height < kBoardSize && (occupancy & cell_mask(action, height)) != 0) {
      ++height;
    }
    out.heights[action] = height;
  }
  return out;
}

std::array<float, kNumActions> transform_policy(const std::array<float, kNumActions>& policy, Symmetry symmetry) {
  std::array<float, kNumActions> out{};
  const auto permutation = action_permutation(symmetry);
  for (Action action = 0; action < kNumActions; ++action) {
    out[permutation[action]] = policy[action];
  }
  return out;
}

std::array<std::uint32_t, kNumActions> transform_visits(const std::array<std::uint32_t, kNumActions>& visits, Symmetry symmetry) {
  std::array<std::uint32_t, kNumActions> out{};
  const auto permutation = action_permutation(symmetry);
  for (Action action = 0; action < kNumActions; ++action) {
    out[permutation[action]] = visits[action];
  }
  return out;
}

std::uint16_t transform_legal_mask(std::uint16_t mask, Symmetry symmetry) {
  std::uint16_t out = 0;
  const auto permutation = action_permutation(symmetry);
  for (Action action = 0; action < kNumActions; ++action) {
    if ((mask & (1u << action)) != 0) {
      out |= static_cast<std::uint16_t>(1u << permutation[action]);
    }
  }
  return out;
}

}  // namespace c4zero::core
