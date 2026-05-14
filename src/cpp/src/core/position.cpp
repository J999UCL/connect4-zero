#include "c4zero/core/position.hpp"

#include <algorithm>
#include <sstream>

namespace c4zero::core {
namespace {

bool in_bounds(int x, int y, int z) {
  return x >= 0 && x < kBoardSize && y >= 0 && y < kBoardSize && z >= 0 && z < kBoardSize;
}

std::vector<Bitboard> build_winning_masks() {
  std::vector<Bitboard> masks;
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -1; dz <= 1; ++dz) {
        if (dx == 0 && dy == 0 && dz == 0) {
          continue;
        }
        if (dx < 0 || (dx == 0 && dy < 0) || (dx == 0 && dy == 0 && dz < 0)) {
          continue;
        }
        for (int x = 0; x < kBoardSize; ++x) {
          for (int y = 0; y < kBoardSize; ++y) {
            for (int z = 0; z < kBoardSize; ++z) {
              const int px = x - dx;
              const int py = y - dy;
              const int pz = z - dz;
              if (in_bounds(px, py, pz)) {
                continue;
              }
              int ex = x + 3 * dx;
              int ey = y + 3 * dy;
              int ez = z + 3 * dz;
              if (!in_bounds(ex, ey, ez)) {
                continue;
              }
              Bitboard mask = 0;
              for (int i = 0; i < 4; ++i) {
                mask |= cell_mask(x + i * dx, y + i * dy, z + i * dz);
              }
              masks.push_back(mask);
            }
          }
        }
      }
    }
  }
  std::sort(masks.begin(), masks.end());
  masks.erase(std::unique(masks.begin(), masks.end()), masks.end());
  return masks;
}

}  // namespace

Position Position::empty() {
  return Position{};
}

Bitboard Position::occupancy() const {
  return current | opponent;
}

bool Position::is_legal(Action action) const {
  return action >= 0 && action < kNumActions && heights[action] < kBoardSize && !is_terminal();
}

std::uint16_t Position::legal_mask() const {
  if (is_terminal()) {
    return 0;
  }
  std::uint16_t mask = 0;
  for (Action action = 0; action < kNumActions; ++action) {
    if (heights[action] < kBoardSize) {
      mask |= static_cast<std::uint16_t>(1u << action);
    }
  }
  return mask;
}

bool Position::is_full() const {
  return ply >= kMaxPlies;
}

std::optional<float> Position::terminal_value() const {
  if (has_winning_line(opponent)) {
    return -1.0f;
  }
  if (is_full()) {
    return 0.0f;
  }
  return std::nullopt;
}

bool Position::is_terminal() const {
  return terminal_value().has_value();
}

Position Position::play(Action action) const {
  if (!is_legal(action)) {
    throw std::invalid_argument("illegal action " + std::to_string(action));
  }
  Position next;
  next.heights = heights;
  const int z = next.heights[action]++;
  const Bitboard placed = current | cell_mask(action, z);
  next.current = opponent;
  next.opponent = placed;
  next.ply = static_cast<std::uint8_t>(ply + 1);
  return next;
}

std::string Position::compact_string() const {
  std::ostringstream out;
  out << "ply=" << static_cast<int>(ply) << " current=" << current << " opponent=" << opponent << " heights=";
  for (int i = 0; i < kNumActions; ++i) {
    out << static_cast<int>(heights[i]);
    if (i + 1 < kNumActions) {
      out << ',';
    }
  }
  return out.str();
}

const std::vector<Bitboard>& winning_masks() {
  static const std::vector<Bitboard> masks = build_winning_masks();
  return masks;
}

bool has_winning_line(Bitboard bits) {
  for (Bitboard mask : winning_masks()) {
    if ((bits & mask) == mask) {
      return true;
    }
  }
  return false;
}

std::vector<Action> legal_actions(const Position& position) {
  std::vector<Action> actions;
  const std::uint16_t mask = position.legal_mask();
  for (Action action = 0; action < kNumActions; ++action) {
    if ((mask & (1u << action)) != 0) {
      actions.push_back(action);
    }
  }
  return actions;
}

Position from_actions(const std::vector<Action>& actions) {
  Position position = Position::empty();
  for (Action action : actions) {
    position = position.play(action);
  }
  return position;
}

}  // namespace c4zero::core
