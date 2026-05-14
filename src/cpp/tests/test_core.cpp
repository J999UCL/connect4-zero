#include "c4zero/core/position.hpp"
#include "c4zero/core/symmetry.hpp"
#include "test_support.hpp"

#include <set>

using namespace c4zero;

int main() {
  C4ZERO_CHECK_EQ(core::winning_masks().size(), 76);

  auto empty = core::Position::empty();
  C4ZERO_CHECK_EQ(empty.legal_mask(), 0xFFFF);
  C4ZERO_CHECK(!empty.is_terminal());

  auto p = empty;
  for (int i = 0; i < 4; ++i) {
    C4ZERO_CHECK(p.is_legal(0));
    p = p.play(0);
  }
  C4ZERO_CHECK(!p.is_legal(0));

  auto vertical = core::from_actions({0, 1, 0, 1, 0, 1, 0});
  C4ZERO_CHECK(vertical.is_terminal());
  C4ZERO_CHECK_EQ(*vertical.terminal_value(), -1.0f);

  auto horizontal = core::from_actions({0, 4, 1, 5, 2, 6, 3});
  C4ZERO_CHECK(horizontal.is_terminal());

  std::array<float, core::kNumActions> policy{};
  policy[0] = 1.0f;
  auto rotated = core::transform_policy(policy, core::Symmetry::Rot90);
  C4ZERO_CHECK_EQ(rotated[3], 1.0f);

  std::set<int> images;
  for (int i = 0; i < 8; ++i) {
    const auto perm = core::action_permutation(static_cast<core::Symmetry>(i));
    images.insert(perm[0]);
    auto transformed = core::transform(horizontal, static_cast<core::Symmetry>(i));
    C4ZERO_CHECK(transformed.is_terminal());
  }
  C4ZERO_CHECK_EQ(images.size(), 4);
  return 0;
}
