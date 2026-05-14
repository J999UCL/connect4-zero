#include "c4zero/data/shard.hpp"
#include "test_support.hpp"

#include <filesystem>

using namespace c4zero;

int main() {
  auto position = core::from_actions({0, 4, 1});
  std::array<float, core::kNumActions> policy{};
  policy[2] = 1.0f;
  std::array<std::uint32_t, core::kNumActions> visits{};
  visits[2] = 9;
  auto sample = data::SelfPlaySample::from_position(position, policy, visits, 0.5f, 2, 42);

  const auto path = std::filesystem::temp_directory_path() / "c4zero-test-shard.c4az";
  data::write_shard(path.string(), {sample});
  auto loaded = data::read_shard(path.string());
  C4ZERO_CHECK_EQ(loaded.size(), 1);
  C4ZERO_CHECK_EQ(loaded[0].current_bits, sample.current_bits);
  C4ZERO_CHECK_EQ(loaded[0].opponent_bits, sample.opponent_bits);
  C4ZERO_CHECK_EQ(loaded[0].visit_counts[2], 9);

  data::ReplayBuffer replay(2);
  replay.add_game({sample});
  replay.add_game({sample, sample});
  replay.add_game({sample});
  C4ZERO_CHECK_EQ(replay.num_games(), 2);
  C4ZERO_CHECK_EQ(replay.num_samples(), 3);
  C4ZERO_CHECK_EQ(replay.sample_batch(4, 1).size(), 4);
  return 0;
}
