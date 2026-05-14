#include "c4zero/data/shard.hpp"
#include "test_support.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

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

  bool missing_threw = false;
  try {
    (void)data::read_shard((std::filesystem::temp_directory_path() / "c4zero-missing-shard.c4az").string());
  } catch (const std::runtime_error&) {
    missing_threw = true;
  }
  C4ZERO_CHECK(missing_threw);

  const auto corrupt_path = std::filesystem::temp_directory_path() / "c4zero-corrupt-shard.c4az";
  {
    std::ofstream corrupt(corrupt_path, std::ios::binary);
    corrupt << "not a c4zero shard";
  }
  bool corrupt_threw = false;
  try {
    (void)data::read_shard(corrupt_path.string());
  } catch (const std::runtime_error&) {
    corrupt_threw = true;
  }
  C4ZERO_CHECK(corrupt_threw);

  data::ReplayBuffer replay(2);
  replay.add_game({sample});
  replay.add_game({sample, sample});
  replay.add_game({sample});
  C4ZERO_CHECK_EQ(replay.num_games(), 2);
  C4ZERO_CHECK_EQ(replay.num_samples(), 3);
  C4ZERO_CHECK_EQ(replay.sample_batch(4, 1).size(), 4);

  const auto manifest_path = std::filesystem::temp_directory_path() / "c4zero-test-manifest.json";
  data::SelfPlayManifestConfig manifest_config;
  manifest_config.model_checkpoint = "model\"with-quote.ts";
  manifest_config.device = "cpu";
  manifest_config.simulations_per_move = 800;
  manifest_config.git_commit = "abc123";
  data::write_manifest(manifest_path.string(), "shards/shard-000000.c4az", 1, 1, manifest_config);
  std::ifstream manifest(manifest_path);
  const std::string text((std::istreambuf_iterator<char>(manifest)), std::istreambuf_iterator<char>());
  C4ZERO_CHECK(text.find("\"simulations_per_move\": 800") != std::string::npos);
  C4ZERO_CHECK(text.find("model\\\"with-quote.ts") != std::string::npos);
  C4ZERO_CHECK(text.find("\"git_commit\": \"abc123\"") != std::string::npos);
  return 0;
}
