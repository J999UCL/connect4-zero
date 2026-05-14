#pragma once

#include "c4zero/data/shard.hpp"

#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace c4zero::curriculum {

enum class Stage0Category {
  ImmediateWin,
  ImmediateBlock,
  SafeMoveVsBlunder,
  PlayableVsFloatingThreat,
  ForkCreate,
  ForkBlock,
  MinimaxDepth3Policy,
};

struct Stage0Config {
  std::uint64_t samples = 1'000'000;
  std::uint64_t shard_size = 100'000;
  std::uint64_t seed = 1;
  bool use_symmetries = true;
  std::string output_dir = "/tmp/thakwani/rl-data/curriculum/stage0-v1";
  std::string git_commit;
};

struct LabeledSample {
  data::SelfPlaySample sample;
  Stage0Category category = Stage0Category::ImmediateWin;
};

struct Stage0Result {
  std::uint64_t samples = 0;
  std::uint64_t shards = 0;
  std::map<std::string, std::uint64_t> category_counts;
  std::string manifest_path;
};

[[nodiscard]] std::string category_name(Stage0Category category);
[[nodiscard]] LabeledSample generate_stage0_sample(
    Stage0Category category,
    std::mt19937_64& rng,
    std::uint64_t sample_id);
[[nodiscard]] std::vector<LabeledSample> generate_stage0_samples(const Stage0Config& config);
[[nodiscard]] Stage0Result write_stage0_dataset(const Stage0Config& config);

}  // namespace c4zero::curriculum
