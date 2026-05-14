#pragma once

#include "c4zero/core/position.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace c4zero::data {

struct SelfPlaySample {
  core::Bitboard current_bits = 0;
  core::Bitboard opponent_bits = 0;
  std::array<std::uint8_t, core::kNumActions> heights{};
  std::uint8_t ply = 0;
  std::uint64_t game_id = 0;
  std::uint16_t legal_mask = 0;
  std::uint8_t action = 0;
  std::array<float, core::kNumActions> policy{};
  std::array<std::uint32_t, core::kNumActions> visit_counts{};
  float value = 0.0f;

  [[nodiscard]] static SelfPlaySample from_position(
      const core::Position& position,
      const std::array<float, core::kNumActions>& policy,
      const std::array<std::uint32_t, core::kNumActions>& visit_counts,
      float value,
      core::Action action,
      std::uint64_t game_id);
};

struct ShardHeader {
  char magic[8] = {'C', '4', 'A', 'Z', 'S', 'P', '0', '1'};
  std::uint32_t schema_major = 1;
  std::uint32_t schema_minor = 0;
  std::uint64_t sample_count = 0;
};

void write_shard(const std::string& path, const std::vector<SelfPlaySample>& samples);
[[nodiscard]] std::vector<SelfPlaySample> read_shard(const std::string& path);
void write_manifest(
    const std::string& path,
    const std::string& shard_path,
    std::uint64_t num_games,
    std::uint64_t num_samples,
    const std::string& model_checkpoint,
    const std::string& config_json);

class ReplayBuffer {
 public:
  explicit ReplayBuffer(std::size_t max_games);
  void add_game(std::vector<SelfPlaySample> game);
  [[nodiscard]] std::size_t num_games() const;
  [[nodiscard]] std::size_t num_samples() const;
  [[nodiscard]] const SelfPlaySample& sample(std::uint64_t draw) const;
  [[nodiscard]] std::vector<SelfPlaySample> sample_batch(std::size_t batch_size, std::uint64_t seed) const;

 private:
  std::size_t max_games_;
  std::vector<std::vector<SelfPlaySample>> games_;
};

}  // namespace c4zero::data
