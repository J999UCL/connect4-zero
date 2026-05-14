#include "c4zero/data/shard.hpp"
#include "c4zero/version/info.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace c4zero::data {
namespace {

template <typename T>
void write_value(std::ofstream& out, const T& value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
void read_value(std::ifstream& in, T& value) {
  in.read(reinterpret_cast<char*>(&value), sizeof(T));
}

}  // namespace

SelfPlaySample SelfPlaySample::from_position(
    const core::Position& position,
    const std::array<float, core::kNumActions>& policy,
    const std::array<std::uint32_t, core::kNumActions>& visit_counts,
    float value,
    core::Action action,
    std::uint64_t game_id) {
  SelfPlaySample sample;
  sample.current_bits = position.current;
  sample.opponent_bits = position.opponent;
  sample.heights = position.heights;
  sample.ply = position.ply;
  sample.game_id = game_id;
  sample.legal_mask = position.legal_mask();
  sample.action = static_cast<std::uint8_t>(action);
  sample.policy = policy;
  sample.visit_counts = visit_counts;
  sample.value = value;
  return sample;
}

void write_shard(const std::string& path, const std::vector<SelfPlaySample>& samples) {
  const auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to open shard for writing: " + path);
  }
  ShardHeader header;
  header.sample_count = samples.size();
  out.write(reinterpret_cast<const char*>(&header), sizeof(header));
  for (const auto& sample : samples) {
    write_value(out, sample.current_bits);
    write_value(out, sample.opponent_bits);
    out.write(reinterpret_cast<const char*>(sample.heights.data()), sample.heights.size());
    write_value(out, sample.ply);
    write_value(out, sample.game_id);
    write_value(out, sample.legal_mask);
    write_value(out, sample.action);
    out.write(reinterpret_cast<const char*>(sample.policy.data()), sizeof(float) * sample.policy.size());
    out.write(reinterpret_cast<const char*>(sample.visit_counts.data()), sizeof(std::uint32_t) * sample.visit_counts.size());
    write_value(out, sample.value);
  }
}

std::vector<SelfPlaySample> read_shard(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open shard for reading: " + path);
  }
  ShardHeader header;
  in.read(reinterpret_cast<char*>(&header), sizeof(header));
  const char expected[8] = {'C', '4', 'A', 'Z', 'S', 'P', '0', '1'};
  if (std::memcmp(header.magic, expected, sizeof(expected)) != 0) {
    throw std::runtime_error("invalid c4zero shard magic: " + path);
  }
  if (header.schema_major != 1) {
    throw std::runtime_error("unsupported c4zero shard schema");
  }
  std::vector<SelfPlaySample> samples(header.sample_count);
  for (auto& sample : samples) {
    read_value(in, sample.current_bits);
    read_value(in, sample.opponent_bits);
    in.read(reinterpret_cast<char*>(sample.heights.data()), sample.heights.size());
    read_value(in, sample.ply);
    read_value(in, sample.game_id);
    read_value(in, sample.legal_mask);
    read_value(in, sample.action);
    in.read(reinterpret_cast<char*>(sample.policy.data()), sizeof(float) * sample.policy.size());
    in.read(reinterpret_cast<char*>(sample.visit_counts.data()), sizeof(std::uint32_t) * sample.visit_counts.size());
    read_value(in, sample.value);
  }
  return samples;
}

void write_manifest(
    const std::string& path,
    const std::string& shard_path,
    std::uint64_t num_games,
    std::uint64_t num_samples,
    const std::string& model_checkpoint,
    const std::string& config_json) {
  const auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open manifest for writing: " + path);
  }
  out << "{\n";
  out << "  \"schema_version\": \"" << c4zero::version::version_field("dataset_schema_version") << "\",\n";
  out << "  \"num_games\": " << num_games << ",\n";
  out << "  \"num_samples\": " << num_samples << ",\n";
  out << "  \"model_checkpoint\": \"" << model_checkpoint << "\",\n";
  out << "  \"shard_paths\": [\"" << shard_path << "\"],\n";
  out << "  \"config\": " << config_json << ",\n";
  out << "  \"version\": " << c4zero::version::current_version_json() << "\n";
  out << "}\n";
}

}  // namespace c4zero::data
