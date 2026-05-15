#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>

namespace c4zero::play {

enum class ValueMode {
  Model,
  Zero,
};

struct TerminalPlayConfig {
  std::string model_path;
  std::string device = "cpu";
  int simulations = 800;
  int search_threads = 4;
  float virtual_loss = 1.0f;
  int inference_batch_size = 128;
  int inference_max_wait_us = 2000;
  std::uint64_t seed = 1;
  bool bot_first = false;
  ValueMode value_mode = ValueMode::Zero;
};

[[nodiscard]] ValueMode parse_value_mode(const std::string& value);

int run_terminal_game(std::istream& input, std::ostream& output, const TerminalPlayConfig& config);

}  // namespace c4zero::play
