#pragma once

#include "c4zero/core/position.hpp"
#include "c4zero/search/puct.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace c4zero::web {

enum class WebValueMode {
  Model,
  Zero,
};

struct WebPlayConfig {
  std::string model_path;
  std::string device = "cpu";
  std::string host = "127.0.0.1";
  std::string web_root = "src/web/play";
  int port = 8080;
  int simulations = 800;
  int search_threads = 4;
  float virtual_loss = 1.0f;
  int inference_batch_size = 128;
  int inference_max_wait_us = 2000;
  std::uint64_t seed = 1;
  bool bot_first = false;
  WebValueMode value_mode = WebValueMode::Model;
};

[[nodiscard]] WebValueMode parse_web_value_mode(const std::string& value);

class WebGameSession {
 public:
  WebGameSession(search::Evaluator& evaluator, search::PuctConfig mcts_config, bool bot_first);

  void reset();
  void play_human_action(core::Action action);
  void play_bot_action();

  [[nodiscard]] const core::Position& position() const;
  [[nodiscard]] bool human_to_move() const;
  [[nodiscard]] bool has_last_search() const;
  [[nodiscard]] const search::SearchResult& last_search() const;
  [[nodiscard]] std::string state_json() const;

 private:
  search::Evaluator& evaluator_;
  search::PuctMcts mcts_;
  bool bot_first_ = false;
  core::Position position_ = core::Position::empty();
  search::SearchTree tree_;
  search::SearchResult last_search_;
  bool has_last_search_ = false;

  void reset_tree();
};

int run_web_server(const WebPlayConfig& config);

}  // namespace c4zero::web
