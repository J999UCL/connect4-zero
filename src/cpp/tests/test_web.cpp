#include "c4zero/web/server.hpp"

#include <cassert>
#include <stdexcept>
#include <string>

namespace {

bool contains(const std::string& text, const std::string& needle) {
  return text.find(needle) != std::string::npos;
}

void test_web_session_human_and_bot_moves() {
  c4zero::search::UniformEvaluator evaluator;
  c4zero::search::PuctConfig config;
  config.simulations_per_move = 8;
  config.search_threads = 2;
  c4zero::web::WebGameSession session(evaluator, config, false);

  assert(session.human_to_move());
  assert(session.position().ply == 0);
  assert(contains(session.state_json(), "\"legalMask\":65535"));
  assert(contains(session.state_json(), "\"lastSearch\":null"));

  session.play_human_action(5);
  assert(!session.human_to_move());
  assert(session.position().ply == 1);

  session.play_bot_action();
  assert(session.human_to_move());
  assert(session.position().ply == 2);
  assert(session.has_last_search());
  assert(session.last_search().completed_simulations == 8);
  assert(session.last_search().selected_action >= 0);
  assert(contains(session.state_json(), "\"lastSearch\":{"));
}

void test_web_session_rejects_bad_human_moves() {
  c4zero::search::UniformEvaluator evaluator;
  c4zero::search::PuctConfig config;
  config.simulations_per_move = 4;
  c4zero::web::WebGameSession session(evaluator, config, false);

  bool rejected = false;
  try {
    session.play_human_action(-1);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
  assert(session.position().ply == 0);

  session.play_human_action(0);
  rejected = false;
  try {
    session.play_human_action(1);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
  assert(session.position().ply == 1);
}

void test_bot_first_can_move_from_empty_board() {
  c4zero::search::UniformEvaluator evaluator;
  c4zero::search::PuctConfig config;
  config.simulations_per_move = 4;
  c4zero::web::WebGameSession session(evaluator, config, true);

  assert(!session.human_to_move());
  session.play_bot_action();
  assert(session.human_to_move());
  assert(session.position().ply == 1);
}

}  // namespace

int main() {
  test_web_session_human_and_bot_moves();
  test_web_session_rejects_bad_human_moves();
  test_bot_first_can_move_from_empty_board();
  return 0;
}
