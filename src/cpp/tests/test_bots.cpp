#include "c4zero/bots/heuristic.hpp"
#include "test_support.hpp"

using namespace c4zero;

int main() {
  const auto empty = core::Position::empty();
  C4ZERO_CHECK_EQ(bots::FirstLegalBot{}.select_move(empty), 0);
  C4ZERO_CHECK_EQ(bots::CenterOrderBot{}.select_move(empty), 5);

  const auto win = core::from_actions({0, 4, 1, 5, 2, 6});
  C4ZERO_CHECK_EQ(bots::OnePlyTacticalBot{}.select_move(win), 3);
  C4ZERO_CHECK_EQ(bots::LineScoreBot{}.select_move(win), 3);
  C4ZERO_CHECK_EQ(bots::ForkThreatBot{}.select_move(win), 3);

  const auto vertical = core::from_actions({0, 1, 0, 1, 0, 2});
  C4ZERO_CHECK_EQ(bots::OnePlyTacticalBot{}.select_move(vertical), 0);

  const auto block = core::from_actions({4, 0, 5, 1, 8, 2});
  C4ZERO_CHECK_EQ(bots::OnePlyTacticalBot{}.select_move(block), 3);
  C4ZERO_CHECK_EQ(bots::LineScoreBot{}.select_move(block), 3);
  C4ZERO_CHECK_EQ(bots::ForkThreatBot{}.select_move(block), 3);
  C4ZERO_CHECK_EQ(bots::DepthLimitedMinimaxBot(3).select_move(block), 3);

  const auto match = bots::play_bot_match(bots::FirstLegalBot{}, bots::CenterOrderBot{}, 2, true);
  C4ZERO_CHECK_EQ(match.games, 2);
  C4ZERO_CHECK(match.total_plies > 0);

  for (const auto& name : bots::bot_names()) {
    auto bot = bots::make_bot(name);
    C4ZERO_CHECK(empty.is_legal(bot->select_move(empty)));
  }
  return 0;
}
