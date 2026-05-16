#include "c4zero/bots/heuristic.hpp"
#include "test_support.hpp"

using namespace c4zero;

class IllegalBot final : public bots::Bot {
 public:
  core::Action select_move(const core::Position&) const override {
    return core::kNumActions;
  }

  std::string name() const override {
    return "illegal";
  }
};

int main() {
  const auto empty = core::Position::empty();
  C4ZERO_CHECK_EQ(bots::FirstLegalBot{}.select_move(empty), 0);
  C4ZERO_CHECK_EQ(bots::CenterOrderBot{}.select_move(empty), 5);

  const auto win = core::from_actions({0, 4, 1, 5, 2, 6});
  C4ZERO_CHECK_EQ(bots::OnePlyTacticalBot{}.select_move(win), 3);
  C4ZERO_CHECK_EQ(bots::LineScoreBot{}.select_move(win), 3);
  C4ZERO_CHECK_EQ(bots::ForkThreatBot{}.select_move(win), 3);
  C4ZERO_CHECK_EQ(bots::OracleBot(4).select_move(win), 3);

  const auto vertical = core::from_actions({0, 1, 0, 1, 0, 2});
  C4ZERO_CHECK_EQ(bots::OnePlyTacticalBot{}.select_move(vertical), 0);

  const auto block = core::from_actions({4, 0, 5, 1, 8, 2});
  C4ZERO_CHECK_EQ(bots::OnePlyTacticalBot{}.select_move(block), 3);
  C4ZERO_CHECK_EQ(bots::LineScoreBot{}.select_move(block), 3);
  C4ZERO_CHECK_EQ(bots::ForkThreatBot{}.select_move(block), 3);
  C4ZERO_CHECK_EQ(bots::DepthLimitedMinimaxBot(3).select_move(block), 3);
  C4ZERO_CHECK_EQ(bots::make_bot("oracle-d4")->select_move(block), 3);
  C4ZERO_CHECK_EQ(bots::make_bot("oracle_d4")->select_move(block), 3);

  const auto match = bots::play_bot_match(bots::FirstLegalBot{}, bots::CenterOrderBot{}, 2, true);
  C4ZERO_CHECK_EQ(match.games, 2);
  C4ZERO_CHECK(match.total_plies > 0);

  const auto ladder = bots::play_bot_match(
      bots::OracleBot(3),
      bots::OnePlyTacticalBot{},
      4,
      true);
  C4ZERO_CHECK_EQ(ladder.games, 4);
  C4ZERO_CHECK_EQ(ladder.first_wins + ladder.second_wins + ladder.draws, 4);
  C4ZERO_CHECK(ladder.first_score_rate() >= 0.5);

  bool illegal_threw = false;
  try {
    (void)bots::play_bot_match(IllegalBot{}, bots::CenterOrderBot{}, 1, true);
  } catch (const std::runtime_error&) {
    illegal_threw = true;
  }
  C4ZERO_CHECK(illegal_threw);

  bool unknown_threw = false;
  try {
    (void)bots::make_bot("does-not-exist");
  } catch (const std::invalid_argument&) {
    unknown_threw = true;
  }
  C4ZERO_CHECK(unknown_threw);

  for (const auto& name : bots::bot_names()) {
    auto bot = bots::make_bot(name);
    if (name == "oracle-d6" || name == "oracle-d8" || name == "oracle-d16") {
      continue;
    }
    C4ZERO_CHECK(empty.is_legal(bot->select_move(empty)));
  }
  return 0;
}
