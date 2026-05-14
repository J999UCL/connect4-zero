#include "c4zero/arena/arena.hpp"
#include "test_support.hpp"

#include <cstdlib>
#include <iostream>

int main() {
  c4zero::arena::ArenaConfig invalid;
  bool threw = false;
  try {
    (void)c4zero::arena::play_checkpoint_match(invalid);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  C4ZERO_CHECK(threw);

  const char* fixture = std::getenv("C4ZERO_TORCHSCRIPT_FIXTURE");
  if (fixture == nullptr || std::string(fixture).empty()) {
    std::cout << "C4ZERO_TORCHSCRIPT_FIXTURE unset; skipping optional arena checkpoint fixture test\n";
    return 0;
  }

  c4zero::arena::ArenaConfig config;
  config.model_a = fixture;
  config.model_b = fixture;
  config.games = 2;
  config.simulations = 1;
  const auto result = c4zero::arena::play_checkpoint_match(config);
  C4ZERO_CHECK_EQ(result.games, 2);
  C4ZERO_CHECK_EQ(result.model_a_wins + result.model_b_wins + result.draws, 2);
  C4ZERO_CHECK(result.total_plies > 0);
  return 0;
}
