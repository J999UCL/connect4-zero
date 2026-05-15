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

  c4zero::arena::ArenaConfig negative_games;
  negative_games.model_a = "unused-a.ts";
  negative_games.model_b = "unused-b.ts";
  negative_games.games = -1;
  threw = false;
  try {
    (void)c4zero::arena::play_checkpoint_match(negative_games);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  C4ZERO_CHECK(threw);

  c4zero::arena::ArenaConfig zero_simulations;
  zero_simulations.model_a = "unused-a.ts";
  zero_simulations.model_b = "unused-b.ts";
  zero_simulations.simulations = 0;
  threw = false;
  try {
    (void)c4zero::arena::play_checkpoint_match(zero_simulations);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  C4ZERO_CHECK(threw);

  c4zero::arena::ArenaConfig invalid_device;
  invalid_device.model_a = "unused-a.ts";
  invalid_device.model_b = "unused-b.ts";
  invalid_device.games = 0;
  invalid_device.device = "gpu";
  threw = false;
  try {
    (void)c4zero::arena::play_checkpoint_match(invalid_device);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  C4ZERO_CHECK(threw);

  c4zero::arena::ArenaConfig bad_noise_alpha;
  bad_noise_alpha.model_a = "unused-a.ts";
  bad_noise_alpha.model_b = "unused-b.ts";
  bad_noise_alpha.root_dirichlet_alpha = 0.0;
  threw = false;
  try {
    (void)c4zero::arena::play_checkpoint_match(bad_noise_alpha);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  C4ZERO_CHECK(threw);

  c4zero::arena::ArenaConfig bad_noise_fraction;
  bad_noise_fraction.model_a = "unused-a.ts";
  bad_noise_fraction.model_b = "unused-b.ts";
  bad_noise_fraction.root_exploration_fraction = 1.5;
  threw = false;
  try {
    (void)c4zero::arena::play_checkpoint_match(bad_noise_fraction);
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
  C4ZERO_CHECK(result.root_noise);
  return 0;
}
