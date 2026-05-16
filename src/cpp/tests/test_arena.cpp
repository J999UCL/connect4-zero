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

  c4zero::arena::ArenaConfig bad_promotion_threshold;
  bad_promotion_threshold.model_a = "unused-a.ts";
  bad_promotion_threshold.model_b = "unused-b.ts";
  bad_promotion_threshold.promotion_threshold = 1.5;
  threw = false;
  try {
    (void)c4zero::arena::play_checkpoint_match(bad_promotion_threshold);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  C4ZERO_CHECK(threw);

  c4zero::arena::ArenaConfig bad_workers;
  bad_workers.model_a = "unused-a.ts";
  bad_workers.model_b = "unused-b.ts";
  bad_workers.arena_workers = 0;
  threw = false;
  try {
    (void)c4zero::arena::play_checkpoint_match(bad_workers);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  C4ZERO_CHECK(threw);

  c4zero::arena::ArenaConfig bad_games_per_opening;
  bad_games_per_opening.model_a = "unused-a.ts";
  bad_games_per_opening.model_b = "unused-b.ts";
  bad_games_per_opening.games_per_opening = 0;
  threw = false;
  try {
    (void)c4zero::arena::play_checkpoint_match(bad_games_per_opening);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  C4ZERO_CHECK(threw);

  c4zero::arena::ArenaConfig ambiguous_side;
  ambiguous_side.model_a = "unused-a.ts";
  ambiguous_side.bot_a = "minimax3";
  ambiguous_side.model_b = "unused-b.ts";
  ambiguous_side.games = 0;
  threw = false;
  try {
    (void)c4zero::arena::play_checkpoint_match(ambiguous_side);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  C4ZERO_CHECK(threw);

  c4zero::arena::ArenaResult promoted;
  promoted.games = 20;
  promoted.model_a_wins = 11;
  promoted.promotion_threshold = c4zero::arena::kDefaultPromotionThreshold;
  C4ZERO_CHECK(promoted.model_a_promoted());
  C4ZERO_CHECK(promoted.summary().find("promotion_threshold=0.55") != std::string::npos);
  C4ZERO_CHECK(promoted.summary().find("promote_model_a=1") != std::string::npos);

  c4zero::arena::ArenaResult not_promoted;
  not_promoted.games = 20;
  not_promoted.model_a_wins = 10;
  not_promoted.promotion_threshold = c4zero::arena::kDefaultPromotionThreshold;
  C4ZERO_CHECK(!not_promoted.model_a_promoted());
  C4ZERO_CHECK(not_promoted.summary().find("promote_model_a=0") != std::string::npos);

  c4zero::arena::ArenaConfig bot_config;
  bot_config.bot_a = "oracle-d2";
  bot_config.bot_b = "one-ply-tactical";
  bot_config.games = 8;
  bot_config.simulations = 1;
  bot_config.opening_count = 2;
  bot_config.games_per_opening = 4;
  bot_config.opening_plies = 2;
  bot_config.arena_workers = 2;
  const auto bot_result = c4zero::arena::play_checkpoint_match(bot_config);
  C4ZERO_CHECK_EQ(bot_result.games, 8);
  C4ZERO_CHECK_EQ(bot_result.model_a_wins + bot_result.model_b_wins + bot_result.draws, 8);
  C4ZERO_CHECK(bot_result.total_plies > 0);
  C4ZERO_CHECK(!bot_result.root_noise);
  C4ZERO_CHECK_EQ(bot_result.opening_count, 2);
  C4ZERO_CHECK_EQ(bot_result.games_per_opening, 4);
  C4ZERO_CHECK_EQ(bot_result.opening_plies, 2);
  C4ZERO_CHECK_EQ(bot_result.arena_workers, 2);
  C4ZERO_CHECK(bot_result.summary().find("player_a=bot:oracle-d2") != std::string::npos);
  C4ZERO_CHECK(bot_result.summary().find("player_b=bot:one-ply-tactical") != std::string::npos);
  C4ZERO_CHECK(bot_result.summary().find("root_noise=0") != std::string::npos);
  C4ZERO_CHECK(bot_result.summary().find("opening_count=2") != std::string::npos);

  const char* fixture = std::getenv("C4ZERO_TORCHSCRIPT_FIXTURE");
  if (fixture == nullptr || std::string(fixture).empty()) {
    std::cout << "C4ZERO_TORCHSCRIPT_FIXTURE unset; skipping optional arena checkpoint fixture test\n";
    return 0;
  }

  c4zero::arena::ArenaConfig config;
  config.model_a = fixture;
  config.model_b = fixture;
  config.games = 4;
  config.simulations = 1;
  config.arena_workers = 2;
  config.opening_count = 1;
  const auto result = c4zero::arena::play_checkpoint_match(config);
  C4ZERO_CHECK_EQ(result.games, 4);
  C4ZERO_CHECK_EQ(result.model_a_wins + result.model_b_wins + result.draws, 4);
  C4ZERO_CHECK(result.total_plies > 0);
  C4ZERO_CHECK(!result.root_noise);
  return 0;
}
