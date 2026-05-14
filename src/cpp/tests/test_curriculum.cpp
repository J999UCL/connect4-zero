#include "c4zero/bots/heuristic.hpp"
#include "c4zero/core/symmetry.hpp"
#include "c4zero/curriculum/stage0.hpp"
#include "c4zero/data/shard.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace c4zero;

namespace {

core::Position sample_position(const data::SelfPlaySample& sample) {
  core::Position position;
  position.current = sample.current_bits;
  position.opponent = sample.opponent_bits;
  position.heights = sample.heights;
  position.ply = sample.ply;
  return position;
}

core::Position opponent_to_move_view(const core::Position& position) {
  core::Position view;
  view.current = position.opponent;
  view.opponent = position.current;
  view.heights = position.heights;
  view.ply = position.ply;
  return view;
}

int count_bits(core::Bitboard bits) {
#if defined(__GNUG__) || defined(__clang__)
  return __builtin_popcountll(bits);
#else
  int count = 0;
  while (bits != 0) {
    bits &= bits - 1;
    ++count;
  }
  return count;
#endif
}

std::vector<core::Action> policy_targets(const data::SelfPlaySample& sample) {
  std::vector<core::Action> targets;
  for (core::Action action = 0; action < core::kNumActions; ++action) {
    if (sample.policy[action] > 0.0f) {
      targets.push_back(action);
    }
  }
  return targets;
}

std::vector<core::Action> floating_threat_columns(const core::Position& position) {
  std::vector<core::Action> out;
  const core::Bitboard occupied = position.occupancy();
  for (core::Bitboard mask : core::winning_masks()) {
    if ((mask & position.current) != 0 || count_bits(mask & position.opponent) != 3) {
      continue;
    }
    const core::Bitboard empty = mask & ~occupied;
    if (empty == 0 || (empty & (empty - 1)) != 0) {
      continue;
    }
    for (core::Action action = 0; action < core::kNumActions; ++action) {
      for (int z = 0; z < core::kBoardSize; ++z) {
        if ((empty & core::cell_mask(action, z)) != 0 && position.heights[action] < z) {
          out.push_back(action);
        }
      }
    }
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

int opponent_fork_reply_count_after(const core::Position& position, core::Action action) {
  if (!position.is_legal(action)) {
    return std::numeric_limits<int>::max() / 4;
  }
  const core::Position child = position.play(action);
  int count = 0;
  for (core::Action reply : bots::ordered_legal_actions(child)) {
    if (bots::playable_threat_count_after(child, reply) >= 2) {
      ++count;
    }
  }
  return count;
}

bool basic_sample_invariants(const data::SelfPlaySample& sample) {
  const core::Position position = sample_position(sample);
  if (position.is_terminal() || position.legal_mask() == 0 || sample.legal_mask != position.legal_mask()) {
    return false;
  }
  float policy_sum = 0.0f;
  std::uint32_t visit_sum = 0;
  for (core::Action action = 0; action < core::kNumActions; ++action) {
    const bool legal = position.is_legal(action);
    if (!legal && (sample.policy[action] != 0.0f || sample.visit_counts[action] != 0)) {
      return false;
    }
    if (sample.policy[action] == 0.0f && sample.visit_counts[action] != 0) {
      return false;
    }
    policy_sum += sample.policy[action];
    visit_sum += sample.visit_counts[action];
  }
  return std::fabs(policy_sum - 1.0f) < 1e-5f &&
         visit_sum == 256 &&
         sample.value == 0.0f &&
         position.is_legal(sample.action) &&
         sample.policy[sample.action] > 0.0f;
}

bool category_invariants(const curriculum::LabeledSample& labeled) {
  const auto& sample = labeled.sample;
  const core::Position position = sample_position(sample);
  const std::vector<core::Action> targets = policy_targets(sample);
  if (targets.empty()) {
    return false;
  }

  switch (labeled.category) {
    case curriculum::Stage0Category::ImmediateWin:
      return std::all_of(targets.begin(), targets.end(), [&](core::Action action) {
        return bots::is_immediate_win(position, action);
      });

    case curriculum::Stage0Category::ImmediateBlock: {
      const auto opponent_wins = bots::immediate_winning_actions(opponent_to_move_view(position));
      return opponent_wins.size() == 1 &&
             std::all_of(targets.begin(), targets.end(), [&](core::Action action) {
               return action == opponent_wins.front() &&
                      bots::opponent_winning_replies_after(position, action) == 0;
             });
    }

    case curriculum::Stage0Category::SafeMoveVsBlunder: {
      bool has_unsafe = false;
      for (core::Action action : bots::ordered_legal_actions(position)) {
        has_unsafe = has_unsafe || bots::opponent_winning_replies_after(position, action) > 0;
      }
      return has_unsafe && std::all_of(targets.begin(), targets.end(), [&](core::Action action) {
        return bots::opponent_winning_replies_after(position, action) == 0;
      });
    }

    case curriculum::Stage0Category::PlayableVsFloatingThreat: {
      const auto floating = floating_threat_columns(position);
      if (floating.empty()) {
        return false;
      }
      return std::all_of(targets.begin(), targets.end(), [&](core::Action action) {
        return std::find(floating.begin(), floating.end(), action) == floating.end() &&
               bots::opponent_winning_replies_after(position, action) == 0;
      });
    }

    case curriculum::Stage0Category::ForkCreate:
      return std::all_of(targets.begin(), targets.end(), [&](core::Action action) {
        return bots::opponent_winning_replies_after(position, action) == 0 &&
               bots::playable_threat_count_after(position, action) >= 2;
      });

    case curriculum::Stage0Category::ForkBlock: {
      int best = std::numeric_limits<int>::max();
      for (core::Action action : bots::ordered_legal_actions(position)) {
        best = std::min(best, opponent_fork_reply_count_after(position, action));
      }
      return best == 0 && std::all_of(targets.begin(), targets.end(), [&](core::Action action) {
        return opponent_fork_reply_count_after(position, action) == best &&
               bots::opponent_winning_replies_after(position, action) == 0;
      });
    }

    case curriculum::Stage0Category::MinimaxDepth3Policy: {
      if (targets.size() != 1) {
        return false;
      }
      return targets.front() == bots::DepthLimitedMinimaxBot(3).select_move(position);
    }
  }
  return false;
}

}  // namespace

int main() {
  std::mt19937_64 rng(7);
  const std::array<curriculum::Stage0Category, 7> categories{{
      curriculum::Stage0Category::ImmediateWin,
      curriculum::Stage0Category::ImmediateBlock,
      curriculum::Stage0Category::SafeMoveVsBlunder,
      curriculum::Stage0Category::PlayableVsFloatingThreat,
      curriculum::Stage0Category::ForkCreate,
      curriculum::Stage0Category::ForkBlock,
      curriculum::Stage0Category::MinimaxDepth3Policy,
  }};
  for (std::size_t index = 0; index < categories.size(); ++index) {
    const auto labeled = curriculum::generate_stage0_sample(categories[index], rng, index);
    C4ZERO_CHECK(basic_sample_invariants(labeled.sample));
    C4ZERO_CHECK(category_invariants(labeled));
  }

  curriculum::Stage0Config config;
  config.samples = 56;
  config.shard_size = 20;
  config.seed = 3;
  config.use_symmetries = false;
  config.git_commit = "test";
  config.output_dir = (std::filesystem::temp_directory_path() / "c4zero-stage0-test").string();
  std::filesystem::remove_all(config.output_dir);
  const auto result = curriculum::write_stage0_dataset(config);
  C4ZERO_CHECK_EQ(result.samples, 56);
  C4ZERO_CHECK_EQ(result.shards, 3);
  C4ZERO_CHECK_EQ(result.category_counts.at("immediate_win"), 8);
  C4ZERO_CHECK_EQ(result.category_counts.at("playable_vs_floating_threat"), 5);

  std::size_t shard_sample_count = 0;
  for (int shard_index = 0; shard_index < 3; ++shard_index) {
    char filename[64];
    std::snprintf(filename, sizeof(filename), "shards/shard-%06d.c4az", shard_index);
    const auto shard_path = std::filesystem::path(config.output_dir) / filename;
    const auto samples = data::read_shard(shard_path.string());
    shard_sample_count += samples.size();
    for (const auto& sample : samples) {
      C4ZERO_CHECK(basic_sample_invariants(sample));
    }
  }
  C4ZERO_CHECK_EQ(shard_sample_count, 56);

  std::ifstream manifest(result.manifest_path);
  const std::string text((std::istreambuf_iterator<char>(manifest)), std::istreambuf_iterator<char>());
  C4ZERO_CHECK(text.find("\"dataset_kind\": \"curriculum_stage0\"") != std::string::npos);
  C4ZERO_CHECK(text.find("\"policy_source\": \"tactical_rules_and_minimax\"") != std::string::npos);
  C4ZERO_CHECK(text.find("\"value_loss_weight\": 0") != std::string::npos);
  C4ZERO_CHECK(text.find("\"category_counts\"") != std::string::npos);
  return 0;
}
