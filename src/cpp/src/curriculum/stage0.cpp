#include "c4zero/curriculum/stage0.hpp"

#include "c4zero/bots/heuristic.hpp"
#include "c4zero/core/symmetry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace c4zero::curriculum {
namespace {

constexpr std::uint32_t kPseudoVisitCount = 256;

const std::array<std::pair<Stage0Category, std::uint64_t>, 7>& category_weights() {
  static const std::array<std::pair<Stage0Category, std::uint64_t>, 7> weights{{
      {Stage0Category::ImmediateWin, 150},
      {Stage0Category::ImmediateBlock, 150},
      {Stage0Category::SafeMoveVsBlunder, 150},
      {Stage0Category::PlayableVsFloatingThreat, 100},
      {Stage0Category::ForkCreate, 150},
      {Stage0Category::ForkBlock, 150},
      {Stage0Category::MinimaxDepth3Policy, 150},
  }};
  return weights;
}

core::Position opponent_to_move_view(const core::Position& position) {
  core::Position view;
  view.current = position.opponent;
  view.opponent = position.current;
  view.heights = position.heights;
  view.ply = position.ply;
  return view;
}

bool contains_action(const std::vector<core::Action>& actions, core::Action action) {
  return std::find(actions.begin(), actions.end(), action) != actions.end();
}

std::vector<core::Action> first_center_ordered(const std::vector<core::Action>& actions) {
  std::vector<core::Action> ordered;
  for (core::Action action : bots::center_order()) {
    if (contains_action(actions, action)) {
      ordered.push_back(action);
    }
  }
  return ordered;
}

core::Action first_target_action(const std::vector<core::Action>& targets) {
  const auto ordered = first_center_ordered(targets);
  if (ordered.empty()) {
    throw std::runtime_error("curriculum sample has no target action");
  }
  return ordered.front();
}

std::vector<core::Action> safe_actions(const core::Position& position) {
  std::vector<core::Action> out;
  for (core::Action action : bots::ordered_legal_actions(position)) {
    if (bots::opponent_winning_replies_after(position, action) == 0) {
      out.push_back(action);
    }
  }
  return out;
}

std::vector<core::Action> unsafe_actions(const core::Position& position) {
  std::vector<core::Action> out;
  for (core::Action action : bots::ordered_legal_actions(position)) {
    if (bots::opponent_winning_replies_after(position, action) > 0) {
      out.push_back(action);
    }
  }
  return out;
}

std::vector<core::Action> opponent_floating_threat_columns(const core::Position& position) {
  std::vector<core::Action> out;
  const core::Bitboard occupied = position.occupancy();
  for (core::Bitboard mask : core::winning_masks()) {
    const core::Bitboard opponent_bits = mask & position.opponent;
    const core::Bitboard current_bits = mask & position.current;
    if (current_bits != 0) {
      continue;
    }
#if defined(__GNUG__) || defined(__clang__)
    if (__builtin_popcountll(opponent_bits) != 3) {
      continue;
    }
#else
    int count = 0;
    core::Bitboard cursor = opponent_bits;
    while (cursor != 0) {
      cursor &= cursor - 1;
      ++count;
    }
    if (count != 3) {
      continue;
    }
#endif
    const core::Bitboard empty = mask & ~occupied;
    if (empty == 0 || (empty & (empty - 1)) != 0) {
      continue;
    }
    for (core::Action action = 0; action < core::kNumActions; ++action) {
      for (int z = 0; z < core::kBoardSize; ++z) {
        if ((empty & core::cell_mask(action, z)) == 0) {
          continue;
        }
        if (position.heights[action] < z) {
          out.push_back(action);
        }
      }
    }
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return first_center_ordered(out);
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

core::Position random_nonterminal_position(std::mt19937_64& rng) {
  std::uniform_int_distribution<int> ply_dist(4, 28);
  for (int attempt = 0; attempt < 200; ++attempt) {
    core::Position position = core::Position::empty();
    const int target_plies = ply_dist(rng);
    bool failed = false;
    for (int ply = 0; ply < target_plies; ++ply) {
      const auto legal = bots::ordered_legal_actions(position);
      if (legal.empty() || position.is_terminal()) {
        failed = true;
        break;
      }
      std::uniform_int_distribution<std::size_t> action_dist(0, legal.size() - 1);
      const core::Action action = legal[action_dist(rng)];
      const core::Position next = position.play(action);
      if (next.is_terminal() && ply + 1 < target_plies) {
        failed = true;
        break;
      }
      position = next;
    }
    if (!failed && !position.is_terminal() && position.legal_mask() != 0) {
      return position;
    }
  }
  return core::Position::empty();
}

std::vector<core::Action> targets_for_category(const core::Position& position, Stage0Category category) {
  if (position.is_terminal() || position.legal_mask() == 0) {
    return {};
  }

  const auto current_wins = bots::immediate_winning_actions(position);
  const core::Position opponent_view = opponent_to_move_view(position);
  const auto opponent_wins = bots::immediate_winning_actions(opponent_view);

  switch (category) {
    case Stage0Category::ImmediateWin:
      return first_center_ordered(current_wins);

    case Stage0Category::ImmediateBlock: {
      if (!current_wins.empty() || opponent_wins.size() != 1) {
        return {};
      }
      const core::Action block = opponent_wins.front();
      if (position.is_legal(block) && bots::opponent_winning_replies_after(position, block) == 0) {
        return {block};
      }
      return {};
    }

    case Stage0Category::SafeMoveVsBlunder: {
      if (!current_wins.empty() || !opponent_wins.empty()) {
        return {};
      }
      const auto safe = safe_actions(position);
      const auto unsafe = unsafe_actions(position);
      return !safe.empty() && !unsafe.empty() ? safe : std::vector<core::Action>{};
    }

    case Stage0Category::PlayableVsFloatingThreat: {
      if (!current_wins.empty() || !opponent_wins.empty()) {
        return {};
      }
      const auto floating = opponent_floating_threat_columns(position);
      if (floating.empty()) {
        return {};
      }
      std::vector<core::Action> targets;
      for (core::Action action : safe_actions(position)) {
        if (!contains_action(floating, action)) {
          targets.push_back(action);
        }
      }
      return !targets.empty() ? first_center_ordered(targets) : std::vector<core::Action>{};
    }

    case Stage0Category::ForkCreate: {
      if (!current_wins.empty() || !opponent_wins.empty()) {
        return {};
      }
      std::vector<core::Action> targets;
      for (core::Action action : safe_actions(position)) {
        if (bots::playable_threat_count_after(position, action) >= 2) {
          targets.push_back(action);
        }
      }
      return first_center_ordered(targets);
    }

    case Stage0Category::ForkBlock: {
      if (!current_wins.empty()) {
        return {};
      }
      int best = std::numeric_limits<int>::max();
      int worst = 0;
      std::vector<std::pair<core::Action, int>> scored;
      for (core::Action action : bots::ordered_legal_actions(position)) {
        const int count = opponent_fork_reply_count_after(position, action);
        scored.push_back({action, count});
        best = std::min(best, count);
        worst = std::max(worst, count);
      }
      if (scored.empty() || best >= worst || best > 0) {
        return {};
      }
      std::vector<core::Action> targets;
      for (const auto& [action, count] : scored) {
        if (count == best && bots::opponent_winning_replies_after(position, action) == 0) {
          targets.push_back(action);
        }
      }
      return first_center_ordered(targets);
    }

    case Stage0Category::MinimaxDepth3Policy:
      return {bots::DepthLimitedMinimaxBot(3).select_move(position)};
  }
  return {};
}

data::SelfPlaySample make_sample(
    const core::Position& position,
    const std::vector<core::Action>& targets,
    std::uint64_t sample_id) {
  if (targets.empty()) {
    throw std::runtime_error("cannot make curriculum sample without targets");
  }
  std::array<float, core::kNumActions> policy{};
  std::array<std::uint32_t, core::kNumActions> visits{};
  const float probability = 1.0f / static_cast<float>(targets.size());
  const std::uint32_t base_visits = kPseudoVisitCount / static_cast<std::uint32_t>(targets.size());
  std::uint32_t remainder = kPseudoVisitCount - base_visits * static_cast<std::uint32_t>(targets.size());
  const auto ordered_targets = first_center_ordered(targets);
  for (core::Action action : ordered_targets) {
    if (!position.is_legal(action)) {
      throw std::runtime_error("curriculum target action is illegal");
    }
    policy[action] = probability;
    visits[action] = base_visits + (remainder > 0 ? 1 : 0);
    if (remainder > 0) {
      --remainder;
    }
  }
  return data::SelfPlaySample::from_position(
      position,
      policy,
      visits,
      0.0f,
      first_target_action(targets),
      sample_id);
}

void validate_sample(const data::SelfPlaySample& sample) {
  float policy_sum = 0.0f;
  std::uint32_t visit_sum = 0;
  for (core::Action action = 0; action < core::kNumActions; ++action) {
    const bool legal = (sample.legal_mask & (1u << action)) != 0;
    if (!legal && (sample.policy[action] != 0.0f || sample.visit_counts[action] != 0)) {
      throw std::runtime_error("curriculum sample gives mass to an illegal action");
    }
    policy_sum += sample.policy[action];
    visit_sum += sample.visit_counts[action];
  }
  if (std::fabs(policy_sum - 1.0f) > 1e-5f) {
    throw std::runtime_error("curriculum sample policy does not sum to one");
  }
  if (visit_sum != kPseudoVisitCount) {
    throw std::runtime_error("curriculum sample pseudo visits do not sum to 256");
  }
  if ((sample.legal_mask & (1u << sample.action)) == 0 || sample.policy[sample.action] <= 0.0f) {
    throw std::runtime_error("curriculum sample action is not a legal target");
  }
}

data::SelfPlaySample transform_sample(
    const data::SelfPlaySample& sample,
    core::Symmetry symmetry,
    std::uint64_t sample_id) {
  core::Position position;
  position.current = sample.current_bits;
  position.opponent = sample.opponent_bits;
  position.heights = sample.heights;
  position.ply = sample.ply;
  const core::Position transformed = core::transform(position, symmetry);
  const auto permutation = core::action_permutation(symmetry);
  auto policy = core::transform_policy(sample.policy, symmetry);
  auto visits = core::transform_visits(sample.visit_counts, symmetry);
  const core::Action action = permutation[sample.action];
  auto transformed_sample = data::SelfPlaySample::from_position(
      transformed,
      policy,
      visits,
      sample.value,
      action,
      sample_id);
  validate_sample(transformed_sample);
  return transformed_sample;
}

std::map<Stage0Category, std::uint64_t> target_counts(std::uint64_t samples, bool use_symmetries) {
  std::map<Stage0Category, std::uint64_t> counts;
  std::uint64_t assigned = 0;
  for (std::size_t index = 0; index < category_weights().size(); ++index) {
    const auto [category, weight] = category_weights()[index];
    std::uint64_t count = samples * weight / 1000;
    if (index + 1 == category_weights().size()) {
      count = samples - assigned;
    }
    if (use_symmetries && count % 8 != 0) {
      throw std::invalid_argument("Stage 0 sample count must make every category count divisible by 8 when symmetries are enabled");
    }
    counts[category] = count;
    assigned += count;
  }
  return counts;
}

}  // namespace

std::string category_name(Stage0Category category) {
  switch (category) {
    case Stage0Category::ImmediateWin:
      return "immediate_win";
    case Stage0Category::ImmediateBlock:
      return "immediate_block";
    case Stage0Category::SafeMoveVsBlunder:
      return "safe_move_vs_blunder";
    case Stage0Category::PlayableVsFloatingThreat:
      return "playable_vs_floating_threat";
    case Stage0Category::ForkCreate:
      return "fork_create";
    case Stage0Category::ForkBlock:
      return "fork_block";
    case Stage0Category::MinimaxDepth3Policy:
      return "minimax_depth3_policy";
  }
  return "unknown";
}

LabeledSample generate_stage0_sample(
    Stage0Category category,
    std::mt19937_64& rng,
    std::uint64_t sample_id) {
  constexpr int kMaxAttempts = 2'000'000;
  for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
    const core::Position position = random_nonterminal_position(rng);
    const auto targets = targets_for_category(position, category);
    if (targets.empty()) {
      continue;
    }
    auto sample = make_sample(position, targets, sample_id);
    validate_sample(sample);
    return LabeledSample{sample, category};
  }
  throw std::runtime_error("failed to generate Stage 0 sample for category: " + category_name(category));
}

std::vector<LabeledSample> generate_stage0_samples(const Stage0Config& config) {
  if (config.samples == 0) {
    throw std::invalid_argument("Stage 0 samples must be positive");
  }
  if (config.shard_size == 0) {
    throw std::invalid_argument("Stage 0 shard size must be positive");
  }

  std::mt19937_64 rng(config.seed);
  const auto counts = target_counts(config.samples, config.use_symmetries);
  std::vector<LabeledSample> samples;
  samples.reserve(static_cast<std::size_t>(config.samples));
  std::uint64_t sample_id = 0;

  for (const auto& [category, count] : counts) {
    const std::uint64_t base_count = config.use_symmetries ? count / 8 : count;
    for (std::uint64_t base = 0; base < base_count; ++base) {
      auto labeled = generate_stage0_sample(category, rng, sample_id);
      if (!config.use_symmetries) {
        samples.push_back(labeled);
        ++sample_id;
        continue;
      }
      for (int symmetry_index = 0; symmetry_index < 8; ++symmetry_index) {
        auto transformed = transform_sample(
            labeled.sample,
            static_cast<core::Symmetry>(symmetry_index),
            sample_id);
        samples.push_back(LabeledSample{transformed, category});
        ++sample_id;
      }
    }
  }
  if (samples.size() != config.samples) {
    throw std::runtime_error("Stage 0 generated sample count mismatch");
  }
  return samples;
}

Stage0Result write_stage0_dataset(const Stage0Config& config) {
  const auto samples = generate_stage0_samples(config);
  const std::filesystem::path root(config.output_dir);
  const std::filesystem::path shard_root = root / "shards";
  std::filesystem::create_directories(shard_root);

  std::vector<std::string> shard_paths;
  std::map<std::string, std::uint64_t> category_counts;
  std::uint64_t written = 0;
  int shard_index = 0;
  while (written < samples.size()) {
    const std::uint64_t count = std::min<std::uint64_t>(config.shard_size, samples.size() - written);
    std::vector<data::SelfPlaySample> shard;
    shard.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t offset = 0; offset < count; ++offset) {
      const auto& labeled = samples[static_cast<std::size_t>(written + offset)];
      shard.push_back(labeled.sample);
      category_counts[category_name(labeled.category)] += 1;
    }

    char filename[64];
    std::snprintf(filename, sizeof(filename), "shard-%06d.c4az", shard_index);
    const std::string relative = std::string("shards/") + filename;
    data::write_shard((root / relative).string(), shard);
    shard_paths.push_back(relative);
    written += count;
    ++shard_index;
  }

  data::SelfPlayManifestConfig manifest_config;
  manifest_config.dataset_kind = "curriculum_stage0";
  manifest_config.model_checkpoint = "none";
  manifest_config.device = "cpu";
  manifest_config.curriculum_stage = 0;
  manifest_config.policy_source = "tactical_rules_and_minimax";
  manifest_config.value_source = "unused_policy_only";
  manifest_config.value_loss_weight = 0.0f;
  manifest_config.shard_size = static_cast<int>(config.shard_size);
  manifest_config.symmetry_augmentation = config.use_symmetries;
  manifest_config.category_counts = category_counts;
  manifest_config.simulations_per_move = static_cast<int>(kPseudoVisitCount);
  manifest_config.add_root_noise = false;
  manifest_config.evaluator_type = "curriculum_stage0";
  manifest_config.seed = config.seed;
  manifest_config.git_commit = config.git_commit;

  const std::string manifest_path = (root / "manifest.json").string();
  data::write_manifest(
      manifest_path,
      shard_paths,
      config.samples,
      config.samples,
      manifest_config);

  return Stage0Result{
      config.samples,
      static_cast<std::uint64_t>(shard_paths.size()),
      category_counts,
      manifest_path,
  };
}

}  // namespace c4zero::curriculum
