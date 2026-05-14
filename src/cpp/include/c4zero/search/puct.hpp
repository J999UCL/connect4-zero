#pragma once

#include "c4zero/core/position.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>

namespace c4zero::search {

struct Evaluation {
  std::array<float, core::kNumActions> priors{};
  float value = 0.0f;
};

class Evaluator {
 public:
  virtual ~Evaluator() = default;
  [[nodiscard]] virtual Evaluation evaluate(const core::Position& position) = 0;
};

class UniformEvaluator final : public Evaluator {
 public:
  [[nodiscard]] Evaluation evaluate(const core::Position& position) override;
};

struct PuctConfig {
  int simulations_per_move = 800;
  int search_threads = 1;
  float virtual_loss = 1.0f;
  double c_base = 19652.0;
  double c_init = 1.25;
  double root_dirichlet_alpha = 0.625;
  double root_exploration_fraction = 0.25;
  int temperature_sampling_plies = 30;
  std::uint64_t seed = 1;
};

struct SearchResult {
  std::array<float, core::kNumActions> policy{};
  std::array<std::uint32_t, core::kNumActions> visit_counts{};
  std::array<float, core::kNumActions> q_values{};
  float root_value = 0.0f;
  core::Action selected_action = -1;
  int max_depth = 0;
  int expanded_nodes = 0;
  int completed_simulations = 0;
  int leaf_evaluations = 0;
  int terminal_evaluations = 0;
  int virtual_loss_waits = 0;
  int pending_eval_waits = 0;
  std::uint32_t root_real_visits = 0;
  double search_time_ms = 0.0;
};

struct EdgeStats {
  float prior = 0.0f;
  std::uint32_t visits = 0;
  float value_sum = 0.0f;
  std::uint32_t virtual_visits = 0;
  float virtual_value_sum = 0.0f;
};

struct Node {
  core::Position position;
  int parent = -1;
  core::Action parent_action = -1;
  std::array<int, core::kNumActions> children{};
  std::array<EdgeStats, core::kNumActions> edges{};
  bool expanded = false;
  bool pending_eval = false;
  bool root_noise_applied = false;
  std::optional<float> terminal_value;
};

class SearchTree {
 public:
  explicit SearchTree(core::Position root);

  [[nodiscard]] int root_index() const;
  [[nodiscard]] const Node& root() const;
  [[nodiscard]] Node& root();
  [[nodiscard]] const std::vector<Node>& nodes() const;
  [[nodiscard]] std::vector<Node>& nodes();
  [[nodiscard]] int node_count() const;
  [[nodiscard]] int max_depth() const;
  [[nodiscard]] bool advance(core::Action action);
  [[nodiscard]] bool has_pending_or_virtual_stats() const;

 private:
  int root_index_ = 0;
  std::vector<Node> nodes_;
};

class PuctMcts {
 public:
  explicit PuctMcts(PuctConfig config = {});

  [[nodiscard]] SearchResult search(SearchTree& tree, Evaluator& evaluator, bool add_root_noise, double temperature);
  [[nodiscard]] SearchTree make_tree(const core::Position& root) const;
  [[nodiscard]] const PuctConfig& config() const;

 private:
  PuctConfig config_;
  std::mt19937_64 rng_;

  float expand_node(SearchTree& tree, int node_index, Evaluator& evaluator);
  void add_root_noise(Node& root);
  [[nodiscard]] core::Action select_action(const Node& node) const;
  [[nodiscard]] SearchResult build_result(
      const SearchTree& tree,
      double temperature,
      const std::array<std::uint32_t, core::kNumActions>& visit_baseline,
      const std::array<float, core::kNumActions>& value_baseline);
};

[[nodiscard]] std::array<float, core::kNumActions> normalize_priors(
    const std::array<float, core::kNumActions>& priors,
    std::uint16_t legal_mask);

[[nodiscard]] core::Action sample_action(
    const std::array<float, core::kNumActions>& policy,
    std::uint16_t legal_mask,
    std::mt19937_64& rng);

}  // namespace c4zero::search
