#include "c4zero/search/puct.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace c4zero::search {
namespace {

float q_value(const EdgeStats& edge) {
  if (edge.visits == 0) {
    return 0.0f;
  }
  return edge.value_sum / static_cast<float>(edge.visits);
}

std::uint32_t effective_visits(const EdgeStats& edge) {
  return edge.visits + edge.virtual_visits;
}

float effective_q_value(const EdgeStats& edge) {
  const std::uint32_t visits = effective_visits(edge);
  if (visits == 0) {
    return 0.0f;
  }
  return (edge.value_sum + edge.virtual_value_sum) / static_cast<float>(visits);
}

std::uint32_t total_visits(const Node& node) {
  std::uint32_t total = 0;
  for (const auto& edge : node.edges) {
    total += edge.visits;
  }
  return total;
}

std::uint32_t total_effective_visits(const Node& node) {
  std::uint32_t total = 0;
  for (const auto& edge : node.edges) {
    total += effective_visits(edge);
  }
  return total;
}

double exploration_constant(const PuctConfig& config, std::uint32_t parent_visits) {
  return std::log((1.0 + static_cast<double>(parent_visits) + config.c_base) / config.c_base) + config.c_init;
}

std::array<float, core::kNumActions> policy_from_visits(const std::array<std::uint32_t, core::kNumActions>& visits, double temperature) {
  std::array<float, core::kNumActions> policy{};
  std::uint32_t max_visits = 0;
  core::Action best = -1;
  for (core::Action action = 0; action < core::kNumActions; ++action) {
    if (visits[action] > max_visits) {
      max_visits = visits[action];
      best = action;
    }
  }
  if (best < 0 || max_visits == 0) {
    return policy;
  }
  if (temperature <= 1e-8) {
    policy[best] = 1.0f;
    return policy;
  }
  double sum = 0.0;
  for (core::Action action = 0; action < core::kNumActions; ++action) {
    if (visits[action] == 0) {
      continue;
    }
    const double value = std::pow(static_cast<double>(visits[action]), 1.0 / temperature);
    policy[action] = static_cast<float>(value);
    sum += value;
  }
  if (sum > 0.0) {
    for (float& value : policy) {
      value = static_cast<float>(static_cast<double>(value) / sum);
    }
  }
  return policy;
}

struct PathStep {
  int node_index = -1;
  core::Action action = -1;
};

struct Reservation {
  std::vector<PathStep> path;
  int node_index = -1;
  core::Position position;
  bool needs_evaluation = false;
  float value = 0.0f;
};

struct SearchRuntime {
  SearchTree& tree;
  const PuctConfig& config;
  std::mutex mutex;
  std::condition_variable cv;
  int launched_simulations = 0;
  int completed_simulations = 0;
  int leaf_evaluations = 0;
  int terminal_evaluations = 0;
  int virtual_loss_waits = 0;
  int pending_eval_waits = 0;
  bool stop = false;
  std::exception_ptr error;
};

void apply_virtual_loss(SearchTree& tree, const std::vector<PathStep>& path, float virtual_loss) {
  for (const PathStep& step : path) {
    EdgeStats& edge = tree.nodes().at(step.node_index).edges[step.action];
    edge.virtual_visits += 1;
    edge.virtual_value_sum -= virtual_loss;
  }
}

void remove_virtual_loss(SearchTree& tree, const std::vector<PathStep>& path, float virtual_loss) {
  for (const PathStep& step : path) {
    EdgeStats& edge = tree.nodes().at(step.node_index).edges[step.action];
    if (edge.virtual_visits == 0) {
      throw std::runtime_error("virtual loss underflow");
    }
    edge.virtual_visits -= 1;
    edge.virtual_value_sum += virtual_loss;
    if (edge.virtual_visits == 0 && std::fabs(edge.virtual_value_sum) < 1e-6f) {
      edge.virtual_value_sum = 0.0f;
    }
  }
}

void backup_real_value(SearchTree& tree, const std::vector<PathStep>& path, float leaf_value) {
  float value = leaf_value;
  for (auto it = path.rbegin(); it != path.rend(); ++it) {
    value = -value;
    EdgeStats& edge = tree.nodes().at(it->node_index).edges[it->action];
    edge.visits += 1;
    edge.value_sum += value;
  }
}

struct ActionSelection {
  core::Action action = -1;
  bool blocked_by_pending = false;
};

ActionSelection select_effective_action(const SearchTree& tree, int node_index, const PuctConfig& config) {
  const Node& node = tree.nodes().at(node_index);
  const std::uint16_t legal = node.position.legal_mask();
  const std::uint32_t parent_visits = total_effective_visits(node);
  const double c = exploration_constant(config, parent_visits);
  const double sqrt_parent = std::sqrt(static_cast<double>(parent_visits));
  double best_score = -std::numeric_limits<double>::infinity();
  core::Action best_action = -1;
  bool saw_pending = false;
  for (core::Action action = 0; action < core::kNumActions; ++action) {
    if ((legal & (1u << action)) == 0) {
      continue;
    }
    const int child_index = node.children[action];
    if (child_index >= 0 && tree.nodes().at(child_index).pending_eval) {
      saw_pending = true;
      continue;
    }
    const EdgeStats& edge = node.edges[action];
    const double q = effective_q_value(edge);
    const double u = c * static_cast<double>(edge.prior) * sqrt_parent /
        (1.0 + static_cast<double>(effective_visits(edge)));
    const double score = q + u;
    if (score > best_score) {
      best_score = score;
      best_action = action;
    }
  }
  return ActionSelection{best_action, best_action < 0 && saw_pending};
}

bool reserve_simulation_locked(SearchRuntime& runtime, Reservation& reservation) {
  SearchTree& tree = runtime.tree;
  const PuctConfig& config = runtime.config;
  int node_index = tree.root_index();
  std::vector<PathStep> path;

  while (true) {
    Node& node = tree.nodes().at(node_index);
    if (node.terminal_value.has_value()) {
      reservation.path = std::move(path);
      reservation.node_index = node_index;
      reservation.position = node.position;
      reservation.needs_evaluation = false;
      reservation.value = *node.terminal_value;
      apply_virtual_loss(tree, reservation.path, config.virtual_loss);
      return true;
    }

    if (!node.expanded) {
      if (node.pending_eval) {
        runtime.pending_eval_waits += 1;
        return false;
      }
      node.pending_eval = true;
      reservation.path = std::move(path);
      reservation.node_index = node_index;
      reservation.position = node.position;
      reservation.needs_evaluation = true;
      apply_virtual_loss(tree, reservation.path, config.virtual_loss);
      return true;
    }

    const ActionSelection selected = select_effective_action(tree, node_index, config);
    if (selected.action < 0 && selected.blocked_by_pending) {
      runtime.virtual_loss_waits += 1;
      return false;
    }
    const core::Action action = selected.action;
    if (action < 0) {
      reservation.path = std::move(path);
      reservation.node_index = node_index;
      reservation.position = node.position;
      reservation.needs_evaluation = false;
      reservation.value = 0.0f;
      apply_virtual_loss(tree, reservation.path, config.virtual_loss);
      return true;
    }

    int child_index = node.children[action];
    path.push_back(PathStep{node_index, action});
    if (child_index < 0) {
      Node child;
      child.position = node.position.play(action);
      child.parent = node_index;
      child.parent_action = action;
      child.children.fill(-1);
      child.terminal_value = child.position.terminal_value();
      child_index = static_cast<int>(tree.nodes().size());
      tree.nodes().push_back(child);
      tree.nodes().at(node_index).children[action] = child_index;
    }

    Node& child = tree.nodes().at(child_index);
    if (child.terminal_value.has_value()) {
      reservation.path = std::move(path);
      reservation.node_index = child_index;
      reservation.position = child.position;
      reservation.needs_evaluation = false;
      reservation.value = *child.terminal_value;
      apply_virtual_loss(tree, reservation.path, config.virtual_loss);
      return true;
    }
    if (!child.expanded) {
      if (child.pending_eval) {
        runtime.pending_eval_waits += 1;
        return false;
      }
      child.pending_eval = true;
      reservation.path = std::move(path);
      reservation.node_index = child_index;
      reservation.position = child.position;
      reservation.needs_evaluation = true;
      apply_virtual_loss(tree, reservation.path, config.virtual_loss);
      return true;
    }
    node_index = child_index;
  }
}

void complete_simulation_locked(SearchRuntime& runtime, Reservation& reservation, const Evaluation* evaluation) {
  SearchTree& tree = runtime.tree;
  float leaf_value = reservation.value;
  if (reservation.needs_evaluation) {
    Node& node = tree.nodes().at(reservation.node_index);
    if (!node.pending_eval) {
      throw std::runtime_error("completed evaluation for a node that is not pending");
    }
    const auto priors = normalize_priors(evaluation->priors, node.position.legal_mask());
    for (core::Action action = 0; action < core::kNumActions; ++action) {
      node.edges[action].prior = priors[action];
    }
    node.expanded = true;
    node.pending_eval = false;
    leaf_value = evaluation->value;
    runtime.leaf_evaluations += 1;
  } else {
    runtime.terminal_evaluations += 1;
  }
  remove_virtual_loss(tree, reservation.path, runtime.config.virtual_loss);
  backup_real_value(tree, reservation.path, leaf_value);
  runtime.completed_simulations += 1;
}

}  // namespace

Evaluation UniformEvaluator::evaluate(const core::Position& position) {
  Evaluation evaluation;
  evaluation.value = 0.0f;
  evaluation.priors = normalize_priors(evaluation.priors, position.legal_mask());
  return evaluation;
}

SearchTree::SearchTree(core::Position root) {
  Node node;
  node.position = root;
  node.terminal_value = root.terminal_value();
  node.children.fill(-1);
  nodes_.push_back(node);
}

int SearchTree::root_index() const {
  return root_index_;
}

const Node& SearchTree::root() const {
  return nodes_.at(root_index_);
}

Node& SearchTree::root() {
  return nodes_.at(root_index_);
}

const std::vector<Node>& SearchTree::nodes() const {
  return nodes_;
}

std::vector<Node>& SearchTree::nodes() {
  return nodes_;
}

int SearchTree::node_count() const {
  return static_cast<int>(nodes_.size());
}

int SearchTree::max_depth() const {
  int best = 0;
  for (int index = 0; index < static_cast<int>(nodes_.size()); ++index) {
    int depth = 0;
    int cursor = index;
    while (cursor >= 0 && cursor != root_index_) {
      cursor = nodes_[cursor].parent;
      ++depth;
    }
    if (cursor == root_index_) {
      best = std::max(best, depth);
    }
  }
  return best;
}

bool SearchTree::has_pending_or_virtual_stats() const {
  for (const Node& node : nodes_) {
    if (node.pending_eval) {
      return true;
    }
    for (const EdgeStats& edge : node.edges) {
      if (edge.virtual_visits != 0 || std::fabs(edge.virtual_value_sum) > 1e-6f) {
        return true;
      }
    }
  }
  return false;
}

bool SearchTree::advance(core::Action action) {
  if (action < 0 || action >= core::kNumActions) {
    return false;
  }
  const int child = root().children[action];
  if (child < 0) {
    return false;
  }
  std::vector<Node> rebuilt;
  std::function<int(int, int, core::Action)> copy_subtree =
      [&](int old_index, int parent, core::Action parent_action) {
        Node node = nodes_.at(old_index);
        const auto old_children = node.children;
        node.children.fill(-1);
        node.parent = parent;
        node.parent_action = parent_action;
        if (parent < 0) {
          node.root_noise_applied = false;
        }
        const int new_index = static_cast<int>(rebuilt.size());
        rebuilt.push_back(node);
        for (core::Action child_action = 0; child_action < core::kNumActions; ++child_action) {
          const int old_child = old_children[child_action];
          if (old_child >= 0) {
            rebuilt[new_index].children[child_action] =
                copy_subtree(old_child, new_index, child_action);
          }
        }
        return new_index;
      };
  copy_subtree(child, -1, -1);
  nodes_ = std::move(rebuilt);
  root_index_ = 0;
  return true;
}

PuctMcts::PuctMcts(PuctConfig config) : config_(config), rng_(config.seed) {}

SearchTree PuctMcts::make_tree(const core::Position& root) const {
  return SearchTree(root);
}

const PuctConfig& PuctMcts::config() const {
  return config_;
}

SearchResult PuctMcts::search(SearchTree& tree, Evaluator& evaluator, bool add_noise, double temperature) {
  const auto search_started = std::chrono::steady_clock::now();
  if (config_.search_threads <= 0) {
    throw std::invalid_argument("search_threads must be positive");
  }
  if (config_.simulations_per_move < 0) {
    throw std::invalid_argument("simulations_per_move must be non-negative");
  }
  if (config_.virtual_loss < 0.0f) {
    throw std::invalid_argument("virtual_loss must be non-negative");
  }

  if (!tree.root().expanded && !tree.root().terminal_value.has_value()) {
    expand_node(tree, tree.root_index(), evaluator);
  }
  if (add_noise && tree.root().expanded && !tree.root().root_noise_applied) {
    add_root_noise(tree.root());
  }

  std::uint32_t root_visits_before = 0;
  std::array<std::uint32_t, core::kNumActions> root_visit_baseline{};
  std::array<float, core::kNumActions> root_value_baseline{};
  for (const EdgeStats& edge : tree.root().edges) {
    root_visits_before += edge.visits;
  }
  for (core::Action action = 0; action < core::kNumActions; ++action) {
    root_visit_baseline[action] = tree.root().edges[action].visits;
    root_value_baseline[action] = tree.root().edges[action].value_sum;
  }
  int completed_simulations = 0;
  int leaf_evaluations = 0;
  int terminal_evaluations = 0;
  int virtual_loss_waits = 0;
  int pending_eval_waits = 0;

  if (config_.simulations_per_move > 0 && !tree.root().terminal_value.has_value()) {
    SearchRuntime runtime{tree, config_};
    const int worker_count = std::max(1, std::min(config_.search_threads, config_.simulations_per_move));
    auto worker = [&]() {
      while (true) {
        Reservation reservation;
        bool reserved = false;
        try {
          {
            std::unique_lock<std::mutex> lock(runtime.mutex);
            while (true) {
              if (runtime.stop || runtime.launched_simulations >= config_.simulations_per_move) {
                return;
              }
              if (reserve_simulation_locked(runtime, reservation)) {
                runtime.launched_simulations += 1;
                reserved = true;
                break;
              }
              runtime.cv.wait(lock);
            }
          }

          Evaluation evaluation;
          Evaluation* evaluation_ptr = nullptr;
          if (reservation.needs_evaluation) {
            evaluation = evaluator.evaluate(reservation.position);
            evaluation_ptr = &evaluation;
          }

          {
            std::lock_guard<std::mutex> lock(runtime.mutex);
            complete_simulation_locked(runtime, reservation, evaluation_ptr);
          }
          runtime.cv.notify_all();
        } catch (...) {
          {
            std::lock_guard<std::mutex> lock(runtime.mutex);
            if (reserved) {
              if (reservation.needs_evaluation && reservation.node_index >= 0 &&
                  reservation.node_index < static_cast<int>(runtime.tree.nodes().size())) {
                runtime.tree.nodes().at(reservation.node_index).pending_eval = false;
              }
              remove_virtual_loss(runtime.tree, reservation.path, runtime.config.virtual_loss);
            }
            runtime.stop = true;
            if (!runtime.error) {
              runtime.error = std::current_exception();
            }
          }
          runtime.cv.notify_all();
          return;
        }
      }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(worker_count));
    for (int index = 0; index < worker_count; ++index) {
      workers.emplace_back(worker);
    }
    for (auto& thread : workers) {
      thread.join();
    }
    if (runtime.error) {
      std::rethrow_exception(runtime.error);
    }

    if (runtime.completed_simulations != config_.simulations_per_move) {
      throw std::runtime_error("parallel PUCT completed an unexpected number of simulations");
    }
    if (tree.has_pending_or_virtual_stats()) {
      throw std::runtime_error("parallel PUCT left pending evaluations or virtual stats behind");
    }
    completed_simulations = runtime.completed_simulations;
    leaf_evaluations = runtime.leaf_evaluations;
    terminal_evaluations = runtime.terminal_evaluations;
    virtual_loss_waits = runtime.virtual_loss_waits;
    pending_eval_waits = runtime.pending_eval_waits;
  }

  SearchResult result = build_result(tree, temperature, root_visit_baseline, root_value_baseline);
  result.max_depth = tree.max_depth();
  result.expanded_nodes = tree.node_count();
  result.completed_simulations = completed_simulations;
  result.leaf_evaluations = leaf_evaluations;
  result.terminal_evaluations = terminal_evaluations;
  result.virtual_loss_waits = virtual_loss_waits;
  result.pending_eval_waits = pending_eval_waits;
  std::uint32_t root_visits = 0;
  for (const EdgeStats& edge : tree.root().edges) {
    root_visits += edge.visits;
  }
  result.root_real_visits = root_visits;
  result.search_time_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - search_started).count();
  if (!tree.root().terminal_value.has_value() &&
      root_visits != root_visits_before + static_cast<std::uint32_t>(completed_simulations)) {
    throw std::runtime_error("root visits do not match completed PUCT simulations");
  }
  return result;
}

float PuctMcts::expand_node(SearchTree& tree, int node_index, Evaluator& evaluator) {
  Node& node = tree.nodes().at(node_index);
  if (node.expanded || node.terminal_value.has_value()) {
    return node.terminal_value.value_or(0.0f);
  }
  const Evaluation evaluation = evaluator.evaluate(node.position);
  const auto priors = normalize_priors(evaluation.priors, node.position.legal_mask());
  for (core::Action action = 0; action < core::kNumActions; ++action) {
    node.edges[action].prior = priors[action];
  }
  node.expanded = true;
  return evaluation.value;
}

void PuctMcts::add_root_noise(Node& root) {
  std::vector<core::Action> legal;
  for (core::Action action = 0; action < core::kNumActions; ++action) {
    if ((root.position.legal_mask() & (1u << action)) != 0) {
      legal.push_back(action);
    }
  }
  if (legal.empty()) {
    return;
  }
  std::gamma_distribution<double> gamma(config_.root_dirichlet_alpha, 1.0);
  std::vector<double> noise;
  noise.reserve(legal.size());
  double sum = 0.0;
  for (std::size_t i = 0; i < legal.size(); ++i) {
    const double sample = gamma(rng_);
    noise.push_back(sample);
    sum += sample;
  }
  if (sum <= 0.0) {
    return;
  }
  for (std::size_t i = 0; i < legal.size(); ++i) {
    EdgeStats& edge = root.edges[legal[i]];
    const double eta = noise[i] / sum;
    edge.prior = static_cast<float>((1.0 - config_.root_exploration_fraction) * edge.prior + config_.root_exploration_fraction * eta);
  }
  root.root_noise_applied = true;
}

core::Action PuctMcts::select_action(const Node& node) const {
  const std::uint16_t legal = node.position.legal_mask();
  const std::uint32_t parent_visits = total_visits(node);
  const double c = exploration_constant(config_, parent_visits);
  const double sqrt_parent = std::sqrt(static_cast<double>(parent_visits));
  double best_score = -std::numeric_limits<double>::infinity();
  core::Action best_action = -1;
  for (core::Action action = 0; action < core::kNumActions; ++action) {
    if ((legal & (1u << action)) == 0) {
      continue;
    }
    const EdgeStats& edge = node.edges[action];
    const double q = q_value(edge);
    const double u = c * static_cast<double>(edge.prior) * sqrt_parent / (1.0 + static_cast<double>(edge.visits));
    const double score = q + u;
    if (score > best_score) {
      best_score = score;
      best_action = action;
    }
  }
  return best_action;
}

SearchResult PuctMcts::build_result(
    const SearchTree& tree,
    double temperature,
    const std::array<std::uint32_t, core::kNumActions>& visit_baseline,
    const std::array<float, core::kNumActions>& value_baseline) {
  SearchResult result;
  const Node& root = tree.root();
  for (core::Action action = 0; action < core::kNumActions; ++action) {
    const EdgeStats& edge = root.edges[action];
    if (edge.visits < visit_baseline[action]) {
      throw std::runtime_error("root visit baseline exceeds current visits");
    }
    result.visit_counts[action] = edge.visits - visit_baseline[action];
    const float value_delta = edge.value_sum - value_baseline[action];
    result.q_values[action] = result.visit_counts[action] == 0
        ? q_value(edge)
        : value_delta / static_cast<float>(result.visit_counts[action]);
  }
  result.policy = policy_from_visits(result.visit_counts, temperature);
  result.selected_action = sample_action(result.policy, root.position.legal_mask(), rng_);
  if (result.selected_action < 0) {
    for (core::Action action = 0; action < core::kNumActions; ++action) {
      if (result.visit_counts[action] > 0) {
        result.selected_action = action;
        break;
      }
    }
  }
  std::uint32_t visits = 0;
  float value_sum = 0.0f;
  for (core::Action action = 0; action < core::kNumActions; ++action) {
    visits += result.visit_counts[action];
    value_sum += root.edges[action].value_sum - value_baseline[action];
  }
  result.root_value = visits == 0 ? 0.0f : value_sum / static_cast<float>(visits);
  return result;
}

std::array<float, core::kNumActions> normalize_priors(
    const std::array<float, core::kNumActions>& priors,
    std::uint16_t legal_mask) {
  std::array<float, core::kNumActions> out{};
  double sum = 0.0;
  for (core::Action action = 0; action < core::kNumActions; ++action) {
    if ((legal_mask & (1u << action)) == 0) {
      continue;
    }
    const float value = std::max(0.0f, priors[action]);
    out[action] = value;
    sum += value;
  }
  const int legal_count = [&]() {
    int count = 0;
    for (core::Action action = 0; action < core::kNumActions; ++action) {
      if ((legal_mask & (1u << action)) != 0) ++count;
    }
    return count;
  }();
  if (legal_count == 0) {
    return out;
  }
  if (sum <= 0.0) {
    const float uniform = 1.0f / static_cast<float>(legal_count);
    for (core::Action action = 0; action < core::kNumActions; ++action) {
      if ((legal_mask & (1u << action)) != 0) {
        out[action] = uniform;
      }
    }
    return out;
  }
  for (float& value : out) {
    value = static_cast<float>(static_cast<double>(value) / sum);
  }
  return out;
}

core::Action sample_action(
    const std::array<float, core::kNumActions>& policy,
    std::uint16_t legal_mask,
    std::mt19937_64& rng) {
  double sum = 0.0;
  for (core::Action action = 0; action < core::kNumActions; ++action) {
    if ((legal_mask & (1u << action)) != 0) {
      sum += std::max(0.0f, policy[action]);
    }
  }
  if (sum <= 0.0) {
    return -1;
  }
  std::uniform_real_distribution<double> dist(0.0, sum);
  double draw = dist(rng);
  for (core::Action action = 0; action < core::kNumActions; ++action) {
    if ((legal_mask & (1u << action)) == 0) {
      continue;
    }
    draw -= std::max(0.0f, policy[action]);
    if (draw <= 0.0) {
      return action;
    }
  }
  for (core::Action action = core::kNumActions - 1; action >= 0; --action) {
    if ((legal_mask & (1u << action)) != 0 && policy[action] > 0.0f) {
      return action;
    }
  }
  return -1;
}

}  // namespace c4zero::search
