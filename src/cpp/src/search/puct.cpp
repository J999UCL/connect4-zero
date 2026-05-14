#include "c4zero/search/puct.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace c4zero::search {
namespace {

float q_value(const EdgeStats& edge) {
  if (edge.visits == 0) {
    return 0.0f;
  }
  return edge.value_sum / static_cast<float>(edge.visits);
}

std::uint32_t total_visits(const Node& node) {
  std::uint32_t total = 0;
  for (const auto& edge : node.edges) {
    total += edge.visits;
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
  if (!tree.root().expanded && !tree.root().terminal_value.has_value()) {
    expand_node(tree, tree.root_index(), evaluator);
  }
  if (add_noise && tree.root().expanded && !tree.root().root_noise_applied) {
    add_root_noise(tree.root());
  }

  for (int simulation = 0; simulation < config_.simulations_per_move; ++simulation) {
    int node_index = tree.root_index();
    std::vector<std::pair<int, core::Action>> path;
    float leaf_value = 0.0f;

    while (true) {
      Node& node = tree.nodes().at(node_index);
      if (node.terminal_value.has_value()) {
        leaf_value = *node.terminal_value;
        break;
      }
      if (!node.expanded) {
        leaf_value = expand_node(tree, node_index, evaluator);
        break;
      }
      const core::Action action = select_action(node);
      if (action < 0) {
        leaf_value = 0.0f;
        break;
      }
      path.push_back({node_index, action});
      int child_index = node.children[action];
      if (child_index < 0) {
        Node child;
        child.position = node.position.play(action);
        child.parent = node_index;
        child.parent_action = action;
        child.children.fill(-1);
        child.terminal_value = child.position.terminal_value();
        child_index = static_cast<int>(tree.nodes().size());
        node.children[action] = child_index;
        tree.nodes().push_back(child);
      }
      Node& child = tree.nodes().at(child_index);
      if (child.terminal_value.has_value()) {
        leaf_value = *child.terminal_value;
        break;
      }
      if (!child.expanded) {
        leaf_value = expand_node(tree, child_index, evaluator);
        break;
      }
      node_index = child_index;
    }

    float value = leaf_value;
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
      value = -value;
      EdgeStats& edge = tree.nodes().at(it->first).edges[it->second];
      edge.visits += 1;
      edge.value_sum += value;
    }
  }

  SearchResult result = build_result(tree, temperature);
  result.max_depth = tree.max_depth();
  result.expanded_nodes = tree.node_count();
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

SearchResult PuctMcts::build_result(const SearchTree& tree, double temperature) {
  SearchResult result;
  const Node& root = tree.root();
  for (core::Action action = 0; action < core::kNumActions; ++action) {
    const EdgeStats& edge = root.edges[action];
    result.visit_counts[action] = edge.visits;
    result.q_values[action] = q_value(edge);
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
    visits += root.edges[action].visits;
    value_sum += root.edges[action].value_sum;
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
