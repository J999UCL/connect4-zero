#include "c4zero/search/puct.hpp"
#include "test_support.hpp"

#include <cmath>

using namespace c4zero;

class WinningPriorEvaluator final : public search::Evaluator {
 public:
  search::Evaluation evaluate(const core::Position& position) override {
    search::Evaluation evaluation;
    evaluation.value = 0.0f;
    for (auto action : core::legal_actions(position)) {
      evaluation.priors[action] = search::normalize_priors(evaluation.priors, position.legal_mask())[action];
    }
    for (auto action : core::legal_actions(position)) {
      if (position.play(action).terminal_value().value_or(0.0f) == -1.0f) {
        evaluation.priors = {};
        evaluation.priors[action] = 1.0f;
        break;
      }
    }
    return evaluation;
  }
};

class IllegalPriorEvaluator final : public search::Evaluator {
 public:
  search::Evaluation evaluate(const core::Position&) override {
    search::Evaluation evaluation;
    evaluation.value = 0.0f;
    evaluation.priors.fill(0.0f);
    evaluation.priors[0] = 1000.0f;
    evaluation.priors[5] = 1.0f;
    return evaluation;
  }
};

int main() {
  search::PuctConfig config;
  config.simulations_per_move = 32;
  config.seed = 7;

  WinningPriorEvaluator evaluator;
  search::PuctMcts mcts(config);
  auto position = core::from_actions({0, 4, 1, 5, 2, 6});
  auto tree = mcts.make_tree(position);
  auto result = mcts.search(tree, evaluator, false, 0.0);

  C4ZERO_CHECK_EQ(result.selected_action, 3);
  C4ZERO_CHECK(position.is_legal(result.selected_action));
  std::uint32_t visits = 0;
  for (auto value : result.visit_counts) {
    visits += value;
  }
  C4ZERO_CHECK_EQ(visits, 32);
  C4ZERO_CHECK(tree.max_depth() >= 1);
  const int before_advance_nodes = tree.node_count();
  C4ZERO_CHECK(tree.advance(3));
  C4ZERO_CHECK(tree.root().position.is_terminal());
  C4ZERO_CHECK(tree.node_count() < before_advance_nodes);
  C4ZERO_CHECK(!tree.advance(3));

  search::PuctConfig noise_config;
  noise_config.simulations_per_move = 0;
  noise_config.seed = 11;
  search::PuctMcts noisy(noise_config);
  search::UniformEvaluator uniform;
  auto noise_tree = noisy.make_tree(core::Position::empty());
  (void)noisy.search(noise_tree, uniform, true, 1.0);
  const auto priors_after_first_noise = noise_tree.root().edges;
  (void)noisy.search(noise_tree, uniform, true, 1.0);
  for (core::Action action = 0; action < core::kNumActions; ++action) {
    C4ZERO_CHECK_EQ(noise_tree.root().edges[action].prior, priors_after_first_noise[action].prior);
  }

  search::PuctMcts arena_style(noise_config);
  auto arena_tree = arena_style.make_tree(core::Position::empty());
  (void)arena_style.search(arena_tree, uniform, false, 1.0);
  C4ZERO_CHECK(!arena_tree.root().root_noise_applied);
  C4ZERO_CHECK(std::fabs(arena_tree.root().edges[0].prior - (1.0f / core::kNumActions)) < 1e-6f);

  search::PuctConfig reusable_config = noise_config;
  reusable_config.simulations_per_move = 4;
  search::PuctMcts reusable_mcts(reusable_config);
  auto reusable_tree = reusable_mcts.make_tree(core::Position::empty());
  const auto reusable_result = reusable_mcts.search(reusable_tree, uniform, true, 1.0);
  C4ZERO_CHECK(reusable_tree.root().root_noise_applied);
  C4ZERO_CHECK(reusable_result.selected_action >= 0);
  C4ZERO_CHECK(reusable_tree.advance(reusable_result.selected_action));
  C4ZERO_CHECK_EQ(reusable_tree.root().parent, -1);
  C4ZERO_CHECK_EQ(reusable_tree.root().parent_action, -1);
  C4ZERO_CHECK(!reusable_tree.root().root_noise_applied);
  C4ZERO_CHECK(reusable_tree.node_count() <= noise_tree.node_count());

  IllegalPriorEvaluator illegal_prior;
  search::PuctConfig legality_config;
  legality_config.simulations_per_move = 1;
  legality_config.seed = 3;
  search::PuctMcts legality_mcts(legality_config);
  auto full_zero = core::from_actions({0, 0, 0, 0});
  C4ZERO_CHECK(!full_zero.is_legal(0));
  auto legality_tree = legality_mcts.make_tree(full_zero);
  const auto legality = legality_mcts.search(legality_tree, illegal_prior, false, 0.0);
  C4ZERO_CHECK(legality.selected_action >= 0);
  C4ZERO_CHECK(full_zero.is_legal(legality.selected_action));
  C4ZERO_CHECK_EQ(legality.visit_counts[0], 0);
  C4ZERO_CHECK_EQ(legality.policy[0], 0.0f);
  return 0;
}
