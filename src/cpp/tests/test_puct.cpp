#include "c4zero/search/puct.hpp"
#include "test_support.hpp"

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
  return 0;
}
