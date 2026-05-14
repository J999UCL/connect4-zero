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
  C4ZERO_CHECK(tree.advance(3));
  C4ZERO_CHECK(tree.root().position.is_terminal());
  return 0;
}
