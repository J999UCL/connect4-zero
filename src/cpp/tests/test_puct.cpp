#include "c4zero/search/puct.hpp"
#include "test_support.hpp"

#include <cmath>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <set>
#include <thread>

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

class BlockingUniformEvaluator final : public search::Evaluator {
 public:
  explicit BlockingUniformEvaluator(int wait_for_calls) : wait_for_calls_(wait_for_calls) {}

  search::Evaluation evaluate(const core::Position& position) override {
    std::unique_lock<std::mutex> lock(mutex_);
    calls_ += 1;
    positions_.insert(position.compact_string());
    if (calls_ >= wait_for_calls_) {
      reached_.notify_all();
    }
    if (calls_ > 1 && !released_) {
      release_.wait(lock, [&]() { return released_; });
    }
    search::Evaluation evaluation;
    evaluation.value = 0.0f;
    evaluation.priors = search::normalize_priors(evaluation.priors, position.legal_mask());
    return evaluation;
  }

  bool wait_until_blocked() {
    std::unique_lock<std::mutex> lock(mutex_);
    return reached_.wait_for(lock, std::chrono::seconds(5), [&]() {
      return calls_ >= wait_for_calls_;
    });
  }

  void release() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      released_ = true;
    }
    release_.notify_all();
  }

  int calls() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return calls_;
  }

  std::size_t unique_positions() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return positions_.size();
  }

 private:
  int wait_for_calls_;
  mutable std::mutex mutex_;
  std::condition_variable reached_;
  std::condition_variable release_;
  int calls_ = 0;
  bool released_ = false;
  std::set<std::string> positions_;
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

  search::PuctConfig parallel_config;
  parallel_config.simulations_per_move = 64;
  parallel_config.search_threads = 4;
  parallel_config.seed = 17;
  search::PuctMcts parallel_mcts(parallel_config);
  auto parallel_tree = parallel_mcts.make_tree(core::Position::empty());
  const auto parallel = parallel_mcts.search(parallel_tree, uniform, false, 1.0);
  std::uint32_t parallel_visits = 0;
  for (auto value : parallel.visit_counts) {
    parallel_visits += value;
  }
  C4ZERO_CHECK_EQ(parallel_visits, 64);
  C4ZERO_CHECK_EQ(parallel.completed_simulations, 64);
  C4ZERO_CHECK(parallel.root_real_visits >= 64);
  C4ZERO_CHECK(parallel.search_time_ms >= 0.0);
  C4ZERO_CHECK(!parallel_tree.has_pending_or_virtual_stats());

  search::PuctConfig blocking_config;
  blocking_config.simulations_per_move = 8;
  blocking_config.search_threads = 4;
  blocking_config.seed = 23;
  BlockingUniformEvaluator blocking(/*wait_for_calls=*/5);
  search::PuctMcts blocking_mcts(blocking_config);
  auto blocking_tree = blocking_mcts.make_tree(core::Position::empty());
  search::SearchResult blocking_result;
  std::thread search_thread([&]() {
    blocking_result = blocking_mcts.search(blocking_tree, blocking, false, 1.0);
  });
  C4ZERO_CHECK(blocking.wait_until_blocked());
  C4ZERO_CHECK(blocking.calls() >= 5);
  blocking.release();
  search_thread.join();
  C4ZERO_CHECK(blocking.unique_positions() >= 5);
  C4ZERO_CHECK(!blocking_tree.has_pending_or_virtual_stats());
  std::uint32_t blocking_visits = 0;
  for (auto value : blocking_result.visit_counts) {
    blocking_visits += value;
  }
  C4ZERO_CHECK_EQ(blocking_visits, 8);

  return 0;
}
