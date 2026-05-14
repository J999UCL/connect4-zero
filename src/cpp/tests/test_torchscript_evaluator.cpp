#include "c4zero/core/position.hpp"
#include "c4zero/model/torchscript.hpp"
#include "test_support.hpp"

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

int main() {
  const char* fixture = std::getenv("C4ZERO_TORCHSCRIPT_FIXTURE");
  if (fixture == nullptr || std::string(fixture).empty()) {
    std::cout << "C4ZERO_TORCHSCRIPT_FIXTURE unset; skipping optional TorchScript fixture test\n";
    return 0;
  }

  c4zero::model::TorchScriptEvaluator evaluator(fixture, torch::kCPU);
  auto evaluation = evaluator.evaluate(c4zero::core::Position::empty());
  float prior_sum = 0.0f;
  for (float prior : evaluation.priors) {
    C4ZERO_CHECK(prior >= 0.0f);
    prior_sum += prior;
  }
  C4ZERO_CHECK(prior_sum > 0.999f && prior_sum < 1.001f);
  C4ZERO_CHECK(evaluation.value >= -1.0f && evaluation.value <= 1.0f);

  c4zero::model::AsyncBatchedTorchScriptConfig config;
  config.max_batch_size = 8;
  config.max_wait_us = 5000;
  c4zero::model::AsyncBatchedTorchScriptEvaluator async(fixture, torch::kCPU, config);
  std::vector<c4zero::search::Evaluation> async_results(8);
  std::vector<std::thread> threads;
  for (int index = 0; index < 8; ++index) {
    threads.emplace_back([&, index]() {
      auto position = c4zero::core::from_actions({index % c4zero::core::kNumActions});
      async_results[index] = async.evaluate(position);
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  const auto stats = async.stats();
  C4ZERO_CHECK_EQ(stats.requests, 8);
  C4ZERO_CHECK(stats.batches >= 1);
  C4ZERO_CHECK(stats.max_batch_size > 1);
  for (int index = 0; index < 8; ++index) {
    auto position = c4zero::core::from_actions({index % c4zero::core::kNumActions});
    auto direct = evaluator.evaluate(position);
    float async_prior_sum = 0.0f;
    for (int action = 0; action < c4zero::core::kNumActions; ++action) {
      C4ZERO_CHECK(std::fabs(async_results[index].priors[action] - direct.priors[action]) < 1e-5f);
      async_prior_sum += async_results[index].priors[action];
    }
    C4ZERO_CHECK(async_prior_sum > 0.999f && async_prior_sum < 1.001f);
    C4ZERO_CHECK(std::fabs(async_results[index].value - direct.value) < 1e-5f);
  }
  async.stop();
  bool stopped_rejected = false;
  try {
    (void)async.evaluate(c4zero::core::Position::empty());
  } catch (const std::runtime_error&) {
    stopped_rejected = true;
  }
  C4ZERO_CHECK(stopped_rejected);
  return 0;
}
