#include "c4zero/core/position.hpp"
#include "c4zero/model/torchscript.hpp"
#include "test_support.hpp"

#include <cstdlib>
#include <iostream>

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
  return 0;
}
