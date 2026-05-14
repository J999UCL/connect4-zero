#pragma once

#include "c4zero/core/position.hpp"
#include "c4zero/search/puct.hpp"

#include <string>
#include <torch/script.h>
#include <torch/torch.h>

namespace c4zero::model {

[[nodiscard]] torch::Tensor encode_positions(
    const std::vector<core::Position>& positions,
    torch::Device device = torch::kCPU);

class TorchScriptEvaluator final : public search::Evaluator {
 public:
  TorchScriptEvaluator(const std::string& model_path, torch::Device device);
  [[nodiscard]] search::Evaluation evaluate(const core::Position& position) override;

 private:
  torch::jit::script::Module module_;
  torch::Device device_;
};

}  // namespace c4zero::model
