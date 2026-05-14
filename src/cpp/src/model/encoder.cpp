#include "c4zero/model/torchscript.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace c4zero::model {

torch::Tensor encode_positions(const std::vector<core::Position>& positions, torch::Device device) {
  auto tensor = torch::zeros(
      {static_cast<long>(positions.size()), 2, core::kBoardSize, core::kBoardSize, core::kBoardSize},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  auto accessor = tensor.accessor<float, 5>();
  for (std::size_t b = 0; b < positions.size(); ++b) {
    for (int z = 0; z < core::kBoardSize; ++z) {
      for (int y = 0; y < core::kBoardSize; ++y) {
        for (int x = 0; x < core::kBoardSize; ++x) {
          const core::Bitboard mask = core::cell_mask(x, y, z);
          if ((positions[b].current & mask) != 0) {
            accessor[static_cast<long>(b)][0][z][y][x] = 1.0f;
          }
          if ((positions[b].opponent & mask) != 0) {
            accessor[static_cast<long>(b)][1][z][y][x] = 1.0f;
          }
        }
      }
    }
  }
  return tensor.to(device);
}

TorchScriptEvaluator::TorchScriptEvaluator(const std::string& model_path, torch::Device device)
    : module_(torch::jit::load(model_path, device)), device_(device) {
  module_.to(device_);
  module_.eval();
}

search::Evaluation TorchScriptEvaluator::evaluate(const core::Position& position) {
  torch::NoGradGuard no_grad;
  auto input = encode_positions({position}, device_);
  auto output = module_.forward({input});
  torch::Tensor logits;
  torch::Tensor values;
  if (output.isTuple()) {
    auto elements = output.toTuple()->elements();
    if (elements.size() != 2) {
      throw std::runtime_error("TorchScript model must return exactly (policy_logits, value)");
    }
    logits = elements.at(0).toTensor();
    values = elements.at(1).toTensor();
  } else {
    throw std::runtime_error("TorchScript model must return (policy_logits, value)");
  }
  if (logits.dim() != 2 || logits.size(0) != 1 || logits.size(1) != core::kNumActions) {
    throw std::runtime_error("TorchScript policy logits must have shape [1,16]");
  }
  if (values.numel() != 1) {
    throw std::runtime_error("TorchScript value output must contain one scalar for batch size 1");
  }
  logits = logits.squeeze(0).to(torch::kCPU);
  values = values.to(torch::kCPU).reshape({-1});
  const float value = values[0].item<float>();
  if (!std::isfinite(value) || value < -1.0001f || value > 1.0001f) {
    throw std::runtime_error("TorchScript value output must be finite and in [-1,1]");
  }
  search::Evaluation evaluation;
  evaluation.value = value;
  std::array<float, core::kNumActions> raw{};
  const std::uint16_t legal = position.legal_mask();
  float max_logit = -std::numeric_limits<float>::infinity();
  for (core::Action action = 0; action < core::kNumActions; ++action) {
    if ((legal & (1u << action)) == 0) {
      continue;
    }
    raw[action] = logits[action].item<float>();
    max_logit = std::max(max_logit, raw[action]);
  }
  double sum = 0.0;
  for (core::Action action = 0; action < core::kNumActions; ++action) {
    if ((legal & (1u << action)) == 0) {
      raw[action] = 0.0f;
      continue;
    }
    raw[action] = std::exp(raw[action] - max_logit);
    sum += raw[action];
  }
  if (sum > 0.0) {
    for (core::Action action = 0; action < core::kNumActions; ++action) {
      evaluation.priors[action] = static_cast<float>(raw[action] / sum);
    }
  }
  return evaluation;
}

}  // namespace c4zero::model
