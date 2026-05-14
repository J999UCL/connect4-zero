#include "c4zero/model/torchscript.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>

namespace c4zero::model {
namespace {

void validate_config(const AsyncBatchedTorchScriptConfig& config) {
  if (config.max_batch_size <= 0) {
    throw std::invalid_argument("inference max batch size must be positive");
  }
  if (config.max_wait_us < 0) {
    throw std::invalid_argument("inference max wait must be non-negative");
  }
}

std::vector<search::Evaluation> evaluations_from_outputs(
    const std::vector<core::Position>& positions,
    torch::Tensor logits,
    torch::Tensor values) {
  const long batch_size = static_cast<long>(positions.size());
  if (logits.dim() != 2 || logits.size(0) != batch_size || logits.size(1) != core::kNumActions) {
    throw std::runtime_error("TorchScript policy logits must have shape [B,16]");
  }
  if (values.numel() != batch_size) {
    throw std::runtime_error("TorchScript value output must contain one scalar per batch item");
  }

  logits = logits.to(torch::kCPU).contiguous();
  values = values.to(torch::kCPU).reshape({batch_size}).contiguous();
  auto logits_accessor = logits.accessor<float, 2>();
  auto values_accessor = values.accessor<float, 1>();

  std::vector<search::Evaluation> evaluations;
  evaluations.resize(positions.size());
  for (std::size_t index = 0; index < positions.size(); ++index) {
    const float value = values_accessor[static_cast<long>(index)];
    if (!std::isfinite(value) || value < -1.0001f || value > 1.0001f) {
      throw std::runtime_error("TorchScript value output must be finite and in [-1,1]");
    }

    search::Evaluation evaluation;
    evaluation.value = value;
    const std::uint16_t legal = positions[index].legal_mask();
    float max_logit = -std::numeric_limits<float>::infinity();
    for (core::Action action = 0; action < core::kNumActions; ++action) {
      if ((legal & (1u << action)) == 0) {
        continue;
      }
      const float logit = logits_accessor[static_cast<long>(index)][action];
      if (!std::isfinite(logit)) {
        throw std::runtime_error("TorchScript policy logits must be finite");
      }
      max_logit = std::max(max_logit, logit);
    }

    double sum = 0.0;
    for (core::Action action = 0; action < core::kNumActions; ++action) {
      if ((legal & (1u << action)) == 0) {
        evaluation.priors[action] = 0.0f;
        continue;
      }
      const float probability = std::exp(logits_accessor[static_cast<long>(index)][action] - max_logit);
      evaluation.priors[action] = probability;
      sum += probability;
    }
    if (sum > 0.0) {
      for (float& prior : evaluation.priors) {
        prior = static_cast<float>(static_cast<double>(prior) / sum);
      }
    }
    evaluations[index] = evaluation;
  }
  return evaluations;
}

}  // namespace

double TorchScriptBatchStats::mean_batch_size() const {
  if (batches == 0) {
    return 0.0;
  }
  return static_cast<double>(batch_items) / static_cast<double>(batches);
}

double TorchScriptBatchStats::mean_wait_ms() const {
  if (requests == 0) {
    return 0.0;
  }
  return total_wait_ms / static_cast<double>(requests);
}

TorchScriptRunner::TorchScriptRunner(const std::string& model_path, torch::Device device)
    : module_(torch::jit::load(model_path, device)), device_(device) {
  module_.to(device_);
  module_.eval();
}

std::vector<search::Evaluation> TorchScriptRunner::evaluate_batch(const std::vector<core::Position>& positions) {
  if (positions.empty()) {
    return {};
  }
  torch::NoGradGuard no_grad;
  auto input = encode_positions(positions, device_);
  auto output = module_.forward({input});
  if (!output.isTuple()) {
    throw std::runtime_error("TorchScript model must return (policy_logits, value)");
  }
  auto elements = output.toTuple()->elements();
  if (elements.size() != 2) {
    throw std::runtime_error("TorchScript model must return exactly (policy_logits, value)");
  }
  return evaluations_from_outputs(positions, elements.at(0).toTensor(), elements.at(1).toTensor());
}

TorchScriptEvaluator::TorchScriptEvaluator(const std::string& model_path, torch::Device device)
    : runner_(model_path, device) {}

search::Evaluation TorchScriptEvaluator::evaluate(const core::Position& position) {
  return runner_.evaluate_batch({position}).at(0);
}

AsyncBatchedTorchScriptEvaluator::AsyncBatchedTorchScriptEvaluator(
    const std::string& model_path,
    torch::Device device,
    AsyncBatchedTorchScriptConfig config)
    : runner_(model_path, device), config_(config) {
  validate_config(config_);
  server_ = std::thread(&AsyncBatchedTorchScriptEvaluator::server_loop, this);
}

AsyncBatchedTorchScriptEvaluator::~AsyncBatchedTorchScriptEvaluator() {
  stop();
}

search::Evaluation AsyncBatchedTorchScriptEvaluator::evaluate(const core::Position& position) {
  auto request = std::make_shared<Request>();
  request->position = position;
  request->enqueued_at = std::chrono::steady_clock::now();
  auto future = request->promise.get_future();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) {
      throw std::runtime_error("async TorchScript evaluator has stopped");
    }
    queue_.push_back(request);
  }
  cv_.notify_one();
  return future.get();
}

void AsyncBatchedTorchScriptEvaluator::stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) {
      return;
    }
    stopping_ = true;
  }
  cv_.notify_all();
  if (server_.joinable()) {
    server_.join();
  }
}

TorchScriptBatchStats AsyncBatchedTorchScriptEvaluator::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

std::vector<std::shared_ptr<AsyncBatchedTorchScriptEvaluator::Request>>
AsyncBatchedTorchScriptEvaluator::take_batch(std::unique_lock<std::mutex>& lock) {
  cv_.wait(lock, [&]() { return stopping_ || !queue_.empty(); });
  if (queue_.empty()) {
    return {};
  }

  const auto deadline = queue_.front()->enqueued_at + std::chrono::microseconds(config_.max_wait_us);
  cv_.wait_until(lock, deadline, [&]() {
    return stopping_ || static_cast<int>(queue_.size()) >= config_.max_batch_size;
  });

  const int count = std::min<int>(config_.max_batch_size, static_cast<int>(queue_.size()));
  std::vector<std::shared_ptr<Request>> batch;
  batch.reserve(static_cast<std::size_t>(count));
  const auto now = std::chrono::steady_clock::now();
  for (int index = 0; index < count; ++index) {
    auto request = queue_.front();
    queue_.pop_front();
    stats_.total_wait_ms +=
        std::chrono::duration<double, std::milli>(now - request->enqueued_at).count();
    batch.push_back(std::move(request));
  }
  stats_.requests += static_cast<std::uint64_t>(batch.size());
  stats_.batches += 1;
  stats_.batch_items += static_cast<std::uint64_t>(batch.size());
  stats_.max_batch_size = std::max(stats_.max_batch_size, static_cast<int>(batch.size()));
  return batch;
}

void AsyncBatchedTorchScriptEvaluator::fail_pending_locked(const std::exception_ptr& error) {
  while (!queue_.empty()) {
    queue_.front()->promise.set_exception(error);
    queue_.pop_front();
  }
}

void AsyncBatchedTorchScriptEvaluator::server_loop() {
  while (true) {
    std::vector<std::shared_ptr<Request>> batch;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      batch = take_batch(lock);
      if (batch.empty() && stopping_) {
        return;
      }
    }

    try {
      std::vector<core::Position> positions;
      positions.reserve(batch.size());
      for (const auto& request : batch) {
        positions.push_back(request->position);
      }
      const auto started = std::chrono::steady_clock::now();
      const auto evaluations = runner_.evaluate_batch(positions);
      const auto elapsed =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
      {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.total_inference_ms += elapsed;
      }
      if (evaluations.size() != batch.size()) {
        throw std::runtime_error("TorchScript batch evaluator returned the wrong batch size");
      }
      for (std::size_t index = 0; index < batch.size(); ++index) {
        batch[index]->promise.set_value(evaluations[index]);
      }
    } catch (...) {
      const auto error = std::current_exception();
      for (const auto& request : batch) {
        request->promise.set_exception(error);
      }
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
      fail_pending_locked(error);
      cv_.notify_all();
      return;
    }
  }
}

}  // namespace c4zero::model
