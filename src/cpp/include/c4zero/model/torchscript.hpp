#pragma once

#include "c4zero/core/position.hpp"
#include "c4zero/search/puct.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <torch/script.h>
#include <torch/torch.h>

namespace c4zero::model {

[[nodiscard]] torch::Tensor encode_positions(
    const std::vector<core::Position>& positions,
    torch::Device device = torch::kCPU);

struct TorchScriptBatchStats {
  std::uint64_t requests = 0;
  std::uint64_t batches = 0;
  std::uint64_t batch_items = 0;
  int max_batch_size = 0;
  double total_wait_ms = 0.0;
  double total_inference_ms = 0.0;

  [[nodiscard]] double mean_batch_size() const;
  [[nodiscard]] double mean_wait_ms() const;
};

struct AsyncBatchedTorchScriptConfig {
  int max_batch_size = 128;
  int max_wait_us = 2000;
};

class TorchScriptRunner {
 public:
  TorchScriptRunner(const std::string& model_path, torch::Device device);
  [[nodiscard]] std::vector<search::Evaluation> evaluate_batch(const std::vector<core::Position>& positions);

 private:
  torch::jit::script::Module module_;
  torch::Device device_;
};

class TorchScriptEvaluator final : public search::Evaluator {
 public:
  TorchScriptEvaluator(const std::string& model_path, torch::Device device);
  [[nodiscard]] search::Evaluation evaluate(const core::Position& position) override;

 private:
  TorchScriptRunner runner_;
};

class AsyncBatchedTorchScriptEvaluator final : public search::Evaluator {
 public:
  AsyncBatchedTorchScriptEvaluator(
      const std::string& model_path,
      torch::Device device,
      AsyncBatchedTorchScriptConfig config = {});
  ~AsyncBatchedTorchScriptEvaluator() override;

  AsyncBatchedTorchScriptEvaluator(const AsyncBatchedTorchScriptEvaluator&) = delete;
  AsyncBatchedTorchScriptEvaluator& operator=(const AsyncBatchedTorchScriptEvaluator&) = delete;

  [[nodiscard]] search::Evaluation evaluate(const core::Position& position) override;
  void stop();
  [[nodiscard]] TorchScriptBatchStats stats() const;

 private:
  struct Request {
    core::Position position;
    std::promise<search::Evaluation> promise;
    std::chrono::steady_clock::time_point enqueued_at;
  };

  void server_loop();
  std::vector<std::shared_ptr<Request>> take_batch(std::unique_lock<std::mutex>& lock);
  void fail_pending_locked(const std::exception_ptr& error);

  TorchScriptRunner runner_;
  AsyncBatchedTorchScriptConfig config_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<Request>> queue_;
  bool stopping_ = false;
  std::thread server_;
  TorchScriptBatchStats stats_;
};

}  // namespace c4zero::model
