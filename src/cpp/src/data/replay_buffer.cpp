#include "c4zero/data/shard.hpp"

#include <random>
#include <stdexcept>

namespace c4zero::data {

ReplayBuffer::ReplayBuffer(std::size_t max_games) : max_games_(max_games) {
  if (max_games_ == 0) {
    throw std::invalid_argument("ReplayBuffer max_games must be positive");
  }
}

void ReplayBuffer::add_game(std::vector<SelfPlaySample> game) {
  if (games_.size() == max_games_) {
    games_.erase(games_.begin());
  }
  games_.push_back(std::move(game));
}

std::size_t ReplayBuffer::num_games() const {
  return games_.size();
}

std::size_t ReplayBuffer::num_samples() const {
  std::size_t total = 0;
  for (const auto& game : games_) {
    total += game.size();
  }
  return total;
}

const SelfPlaySample& ReplayBuffer::sample(std::uint64_t draw) const {
  if (games_.empty()) {
    throw std::runtime_error("cannot sample from empty replay buffer");
  }
  const std::size_t total = num_samples();
  std::size_t index = static_cast<std::size_t>(draw % total);
  for (const auto& game : games_) {
    if (index < game.size()) {
      return game[index];
    }
    index -= game.size();
  }
  return games_.back().back();
}

std::vector<SelfPlaySample> ReplayBuffer::sample_batch(std::size_t batch_size, std::uint64_t seed) const {
  std::vector<SelfPlaySample> batch;
  batch.reserve(batch_size);
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<std::uint64_t> dist(0, static_cast<std::uint64_t>(num_samples() - 1));
  for (std::size_t i = 0; i < batch_size; ++i) {
    batch.push_back(sample(dist(rng)));
  }
  return batch;
}

}  // namespace c4zero::data
