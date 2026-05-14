#pragma once

#include "c4zero/core/position.hpp"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace c4zero::bots {

struct BotMatchResult {
  int games = 0;
  int first_wins = 0;
  int second_wins = 0;
  int draws = 0;
  int total_plies = 0;

  [[nodiscard]] double first_score_rate() const;
  [[nodiscard]] std::string summary() const;
};

class Bot {
 public:
  virtual ~Bot() = default;
  [[nodiscard]] virtual core::Action select_move(const core::Position& position) const = 0;
  [[nodiscard]] virtual std::string name() const = 0;
};

[[nodiscard]] const std::array<core::Action, core::kNumActions>& center_order();
[[nodiscard]] bool is_immediate_win(const core::Position& position, core::Action action);
[[nodiscard]] std::vector<core::Action> immediate_winning_actions(const core::Position& position);
[[nodiscard]] int opponent_winning_replies_after(const core::Position& position, core::Action action);
[[nodiscard]] int playable_threat_count_after(const core::Position& position, core::Action action);
[[nodiscard]] int evaluate_lines_for_bits(core::Bitboard mine, core::Bitboard theirs);
[[nodiscard]] int evaluate_position_for_side_to_move(const core::Position& position);
[[nodiscard]] std::vector<core::Action> ordered_legal_actions(const core::Position& position);

class FirstLegalBot final : public Bot {
 public:
  [[nodiscard]] core::Action select_move(const core::Position& position) const override;
  [[nodiscard]] std::string name() const override;
};

class CenterOrderBot final : public Bot {
 public:
  [[nodiscard]] core::Action select_move(const core::Position& position) const override;
  [[nodiscard]] std::string name() const override;
};

class OnePlyTacticalBot final : public Bot {
 public:
  [[nodiscard]] core::Action select_move(const core::Position& position) const override;
  [[nodiscard]] std::string name() const override;
};

class LineScoreBot final : public Bot {
 public:
  [[nodiscard]] core::Action select_move(const core::Position& position) const override;
  [[nodiscard]] std::string name() const override;
};

class ForkThreatBot final : public Bot {
 public:
  [[nodiscard]] core::Action select_move(const core::Position& position) const override;
  [[nodiscard]] std::string name() const override;
};

class DepthLimitedMinimaxBot final : public Bot {
 public:
  explicit DepthLimitedMinimaxBot(int depth = 3);
  [[nodiscard]] core::Action select_move(const core::Position& position) const override;
  [[nodiscard]] std::string name() const override;

 private:
  int depth_;
};

[[nodiscard]] std::unique_ptr<Bot> make_bot(const std::string& name);
[[nodiscard]] std::vector<std::string> bot_names();
[[nodiscard]] BotMatchResult play_bot_match(
    const Bot& first,
    const Bot& second,
    int games,
    bool alternate_starts = true);

}  // namespace c4zero::bots
