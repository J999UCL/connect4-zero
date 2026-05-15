#include "c4zero/play/terminal.hpp"
#include "test_support.hpp"

#include <sstream>
#include <string>

int main() {
  C4ZERO_CHECK(c4zero::play::parse_value_mode("model") == c4zero::play::ValueMode::Model);
  C4ZERO_CHECK(c4zero::play::parse_value_mode("zero") == c4zero::play::ValueMode::Zero);

  {
    c4zero::play::TerminalPlayConfig config;
    config.simulations = 4;
    config.search_threads = 2;
    std::istringstream input("bad\n99\n0\nquit\n");
    std::ostringstream output;
    const int status = c4zero::play::run_terminal_game(input, output, config);
    const std::string text = output.str();
    C4ZERO_CHECK_EQ(status, 0);
    C4ZERO_CHECK(text.find("Invalid input") != std::string::npos);
    C4ZERO_CHECK(text.find("Illegal action 99") != std::string::npos);
    C4ZERO_CHECK(text.find("You played action 0") != std::string::npos);
    C4ZERO_CHECK(text.find("Bot played action") != std::string::npos);
  }

  {
    c4zero::play::TerminalPlayConfig config;
    config.simulations = 4;
    config.search_threads = 2;
    config.bot_first = true;
    std::istringstream input("quit\n");
    std::ostringstream output;
    const int status = c4zero::play::run_terminal_game(input, output, config);
    const std::string text = output.str();
    C4ZERO_CHECK_EQ(status, 0);
    C4ZERO_CHECK(text.find("Bot played action") != std::string::npos);
    C4ZERO_CHECK(text.find("connect4>") != std::string::npos);
  }

  return 0;
}
