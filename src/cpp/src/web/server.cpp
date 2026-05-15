#include "c4zero/web/server.hpp"

#include "c4zero/model/torchscript.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <netinet/in.h>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <torch/torch.h>
#include <unistd.h>
#include <vector>

namespace c4zero::web {
namespace {

torch::Device parse_device(const std::string& device) {
  if (device == "cuda") {
    return torch::Device(torch::kCUDA);
  }
  if (device == "cpu") {
    return torch::Device(torch::kCPU);
  }
  throw std::invalid_argument("unsupported web device: " + device);
}

std::string json_escape(const std::string& value) {
  std::ostringstream out;
  for (const char ch : value) {
    switch (ch) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << ch;
        break;
    }
  }
  return out.str();
}

template <typename T, std::size_t N>
std::string array_json(const std::array<T, N>& values) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << values[i];
  }
  out << "]";
  return out.str();
}

bool bit_is_set(core::Bitboard bits, int x, int y, int z) {
  return (bits & core::cell_mask(x, y, z)) != 0;
}

bool session_human_to_move(const core::Position& position, bool bot_first) {
  const bool human_is_first = !bot_first;
  const bool first_player_to_move = (position.ply % 2 == 0);
  return human_is_first == first_player_to_move;
}

std::string terminal_message(const core::Position& position, bool bot_first) {
  const auto value = position.terminal_value();
  if (!value.has_value()) {
    return "";
  }
  if (*value == 0.0f) {
    return "Draw.";
  }
  return session_human_to_move(position, bot_first) ? "Bot wins." : "You win.";
}

std::string cells_json(const core::Position& position, bool bot_first) {
  const bool human_turn = session_human_to_move(position, bot_first);
  const core::Bitboard human_bits = human_turn ? position.current : position.opponent;
  const core::Bitboard bot_bits = human_turn ? position.opponent : position.current;

  std::ostringstream out;
  out << "[";
  bool first = true;
  for (int z = 0; z < core::kBoardSize; ++z) {
    for (int y = 0; y < core::kBoardSize; ++y) {
      for (int x = 0; x < core::kBoardSize; ++x) {
        const bool human = bit_is_set(human_bits, x, y, z);
        const bool bot = bit_is_set(bot_bits, x, y, z);
        if (!human && !bot) {
          continue;
        }
        if (!first) {
          out << ",";
        }
        first = false;
        out << "{\"x\":" << x
            << ",\"y\":" << y
            << ",\"z\":" << z
            << ",\"owner\":\"" << (human ? "human" : "bot") << "\"}";
      }
    }
  }
  out << "]";
  return out.str();
}

std::string heights_json(const core::Position& position) {
  std::ostringstream out;
  out << "[";
  for (int i = 0; i < core::kNumActions; ++i) {
    if (i != 0) {
      out << ",";
    }
    out << static_cast<int>(position.heights[i]);
  }
  out << "]";
  return out.str();
}

std::string search_json(const search::SearchResult& result) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(6);
  out << "{"
      << "\"selectedAction\":" << result.selected_action
      << ",\"rootValue\":" << result.root_value
      << ",\"searchMs\":" << result.search_time_ms
      << ",\"maxDepth\":" << result.max_depth
      << ",\"expandedNodes\":" << result.expanded_nodes
      << ",\"completedSimulations\":" << result.completed_simulations
      << ",\"leafEvaluations\":" << result.leaf_evaluations
      << ",\"terminalEvaluations\":" << result.terminal_evaluations
      << ",\"rootVisits\":" << result.root_real_visits
      << ",\"pendingEvalWaits\":" << result.pending_eval_waits
      << ",\"virtualLossWaits\":" << result.virtual_loss_waits
      << ",\"policy\":" << array_json(result.policy)
      << ",\"visitCounts\":" << array_json(result.visit_counts)
      << ",\"qValues\":" << array_json(result.q_values)
      << "}";
  return out.str();
}

class ValueOverrideEvaluator final : public search::Evaluator {
 public:
  ValueOverrideEvaluator(search::Evaluator& evaluator, WebValueMode mode) : evaluator_(evaluator), mode_(mode) {}

  search::Evaluation evaluate(const core::Position& position) override {
    search::Evaluation evaluation = evaluator_.evaluate(position);
    if (mode_ == WebValueMode::Zero) {
      evaluation.value = 0.0f;
    }
    return evaluation;
  }

 private:
  search::Evaluator& evaluator_;
  WebValueMode mode_;
};

struct HttpRequest {
  std::string method;
  std::string path;
  std::string body;
};

std::string status_text(int status) {
  if (status == 200) return "OK";
  if (status == 400) return "Bad Request";
  if (status == 404) return "Not Found";
  if (status == 405) return "Method Not Allowed";
  if (status == 500) return "Internal Server Error";
  return "OK";
}

std::string content_type_for(const std::filesystem::path& path) {
  const auto ext = path.extension().string();
  if (ext == ".html") return "text/html; charset=utf-8";
  if (ext == ".js") return "text/javascript; charset=utf-8";
  if (ext == ".css") return "text/css; charset=utf-8";
  if (ext == ".json") return "application/json; charset=utf-8";
  if (ext == ".svg") return "image/svg+xml";
  return "application/octet-stream";
}

void write_all(int fd, const std::string& data) {
  const char* cursor = data.data();
  std::size_t remaining = data.size();
  while (remaining > 0) {
    const ssize_t written = send(fd, cursor, remaining, 0);
    if (written <= 0) {
      return;
    }
    cursor += written;
    remaining -= static_cast<std::size_t>(written);
  }
}

void send_response(int fd, int status, const std::string& content_type, const std::string& body) {
  std::ostringstream headers;
  headers << "HTTP/1.1 " << status << " " << status_text(status) << "\r\n"
          << "Content-Type: " << content_type << "\r\n"
          << "Content-Length: " << body.size() << "\r\n"
          << "Cache-Control: no-store\r\n"
          << "Connection: close\r\n"
          << "\r\n";
  write_all(fd, headers.str());
  write_all(fd, body);
}

void send_json(int fd, int status, const std::string& body) {
  send_response(fd, status, "application/json; charset=utf-8", body);
}

void send_error(int fd, int status, const std::string& message) {
  send_json(fd, status, "{\"error\":\"" + json_escape(message) + "\"}");
}

std::optional<HttpRequest> read_request(int fd) {
  std::string raw;
  std::array<char, 4096> buffer{};
  while (raw.find("\r\n\r\n") == std::string::npos) {
    const ssize_t read = recv(fd, buffer.data(), buffer.size(), 0);
    if (read <= 0) {
      return std::nullopt;
    }
    raw.append(buffer.data(), static_cast<std::size_t>(read));
    if (raw.size() > 1'000'000) {
      throw std::runtime_error("HTTP request is too large");
    }
  }

  const std::size_t header_end = raw.find("\r\n\r\n");
  const std::string headers = raw.substr(0, header_end);
  std::istringstream input(headers);
  HttpRequest request;
  std::string version;
  input >> request.method >> request.path >> version;
  if (request.method.empty() || request.path.empty()) {
    throw std::runtime_error("malformed HTTP request");
  }

  int content_length = 0;
  std::string line;
  std::getline(input, line);
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    const std::string prefix = "Content-Length:";
    if (line.rfind(prefix, 0) == 0) {
      content_length = std::stoi(line.substr(prefix.size()));
    }
  }

  request.body = raw.substr(header_end + 4);
  while (static_cast<int>(request.body.size()) < content_length) {
    const ssize_t read = recv(fd, buffer.data(), buffer.size(), 0);
    if (read <= 0) {
      break;
    }
    request.body.append(buffer.data(), static_cast<std::size_t>(read));
  }
  if (static_cast<int>(request.body.size()) > content_length) {
    request.body.resize(static_cast<std::size_t>(content_length));
  }
  return request;
}

int parse_action_body(const std::string& body) {
  const std::size_t key = body.find("action");
  if (key == std::string::npos) {
    throw std::invalid_argument("missing action");
  }
  const std::size_t colon = body.find(':', key);
  if (colon == std::string::npos) {
    throw std::invalid_argument("missing action value");
  }
  std::size_t start = colon + 1;
  while (start < body.size() && std::isspace(static_cast<unsigned char>(body[start])) != 0) {
    ++start;
  }
  std::size_t end = start;
  while (end < body.size() && (std::isdigit(static_cast<unsigned char>(body[end])) != 0 || body[end] == '-')) {
    ++end;
  }
  if (start == end) {
    throw std::invalid_argument("invalid action value");
  }
  return std::stoi(body.substr(start, end - start));
}

std::filesystem::path resolve_static_path(const std::filesystem::path& web_root, std::string request_path) {
  const std::size_t query = request_path.find('?');
  if (query != std::string::npos) {
    request_path.resize(query);
  }
  if (request_path == "/") {
    request_path = "/index.html";
  }
  if (request_path.find("..") != std::string::npos) {
    throw std::invalid_argument("path traversal is not allowed");
  }
  while (!request_path.empty() && request_path.front() == '/') {
    request_path.erase(request_path.begin());
  }
  return web_root / request_path;
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("could not open static file");
  }
  std::ostringstream out;
  out << input.rdbuf();
  return out.str();
}

int create_server_socket(const std::string& host, int port) {
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    throw std::runtime_error("socket failed: " + std::string(std::strerror(errno)));
  }
  int reuse = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<std::uint16_t>(port));
  if (host == "0.0.0.0") {
    address.sin_addr.s_addr = INADDR_ANY;
  } else if (host == "127.0.0.1" || host == "localhost") {
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  } else {
    close(fd);
    throw std::invalid_argument("web host must be 127.0.0.1, localhost, or 0.0.0.0");
  }

  if (bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
    close(fd);
    throw std::runtime_error("bind failed: " + std::string(std::strerror(errno)));
  }
  if (listen(fd, 16) < 0) {
    close(fd);
    throw std::runtime_error("listen failed: " + std::string(std::strerror(errno)));
  }
  return fd;
}

void handle_request(
    int client_fd,
    const HttpRequest& request,
    WebGameSession& session,
    const std::filesystem::path& web_root) {
  try {
    if (request.path == "/api/health") {
      send_json(client_fd, 200, "{\"ok\":true}");
      return;
    }
    if (request.path == "/api/state") {
      if (request.method != "GET") {
        send_error(client_fd, 405, "GET required");
        return;
      }
      send_json(client_fd, 200, session.state_json());
      return;
    }
    if (request.path == "/api/new") {
      if (request.method != "POST") {
        send_error(client_fd, 405, "POST required");
        return;
      }
      session.reset();
      if (!session.human_to_move() && !session.position().is_terminal()) {
        session.play_bot_action();
      }
      send_json(client_fd, 200, session.state_json());
      return;
    }
    if (request.path == "/api/move") {
      if (request.method != "POST") {
        send_error(client_fd, 405, "POST required");
        return;
      }
      session.play_human_action(parse_action_body(request.body));
      send_json(client_fd, 200, session.state_json());
      return;
    }
    if (request.path == "/api/bot") {
      if (request.method != "POST") {
        send_error(client_fd, 405, "POST required");
        return;
      }
      if (!session.human_to_move() && !session.position().is_terminal()) {
        session.play_bot_action();
      }
      send_json(client_fd, 200, session.state_json());
      return;
    }

    if (request.method != "GET") {
      send_error(client_fd, 405, "GET required");
      return;
    }
    const auto static_path = resolve_static_path(web_root, request.path);
    if (!std::filesystem::exists(static_path) || !std::filesystem::is_regular_file(static_path)) {
      send_error(client_fd, 404, "not found");
      return;
    }
    send_response(client_fd, 200, content_type_for(static_path), read_file(static_path));
  } catch (const std::invalid_argument& error) {
    send_error(client_fd, 400, error.what());
  } catch (const std::exception& error) {
    send_error(client_fd, 500, error.what());
  }
}

}  // namespace

WebValueMode parse_web_value_mode(const std::string& value) {
  if (value == "model") {
    return WebValueMode::Model;
  }
  if (value == "zero") {
    return WebValueMode::Zero;
  }
  throw std::invalid_argument("web value mode must be 'model' or 'zero'");
}

WebGameSession::WebGameSession(search::Evaluator& evaluator, search::PuctConfig mcts_config, bool bot_first)
    : evaluator_(evaluator), mcts_(mcts_config), bot_first_(bot_first), tree_(position_) {}

void WebGameSession::reset_tree() {
  tree_ = search::SearchTree(position_);
}

void WebGameSession::reset() {
  position_ = core::Position::empty();
  reset_tree();
  has_last_search_ = false;
}

void WebGameSession::play_human_action(core::Action action) {
  if (position_.is_terminal()) {
    throw std::invalid_argument("game is already over");
  }
  if (!human_to_move()) {
    throw std::invalid_argument("it is not the human turn");
  }
  if (!position_.is_legal(action)) {
    throw std::invalid_argument("illegal action " + std::to_string(action));
  }
  position_ = position_.play(action);
  if (!tree_.advance(action)) {
    reset_tree();
  }
}

void WebGameSession::play_bot_action() {
  if (position_.is_terminal()) {
    return;
  }
  if (human_to_move()) {
    throw std::invalid_argument("it is not the bot turn");
  }
  if (tree_.root().position.compact_string() != position_.compact_string()) {
    reset_tree();
  }
  last_search_ = mcts_.search(tree_, evaluator_, false, 0.0);
  has_last_search_ = true;
  if (last_search_.selected_action < 0 || !position_.is_legal(last_search_.selected_action)) {
    throw std::runtime_error("web play selected an illegal bot action");
  }
  position_ = position_.play(last_search_.selected_action);
  if (!tree_.advance(last_search_.selected_action)) {
    reset_tree();
  }
}

const core::Position& WebGameSession::position() const {
  return position_;
}

bool WebGameSession::human_to_move() const {
  return session_human_to_move(position_, bot_first_);
}

bool WebGameSession::has_last_search() const {
  return has_last_search_;
}

const search::SearchResult& WebGameSession::last_search() const {
  if (!has_last_search_) {
    throw std::logic_error("no last search is available");
  }
  return last_search_;
}

std::string WebGameSession::state_json() const {
  std::ostringstream out;
  out << std::fixed << std::setprecision(6);
  out << "{"
      << "\"ply\":" << static_cast<int>(position_.ply)
      << ",\"botFirst\":" << (bot_first_ ? "true" : "false")
      << ",\"humanToMove\":" << (human_to_move() ? "true" : "false")
      << ",\"terminal\":" << (position_.is_terminal() ? "true" : "false")
      << ",\"terminalValue\":";
  const auto terminal = position_.terminal_value();
  if (terminal.has_value()) {
    out << *terminal;
  } else {
    out << "null";
  }
  out << ",\"message\":\"" << json_escape(terminal_message(position_, bot_first_)) << "\""
      << ",\"legalMask\":" << position_.legal_mask()
      << ",\"heights\":" << heights_json(position_)
      << ",\"cells\":" << cells_json(position_, bot_first_)
      << ",\"lastSearch\":";
  if (has_last_search_) {
    out << search_json(last_search_);
  } else {
    out << "null";
  }
  out << "}";
  return out.str();
}

int run_web_server(const WebPlayConfig& config) {
  if (config.port <= 0 || config.port > 65535) {
    throw std::invalid_argument("web port must be in 1..65535");
  }
  if (config.simulations <= 0) {
    throw std::invalid_argument("web simulations must be positive");
  }
  if (config.search_threads <= 0) {
    throw std::invalid_argument("web search threads must be positive");
  }

  std::signal(SIGPIPE, SIG_IGN);

  search::UniformEvaluator uniform_evaluator;
  std::unique_ptr<model::AsyncBatchedTorchScriptEvaluator> torchscript_evaluator;
  search::Evaluator* base_evaluator = &uniform_evaluator;
  if (!config.model_path.empty()) {
    model::AsyncBatchedTorchScriptConfig inference_config;
    inference_config.max_batch_size = config.inference_batch_size;
    inference_config.max_wait_us = config.inference_max_wait_us;
    torchscript_evaluator = std::make_unique<model::AsyncBatchedTorchScriptEvaluator>(
        config.model_path,
        parse_device(config.device),
        inference_config);
    base_evaluator = torchscript_evaluator.get();
  }
  ValueOverrideEvaluator evaluator(*base_evaluator, config.value_mode);

  search::PuctConfig mcts_config;
  mcts_config.simulations_per_move = config.simulations;
  mcts_config.search_threads = config.search_threads;
  mcts_config.virtual_loss = config.virtual_loss;
  mcts_config.seed = config.seed;
  WebGameSession session(evaluator, mcts_config, config.bot_first);
  if (config.bot_first) {
    session.play_bot_action();
  }

  const std::filesystem::path web_root = std::filesystem::absolute(config.web_root);
  if (!std::filesystem::exists(web_root / "index.html")) {
    throw std::runtime_error("web root does not contain index.html: " + web_root.string());
  }
  const int server_fd = create_server_socket(config.host, config.port);
  std::cout << "c4zero web play server listening at http://" << config.host << ":" << config.port << "\n"
            << "web_root=" << web_root.string() << "\n"
            << "model=" << (config.model_path.empty() ? "uniform-evaluator" : config.model_path) << "\n"
            << "simulations=" << config.simulations
            << " search_threads=" << config.search_threads
            << " value_mode=" << (config.value_mode == WebValueMode::Zero ? "zero" : "model") << "\n";

  while (true) {
    sockaddr_in client_address{};
    socklen_t client_length = sizeof(client_address);
    const int client_fd = accept(server_fd, reinterpret_cast<sockaddr*>(&client_address), &client_length);
    if (client_fd < 0) {
      continue;
    }
    try {
      const auto request = read_request(client_fd);
      if (request.has_value()) {
        handle_request(client_fd, *request, session, web_root);
      }
    } catch (const std::exception& error) {
      send_error(client_fd, 500, error.what());
    }
    close(client_fd);
  }
}

}  // namespace c4zero::web
