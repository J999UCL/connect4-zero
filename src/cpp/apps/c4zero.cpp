#include "c4zero/arena/arena.hpp"
#include "c4zero/bots/heuristic.hpp"
#include "c4zero/data/shard.hpp"
#include "c4zero/model/torchscript.hpp"
#include "c4zero/search/puct.hpp"
#include "c4zero/selfplay/selfplay.hpp"
#include "c4zero/version/info.hpp"

#include <filesystem>
#include <array>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

int to_int(const std::string& value) {
  return std::stoi(value);
}

std::string arg_value(int argc, char** argv, const std::string& name, const std::string& fallback) {
  for (int i = 0; i + 1 < argc; ++i) {
    if (argv[i] == name) {
      return argv[i + 1];
    }
  }
  return fallback;
}

bool has_arg(int argc, char** argv, const std::string& name) {
  for (int i = 0; i < argc; ++i) {
    if (argv[i] == name) {
      return true;
    }
  }
  return false;
}

std::string git_commit() {
  std::array<char, 128> buffer{};
  std::string output;
  FILE* pipe = popen("git rev-parse --short HEAD 2>/dev/null", "r");
  if (pipe == nullptr) {
    return "";
  }
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    output += buffer.data();
  }
  pclose(pipe);
  while (!output.empty() && (output.back() == '\n' || output.back() == '\r' || output.back() == ' ')) {
    output.pop_back();
  }
  return output;
}

void usage() {
  std::cout
      << "c4zero commands:\n"
      << "  version --json\n"
      << "  bots\n"
      << "  botmatch --bot-a center --bot-b tactical --games 20\n"
      << "  arena --model-a checkpoints/a/inference.ts --model-b checkpoints/b/inference.ts --games 20 --simulations 800\n"
      << "  selfplay --model checkpoints/current/inference.ts --games 2 --simulations 32 --out runs/smoke\n";
}

int run_botmatch(int argc, char** argv) {
  const auto bot_a = c4zero::bots::make_bot(arg_value(argc, argv, "--bot-a", "center"));
  const auto bot_b = c4zero::bots::make_bot(arg_value(argc, argv, "--bot-b", "tactical"));
  const int games = to_int(arg_value(argc, argv, "--games", "2"));
  const auto result = c4zero::bots::play_bot_match(*bot_a, *bot_b, games, true);
  std::cout << result.summary() << "\n";
  return 0;
}

int run_arena(int argc, char** argv) {
  c4zero::arena::ArenaConfig config;
  config.model_a = arg_value(argc, argv, "--model-a", "");
  config.model_b = arg_value(argc, argv, "--model-b", "");
  config.device = arg_value(argc, argv, "--device", "cpu");
  config.games = to_int(arg_value(argc, argv, "--games", "2"));
  config.simulations = to_int(arg_value(argc, argv, "--simulations", "800"));
  config.seed = static_cast<std::uint64_t>(std::stoull(arg_value(argc, argv, "--seed", "1")));
  const auto result = c4zero::arena::play_checkpoint_match(config);
  std::cout << result.summary() << "\n";
  return 0;
}

int run_selfplay(int argc, char** argv) {
  const int games = to_int(arg_value(argc, argv, "--games", "2"));
  const int simulations = to_int(arg_value(argc, argv, "--simulations", "800"));
  const std::string out_dir = arg_value(argc, argv, "--out", "runs/c4zero-smoke");
  const std::string model_path = arg_value(argc, argv, "--model", "");
  const std::string device_name = arg_value(argc, argv, "--device", "cpu");
  const auto seed = static_cast<std::uint64_t>(std::stoull(arg_value(argc, argv, "--seed", "1")));

  c4zero::search::UniformEvaluator evaluator;
  std::unique_ptr<c4zero::model::TorchScriptEvaluator> torchscript_evaluator;
  c4zero::search::Evaluator* active_evaluator = &evaluator;
  if (!model_path.empty()) {
    torch::Device device(device_name == "cuda" ? torch::kCUDA : torch::kCPU);
    torchscript_evaluator = std::make_unique<c4zero::model::TorchScriptEvaluator>(model_path, device);
    active_evaluator = torchscript_evaluator.get();
  }
  c4zero::selfplay::SelfPlayConfig config;
  config.games = games;
  config.mcts.simulations_per_move = simulations;
  config.seed = seed;

  std::vector<c4zero::data::SelfPlaySample> all_samples;
  for (int game = 0; game < games; ++game) {
    auto generated = c4zero::selfplay::generate_game(*active_evaluator, config, static_cast<std::uint64_t>(game));
    all_samples.insert(all_samples.end(), generated.samples.begin(), generated.samples.end());
    std::cout << "game=" << game << " plies=" << generated.plies << " terminal=" << generated.terminal_value << "\n";
  }

  std::filesystem::create_directories(out_dir + "/shards");
  const std::string shard_path = out_dir + "/shards/shard-000000.c4az";
  c4zero::data::write_shard(shard_path, all_samples);
  c4zero::data::SelfPlayManifestConfig manifest_config;
  manifest_config.model_checkpoint = model_path.empty() ? "uniform-evaluator" : model_path;
  manifest_config.device = device_name;
  manifest_config.simulations_per_move = config.mcts.simulations_per_move;
  manifest_config.c_base = config.mcts.c_base;
  manifest_config.c_init = config.mcts.c_init;
  manifest_config.root_dirichlet_alpha = config.mcts.root_dirichlet_alpha;
  manifest_config.root_exploration_fraction = config.mcts.root_exploration_fraction;
  manifest_config.temperature_sampling_plies = config.temperature_sampling_plies;
  manifest_config.add_root_noise = config.add_root_noise;
  manifest_config.seed = config.seed;
  manifest_config.git_commit = git_commit();
  c4zero::data::write_manifest(
      out_dir + "/manifest.json",
      "shards/shard-000000.c4az",
      static_cast<std::uint64_t>(games),
      static_cast<std::uint64_t>(all_samples.size()),
      manifest_config);
  std::cout << "samples=" << all_samples.size() << " manifest=" << out_dir << "/manifest.json\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      usage();
      return 0;
    }
    const std::string command = argv[1];
    if (command == "version") {
      if (has_arg(argc, argv, "--json")) {
        std::cout << c4zero::version::current_version_json() << "\n";
      } else {
        std::cout << "c4zero " << c4zero::version::version_field("project_version") << "\n";
      }
      return 0;
    }
    if (command == "bots") {
      for (const auto& name : c4zero::bots::bot_names()) {
        std::cout << name << "\n";
      }
      return 0;
    }
    if (command == "botmatch") {
      return run_botmatch(argc, argv);
    }
    if (command == "arena") {
      return run_arena(argc, argv);
    }
    if (command == "selfplay") {
      return run_selfplay(argc, argv);
    }
    usage();
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 2;
  }
}
