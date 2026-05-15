#include "c4zero/arena/arena.hpp"
#include "c4zero/bots/heuristic.hpp"
#include "c4zero/curriculum/stage0.hpp"
#include "c4zero/data/shard.hpp"
#include "c4zero/model/torchscript.hpp"
#include "c4zero/play/terminal.hpp"
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

float to_float(const std::string& value) {
  return std::stof(value);
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
      << "  play --model checkpoints/current/inference.ts --simulations 800 --search-threads 4\n"
      << "  curriculum --stage 0 --samples 1000000 --shard-size 100000 --out /tmp/thakwani/rl-data/curriculum/stage0-v1\n"
      << "  selfplay --model checkpoints/current/inference.ts --games 2 --simulations 32 --search-threads 4 --out runs/smoke\n";
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
  config.search_threads = to_int(arg_value(argc, argv, "--search-threads", "1"));
  config.seed = static_cast<std::uint64_t>(std::stoull(arg_value(argc, argv, "--seed", "1")));
  const auto result = c4zero::arena::play_checkpoint_match(config);
  std::cout << result.summary() << "\n";
  return 0;
}

int run_curriculum(int argc, char** argv) {
  const int stage = to_int(arg_value(argc, argv, "--stage", "0"));
  if (stage != 0) {
    throw std::invalid_argument("only curriculum --stage 0 is supported");
  }
  c4zero::curriculum::Stage0Config config;
  config.samples = static_cast<std::uint64_t>(std::stoull(arg_value(argc, argv, "--samples", "1000000")));
  config.shard_size = static_cast<std::uint64_t>(std::stoull(arg_value(argc, argv, "--shard-size", "100000")));
  config.seed = static_cast<std::uint64_t>(std::stoull(arg_value(argc, argv, "--seed", "1")));
  config.output_dir = arg_value(argc, argv, "--out", "/tmp/thakwani/rl-data/curriculum/stage0-v1");
  config.use_symmetries = !has_arg(argc, argv, "--no-symmetries");
  config.git_commit = git_commit();

  const auto result = c4zero::curriculum::write_stage0_dataset(config);
  std::cout << "curriculum_stage=0"
            << " samples=" << result.samples
            << " shards=" << result.shards
            << " manifest=" << result.manifest_path << "\n";
  for (const auto& [category, count] : result.category_counts) {
    std::cout << "category=" << category << " count=" << count << "\n";
  }
  return 0;
}

int run_selfplay(int argc, char** argv) {
  const int games = to_int(arg_value(argc, argv, "--games", "2"));
  const int simulations = to_int(arg_value(argc, argv, "--simulations", "800"));
  const std::string out_dir = arg_value(argc, argv, "--out", "runs/c4zero-smoke");
  const std::string model_path = arg_value(argc, argv, "--model", "");
  const std::string device_name = arg_value(argc, argv, "--device", "cpu");
  const int search_threads = to_int(arg_value(argc, argv, "--search-threads", "4"));
  const float virtual_loss = to_float(arg_value(argc, argv, "--virtual-loss", "1.0"));
  const int inference_batch_size = to_int(arg_value(argc, argv, "--inference-batch-size", "128"));
  const int inference_max_wait_us = to_int(arg_value(argc, argv, "--inference-max-wait-us", "2000"));
  const auto seed = static_cast<std::uint64_t>(std::stoull(arg_value(argc, argv, "--seed", "1")));

  c4zero::search::UniformEvaluator evaluator;
  std::unique_ptr<c4zero::model::AsyncBatchedTorchScriptEvaluator> torchscript_evaluator;
  c4zero::search::Evaluator* active_evaluator = &evaluator;
  if (!model_path.empty()) {
    torch::Device device(device_name == "cuda" ? torch::kCUDA : torch::kCPU);
    c4zero::model::AsyncBatchedTorchScriptConfig inference_config;
    inference_config.max_batch_size = inference_batch_size;
    inference_config.max_wait_us = inference_max_wait_us;
    torchscript_evaluator =
        std::make_unique<c4zero::model::AsyncBatchedTorchScriptEvaluator>(model_path, device, inference_config);
    active_evaluator = torchscript_evaluator.get();
  }
  c4zero::selfplay::SelfPlayConfig config;
  config.games = games;
  config.mcts.simulations_per_move = simulations;
  config.mcts.search_threads = search_threads;
  config.mcts.virtual_loss = virtual_loss;
  config.seed = seed;

  std::vector<c4zero::data::SelfPlaySample> all_samples;
  for (int game = 0; game < games; ++game) {
    auto generated = c4zero::selfplay::generate_game(*active_evaluator, config, static_cast<std::uint64_t>(game));
    all_samples.insert(all_samples.end(), generated.samples.begin(), generated.samples.end());
    std::cout << "game=" << game
              << " plies=" << generated.plies
              << " terminal=" << generated.terminal_value
              << " simulations=" << generated.completed_simulations
              << " leaf_evals=" << generated.leaf_evaluations
              << " terminal_evals=" << generated.terminal_evaluations
              << " max_depth=" << generated.max_depth
              << " search_ms=" << generated.search_time_ms << "\n";
  }
  if (torchscript_evaluator) {
    const auto stats = torchscript_evaluator->stats();
    std::cout << "inference_requests=" << stats.requests
              << " inference_batches=" << stats.batches
              << " mean_batch_size=" << stats.mean_batch_size()
              << " max_batch_size=" << stats.max_batch_size
              << " mean_wait_ms=" << stats.mean_wait_ms()
              << " total_inference_ms=" << stats.total_inference_ms << "\n";
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
  manifest_config.search_threads = config.mcts.search_threads;
  manifest_config.virtual_loss = config.mcts.virtual_loss;
  manifest_config.inference_batch_size = model_path.empty() ? 1 : inference_batch_size;
  manifest_config.inference_max_wait_us = model_path.empty() ? 0 : inference_max_wait_us;
  manifest_config.evaluator_type = model_path.empty() ? "uniform" : "async_torchscript";
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

int run_play(int argc, char** argv) {
  c4zero::play::TerminalPlayConfig config;
  config.model_path = arg_value(argc, argv, "--model", "");
  config.device = arg_value(argc, argv, "--device", "cpu");
  config.simulations = to_int(arg_value(argc, argv, "--simulations", "800"));
  config.search_threads = to_int(arg_value(argc, argv, "--search-threads", "4"));
  config.virtual_loss = to_float(arg_value(argc, argv, "--virtual-loss", "1.0"));
  config.inference_batch_size = to_int(arg_value(argc, argv, "--inference-batch-size", "128"));
  config.inference_max_wait_us = to_int(arg_value(argc, argv, "--inference-max-wait-us", "2000"));
  config.seed = static_cast<std::uint64_t>(std::stoull(arg_value(argc, argv, "--seed", "1")));
  config.bot_first = has_arg(argc, argv, "--bot-first");
  config.value_mode = c4zero::play::parse_value_mode(arg_value(argc, argv, "--value-mode", "zero"));
  return c4zero::play::run_terminal_game(std::cin, std::cout, config);
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
    if (command == "curriculum") {
      return run_curriculum(argc, argv);
    }
    if (command == "selfplay") {
      return run_selfplay(argc, argv);
    }
    if (command == "play") {
      return run_play(argc, argv);
    }
    usage();
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 2;
  }
}
