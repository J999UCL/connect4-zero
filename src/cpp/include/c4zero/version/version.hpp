#pragma once

namespace c4zero::version {

inline constexpr const char* kVersionManifestJson = R"c4zero({
  "project_version": "0.1.0",
  "cpp_abi_version": "1.0.0",
  "python_tools_version": "0.1.0",
  "dataset_schema_version": "1.0.0",
  "checkpoint_schema_version": "1.0.0",
  "model_config_version": "1.0.0",
  "encoder_version": "1.0.0",
  "game_rules_version": "1.0.0",
  "action_mapping_version": "1.0.0",
  "symmetry_version": "1.0.0"
})c4zero";

}  // namespace c4zero::version
