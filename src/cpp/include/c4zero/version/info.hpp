#pragma once

#include <map>
#include <string>

namespace c4zero::version {

struct VersionInfo {
  std::string manifest_json;
  std::map<std::string, std::string> fields;
};

[[nodiscard]] VersionInfo current_version_info();
[[nodiscard]] std::string current_version_json();
[[nodiscard]] std::string version_field(const std::string& key);

}  // namespace c4zero::version
