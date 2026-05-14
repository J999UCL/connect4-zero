#include "c4zero/version/info.hpp"
#include "c4zero/version/version.hpp"

#include <regex>

namespace c4zero::version {

VersionInfo current_version_info() {
  VersionInfo info;
  info.manifest_json = kVersionManifestJson;
  const std::regex field_re(R"re("([^"]+)"\s*:\s*"([^"]*)")re");
  for (std::sregex_iterator it(info.manifest_json.begin(), info.manifest_json.end(), field_re), end; it != end; ++it) {
    info.fields[(*it)[1].str()] = (*it)[2].str();
  }
  return info;
}

std::string current_version_json() {
  return kVersionManifestJson;
}

std::string version_field(const std::string& key) {
  auto info = current_version_info();
  const auto it = info.fields.find(key);
  if (it == info.fields.end()) {
    return "";
  }
  return it->second;
}

}  // namespace c4zero::version
