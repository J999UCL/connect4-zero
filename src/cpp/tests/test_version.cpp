#include "c4zero/version/info.hpp"
#include "test_support.hpp"

using namespace c4zero;

int main() {
  auto info = version::current_version_info();
  C4ZERO_CHECK(!info.manifest_json.empty());
  C4ZERO_CHECK_EQ(info.fields.at("project_version"), "0.1.0");
  C4ZERO_CHECK_EQ(version::version_field("game_rules_version"), "1.0.0");
  return 0;
}
