cmake_minimum_required(VERSION 3.20)
project(c4zero LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(READ "${CMAKE_CURRENT_LIST_DIR}/../../version_manifest.json" C4ZERO_VERSION_MANIFEST_JSON)
configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/cmake/version.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/c4zero/version/version.hpp"
  @ONLY
)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(c4zero_core
  src/core/position.cpp
  src/core/symmetry.cpp
  src/version/version.cpp
)
target_include_directories(c4zero_core PUBLIC
  "${CMAKE_CURRENT_BINARY_DIR}/generated"
  "${CMAKE_CURRENT_LIST_DIR}/include"
)

add_library(c4zero_bots
  src/bots/heuristic.cpp
)
target_link_libraries(c4zero_bots PUBLIC c4zero_core)

add_library(c4zero_search
  src/search/puct.cpp
)
target_link_libraries(c4zero_search PUBLIC c4zero_core)

add_library(c4zero_data
  src/data/shard.cpp
  src/data/replay_buffer.cpp
)
target_link_libraries(c4zero_data PUBLIC c4zero_core)

add_library(c4zero_model
  src/model/encoder.cpp
)
target_link_libraries(c4zero_model PUBLIC c4zero_core "${TORCH_LIBRARIES}")

add_library(c4zero_selfplay
  src/selfplay/selfplay.cpp
)
target_link_libraries(c4zero_selfplay PUBLIC c4zero_core c4zero_search c4zero_data)

add_library(c4zero_arena
  src/arena/arena.cpp
)
target_link_libraries(c4zero_arena PUBLIC c4zero_core c4zero_bots c4zero_search)

add_executable(c4zero apps/c4zero.cpp)
target_link_libraries(c4zero PRIVATE
  c4zero_core
  c4zero_bots
  c4zero_search
  c4zero_data
  c4zero_selfplay
  c4zero_arena
  c4zero_model
  "${TORCH_LIBRARIES}"
)

enable_testing()

function(c4zero_test name)
  add_executable(${name} tests/${name}.cpp)
  target_link_libraries(${name} PRIVATE c4zero_core c4zero_bots c4zero_search c4zero_data)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

c4zero_test(test_core)
c4zero_test(test_bots)
c4zero_test(test_puct)
c4zero_test(test_data)
c4zero_test(test_version)

add_executable(test_torchscript_evaluator tests/test_torchscript_evaluator.cpp)
target_link_libraries(test_torchscript_evaluator PRIVATE c4zero_core c4zero_search c4zero_model "${TORCH_LIBRARIES}")
add_test(NAME test_torchscript_evaluator COMMAND test_torchscript_evaluator)
