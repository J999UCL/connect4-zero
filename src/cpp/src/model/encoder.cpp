#include "c4zero/model/torchscript.hpp"

namespace c4zero::model {

torch::Tensor encode_positions(const std::vector<core::Position>& positions, torch::Device device) {
  auto tensor = torch::zeros(
      {static_cast<long>(positions.size()), 2, core::kBoardSize, core::kBoardSize, core::kBoardSize},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  auto accessor = tensor.accessor<float, 5>();
  for (std::size_t b = 0; b < positions.size(); ++b) {
    for (int z = 0; z < core::kBoardSize; ++z) {
      for (int y = 0; y < core::kBoardSize; ++y) {
        for (int x = 0; x < core::kBoardSize; ++x) {
          const core::Bitboard mask = core::cell_mask(x, y, z);
          if ((positions[b].current & mask) != 0) {
            accessor[static_cast<long>(b)][0][z][y][x] = 1.0f;
          }
          if ((positions[b].opponent & mask) != 0) {
            accessor[static_cast<long>(b)][1][z][y][x] = 1.0f;
          }
        }
      }
    }
  }
  return tensor.to(device);
}

}  // namespace c4zero::model
