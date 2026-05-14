#pragma once

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#define C4ZERO_CHECK(expr) \
  do { \
    if (!(expr)) { \
      std::cerr << __FILE__ << ":" << __LINE__ << " check failed: " #expr "\n"; \
      return 1; \
    } \
  } while (false)

#define C4ZERO_CHECK_EQ(a, b) \
  do { \
    const auto c4zero_a = (a); \
    const auto c4zero_b = (b); \
    if (!(c4zero_a == c4zero_b)) { \
      std::cerr << __FILE__ << ":" << __LINE__ << " check failed: " #a " == " #b \
                << " (" << c4zero_a << " != " << c4zero_b << ")\n"; \
      return 1; \
    } \
  } while (false)
