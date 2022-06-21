#include <x86intrin.h>

#include "aplusb.h"

void a_plus_b_intrinsic(float* a, float* b, float* c, int n) {
  for (int i = 0; i < n; i += 8) {
    _mm256_store_ps(
        c + i, _mm256_add_ps(_mm256_load_ps(a + i), _mm256_load_ps(b + i)));
  }
}
