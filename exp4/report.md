---
title: "exp4: 自动向量化与基于 intrinsic 的手动向量化"
category: "Course Work"
tags:
  - "Introduction to High Performance Computing"
  - "HPC"
  - "intrinsic"
---

# exp4: 自动向量化与基于 intrinsic 的手动向量化

## Performance

| Method    | Time    |
| --------- | ------- |
| baseline  | 4711 us |
| auto simd | 530 us  |
| intrinsic | 514 us  |

## Implementation

```c++
void a_plus_b_intrinsic(float* a, float* b, float* c, int n) {
  for (int i = 0; i < n; i += 8) {
    _mm256_store_ps(
        c + i, _mm256_add_ps(_mm256_load_ps(a + i), _mm256_load_ps(b + i)));
  }
}
```

