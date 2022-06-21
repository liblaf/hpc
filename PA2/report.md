---
title: "PA2: 模板计算"
category: "Course Work"
tags:
  - "Introduction to High Performance Computing"
  - "HPC"
  - "OMP"
  - "MPI"
---

# PA2: 模板计算

Size: 512 x 512 x 512

## Naive

### No OPT Performance

| Threads | Computation Time (s) | Performance (Gflop/s) | Speedup / Single Thread |
| ------- | -------------------- | --------------------- | ----------------------- |
| 1       | 267.075484           | 0.653310              | 1.                      |
| 2       | 135.902281           | 1.283886              | 1.96520182              |
| 4       | 70.080111            | 2.489766              | 3.81100243              |
| 8       | 36.813651            | 4.739629              | 7.25479328              |
| 16      | 25.793147            | 6.764706              | 10.35451164             |
| 28      | 17.400062            | 10.027726             | 15.34910839             |

### OPT Performance

| Threads | Computation Time (s) | Performance (Gflop/s) | Speedup / Single Thread | Speedup / Naive Single Thread |
| ------- | -------------------- | --------------------- | ----------------------- | ----------------------------- |
| 1       | 33.793919            | 5.163149              | 1.                      | 7.90306133                    |
| 2       | 20.366889            | 8.566996              | 1.65925795              | 13.11321731                   |
| 4       | 11.615848            | 15.021120             | 2.90929431              | 22.99233136                   |
| 8       | 10.494952            | 16.625426             | 3.22001670              | 25.44798947                   |
| 16      | 8.618103             | 20.246109             | 3.92127150              | 30.99004913                   |
| 28      | 7.622385             | 22.890871             | 4.43350967              | 35.03829882                   |

## OMP

使用 Time Skewing + Intrinsic 手动向量化进行优化. Time Skewing 通过在空间维度上进行分块, 在时间维度上进行斜向划分, 使得 `t + 1` 时刻的计算能够利用 `t` 时刻缓存在 cache 中的计算结果, 因而提升性能. 注意, 边界上的块会随 `t` 的增大而逐渐减小. 因此块的大小并不均衡. 好消息是, Time Skewing 并不并行块, 而是在每个块内部进行并行计算, 并不会导致负载的严重不均衡.

使用 Intel Intrinsic 手动向量化的优化效果并不明显, 与自动向量化相比几乎没有提升, 但聊胜于无.

此外, 由于计算顺序的不同, 结果可能会与串行得到的结果产生微小偏差, 在可接受范围之内, 算法本身没有错误.

其余的算法, 如 2D Cache Blocking, Cache Oblivious, Circular Queue 等算法也参考 [StencilProbe](http://people.csail.mit.edu/skamil/projects/stencilprobe/) 进行了实现和测试, 均不如 Time Skewing 高效, 测试结果见 [Performance](#Performance).

```c++
#define INTRINSIC

#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

const char *version_name = "OMP";

void create_dist_grid(dist_grid_info_t *grid_info) {
  // Naive implementation uses Process 0 to do all computations
  if (grid_info->p_id == 0) {
    grid_info->local_size_x = grid_info->global_size_x;
    grid_info->local_size_y = grid_info->global_size_y;
    grid_info->local_size_z = grid_info->global_size_z;
  } else {
    grid_info->local_size_x = 0;
    grid_info->local_size_y = 0;
    grid_info->local_size_z = 0;
  }
  grid_info->offset_x = 0;
  grid_info->offset_y = 0;
  grid_info->offset_z = 0;
  grid_info->halo_size_x = 1;
  grid_info->halo_size_y = 1;
  grid_info->halo_size_z = 1;
}

void destroy_dist_grid(dist_grid_info_t *grid_info) {}

int Min(const int a, const int b) { return (a < b) ? a : b; }

int Max(const int a, const int b) { return (a < b) ? b : a; }

#ifdef INTRINSIC
// calculate 4 elements as a vector
void Kernel7(cptr_t a0, ptr_t a1, const int x, const int y, const int z,
             const int ldx, const int ldy) {
  const __m256d alpha_zzz = _mm256_set1_pd((double)ALPHA_ZZZ);
  const __m256d alpha_nzz = _mm256_set1_pd((double)ALPHA_NZZ);
  const __m256d alpha_pzz = _mm256_set1_pd((double)ALPHA_PZZ);
  const __m256d alpha_znz = _mm256_set1_pd((double)ALPHA_ZNZ);
  const __m256d alpha_zpz = _mm256_set1_pd((double)ALPHA_ZPZ);
  const __m256d alpha_zzn = _mm256_set1_pd((double)ALPHA_ZZN);
  const __m256d alpha_zzp = _mm256_set1_pd((double)ALPHA_ZZP);
  __m256d zzz = _mm256_loadu_pd(a0 + INDEX(x, y, z, ldx, ldy));
  __m256d nzz = _mm256_loadu_pd(a0 + INDEX(x - 1, y, z, ldx, ldy));
  __m256d pzz = _mm256_loadu_pd(a0 + INDEX(x + 1, y, z, ldx, ldy));
  __m256d znz = _mm256_loadu_pd(a0 + INDEX(x, y - 1, z, ldx, ldy));
  __m256d zpz = _mm256_loadu_pd(a0 + INDEX(x, y + 1, z, ldx, ldy));
  __m256d zzn = _mm256_loadu_pd(a0 + INDEX(x, y, z - 1, ldx, ldy));
  __m256d zzp = _mm256_loadu_pd(a0 + INDEX(x, y, z + 1, ldx, ldy));
  __m256d res = _mm256_mul_pd(alpha_zzz, zzz);
  res = _mm256_fmadd_pd(alpha_nzz, nzz, res);
  res = _mm256_fmadd_pd(alpha_pzz, pzz, res);
  res = _mm256_fmadd_pd(alpha_znz, znz, res);
  res = _mm256_fmadd_pd(alpha_zpz, zpz, res);
  res = _mm256_fmadd_pd(alpha_zzn, zzn, res);
  res = _mm256_fmadd_pd(alpha_zzp, zzp, res);
  _mm256_storeu_pd(a1 + INDEX(x, y, z, ldx, ldy), res);
}
#endif

ptr_t stencil_7(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info,
                int nt) {
  omp_set_num_threads(28);

  ptr_t buffer[2] = {grid, aux};
  int x_start = grid_info->halo_size_x,
      x_end = grid_info->local_size_x + grid_info->halo_size_x;
  int y_start = grid_info->halo_size_y,
      y_end = grid_info->local_size_y + grid_info->halo_size_y;
  int z_start = grid_info->halo_size_z,
      z_end = grid_info->local_size_z + grid_info->halo_size_z;
  int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
  int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
  int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;

  const int tx = x_end - x_start;  // block size along x-axis
  const int ty = 16;               // block size along y-axis
  const int tz = 112;              // block size along z-axis
  for (int zz = z_start; zz < z_end; zz += tz) {
    // shrink size
    const int neg_z_slope = (zz == z_start) ? 0 : 1;
    const int pos_z_slope = (zz + tz < z_end) ? -1 : 0;
    for (int yy = y_start; yy < y_end; yy += ty) {
      const int neg_y_slope = (yy == y_start) ? 0 : 1;
      const int pos_y_slope = (yy + ty < y_end) ? -1 : 0;
      for (int xx = x_start; xx < x_end; xx += tx) {
        const int neg_x_slope = (xx == x_start) ? 0 : 1;
        const int pos_x_slope = (xx + tx < x_end) ? -1 : 0;
        for (int t = 0; t < nt; ++t) {
          const int block_min_x = Max(x_start, xx - t * neg_x_slope);
          const int block_min_y = Max(y_start, yy - t * neg_y_slope);
          const int block_min_z = Max(z_start, zz - t * neg_z_slope);
          const int block_max_x =
              Min(x_end, Max(x_start, xx + tx + t * pos_x_slope));
          const int block_max_y =
              Min(y_end, Max(y_start, yy + ty + t * pos_y_slope));
          const int block_max_z =
              Min(z_end, Max(z_start, zz + tz + t * pos_z_slope));
          cptr_t a0 = buffer[t % 2];
          ptr_t a1 = buffer[(t + 1) % 2];
#pragma omp parallel for
          for (int z = block_min_z; z < block_max_z; z++) {
            for (int y = block_min_y; y < block_max_y; y++) {
#ifdef INTRINSIC
              for (int x = block_min_x; x < block_max_x / 4 * 4; x += 4)
                Kernel7(a0, a1, x, y, z, ldx, ldy);
              for (int x = block_max_x / 4 * 4; x < block_max_x; ++x) {
                a1[INDEX(x, y, z, ldx, ldy)] =
                    ALPHA_ZZZ * a0[INDEX(x, y, z, ldx, ldy)] +
                    ALPHA_NZZ * a0[INDEX(x - 1, y, z, ldx, ldy)] +
                    ALPHA_PZZ * a0[INDEX(x + 1, y, z, ldx, ldy)] +
                    ALPHA_ZNZ * a0[INDEX(x, y - 1, z, ldx, ldy)] +
                    ALPHA_ZPZ * a0[INDEX(x, y + 1, z, ldx, ldy)] +
                    ALPHA_ZZN * a0[INDEX(x, y, z - 1, ldx, ldy)] +
                    ALPHA_ZZP * a0[INDEX(x, y, z + 1, ldx, ldy)];
              }
#else
#pragma omp simd
              for (int x = block_min_x; x < block_max_x; ++x) {
                a1[INDEX(x, y, z, ldx, ldy)] =
                    ALPHA_ZZZ * a0[INDEX(x, y, z, ldx, ldy)] +
                    ALPHA_NZZ * a0[INDEX(x - 1, y, z, ldx, ldy)] +
                    ALPHA_PZZ * a0[INDEX(x + 1, y, z, ldx, ldy)] +
                    ALPHA_ZNZ * a0[INDEX(x, y - 1, z, ldx, ldy)] +
                    ALPHA_ZPZ * a0[INDEX(x, y + 1, z, ldx, ldy)] +
                    ALPHA_ZZN * a0[INDEX(x, y, z - 1, ldx, ldy)] +
                    ALPHA_ZZP * a0[INDEX(x, y, z + 1, ldx, ldy)];
              }
#endif
            }
          }
        }
      }
    }
  }
  return buffer[nt % 2];
}
```

### OMP Performance

| Threads | Computation Time (s) | Performance (Gflop/s) | Speedup / Single Thread | Speedup / Naive Single Thread |
| ------- | -------------------- | --------------------- | ----------------------- | ----------------------------- |
| 1       | 23.760512            | 7.3434040             | 1.                      | 11.24030552                   |
| 2       | 12.095517            | 14.425432             | 1.9644067               | 22.08053145                   |
| 4       | 6.763701             | 25.796978             | 3.5129455               | 39.48658064                   |
| 8       | 3.647941             | 47.830557             | 6.513404                | 73.21265096                   |
| 16      | 2.625376             | 66.460208             | 9.05032707              | 101.72844132                  |
| 28      | 2.359089             | 73.962038             | 10.07190099             | 113.21124428                  |

## MPI

考虑对数据进行分块, 并尽量减小通信. 不难发现, 每个块都需要与其相邻的块交换边界上的数据, 因此通信量的大小与分块后内部多出的表面积成正比. 显然, 如果只沿一个方向分块, 无疑是增加面积最大的分块方法, 因此考虑沿 x, y, z 轴进行 3D Blocking, 在进程数为 2, 4, 8, 16, 28 时进行手动分块, 以减少通信.

在测试时, 我们发现非阻塞通信的提升并不显著, 且编程相对复杂, 容易产生死锁等问题, 因此最终选用 `Sendrecv()` 进行通信.

此外, 还使用了 Intel Intrinsic 手动向量化进行优化.

```c++
#define INTRINSIC

#include <immintrin.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

const char *version_name = "MPI";

typedef struct {
  int num_block_x, num_block_y, num_block_z;
  int id_x, id_y, id_z;
} GridId;

enum Direction { kXPred, kXSucc, kYPred, kYSucc, kZPred, kZSucc };

int Min(const int a, const int b) { return (b < a) ? b : a; }

int Ceiling(const int a, const int b) { return (a + (b - 1)) / b; }

void Blocking(const int id, const int global_size, const int num_block,
              int *local_size, int *offset) {
  int block_size = Ceiling(global_size, num_block);
  *offset = block_size * id;
  if ((*offset) < global_size) {
    *local_size = Min(block_size, global_size - (*offset));
  } else {
    *local_size = 0;
  }
}

void create_dist_grid(dist_grid_info_t *grid_info) {
  GridId *grid_id = malloc(sizeof(GridId));
  grid_info->additional_info = grid_id;
  grid_id->num_block_x = 1;
  grid_id->num_block_y = 1;
  grid_id->num_block_z = 1;
  switch (grid_info->p_num) {
    case 4: {
      grid_id->num_block_x = 1;
      grid_id->num_block_y = 2;
      grid_id->num_block_z = 2;
      break;
    }
    case 8: {
      grid_id->num_block_x = 2;
      grid_id->num_block_y = 2;
      grid_id->num_block_z = 2;
      break;
    }
    case 16: {
      grid_id->num_block_x = 2;
      grid_id->num_block_y = 2;
      grid_id->num_block_z = 4;
      break;
    }
    case 28: {
      grid_id->num_block_x = 2;
      grid_id->num_block_y = 2;
      grid_id->num_block_z = 7;
      break;
    }
    default: {
      grid_id->num_block_x = 1;
      grid_id->num_block_y = 1;
      grid_id->num_block_z = grid_info->p_num;
      break;
    }
  }
  grid_id->id_x = grid_info->p_id % grid_id->num_block_x;
  grid_id->id_y =
      (grid_info->p_id / grid_id->num_block_x) % grid_id->num_block_y;
  grid_id->id_z =
      grid_info->p_id / (grid_id->num_block_x * grid_id->num_block_y);
  Blocking(grid_id->id_x, grid_info->global_size_x, grid_id->num_block_x,
           &(grid_info->local_size_x), &(grid_info->offset_x));
  Blocking(grid_id->id_y, grid_info->global_size_y, grid_id->num_block_y,
           &(grid_info->local_size_y), &(grid_info->offset_y));
  Blocking(grid_id->id_z, grid_info->global_size_z, grid_id->num_block_z,
           &(grid_info->local_size_z), &(grid_info->offset_z));
  grid_info->halo_size_x = 1;
  grid_info->halo_size_y = 1;
  grid_info->halo_size_z = 1;
}

void destroy_dist_grid(dist_grid_info_t *grid_info) {
  free(grid_info->additional_info);
}

#ifdef INTRINSIC
void Kernel7(cptr_t a0, ptr_t a1, const int x, const int y, const int z,
             const int ldx, const int ldy) {
  const __m256d alpha_zzz = _mm256_set1_pd((double)ALPHA_ZZZ);
  const __m256d alpha_nzz = _mm256_set1_pd((double)ALPHA_NZZ);
  const __m256d alpha_pzz = _mm256_set1_pd((double)ALPHA_PZZ);
  const __m256d alpha_znz = _mm256_set1_pd((double)ALPHA_ZNZ);
  const __m256d alpha_zpz = _mm256_set1_pd((double)ALPHA_ZPZ);
  const __m256d alpha_zzn = _mm256_set1_pd((double)ALPHA_ZZN);
  const __m256d alpha_zzp = _mm256_set1_pd((double)ALPHA_ZZP);
  __m256d zzz = _mm256_loadu_pd(a0 + INDEX(x, y, z, ldx, ldy));
  __m256d nzz = _mm256_loadu_pd(a0 + INDEX(x - 1, y, z, ldx, ldy));
  __m256d pzz = _mm256_loadu_pd(a0 + INDEX(x + 1, y, z, ldx, ldy));
  __m256d znz = _mm256_loadu_pd(a0 + INDEX(x, y - 1, z, ldx, ldy));
  __m256d zpz = _mm256_loadu_pd(a0 + INDEX(x, y + 1, z, ldx, ldy));
  __m256d zzn = _mm256_loadu_pd(a0 + INDEX(x, y, z - 1, ldx, ldy));
  __m256d zzp = _mm256_loadu_pd(a0 + INDEX(x, y, z + 1, ldx, ldy));
  __m256d res = _mm256_mul_pd(alpha_zzz, zzz);
  res = _mm256_fmadd_pd(alpha_nzz, nzz, res);
  res = _mm256_fmadd_pd(alpha_pzz, pzz, res);
  res = _mm256_fmadd_pd(alpha_znz, znz, res);
  res = _mm256_fmadd_pd(alpha_zpz, zpz, res);
  res = _mm256_fmadd_pd(alpha_zzn, zzn, res);
  res = _mm256_fmadd_pd(alpha_zzp, zzp, res);
  _mm256_storeu_pd(a1 + INDEX(x, y, z, ldx, ldy), res);
}
#endif

ptr_t stencil_7(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info,
                int nt) {
  ptr_t buffer[2] = {grid, aux};
  int x_start = grid_info->halo_size_x,
      x_end = grid_info->local_size_x + grid_info->halo_size_x;
  int y_start = grid_info->halo_size_y,
      y_end = grid_info->local_size_y + grid_info->halo_size_y;
  int z_start = grid_info->halo_size_z,
      z_end = grid_info->local_size_z + grid_info->halo_size_z;
  int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
  int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
  int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;

  MPI_Datatype XY_PLANE, XZ_PLANE, YZ_PLANE;
  MPI_Type_vector(/*count=*/1, /*blocklength=*/ldx * ldy, /*stride=*/0,
                  /*oldtype=*/MPI_DOUBLE, /*newtype=*/&XY_PLANE);
  MPI_Type_vector(/*count=*/ldz, /*blocklength=*/ldx, /*stride=*/ldx * ldy,
                  /*oldtype=*/MPI_DOUBLE, /*newtype=*/&XZ_PLANE);
  MPI_Type_vector(/*count=*/ldy * ldz, /*blocklength=*/1, /*stride=*/ldx,
                  /*oldtype=*/MPI_DOUBLE, /*newtype=*/&YZ_PLANE);
  MPI_Type_commit(&XY_PLANE);
  MPI_Type_commit(&XZ_PLANE);
  MPI_Type_commit(&YZ_PLANE);

  GridId *grid_id = grid_info->additional_info;
  int pid[6];
  int block_size[6] = {0};
  int offset;
  if (grid_id->id_x > 0) {
    pid[kXPred] = INDEX(grid_id->id_x - 1, grid_id->id_y, grid_id->id_z,
                        grid_id->num_block_x, grid_id->num_block_y);
    Blocking(grid_id->id_x - 1, grid_info->global_size_x, grid_id->num_block_x,
             &block_size[kXPred], &offset);
  }
  if (grid_id->id_x + 1 < grid_id->num_block_x) {
    pid[kXSucc] = INDEX(grid_id->id_x + 1, grid_id->id_y, grid_id->id_z,
                        grid_id->num_block_x, grid_id->num_block_y);
    Blocking(grid_id->id_x + 1, grid_info->global_size_x, grid_id->num_block_x,
             &block_size[kXSucc], &offset);
  }
  if (grid_id->id_y > 0) {
    pid[kYPred] = INDEX(grid_id->id_x, grid_id->id_y - 1, grid_id->id_z,
                        grid_id->num_block_x, grid_id->num_block_y);
    Blocking(grid_id->id_y - 1, grid_info->global_size_y, grid_id->num_block_y,
             &block_size[kYPred], &offset);
  }
  if (grid_id->id_y + 1 < grid_id->num_block_y) {
    pid[kYSucc] = INDEX(grid_id->id_x, grid_id->id_y + 1, grid_id->id_z,
                        grid_id->num_block_x, grid_id->num_block_y);
    Blocking(grid_id->id_y + 1, grid_info->global_size_y, grid_id->num_block_y,
             &block_size[kYSucc], &offset);
  }
  if (grid_id->id_z > 0) {
    pid[kZPred] = INDEX(grid_id->id_x, grid_id->id_y, grid_id->id_z - 1,
                        grid_id->num_block_x, grid_id->num_block_y);
    Blocking(grid_id->id_z - 1, grid_info->global_size_z, grid_id->num_block_z,
             &block_size[kZPred], &offset);
  }
  if (grid_id->id_z + 1 < grid_id->num_block_z) {
    pid[kZSucc] = INDEX(grid_id->id_x, grid_id->id_y, grid_id->id_z + 1,
                        grid_id->num_block_x, grid_id->num_block_y);
    Blocking(grid_id->id_z + 1, grid_info->global_size_z, grid_id->num_block_z,
             &block_size[kZSucc], &offset);
  }
  for (int t = 0; t < nt; ++t) {
    ptr_t a0 = buffer[t % 2];
    ptr_t a1 = buffer[(t + 1) % 2];
    if (block_size[kXPred]) {
      MPI_Sendrecv(/*sendbuf=*/&a0[INDEX(x_start, 0, 0, ldx, ldy)],
                   /*sendcount=*/1, /*sendtype=*/YZ_PLANE,
                   /*dest=*/pid[kXPred], /*sendtag=*/kXPred,
                   /*recvbuf=*/&a0[INDEX(x_start - 1, 0, 0, ldx, ldy)],
                   /*recvcount=*/1, /*recvtype=*/YZ_PLANE,
                   /*source=*/pid[kXPred], /*recvtag=*/kXSucc,
                   /*comm=*/MPI_COMM_WORLD, /*status=*/MPI_STATUS_IGNORE);
    }
    if (block_size[kXSucc]) {
      MPI_Sendrecv(/*sendbuf=*/&a0[INDEX(x_end - 1, 0, 0, ldx, ldy)],
                   /*sendcount=*/1, /*sendtype=*/YZ_PLANE,
                   /*dest=*/pid[kXSucc], /*sendtag=*/kXSucc,
                   /*recvbuf=*/&a0[INDEX(x_end, 0, 0, ldx, ldy)],
                   /*recvcount=*/1, /*recvtype=*/YZ_PLANE,
                   /*source=*/pid[kXSucc], /*recvtag=*/kXPred,
                   /*comm=*/MPI_COMM_WORLD, /*status=*/MPI_STATUS_IGNORE);
    }
    if (block_size[kYPred]) {
      MPI_Sendrecv(/*sendbuf=*/&a0[INDEX(0, y_start, 0, ldx, ldy)],
                   /*sendcount=*/1, /*sendtype=*/XZ_PLANE,
                   /*dest=*/pid[kYPred], /*sendtag=*/kYPred,
                   /*recvbuf=*/&a0[INDEX(0, y_start - 1, 0, ldx, ldy)],
                   /*recvcount=*/1, /*recvtype=*/XZ_PLANE,
                   /*source=*/pid[kYPred], /*recvtag=*/kYSucc,
                   /*comm=*/MPI_COMM_WORLD, /*status=*/MPI_STATUS_IGNORE);
    }
    if (block_size[kYSucc]) {
      MPI_Sendrecv(/*sendbuf=*/&a0[INDEX(0, y_end - 1, 0, ldx, ldy)],
                   /*sendcount=*/1, /*sendtype=*/XZ_PLANE,
                   /*dest=*/pid[kYSucc], /*sendtag=*/kYSucc,
                   /*recvbuf=*/&a0[INDEX(0, y_end, 0, ldx, ldy)],
                   /*recvcount=*/1, /*recvtype=*/XZ_PLANE,
                   /*source=*/pid[kYSucc], /*recvtag=*/kYPred,
                   /*comm=*/MPI_COMM_WORLD, /*status=*/MPI_STATUS_IGNORE);
    }
    if (block_size[kZPred]) {
      MPI_Sendrecv(/*sendbuf=*/&a0[INDEX(0, 0, z_start, ldx, ldy)],
                   /*sendcount=*/1, /*sendtype=*/XY_PLANE,
                   /*dest=*/pid[kZPred], /*sendtag=*/kZPred,
                   /*recvbuf=*/&a0[INDEX(0, 0, z_start - 1, ldx, ldy)],
                   /*recvcount=*/1, /*recvtype=*/XY_PLANE,
                   /*source=*/pid[kZPred], /*recvtag=*/kZSucc,
                   /*comm=*/MPI_COMM_WORLD, /*status=*/MPI_STATUS_IGNORE);
    }
    if (block_size[kZSucc]) {
      MPI_Sendrecv(/*sendbuf=*/&a0[INDEX(0, 0, z_end - 1, ldx, ldy)],
                   /*sendcount=*/1, /*sendtype=*/XY_PLANE,
                   /*dest=*/pid[kZSucc], /*sendtag=*/kZSucc,
                   /*recvbuf=*/&a0[INDEX(0, 0, z_end, ldx, ldy)],
                   /*recvcount=*/1, /*recvtype=*/XY_PLANE,
                   /*source=*/pid[kZSucc], /*recvtag=*/kZPred,
                   /*comm=*/MPI_COMM_WORLD, /*status=*/MPI_STATUS_IGNORE);
    }
    for (int z = z_start; z < z_end; ++z) {
      for (int y = y_start; y < y_end; ++y) {
#ifdef INTRINSIC
        for (int x = x_start; x < x_end / 4 * 4; x += 4)
          Kernel7(a0, a1, x, y, z, ldx, ldy);
        for (int x = x_end / 4 * 4; x < x_end; ++x) {
#else
        for (int x = x_start; x < x_end; ++x) {
#endif
          a1[INDEX(x, y, z, ldx, ldy)] =
              ALPHA_ZZZ * a0[INDEX(x, y, z, ldx, ldy)] +
              ALPHA_NZZ * a0[INDEX(x - 1, y, z, ldx, ldy)] +
              ALPHA_PZZ * a0[INDEX(x + 1, y, z, ldx, ldy)] +
              ALPHA_ZNZ * a0[INDEX(x, y - 1, z, ldx, ldy)] +
              ALPHA_ZPZ * a0[INDEX(x, y + 1, z, ldx, ldy)] +
              ALPHA_ZZN * a0[INDEX(x, y, z - 1, ldx, ldy)] +
              ALPHA_ZZP * a0[INDEX(x, y, z + 1, ldx, ldy)];
        }
      }
    }
  }
  return buffer[nt % 2];
}
```

### MPI Performance

| Threads | Computation Time (s) | Performance (Gflop/s) | Speedup / Single Thread | Speedup / Naive Single Thread |
| ------- | -------------------- | --------------------- | ----------------------- | ----------------------------- |
| 1       | 31.195547            | 5.593204              | 1.                      | 8.561332290                   |
| 2       | 16.056561            | 10.866776             | 1.94285351              | 16.63341446                   |
| 4       | 8.557309             | 20.389943             | 3.64548531              | 31.21021108                   |
| 8       | 4.688768             | 37.212980             | 6.65324919              | 56.96067717                   |
| 16      | 3.599826             | 48.469855             | 8.66584788              | 74.19120326                   |
| 28      | 3.494326             | 49.933255             | 8.92748682              | 76.43118122                   |

## Performance

### Naive

#### No OPT

| Size            | Computation Time (s) | Performance (Gflop/s) |
| --------------- | -------------------- | --------------------- |
| 256 x 256 x 256 | 1.310379             | 16.644326             |
| 512 x 512 x 512 | 10.179150            | 17.141220             |
| 768 x 768 x 768 | 33.206675            | 17.733793             |

#### OPT

| Size            | Computation Time (s) | Performance (Gflop/s) | Speedup / Naive |
| --------------- | -------------------- | --------------------- | --------------- |
| 256 x 256 x 256 | 1.042685             | 20.917516             | 1.25673554      |
| 512 x 512 x 512 | 6.357943             | 27.443318             | 1.60101311      |
| 768 x 768 x 768 | 21.682391            | 27.159379             | 1.53150423      |

### OMP

#### 2D Cache Blocking

```c++
static const int tx = 256;
static const int ty = 16;
```

| Size            | Computation Time (s) | Performance (Gflop/s) | Speedup / Naive | Speedup / Naive OPT |
| --------------- | -------------------- | --------------------- | --------------- | ------------------- |
| 256 x 256 x 256 | 0.656014             | 33.246826             | 1.99748707      | 1.58942515          |
| 512 x 512 x 512 | 4.403168             | 39.626707             | 2.31177868      | 1.44394738          |
| 768 x 768 x 768 | 15.625801            | 37.686406             | 2.12511819      | 1.38760190          |

#### Cache Oblivious

```c++
static const int kCutoff = (1 << 20);
static const int ds = 1;
```

| Size            | Computation Time (s) | Performance (Gflop/s) | Speedup / Naive | Speedup / Naive OPT |
| --------------- | -------------------- | --------------------- | --------------- | ------------------- |
| 256 x 256 x 256 | 0.290660             | 75.037455             | 4.50829039      | 3.58730238          |
| 512 x 512 x 512 | 5.862571             | 29.762205             | 1.73629444      | 1.08449733          |
| 768 x 768 x 768 | 17.230060            | 34.177494             | 1.92725234      | 1.25840484          |

#### Time Skewing

```c++
const int tx = x_end - x_start;
const int ty = 32;
const int tz = 64;
```

| Size            | Computation Time (s) | Performance (Gflop/s) | Speedup / Naive | Speedup / Naive OPT |
| --------------- | -------------------- | --------------------- | --------------- | ------------------- |
| 256 x 256 x 256 | 0.244840             | 89.080177             | 5.35198463      | 4.25864032          |
| 512 x 512 x 512 | 3.916044             | 44.555941             | 2.59934480      | 1.62356246          |
| 768 x 768 x 768 | 11.524923            | 51.096243             | 2.88129240      | 1.88134799          |

#### Circular Queue

| Size            | Computation Time (s) | Performance (Gflop/s) | Speedup / Naive | Speedup / Naive OPT |
| --------------- | -------------------- | --------------------- | --------------- | ------------------- |
| 256 x 256 x 256 | 1.461332             | 14.925003             | 0.89670216      | 0.71351699          |
| 512 x 512 x 512 | 12.507371            | 13.950418             | 0.81385211      | 0.50833569          |
| 768 x 768 x 768 | 40.458871            | 14.555035             | 0.82075138      | 0.53591192          |

#### Auto SIMD

| Size            | Computation Time (s) | Performance (Gflop/s) | Speedup / Naive | Speedup / Naive OPT |
| --------------- | -------------------- | --------------------- | --------------- | ------------------- |
| 256 x 256 x 256 | 0.728613             | 29.934117             | 1.79845774      | 1.43105505          |
| 512 x 512 x 512 | 5.759591             | 30.294345             | 1.76733891      | 1.10388784          |
| 768 x 768 x 768 | 16.067388            | 36.650655             | 2.06671269      | 1.34946587          |

#### Intrinsic

| Size            | Computation Time (s) | Performance (Gflop/s) | Speedup / Naive | Speedup / Naive OPT |
| --------------- | -------------------- | --------------------- | --------------- | ------------------- |
| 256 x 256 x 256 | 0.715376             | 30.487998             | 1.83173521      | 1.45753435          |
| 512 x 512 x 512 | 5.494364             | 31.756734             | 1.85265308      | 1.15717546          |
| 768 x 768 x 768 | 16.741729            | 35.174399             | 1.98346733      | 1.29511058          |

#### Time Skewing + Intrinsic

```c++
const int tx = x_end - x_start;
const int ty = 16;
const int tz = 112;
```

| Size            | Computation Time (s) | Performance (Gflop/s) | Speedup / Naive | Speedup / Naive OPT |
| --------------- | -------------------- | --------------------- | --------------- | ------------------- |
| 256 x 256 x 256 | 0.252214             | 86.475845             | 5.19551498      | 4.13413548          |
| 512 x 512 x 512 | 2.525018             | 69.101709             | 4.03131802      | 2.51797939          |
| 768 x 768 x 768 | 8.686356             | 67.793709             | 3.82285442      | 2.49614356          |

### MPI

#### Blocking Communication

| Size            | Computation Time (s) | Performance (Gflop/s) | Speedup / Naive | Speedup / Naive OPT |
| --------------- | -------------------- | --------------------- | --------------- | ------------------- |
| 256 x 256 x 256 | 0.501527             | 43.487911             | 2.61277693      | 2.07901889          |
| 512 x 512 x 512 | 4.659057             | 37.450291             | 2.18480896      | 1.36464151          |
| 768 x 768 x 768 | 15.672776            | 37.573451             | 2.11874871      | 1.38344294          |

#### Non-Blocking Communication

| Size            | Computation Time (s) | Performance (Gflop/s) | Speedup / Naive | Speedup / Naive OPT |
| --------------- | -------------------- | --------------------- | --------------- | ------------------- |
| 256 x 256 x 256 | 0.490962             | 44.423804             | 2.66900588      | 2.12376097          |
| 512 x 512 x 512 | 4.637739             | 37.622439             | 2.19485188      | 1.37091437          |
| 768 x 768 x 768 | 16.453953            | 35.789594             | 2.01815788      | 1.31776187          |

#### 3D Blocking

| Size            | Computation Time (s) | Performance (Gflop/s) | Speedup / Naive | Speedup / Naive OPT |
| --------------- | -------------------- | --------------------- | --------------- | ------------------- |
| 256 x 256 x 256 | 0.377116             | 57.834689             | 3.47473902      | 2.76489278          |
| 512 x 512 x 512 | 3.540040             | 49.288444             | 2.87543384      | 1.79600892          |
| 768 x 768 x 768 | 14.868124            | 39.606897             | 2.23341374      | 1.45831379          |

#### 3D Blocking + Intrinsic

| Size            | Computation Time (s) | Performance (Gflop/s) | Speedup / Naive | Speedup / Naive OPT |
| --------------- | -------------------- | --------------------- | --------------- | ------------------- |
| 256 x 256 x 256 | 0.400361             | 54.476819             | 3.27299640      | 2.60436368          |
| 512 x 512 x 512 | 3.460323             | 50.423912             | 2.94167580      | 1.83738395          |
| 768 x 768 x 768 | 14.525021            | 40.542475             | 2.28617053      | 1.49276149          |
