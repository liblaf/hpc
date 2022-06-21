#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

const char *version_name = "Intrinsic";

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

#ifndef KERNEL_INLINE
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

#ifdef KERNEL_INLINE
  const __m256d alpha_zzz = _mm256_set1_pd((double)ALPHA_ZZZ);
  const __m256d alpha_nzz = _mm256_set1_pd((double)ALPHA_NZZ);
  const __m256d alpha_pzz = _mm256_set1_pd((double)ALPHA_PZZ);
  const __m256d alpha_znz = _mm256_set1_pd((double)ALPHA_ZNZ);
  const __m256d alpha_zpz = _mm256_set1_pd((double)ALPHA_ZPZ);
  const __m256d alpha_zzn = _mm256_set1_pd((double)ALPHA_ZZN);
  const __m256d alpha_zzp = _mm256_set1_pd((double)ALPHA_ZZP);
#endif
  for (int t = 0; t < nt; ++t) {
    cptr_t a0 = buffer[t % 2];
    ptr_t a1 = buffer[(t + 1) % 2];
#pragma omp parallel for
    for (int z = z_start; z < z_end; ++z) {
      for (int y = y_start; y < y_end; ++y) {
        for (int x = x_start; x < x_end / 4 * 4; x += 4) {
#ifndef KERNEL_INLINE
          Kernel7(a0, a1, x, y, z, ldx, ldy);
#else
          __m256d zzz = _mm256_loadu_pd(a0 + INDEX(x, y, z, ldx, ldy));
          __m256d nzz = _mm256_loadu_pd(a0 + INDEX(x - 1, y, z, ldx, ldy));
          __m256d pzz = _mm256_loadu_pd(a0 + INDEX(x + 1, y, z, ldx, ldy));
          __m256d znz = _mm256_loadu_pd(a0 + INDEX(x, y - 1, z, ldx, ldy));
          __m256d zpz = _mm256_loadu_pd(a0 + INDEX(x, y + 1, z, ldx, ldy));
          __m256d zzn = _mm256_loadu_pd(a0 + INDEX(x, y, z - 1, ldx, ldy));
          __m256d zzp = _mm256_loadu_pd(a0 + INDEX(x, y, z + 1, ldx, ldy));
          __m256d res = _mm256_mul_pd(alpha_zzz, zzz);
          res = _mm256_fmadd_pd(alpha_zzz, zzz, res);
          res = _mm256_fmadd_pd(alpha_nzz, nzz, res);
          res = _mm256_fmadd_pd(alpha_pzz, pzz, res);
          res = _mm256_fmadd_pd(alpha_znz, znz, res);
          res = _mm256_fmadd_pd(alpha_zpz, zpz, res);
          res = _mm256_fmadd_pd(alpha_zzn, zzn, res);
          res = _mm256_fmadd_pd(alpha_zzp, zzp, res);
          _mm256_storeu_pd(a1 + INDEX(x, y, z, ldx, ldy), res);
#endif
        }
        for (int x = x_end / 4 * 4; x < x_end; ++x) {
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
