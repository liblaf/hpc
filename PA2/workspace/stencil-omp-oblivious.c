#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

const char *version_name = "Cache Oblivious";

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

void Walk3(ptr_t *A, int nx, int ny, int nz, int t0, int t1, int x0, int dx0,
           int x1, int dx1, int y0, int dy0, int y1, int dy1, int z0, int dz0,
           int z1, int dz1) {
  static const int kCutoff = (1 << 20);
  static const int ds = 1;

  int dt = t1 - t0;
  if (dt == 1 || (x1 - x0) * (y1 - y0) * (z1 - z0) < kCutoff) {
    for (int t = t0; t < t1; t++) {
      cptr_t a0 = A[t % 2];
      ptr_t a1 = A[(t + 1) % 2];
#pragma omp parallel for
      for (int z = z0 + (t - t0) * dz0; z < z1 + (t - t0) * dz1; ++z) {
        for (int y = y0 + (t - t0) * dy0; y < y1 + (t - t0) * dy1; ++y) {
          for (int x = x0 + (t - t0) * dx0; x < x1 + (t - t0) * dx1; ++x) {
            a1[INDEX(x, y, z, nx, ny)] =
                ALPHA_ZZZ * a0[INDEX(x, y, z, nx, ny)] +
                ALPHA_NZZ * a0[INDEX(x - 1, y, z, nx, ny)] +
                ALPHA_PZZ * a0[INDEX(x + 1, y, z, nx, ny)] +
                ALPHA_ZNZ * a0[INDEX(x, y - 1, z, nx, ny)] +
                ALPHA_ZPZ * a0[INDEX(x, y + 1, z, nx, ny)] +
                ALPHA_ZZN * a0[INDEX(x, y, z - 1, nx, ny)] +
                ALPHA_ZZP * a0[INDEX(x, y, z + 1, nx, ny)];
          }
        }
      }
    }
  } else if (dt > 1) {
    if (2 * (z1 - z0) + (dz1 - dz0) * dt >= 4 * ds * dt) {
      int zm = (2 * (z0 + z1) + (2 * ds + dz0 + dz1) * dt) / 4;
      Walk3(A, nx, ny, nz, t0, t1, x0, dx0, x1, dx1, y0, dy0, y1, dy1, z0, dz0,
            zm, -ds);
      Walk3(A, nx, ny, nz, t0, t1, x0, dx0, x1, dx1, y0, dy0, y1, dy1, zm, -ds,
            z1, dz1);
    } else if (2 * (y1 - y0) + (dy1 - dy0) * dt >= 4 * ds * dt) {
      int ym = (2 * (y0 + y1) + (2 * ds + dy0 + dy1) * dt) / 4;
      Walk3(A, nx, ny, nz, t0, t1, x0, dx0, x1, dx1, y0, dy0, ym, -ds, z0, dz0,
            z1, dz1);
      Walk3(A, nx, ny, nz, t0, t1, x0, dx0, x1, dx1, ym, -ds, y1, dy1, z0, dz0,
            z1, dz1);
    } else {
      int s = dt / 2;
      Walk3(A, nx, ny, nz, t0, t0 + s, x0, dx0, x1, dx1, y0, dy0, y1, dy1, z0,
            dz0, z1, dz1);
      Walk3(A, nx, ny, nz, t0 + s, t1, x0 + dx0 * s, dx0, x1 + dx1 * s, dx1,
            y0 + dy0 * s, dy0, y1 + dy1 * s, dy1, z0 + dz0 * s, dz0,
            z1 + dz1 * s, dz1);
    }
  }
}

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

  Walk3(buffer, ldx, ldy, ldz, 0, nt, x_start, 0, x_end, 0, y_start, 0, y_end,
        0, z_start, 0, z_end, 0);
  return buffer[nt % 2];
}
