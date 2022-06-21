#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

const char *version_name = "Time Skewing";

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

  const int tx = x_end - x_start;
  const int ty = 32;
  const int tz = 64;
  for (int zz = z_start; zz < z_end; zz += tz) {
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
              for (int x = block_min_x; x < block_max_x; x++) {
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
      }
    }
  }
  return buffer[nt % 2];
}
