#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

const char *version_name = "2D Cache Blocking";

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

int Min(const int a, const int b) { return (b < a) ? b : a; }

ptr_t stencil_7(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info,
                int nt) {
  omp_set_num_threads(28);

  static const int tx = 256;
  static const int ty = 16;

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

  for (int t = 0; t < nt; ++t) {
    cptr_t a0 = buffer[t % 2];
    ptr_t a1 = buffer[(t + 1) % 2];
    for (int yy = y_start; yy < y_end; yy += ty) {
      const int yy_end = Min(yy + ty, y_end);
      for (int xx = x_start; xx < x_end; xx += tx) {
        const int xx_end = Min(xx + tx, x_end);
#pragma omp parallel for
        for (int z = z_start; z < z_end; ++z) {
          for (int y = yy; y < yy_end; ++y) {
            for (int x = xx; x < xx_end; ++x) {
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
  return buffer[nt % 2];
}