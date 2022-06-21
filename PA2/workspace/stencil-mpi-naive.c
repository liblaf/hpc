#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

const char *version_name = "MPI Blocking";

int Min(const int a, const int b) { return (b < a) ? b : a; }

int Ceiling(const int a, const int b) { return (a + (b - 1)) / b; }

void create_dist_grid(dist_grid_info_t *grid_info) {
  grid_info->local_size_x = grid_info->global_size_x;
  grid_info->local_size_y = grid_info->global_size_y;
  grid_info->offset_x = 0;
  grid_info->offset_y = 0;
  grid_info->halo_size_x = 1;
  grid_info->halo_size_y = 1;
  grid_info->halo_size_z = 1;
  int block_size = Ceiling(grid_info->global_size_z, grid_info->p_num);
  grid_info->offset_z = block_size * grid_info->p_id;
  if (grid_info->offset_z >= grid_info->global_size_z) {
    grid_info->local_size_z = 0;
  } else {
    grid_info->local_size_z =
        Min(block_size, grid_info->global_size_z - grid_info->offset_z);
  }
}

void destroy_dist_grid(dist_grid_info_t *grid_info) {}

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

  for (int t = 0; t < nt; ++t) {
    cptr_t a0 = buffer[t % 2];
    ptr_t a1 = buffer[(t + 1) % 2];
    if (grid_info->p_id > 0) {
      MPI_Sendrecv(
          /*sendbuf=*/&a0[INDEX(0, 0, z_start, ldx, ldy)],
          /*sendcount=*/ldx * ldy, /*sendtype=*/MPI_DOUBLE,
          /*dest=*/grid_info->p_id - 1, /*sendtag=*/t,
          /*recvbuf=*/&a0[INDEX(0, 0, z_start - 1, ldx, ldy)],
          /*recvcount=*/ldx * ldy, /*recvtype=*/MPI_DOUBLE,
          /*source=*/grid_info->p_id - 1, /*recvtag=*/t,
          /*comm=*/MPI_COMM_WORLD, /*status=*/MPI_STATUS_IGNORE);
    }
    if (grid_info->p_id + 1 < grid_info->p_num) {
      MPI_Sendrecv(/*sendbuf=*/&a0[INDEX(0, 0, z_end - 1, ldx, ldy)],
                   /*sendcount=*/ldx * ldy, /*sendtype=*/MPI_DOUBLE,
                   /*dest=*/grid_info->p_id + 1, /*sendtag=*/t,
                   /*recvbuf=*/&a0[INDEX(0, 0, z_end, ldx, ldy)],
                   /*recvcount=*/ldx * ldy, /*recvtype=*/MPI_DOUBLE,
                   /*source=*/grid_info->p_id + 1, /*recvtag=*/t,
                   /*comm=*/MPI_COMM_WORLD, /*status=*/MPI_STATUS_IGNORE);
    }
    for (int z = z_start; z < z_end; ++z) {
      for (int y = y_start; y < y_end; ++y) {
        for (int x = x_start; x < x_end; ++x) {
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
