#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

const char *version_name = "Circular Queue";

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

ptr_t queue_planes, queue_plane_0, queue_plane_1, queue_plane_2;
int *queue_planes_indices;

// This method creates the circular queues that will be needed for the
// circular_queue() method.  It is only called when more than one iteration is
// being performed.
void CircularQueueInit(int ldx, int ty, int timesteps) {
  int num_points_in_queue_plane, t;
  queue_planes_indices = malloc((timesteps - 1) * sizeof(int));
  if (!queue_planes_indices) {
    printf("Error on array queue_planes_indices malloc.\n");
    exit(EXIT_FAILURE);
  }
  int queue_planes_index_ptr = 0;
  for (t = 1; t < timesteps; t++) {
    queue_planes_indices[t - 1] = queue_planes_index_ptr;
    num_points_in_queue_plane = (ty + 2 * (timesteps - t)) * ldx;
    queue_planes_index_ptr += num_points_in_queue_plane;
  }
  queue_planes = malloc(3 * queue_planes_index_ptr * sizeof(double));
  if (!queue_planes) {
    printf("Error on array queue_planes malloc.\n");
    exit(EXIT_FAILURE);
  }
  queue_plane_0 = queue_planes;
  queue_plane_1 = &queue_planes[queue_planes_index_ptr];
  queue_plane_2 = &queue_planes[2 * queue_planes_index_ptr];
}

// This method traverses each slab and uses the circular queues to perform the
// specified number of iterations.The circular queue at a given timestep is
// shrunken in the y-dimension from the circular queue at the previous timestep.
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

  const int ty = 256;
  int num_blocks_y = (ldy - 2) / ty;
  CircularQueueInit(ldx, ty, nt);
  for (int s = 0; s < num_blocks_y; ++s) {
    for (int k = 1; k < (ldz + nt - 2); ++k) {
      for (int t = 0; t < nt; ++t) {
        ptr_t read_queue_plane_0, read_queue_plane_1, read_queue_plane_2,
            write_queue_plane;
        if ((k > t) && (k < (ldz + t - 1))) {
          if (t == 0) {
            read_queue_plane_0 = &grid[INDEX(0, 0, k - 1, ldx, ldy)];
            read_queue_plane_1 = &grid[INDEX(0, 0, k, ldx, ldy)];
            read_queue_plane_2 = &grid[INDEX(0, 0, k + 1, ldx, ldy)];
          } else {
            read_queue_plane_0 = &queue_plane_0[queue_planes_indices[t - 1]];
            read_queue_plane_1 = &queue_plane_1[queue_planes_indices[t - 1]];
            read_queue_plane_2 = &queue_plane_2[queue_planes_indices[t - 1]];
          }

          // determine the edges of the queues
          int write_block_min_y = s * ty - (nt - t) + 2;
          int write_block_max_y = (s + 1) * ty + (nt - t);
          int write_block_real_min_y = write_block_min_y;
          int write_block_real_max_y = write_block_max_y;

          if (write_block_min_y < 1) {
            write_block_min_y = 0;
            write_block_real_min_y = 1;
          }
          if (write_block_max_y > ldy - 1) {
            write_block_max_y = ldy;
            write_block_real_max_y = ldy - 1;
          }

          int write_offset;
          if (t == (nt - 1)) {
            write_queue_plane = aux;
            write_offset = 0;
          } else {
            write_queue_plane = &queue_plane_2[queue_planes_indices[t]];
            write_offset = INDEX(0, write_block_min_y, k - t, ldx, ldy);
          }

          int read_offset;
          if ((write_block_min_y == 0) || (t == 0)) {
            read_offset = INDEX(0, 0, k - t, ldx, ldy);
          } else {
            read_offset = INDEX(0, write_block_min_y - 1, k - t, ldx, ldy);
          }

          // use ghost cells for the bottommost and topmost planes
          if (k == (t + 1)) {
            read_queue_plane_0 = grid;
          }
          if (k == (ldz + t - 2)) {
            read_queue_plane_2 = &grid[INDEX(0, 0, ldz - 1, ldx, ldy)];
          }

          // copy ghost cells
          if (t < (nt - 1)) {
            for (int j = (write_block_min_y + 1); j < (write_block_max_y - 1);
                 j++) {
              write_queue_plane[INDEX(0, j, k - t, ldx, ldy) - write_offset] =
                  read_queue_plane_1[INDEX(0, j, k - t, ldx, ldy) -
                                     read_offset];
              write_queue_plane[INDEX(ldx - 1, j, k - t, ldx, ldy) -
                                write_offset] =
                  read_queue_plane_1[INDEX(ldx - 1, j, k - t, ldx, ldy) -
                                     read_offset];
            }
            if (write_block_min_y == 0) {
              for (int i = 1; i < (ldx - 1); i++) {
                write_queue_plane[INDEX(i, write_block_min_y, k - t, ldx, ldy) -
                                  write_offset] =
                    read_queue_plane_1[INDEX(i, write_block_min_y, k - t, ldx,
                                             ldy) -
                                       read_offset];
              }
            }
            if (write_block_max_y == ldy) {
              for (int i = 1; i < (ldx - 1); i++) {
                write_queue_plane[INDEX(i, write_block_real_max_y, k - t, ldx,
                                        ldy) -
                                  write_offset] =
                    read_queue_plane_1[INDEX(i, write_block_real_max_y, k - t,
                                             ldx, ldy) -
                                       read_offset];
              }
            }
          }

          // actual calculations
#pragma omp parallel for
          for (int j = write_block_real_min_y; j < write_block_real_max_y;
               j++) {
            for (int i = 1; i < (ldx - 1); i++) {
              write_queue_plane[INDEX(i, j, k - t, ldx, ldy) - write_offset] =
                  ALPHA_ZZZ * read_queue_plane_1[INDEX(i, j, k - t, ldx, ldy) -
                                                 read_offset] +
                  ALPHA_ZZN * read_queue_plane_0[INDEX(i, j, k - t, ldx, ldy) -
                                                 read_offset] +
                  ALPHA_ZZP * read_queue_plane_2[INDEX(i, j, k - t, ldx, ldy) -
                                                 read_offset] +
                  ALPHA_ZNZ *
                      read_queue_plane_1[INDEX(i, j - 1, k - t, ldx, ldy) -
                                         read_offset] +
                  ALPHA_NZZ *
                      read_queue_plane_1[INDEX(i - 1, j, k - t, ldx, ldy) -
                                         read_offset] +
                  ALPHA_PZZ *
                      read_queue_plane_1[INDEX(i + 1, j, k - t, ldx, ldy) -
                                         read_offset] +
                  ALPHA_ZPZ *
                      read_queue_plane_1[INDEX(i, j + 1, k - t, ldx, ldy) -
                                         read_offset];
            }
          }
        }
      }
      if (nt > 0) {
        ptr_t temp_queue_plane = queue_plane_0;
        queue_plane_0 = queue_plane_1;
        queue_plane_1 = queue_plane_2;
        queue_plane_2 = temp_queue_plane;
      }
    }
  }
  free(queue_planes_indices);
  free(queue_planes);
  return aux;
}
