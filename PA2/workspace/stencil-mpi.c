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
