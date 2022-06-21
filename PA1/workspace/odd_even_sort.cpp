// odd_even_sort.cpp
#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "worker.h"

size_t CalcBlockLen(const size_t n, const int nprocs, const int rank) {
  if (rank < 0) return 0;
  size_t block_size = ceiling(n, nprocs);
  size_t IO_offset = block_size * rank;
  bool out_of_range = IO_offset >= n;
  return out_of_range ? 0 : std::min(block_size, n - IO_offset);
}

void Worker::sort() {
  if (this->block_len == 0) return;
  std::sort(this->data, this->data + this->block_len);
  if (this->nprocs == 1) return;
  int left_rank = ((this->rank == 0) ? (-1) : (this->rank - 1));
  size_t left_block_len = CalcBlockLen(this->n, this->nprocs, left_rank);
  int right_rank = ((this->rank + 1 == this->nprocs) ? (-1) : (this->rank + 1));
  size_t right_block_len = CalcBlockLen(this->n, this->nprocs, right_rank);
  float* neighbor_data = new float[std::max(left_block_len, right_block_len)];
  float* data_0 = new float[this->block_len];
  float* data_1 = new float[this->block_len];
  std::memcpy(data_0, this->data, this->block_len * sizeof(float));
  MPI_Request requests[2];
  for (int i = 0; i < this->nprocs; ++i) {
    if ((i ^ this->rank) & 1) {
      if (right_block_len == 0)
        // no neighbor
        continue;
      MPI_Sendrecv(/*sendbuf=*/&(data_0[this->block_len - 1]),
                   /*sendcount=*/1, /*sendtype=*/MPI_FLOAT, /*dest=*/right_rank,
                   /*sendtag=*/i, /*recvbuf=*/&(neighbor_data[0]),
                   /*recvcount=*/1, /*recvtype=*/MPI_FLOAT,
                   /*source=*/right_rank, /*recvtag=*/i,
                   /*comm=*/MPI_COMM_WORLD, /*status=*/MPI_STATUS_IGNORE);
      if (data_0[this->block_len - 1] <= neighbor_data[0]) {
        // no need to merge
        continue;
      }
      MPI_Isend(/*buf=*/data_0, /*count=*/this->block_len,
                /*datatype=*/MPI_FLOAT, /*dest=*/right_rank, /*tag=*/i,
                /*comm=*/MPI_COMM_WORLD, /*request=*/&requests[0]);
      MPI_Irecv(/*buf=*/neighbor_data, /*count=*/right_block_len,
                /*datatype=*/MPI_FLOAT,
                /*source=*/right_rank, /*tag=*/i, /*comm=*/MPI_COMM_WORLD,
                /*request=*/&requests[1]);
      size_t j = 0, k = 0, l = 0;
      MPI_Wait(/*request=*/&requests[1], /*status=*/MPI_STATUS_IGNORE);
      while (j < this->block_len && k < right_block_len &&
             l < this->block_len) {
        if (data_0[j] < neighbor_data[k]) {
          data_1[l++] = data_0[j++];
        } else {
          data_1[l++] = neighbor_data[k++];
        }
      }
      while (j < this->block_len && l < this->block_len)
        data_1[l++] = data_0[j++];
      while (k < right_block_len && l < this->block_len)
        data_1[l++] = neighbor_data[k++];
    } else {
      if (left_block_len == 0)
        // no neighbor
        continue;
      MPI_Sendrecv(/*sendbuf=*/&(data_0[0]),
                   /*sendcount=*/1, /*sendtype=*/MPI_FLOAT, /*dest=*/left_rank,
                   /*sendtag=*/i,
                   /*recvbuf=*/&(neighbor_data[left_block_len - 1]),
                   /*recvcount=*/1, /*recvtype=*/MPI_FLOAT,
                   /*source=*/left_rank, /*recvtag=*/i,
                   /*comm=*/MPI_COMM_WORLD, /*status=*/MPI_STATUS_IGNORE);
      if (neighbor_data[left_block_len - 1] <= data_0[0])
        // no need to merge
        continue;
      MPI_Isend(/*buf=*/data_0, /*count=*/this->block_len,
                /*datatype=*/MPI_FLOAT, /*dest=*/left_rank, /*tag=*/i,
                /*comm=*/MPI_COMM_WORLD, /*request=*/&requests[0]);
      MPI_Irecv(/*buf=*/neighbor_data, /*count=*/left_block_len,
                /*datatype=*/MPI_FLOAT,
                /*source=*/left_rank, /*tag=*/i, /*comm=*/MPI_COMM_WORLD,
                /*request=*/&requests[1]);
      int j = left_block_len - 1, k = this->block_len - 1,
          l = this->block_len - 1;
      MPI_Wait(/*request=*/&requests[1], /*status=*/MPI_STATUS_IGNORE);
      while (j > -1 && k > -1 && l > -1) {
        if (neighbor_data[j] < data_0[k]) {
          data_1[l--] = data_0[k--];
        } else {
          data_1[l--] = neighbor_data[j--];
        }
      }
      while (j > -1 && l > -1) data_1[l--] = neighbor_data[j--];
      while (k > -1 && l > -1) data_1[l--] = data_0[k--];
    }
    MPI_Wait(/*request=*/&requests[0], /*status=*/MPI_STATUS_IGNORE);
    std::swap(data_0, data_1);
  }
  std::memcpy(this->data, data_0, (this->block_len) * sizeof(float));
  delete[] neighbor_data;
  delete[] data_0;
  delete[] data_1;
}
