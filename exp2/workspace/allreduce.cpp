#include <mpi.h>
#include <time.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>

#define EPS 1e-8

namespace ch = std::chrono;

void Ring_Allreduce(void* sendbuf, void* recvbuf, int n, MPI_Comm comm,
                    int comm_sz, int my_rank) {
  int count = n / comm_sz;
  int succ = (my_rank + 1) % comm_sz;
  int pred = (my_rank - 1 + comm_sz) % comm_sz;
  float* send_buf_begin = static_cast<float*>(sendbuf);
  float* recv_buf_begin = static_cast<float*>(recvbuf);
  memcpy(recv_buf_begin, send_buf_begin, n * sizeof(float));
  MPI_Request req[comm_sz - 1];
  for (int k = 0; k < comm_sz - 1; ++k) {
    MPI_Isend(recv_buf_begin + (((my_rank - k + comm_sz) % comm_sz) * count),
              count, MPI_FLOAT, succ, k, comm, req + k);
    float* begin = recv_buf_begin + (((pred - k + comm_sz) % comm_sz) * count);
    MPI_Recv(begin, count, MPI_FLOAT, pred, k, comm, nullptr);
    for (float* iter = begin; iter < begin + count; ++iter)
      (*iter) += *(send_buf_begin + (iter - recv_buf_begin));
  }
  MPI_Waitall(comm_sz - 1, req, nullptr);
  for (int k = 0; k < comm_sz - 1; ++k) {
    MPI_Isend(
        recv_buf_begin + (((my_rank + 1 - k + comm_sz) % comm_sz) * count),
        count, MPI_FLOAT, succ, k, comm, req + k);
    MPI_Recv(recv_buf_begin + (((my_rank - k + comm_sz) % comm_sz) * count),
             count, MPI_FLOAT, pred, k, comm, nullptr);
  }
  MPI_Waitall(comm_sz - 1, req, nullptr);
}

// reduce + bcast
void Naive_Allreduce(void* sendbuf, void* recvbuf, int n, MPI_Comm comm,
                     int comm_sz, int my_rank) {
  MPI_Reduce(sendbuf, recvbuf, n, MPI_FLOAT, MPI_SUM, 0, comm);
  MPI_Bcast(recvbuf, n, MPI_FLOAT, 0, comm);
}

int main(int argc, char* argv[]) {
  int ITER = atoi(argv[1]);
  int n = atoi(argv[2]);
  float* mpi_sendbuf = new float[n];
  float* mpi_recvbuf = new float[n];
  float* naive_sendbuf = new float[n];
  float* naive_recvbuf = new float[n];
  float* ring_sendbuf = new float[n];
  float* ring_recvbuf = new float[n];

  MPI_Init(nullptr, nullptr);
  int comm_sz;
  int my_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  srand(time(NULL) + my_rank);
  for (int i = 0; i < n; ++i)
    mpi_sendbuf[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  memcpy(naive_sendbuf, mpi_sendbuf, n * sizeof(float));
  memcpy(ring_sendbuf, mpi_sendbuf, n * sizeof(float));

  // warmup and check
  MPI_Allreduce(mpi_sendbuf, mpi_recvbuf, n, MPI_FLOAT, MPI_SUM,
                MPI_COMM_WORLD);
  Naive_Allreduce(naive_sendbuf, naive_recvbuf, n, MPI_COMM_WORLD, comm_sz,
                  my_rank);
  Ring_Allreduce(ring_sendbuf, ring_recvbuf, n, MPI_COMM_WORLD, comm_sz,
                 my_rank);
  bool correct = true;
  for (int i = 0; i < n; ++i)
    if (abs(mpi_recvbuf[i] - ring_recvbuf[i]) > EPS) {
      correct = false;
      break;
    }

  if (correct) {
    auto beg = ch::high_resolution_clock::now();
    for (int iter = 0; iter < ITER; ++iter)
      MPI_Allreduce(mpi_sendbuf, mpi_recvbuf, n, MPI_FLOAT, MPI_SUM,
                    MPI_COMM_WORLD);
    auto end = ch::high_resolution_clock::now();
    double mpi_dur =
        ch::duration_cast<ch::duration<double>>(end - beg).count() *
        1000;  // ms

    beg = ch::high_resolution_clock::now();
    for (int iter = 0; iter < ITER; ++iter)
      Naive_Allreduce(naive_sendbuf, naive_recvbuf, n, MPI_COMM_WORLD, comm_sz,
                      my_rank);
    end = ch::high_resolution_clock::now();
    double naive_dur =
        ch::duration_cast<ch::duration<double>>(end - beg).count() *
        1000;  // ms

    beg = ch::high_resolution_clock::now();
    for (int iter = 0; iter < ITER; ++iter)
      Ring_Allreduce(ring_sendbuf, ring_recvbuf, n, MPI_COMM_WORLD, comm_sz,
                     my_rank);
    end = ch::high_resolution_clock::now();
    double ring_dur =
        ch::duration_cast<ch::duration<double>>(end - beg).count() *
        1000;  // ms

    if (my_rank == 0) {
      std::cout << "Correct." << std::endl;
      std::cout << "MPI_Allreduce:   " << mpi_dur << " ms." << std::endl;
      std::cout << "Naive_Allreduce: " << naive_dur << " ms." << std::endl;
      std::cout << "Ring_Allreduce:  " << ring_dur << " ms." << std::endl;
    }
  } else if (my_rank == 0)
    std::cout << "Wrong!" << std::endl;

  delete[] mpi_sendbuf;
  delete[] mpi_recvbuf;
  delete[] naive_sendbuf;
  delete[] naive_recvbuf;
  delete[] ring_sendbuf;
  delete[] ring_recvbuf;
  MPI_Finalize();
  return 0;
}
