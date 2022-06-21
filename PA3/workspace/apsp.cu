#include <cassert>
#include <cstdio>

#include "apsp.h"

constexpr int b = 32;
#define D(i, j) (graph[((i) * (n)) + (j)])

int CeilDivide(const int lhs, const int rhs) { return (lhs - 1) / rhs + 1; }

namespace {

__global__ void Phase1Kernel(const int n, int *graph,
                             const int center_block_start,
                             const int center_block_end) {
  __shared__ int block[b][b];
  int i = threadIdx.y + center_block_start;
  int j = threadIdx.x + center_block_start;
  if (i < center_block_end && j < center_block_end) {
    block[threadIdx.y][threadIdx.x] = D(i, j);
    __syncthreads();
    for (int k = center_block_start; k < center_block_end; ++k) {
      int thread_idx_k = k - center_block_start;
      // D[i][j] = min(D[i][j], D[i][k] + D[k][j]);
      block[threadIdx.y][threadIdx.x] = min(
          block[threadIdx.y][threadIdx.x],
          block[threadIdx.y][thread_idx_k] + block[thread_idx_k][threadIdx.x]);
    }
    D(i, j) = block[threadIdx.y][threadIdx.x];
  }
}

__global__ void Phase2KernelHorizontal(const int n, int *graph,
                                       const int center_block_start,
                                       const int center_block_end) {
  __shared__ int shared_memory[2][b][b];
  // current = [ current_block_start, current_block_end )
  // center  = [ center_block_start , center_block_end  )
  int(*block)[b] = shared_memory[0];   // center x current
  int(*center)[b] = shared_memory[1];  // center x center
  // k in center
  // i in center
  // j in current
  int i_start = center_block_start;
  int j_start = blockIdx.x * b;
  if (j_start >= center_block_start) j_start += b;
  int i = i_start + threadIdx.y;
  int j = j_start + threadIdx.x;
  if (i < n && j < n) {
    block[threadIdx.y][threadIdx.x] = D(i, j);
  }
  if (i < n && center_block_start + threadIdx.x < center_block_end) {
    center[threadIdx.y][threadIdx.x] = D(i, center_block_start + threadIdx.x);
  }
  __syncthreads();
  if (i < n && j < n) {
    for (int k = center_block_start; k < center_block_end; ++k) {
      int thread_idx_k = k - center_block_start;
      // D[i][j] = min(D[i][j], D[i][k] + D[k][j]);
      block[threadIdx.y][threadIdx.x] = min(
          block[threadIdx.y][threadIdx.x],
          center[threadIdx.y][thread_idx_k] + block[thread_idx_k][threadIdx.x]);
    }
    D(i, j) = block[threadIdx.y][threadIdx.x];
  }
}

__global__ void Phase2KernelVertical(const int n, int *graph,
                                     const int center_block_start,
                                     const int center_block_end) {
  __shared__ int shared_memory[2][b][b];
  // current = [ current_block_start, current_block_end )
  // center  = [ center_block_start , center_block_end  )
  int(*block)[b] = shared_memory[0];   // current x center
  int(*center)[b] = shared_memory[1];  // center  x center
  // k in center
  // i in current
  // j in center
  int i_start = blockIdx.y * b;
  int j_start = center_block_start;
  if (i_start >= center_block_start) i_start += b;
  int i = i_start + threadIdx.y;
  int j = j_start + threadIdx.x;
  if (i < n && j < n) {
    block[threadIdx.y][threadIdx.x] = D(i, j);
  }
  if (center_block_start + threadIdx.y < center_block_end && j < n) {
    center[threadIdx.y][threadIdx.x] = D(center_block_start + threadIdx.y, j);
  }
  __syncthreads();
  if (i < n && j < n) {
    for (int k = center_block_start; k < center_block_end; ++k) {
      int thread_idx_k = k - center_block_start;
      // D[i][j] = min(D[i][j], D[i][k] + D[k][j]);
      block[threadIdx.y][threadIdx.x] = min(
          block[threadIdx.y][threadIdx.x],
          block[threadIdx.y][thread_idx_k] + center[thread_idx_k][threadIdx.x]);
    }
    D(i, j) = block[threadIdx.y][threadIdx.x];
  }
}

__global__ void Phase3Kernel(const int n, int *graph,
                             const int center_block_start,
                             const int center_block_end) {
  int i_start = blockIdx.y * b;
  if (i_start >= center_block_start) i_start += b;
  // int i_stop = min(i_start + b, n);
  int j_start = blockIdx.x * b;
  if (j_start >= center_block_start) j_start += b;
  // int j_stop = min(j_start + b, n);
  // i = [ i_start            , i_stop           )
  // j = [ j_start            , j_stop           )
  // k = [ center_block_start , center_block_end )
  __shared__ int shared_memory[3][b][b];
  int(*block)[b] = shared_memory[0];     // i x j
  int(*column_k)[b] = shared_memory[1];  // i x k
  int(*row_k)[b] = shared_memory[2];     // k x j
  int i = i_start + threadIdx.y;
  int j = j_start + threadIdx.x;
  if (i < n && j < n) {
    block[threadIdx.y][threadIdx.x] = D(i, j);
  }
  if (i < n && center_block_start + threadIdx.x < center_block_end) {
    column_k[threadIdx.y][threadIdx.x] = D(i, center_block_start + threadIdx.x);
  }
  if (center_block_start + threadIdx.y < center_block_end && j < n) {
    row_k[threadIdx.y][threadIdx.x] = D(center_block_start + threadIdx.y, j);
  }
  __syncthreads();
  if (i < n && j < n) {
    for (int k = center_block_start; k < center_block_end; ++k) {
      int thread_idx_k = k - center_block_start;
      // D[i][j] = min(D[i][j], D[i][k] + D[k][j]);
      block[threadIdx.y][threadIdx.x] =
          min(block[threadIdx.y][threadIdx.x],
              column_k[threadIdx.y][thread_idx_k] +
                  row_k[thread_idx_k][threadIdx.x]);
    }
    D(i, j) = block[threadIdx.y][threadIdx.x];
  }
}

}  // namespace

void Phase1CPU(const int n, int *graph, const int p) {
  int *graph_on_gpu = graph;
  int *graph_on_cpu = new int[n * n];
  cudaMemcpy(graph_on_cpu, graph_on_gpu, n * n * sizeof(int),
             cudaMemcpyDefault);
  graph = graph_on_cpu;
  const int center_block_start = p * b;
  const int center_block_end = min((p + 1) * b, n);
  for (int k = center_block_start; k < center_block_end; ++k) {
    for (int i = center_block_start; i < center_block_end; ++i) {
      for (int j = center_block_start; j < center_block_end; ++j) {
        D(i, j) = min(D(i, j), D(i, k) + D(k, j));
      }
    }
  }
  cudaMemcpy(graph_on_gpu, graph_on_cpu, n * n * sizeof(int),
             cudaMemcpyDefault);
  delete[] graph_on_cpu;
}

void Phase1GPU(const int n, int *graph, const int p) {
  const int center_block_start = p * b;
  const int center_block_end = min((p + 1) * b, n);
  dim3 threads_per_block(b, b);
  dim3 num_blocks(1);
  Phase1Kernel<<<num_blocks, threads_per_block>>>(n, graph, center_block_start,
                                                  center_block_end);
}

void Phase2CPU(const int n, int *graph, const int p) {
  int *graph_on_gpu = graph;
  int *graph_on_cpu = new int[n * n];
  cudaMemcpy(graph_on_cpu, graph_on_gpu, n * n * sizeof(int),
             cudaMemcpyDefault);
  graph = graph_on_cpu;
  const int center_block_start = p * b;
  const int center_block_end = min((p + 1) * b, n);
  for (int k = center_block_start; k < center_block_end; k++) {
    for (int i = center_block_start; i < center_block_end; i++) {
      for (int j = 0; j < center_block_start; j++) {
        D(i, j) = min(D(i, j), D(i, k) + D(k, j));
      }
      for (int j = center_block_end; j < n; j++) {
        D(i, j) = min(D(i, j), D(i, k) + D(k, j));
      }
    }
    for (int j = center_block_start; j < center_block_end; j++) {
      for (int i = 0; i < center_block_start; i++) {
        D(i, j) = min(D(i, j), D(i, k) + D(k, j));
      }
      for (int i = center_block_end; i < n; i++) {
        D(i, j) = min(D(i, j), D(i, k) + D(k, j));
      }
    }
  }
  cudaMemcpy(graph_on_gpu, graph_on_cpu, n * n * sizeof(int),
             cudaMemcpyDefault);
  delete[] graph_on_cpu;
}

void Phase2GPU(const int n, int *graph, const int p) {
  const int center_block_start = p * b;
  const int center_block_end = min((p + 1) * b, n);
  const int center_block_size = center_block_end - center_block_start;
  dim3 threads_per_block(b, b);
  dim3 num_blocks_horizontal(CeilDivide(n - center_block_size, b), 1);
  Phase2KernelHorizontal<<<num_blocks_horizontal, threads_per_block>>>(
      n, graph, center_block_start, center_block_end);
  dim3 num_blocks_vertical(1, CeilDivide(n - center_block_size, b));
  Phase2KernelVertical<<<num_blocks_vertical, threads_per_block>>>(
      n, graph, center_block_start, center_block_end);
}

void Phase3CPU(const int n, int *graph, const int p) {
  int *graph_on_gpu = graph;
  int *graph_on_cpu = new int[n * n];
  cudaMemcpy(graph_on_cpu, graph_on_gpu, n * n * sizeof(int),
             cudaMemcpyDefault);
  graph = graph_on_cpu;
  const int center_block_start = p * b;
  const int center_block_end = min((p + 1) * b, n);
  for (int k = center_block_start; k < center_block_end; k++) {
    for (int i = 0; i < center_block_start; ++i) {
      for (int j = 0; j < center_block_start; ++j) {
        D(i, j) = min(D(i, j), D(i, k) + D(k, j));
      }
      for (int j = center_block_end; j < n; ++j) {
        D(i, j) = min(D(i, j), D(i, k) + D(k, j));
      }
    }
    for (int i = center_block_end; i < n; ++i) {
      for (int j = 0; j < center_block_start; ++j) {
        D(i, j) = min(D(i, j), D(i, k) + D(k, j));
      }
      for (int j = center_block_end; j < n; ++j) {
        D(i, j) = min(D(i, j), D(i, k) + D(k, j));
      }
    }
  }
  cudaMemcpy(graph_on_gpu, graph_on_cpu, n * n * sizeof(int),
             cudaMemcpyDefault);
  delete[] graph_on_cpu;
}

void Phase3GPU(const int n, int *graph, const int p) {
  const int center_block_start = p * b;
  const int center_block_end = min((p + 1) * b, n);
  const int center_block_size = center_block_end - center_block_start;
  dim3 threads_per_block(b, b);
  dim3 num_blocks(CeilDivide(n - center_block_size, b),
                  CeilDivide(n - center_block_size, b));
  Phase3Kernel<<<num_blocks, threads_per_block>>>(n, graph, center_block_start,
                                                  center_block_end);
}

void apsp(int n, /* device */ int *graph) {
  for (int p = 0; p * b < n; ++p) {
    Phase1GPU(n, graph, p);
    Phase2GPU(n, graph, p);
    Phase3GPU(n, graph, p);
  }
}
