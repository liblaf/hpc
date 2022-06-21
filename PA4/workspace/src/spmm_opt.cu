#include <algorithm>
#include <vector>

#include "spmm_opt.h"

constexpr int kBatchSize = 256;
constexpr int kTasksPerBlock = 1;

__global__ void SpmmOptKernel(const Task *tasks, const int *idx,
                              const float *val, const float *vin, float *vout,
                              const int num_tasks, int feat_in) {
  const int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id_x >= num_tasks) return;
  const Task task = tasks[thread_id_x];
  const int row = task.row;
  const int ptr_begin = task.ptr_begin;
  const int ptr_end = task.ptr_end;

  __shared__ float shared_memory[2 * kTasksPerBlock][kBatchSize];
  int(*sub_idx)[kBatchSize] = (int(*)[kBatchSize])(&(shared_memory[0]));
  float(*sub_val)[kBatchSize] = &(shared_memory[kTasksPerBlock]);
  for (int ptr = ptr_begin + threadIdx.y; ptr < ptr_end; ptr += blockDim.y) {
    sub_idx[threadIdx.x][ptr - ptr_begin] = idx[ptr];
    sub_val[threadIdx.x][ptr - ptr_begin] = val[ptr];
  }
  __syncthreads();

  float result = 0.0f;
  const int feat_id = blockIdx.y * blockDim.y + threadIdx.y;
  for (int ptr = ptr_begin; ptr < ptr_end; ++ptr) {
    result += vin[sub_idx[threadIdx.x][ptr - ptr_begin] * feat_in + feat_id] *
              sub_val[threadIdx.x][ptr - ptr_begin];
  }
  atomicAdd(&(vout[row * feat_in + feat_id]), result);
}

void SpMMOpt::preprocess(float *vin, float *vout) {
  int *h_ptr = new int[this->num_v + 1];
  checkCudaErrors(cudaMemcpy(h_ptr, this->d_ptr,
                             (this->num_v + 1) * sizeof(int),
                             cudaMemcpyDeviceToHost));
  std::vector<Task> tasks;
  for (int row = 0; row < this->num_v; ++row) {
    const int begin = h_ptr[row];
    const int end = h_ptr[row + 1];
    for (int ptr_begin = begin; ptr_begin < end; ptr_begin += kBatchSize) {
      Task task = {
          .row = row,
          .ptr_begin = ptr_begin,
          .ptr_end = min(ptr_begin + kBatchSize, end),
      };
      tasks.push_back(task);
    }
  }
  delete[] h_ptr;
  this->num_tasks_ = tasks.size();
  std::random_shuffle(tasks.begin(), tasks.end());
  checkCudaErrors(
      cudaMalloc2((void **)&(this->d_tasks_), this->num_tasks_ * sizeof(Task)));
  checkCudaErrors(cudaMemcpy(this->d_tasks_, tasks.data(),
                             this->num_tasks_ * sizeof(Task),
                             cudaMemcpyHostToDevice));
  this->block.x = kTasksPerBlock;
  this->block.y = this->feat_in;
  this->grid.x = CEIL(this->num_tasks_, this->block.x);
  this->grid.y = this->feat_in / this->block.y;
  checkCudaErrors(
      cudaMemset(vout, 0, this->num_v * this->feat_in * sizeof(float)));
}

void SpMMOpt::run(float *vin, float *vout) {
  SpmmOptKernel<<<this->grid, this->block>>>(this->d_tasks_, this->d_idx,
                                             this->d_val, vin, vout,
                                             this->num_tasks_, this->feat_in);
}
