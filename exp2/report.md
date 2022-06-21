---
title: "exp2: MPI Allreduce"
category: "Course Work"
tags:
  - "Introduction to High Performance Computing"
  - "HPC"
---

# exp2: MPI Allreduce

## Ring Allreduce 算法

首先将每个结点的数据分为 `comm_sz` 个数据块, 每个数据块大小为 `count = n / comm_sz` 个 `float`.

第一阶段共 `comm_sz - 1` 步. 在第 `k` 步, 第 `my_rank` 个进程会将自己的第 `(my_rank - k) % comm_sz` 对应数据块发送给第 `succ = my_rank + 1` 个进程并累加. 注意到对于 `my_rank` 进程, 第 `k` 步的 `Send` 与 `Recv` 使用的数据块不同, 但第 `k + 1` 步的 `Send` 依赖于第 `k` 步的 `Recv` 得到的数据块. 因此 `Send` 可以是非阻塞的, 但 `Recv` 必须是阻塞的, 以确保在第 `k + 1` 步 `Send` 前, 第 `k` 步 `Recv` 已完成.

第二阶段共 `comm_sz - 1` 步. 在第 `k` 步, 第 `my_rank` 个进程会将自己的第 `(my_rank + 1 - k) % comm_sz` 对应数据块发送给第 `succ = my_rand + 1` 个进程. 与第一阶段同理, `Send` 可以是非阻塞的, 但 `Recv` 必须是阻塞的, 以确保在第 `k + 1` 步 `Send` 前, 第 `k` 步 `Recv` 已完成.

```c++
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
```

## 通信时间

| Method            | Comm_size | n         | Time       |
| ----------------- | --------- | --------- | ---------- |
| `MPI_Allreduce`   | 4         | 100000000 | 3195.49 ms |
| `Naive_Allreduce` | 4         | 100000000 | 4526.87 ms |
| `Ring_Allreduce`  | 4         | 100000000 | 1873.94 ms |

