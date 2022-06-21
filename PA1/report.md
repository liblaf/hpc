---
title: "PA1: 奇偶排序（odd_even_sort）"
category: "Course Work"
tags:
  - "Introduction to High Performance Computing"
  - "HPC"
---

# PA1: 奇偶排序

## Performance

| Number of Nodes | Number of Tasks | Number Count | v0          | v1          | v2          | v3          | v4         |
| --------------- | --------------- | ------------ | ----------- | ----------- | ----------- | ----------- | ---------- |
| 1               | 1               | 100000000    | 1.          | 1.          | 1.          | 1.          | 1.         |
| 1               | 2               | 100000000    | 1.82738226  | 1.83557525  | 1.8772568   | 1.86270894  | 1.67925379 |
| 1               | 4               | 100000000    | 3.05696721  | 3.17540441  | 3.37059297  | 3.39672385  | 2.3387731  |
| 1               | 8               | 100000000    | 5.15212684  | 5.59405831  | 6.03427353  | 6.15270254  | 3.29768106 |
| 1               | 16              | 100000000    | 7.65590815  | 8.91877194  | 9.65970998  | 10.31895407 | 4.39598177 |
| 2               | 32              | 100000000    | 10.63745468 | 12.91174115 | 14.70125847 | 15.67950831 | 5.01147717 |

## v0

`AllReduceBitwiseAnd` + Blocking Communication + Naive Merge

每轮归并后 `AllReduce` 检查是否已为有序.

### `AllReduceBitwiseAnd`

```
0 -> 1 -> ... -> (nprocs - 1) -> 0 # Tag 1
0 -> 1 -> ... -> (nprocs - 2) # Tag 2
```

### Source Code

```c++
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

int AllReduceBitwiseAnd(int sendbuf, MPI_Comm comm, const int nprocs,
                        const int current_rank) {
  if (nprocs == 1) return sendbuf;
  int succ_rank = (current_rank + 1) % nprocs;
  int pred_rank = (current_rank - 1 + nprocs) % nprocs;
  if (current_rank == 0) {
    MPI_Send(/*buf=*/&sendbuf, /*count=*/1, /*datatype=*/MPI_INT,
             /*dest=*/succ_rank, /*tag=*/1, /*comm=*/comm);
    MPI_Recv(/*buf=*/&sendbuf, /*count=*/1, /*datatype=*/MPI_INT,
             /*source=*/pred_rank, /*tag=*/1, /*comm=*/comm,
             /*status=*/nullptr);
    if (succ_rank != nprocs - 1)
      MPI_Send(/*buf=*/&sendbuf, /*count=*/1, /*datatype=*/MPI_INT,
               /*dest=*/succ_rank, /*tag=*/2, /*comm=*/comm);
  } else {
    bool recvbuf;
    MPI_Recv(/*buf=*/&recvbuf, /*count=*/1, /*datatype=*/MPI_INT,
             /*source=*/pred_rank, /*tag=*/1, /*comm=*/comm,
             /*status=*/nullptr);
    sendbuf &= recvbuf;
    MPI_Send(/*buf=*/&sendbuf, /*count=*/1, /*datatype=*/MPI_INT,
             /*dest=*/succ_rank, /*tag=*/1, /*comm=*/comm);
    if (current_rank != nprocs - 1) {
      MPI_Recv(/*buf=*/&sendbuf, /*count=*/1, /*datatype=*/MPI_INT,
               /*source=*/pred_rank, /*tag=*/2, /*comm=*/comm,
               /*status=*/nullptr);
      if (succ_rank != nprocs - 1)
        MPI_Send(/*buf=*/&sendbuf, /*count=*/1, /*datatype=*/MPI_INT,
                 /*dest=*/succ_rank, /*tag=*/2, /*comm=*/comm);
    }
  }
  return sendbuf;
}

void Worker::sort() {
  if (this->out_of_range) return;
  std::sort(this->data, this->data + this->block_len);
  if (this->nprocs == 1) return;
  int left_rank = ((this->rank == 0) ? (-1) : (this->rank - 1));
  size_t left_block_len = CalcBlockLen(this->n, this->nprocs, left_rank);
  int right_rank = ((this->rank + 1 == this->nprocs) ? (-1) : (this->rank + 1));
  size_t right_block_len = CalcBlockLen(this->n, this->nprocs, right_rank);
  float* neighbor_data = new float[std::max(left_block_len, right_block_len)];
  float* merged_data = new float[this->block_len];
  bool sorted = false;
  for (int i = 0;; ++i) {
    if ((i & 1) == 0) {  // even stage
      if (AllReduceBitwiseAnd(sorted, MPI_COMM_WORLD, this->nprocs, this->rank))
        break;
      sorted = false;
    }
    int neighbor_rank = 0;
    size_t neighbor_block_len = 0;
    if ((i & 1) ^ ((this->rank) & 1)) {
      neighbor_rank = right_rank;
      neighbor_block_len = right_block_len;
    } else {
      neighbor_rank = left_rank;
      neighbor_block_len = left_block_len;
    }
    if (neighbor_block_len == 0) {
      // no neighbor
      sorted = true;
      continue;
    }
    MPI_Status status;
    if (neighbor_rank == left_rank) {
      MPI_Sendrecv(/*sendbuf=*/this->data, /*sendcount=*/1,
                   /*sendtype=*/MPI_FLOAT, /*dest=*/neighbor_rank,
                   /*sendtag=*/i, /*recvbuf=*/neighbor_data, /*recvcount=*/1,
                   /*recvtype=*/MPI_FLOAT, /*source=*/neighbor_rank,
                   /*recvtag=*/i, /*comm=*/MPI_COMM_WORLD, /*status=*/&status);
      if (neighbor_data[0] <= this->data[0]) {
        // no need to merge
        sorted = true;
        continue;
      }
    } else {  // neighbor_rank == right_rank
      MPI_Sendrecv(/*sendbuf=*/this->data + (this->block_len - 1),
                   /*sendcount=*/1,
                   /*sendtype=*/MPI_FLOAT, /*dest=*/neighbor_rank,
                   /*sendtag=*/i, /*recvbuf=*/neighbor_data, /*recvcount=*/1,
                   /*recvtype=*/MPI_FLOAT, /*source=*/neighbor_rank,
                   /*recvtag=*/i, /*comm=*/MPI_COMM_WORLD, /*status=*/&status);
      if (this->data[this->block_len - 1] <= neighbor_data[0]) {
        // no need to merge
        sorted = true;
        continue;
      }
    }
    MPI_Sendrecv(/*sendbuf=*/this->data, /*sendcount=*/this->block_len,
                 /*sendtype=*/MPI_FLOAT, /*dest=*/neighbor_rank, /*sendtag=*/i,
                 /*recvbuf=*/neighbor_data, /*recvcount=*/neighbor_block_len,
                 /*recvtype=*/MPI_FLOAT, /*source=*/neighbor_rank,
                 /*recvtag=*/i, /*comm=*/MPI_COMM_WORLD, /*status=*/&status);
    size_t skip_count = (neighbor_rank == left_rank) ? neighbor_block_len : 0;
    size_t j = 0, k = 0, l = 0;
    while (j < neighbor_block_len && k < this->block_len && l < skip_count) {
      if (neighbor_data[j] < this->data[k]) {
        ++j;
        ++l;
      } else {
        ++k;
        ++l;
      }
    }
    while (j < neighbor_block_len && l < skip_count) {
      ++j;
      ++l;
    }
    while (k < this->block_len && l < skip_count) {
      ++k;
      ++l;
    }
    l = 0;
    while (j < neighbor_block_len && k < this->block_len &&
           l < this->block_len) {
      if (neighbor_data[j] < this->data[k]) {
        merged_data[l++] = neighbor_data[j++];
      } else {
        merged_data[l++] = this->data[k++];
      }
    }
    while (j < neighbor_block_len && l < this->block_len)
      merged_data[l++] = neighbor_data[j++];
    while (k < this->block_len && l < this->block_len)
      merged_data[l++] = this->data[k++];
    std::memcpy(this->data, merged_data, (this->block_len) * sizeof(float));
  }
  delete[] neighbor_data;
  delete[] merged_data;
}
```

### Performance

| Number of Nodes | Number of Tasks | Number Count | Execution Time  | Speedup     |
| --------------- | --------------- | ------------ | --------------- | ----------- |
| 1               | 1               | 100000000    | 12526.773000 ms | 1.          |
| 1               | 2               | 100000000    | 6855.037000 ms  | 1.82738226  |
| 1               | 4               | 100000000    | 4097.778000 ms  | 3.05696721  |
| 1               | 8               | 100000000    | 2431.379000 ms  | 5.15212684  |
| 1               | 16              | 100000000    | 1636.223000 ms  | 7.65590815  |
| 2               | 32              | 100000000    | 1177.610000 ms  | 10.63745468 |

## v1

Loop `nprocs` times + Blocking Communication + Naive Merge

节约 `AllReduce` 的时间, 进行 `nprocs` 轮排序.

### Proof of Correctness

> Reference: [Odd–even sort - Wikipedia](https://en.wikipedia.org/wiki/Odd–even_sort#Proof_of_correctness)

#### Claim

Let $a_1, \cdots, a_n$ be a sequence of data ordered by `<`. The odd-even sort algorithm correctly sorts this data in $n$ passes. (A pass here is defined to be a full sequence of odd-even, or even-odd comparisons. The passes occur in order pass 1: odd–even, pass 2: even–odd, etc.)

#### Proof

This proof is based loosely on one by Thomas Worsch.[^6]

Since the sorting algorithm only involves comparison-swap operations and is oblivious (the order of comparison-swap operations does not depend on the data), by Knuth's 0-1 sorting principle,[^7][^8] it suffices to check correctness when each $a_i$ is either 0 or 1. Assume that there are $e$ 1s.

Observe that the rightmost 1 can be either in an even or odd position, so it might not be moved by the first odd-even pass. But after the first odd-even pass, the rightmost 1 will be in an even position. It follows that it will be moved to the right by all remaining passes. Since the rightmost one starts in position greater than or equal to $e$, it must be moved at most $n - e$ steps. It follows that it takes at most $n - e + 1$ passes to move the rightmost 1 to its correct position.

Now, consider the second rightmost 1. After two passes, the 1 to its right will have moved right by at least one step. It follows that, for all remaining passes, we can view the second rightmost 1 as the rightmost 1. The second rightmost 1 starts in position at least $e - 1$ and must be moved to position at most $n - 1$, so it must be moved at most $(n - 1) - (e - 1) = n - e$ steps. After at most 2 passes, the rightmost 1 will have already moved, so the entry to the right of the second rightmost 1 will be 0. Hence, for all passes after the first two, the second rightmost 1 will move to the right. It thus takes at most $n - e + 2$ passes to move the second rightmost 1 to its correct position.

Continuing in this manner, by induction it can be shown that the $i$-th rightmost 1 is moved to its correct position in at most $n - e + i$ passes. Since $i \leqslant e$, it follows that the $i$-th rightmost 1 is moved to its correct position in at most $n - e + e = n$ passes. The list is thus correctly sorted in $n$ passes. QED.

We remark that each pass takes $O(n)$ steps, so this algorithm has $O(n^2)$ complexity.

[^6]: ["Five Lectures on CA"](http://liinwww.ira.uka.de/~thw/vl-hiroshima/slides-4.pdf) (PDF). *Liinwww.ira.uka.de*. Retrieved 2017-07-30.

[^7]: Lang, Hans Werner. ["The 0-1-principle"](http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/networks/nulleinsen.htm). *Iti.fh-flensburg.de*. Retrieved 30 July 2017.
[^8]: ["Distributed Sorting"](http://www.net.t-labs.tu-berlin.de/~stefan/netalg13-9-sort.pdf) (PDF). *Net.t-labs.tu-berlin.de*. Retrieved 2017-07-30.

### Source Code

```c++
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
  if (this->out_of_range) return;
  std::sort(this->data, this->data + this->block_len);
  if (this->nprocs == 1) return;
  int left_rank = ((this->rank == 0) ? (-1) : (this->rank - 1));
  size_t left_block_len = CalcBlockLen(this->n, this->nprocs, left_rank);
  int right_rank = ((this->rank + 1 == this->nprocs) ? (-1) : (this->rank + 1));
  size_t right_block_len = CalcBlockLen(this->n, this->nprocs, right_rank);
  float* neighbor_data = new float[std::max(left_block_len, right_block_len)];
  float* merged_data = new float[this->block_len];
  for (int i = 0; i < nprocs; ++i) {
    int neighbor_rank = 0;
    size_t neighbor_block_len = 0;
    if ((i & 1) ^ ((this->rank) & 1)) {
      neighbor_rank = right_rank;
      neighbor_block_len = right_block_len;
    } else {
      neighbor_rank = left_rank;
      neighbor_block_len = left_block_len;
    }
    if (neighbor_block_len == 0)
      // no neighbor
      continue;
    MPI_Status status;
    if (neighbor_rank == left_rank) {
      MPI_Sendrecv(/*sendbuf=*/this->data, /*sendcount=*/1,
                   /*sendtype=*/MPI_FLOAT, /*dest=*/neighbor_rank,
                   /*sendtag=*/i, /*recvbuf=*/neighbor_data, /*recvcount=*/1,
                   /*recvtype=*/MPI_FLOAT, /*source=*/neighbor_rank,
                   /*recvtag=*/i, /*comm=*/MPI_COMM_WORLD, /*status=*/&status);
      if (neighbor_data[0] <= this->data[0])
        // no need to merge
        continue;
    } else {  // neighbor_rank == right_rank
      MPI_Sendrecv(/*sendbuf=*/this->data + (this->block_len - 1),
                   /*sendcount=*/1,
                   /*sendtype=*/MPI_FLOAT, /*dest=*/neighbor_rank,
                   /*sendtag=*/i, /*recvbuf=*/neighbor_data, /*recvcount=*/1,
                   /*recvtype=*/MPI_FLOAT, /*source=*/neighbor_rank,
                   /*recvtag=*/i, /*comm=*/MPI_COMM_WORLD, /*status=*/&status);
      if (this->data[this->block_len - 1] <= neighbor_data[0])
        // no need to merge
        continue;
    }
    MPI_Sendrecv(/*sendbuf=*/this->data, /*sendcount=*/this->block_len,
                 /*sendtype=*/MPI_FLOAT, /*dest=*/neighbor_rank, /*sendtag=*/i,
                 /*recvbuf=*/neighbor_data, /*recvcount=*/neighbor_block_len,
                 /*recvtype=*/MPI_FLOAT, /*source=*/neighbor_rank,
                 /*recvtag=*/i, /*comm=*/MPI_COMM_WORLD, /*status=*/&status);
    size_t skip_count = (neighbor_rank == left_rank) ? neighbor_block_len : 0;
    size_t j = 0, k = 0, l = 0;
    while (j < neighbor_block_len && k < this->block_len && l < skip_count) {
      if (neighbor_data[j] < this->data[k]) {
        ++j;
        ++l;
      } else {
        ++k;
        ++l;
      }
    }
    while (j < neighbor_block_len && l < skip_count) {
      ++j;
      ++l;
    }
    while (k < this->block_len && l < skip_count) {
      ++k;
      ++l;
    }
    l = 0;
    while (j < neighbor_block_len && k < this->block_len &&
           l < this->block_len) {
      if (neighbor_data[j] < this->data[k]) {
        merged_data[l++] = neighbor_data[j++];
      } else {
        merged_data[l++] = this->data[k++];
      }
    }
    while (j < neighbor_block_len && l < this->block_len)
      merged_data[l++] = neighbor_data[j++];
    while (k < this->block_len && l < this->block_len)
      merged_data[l++] = this->data[k++];
    std::memcpy(this->data, merged_data, (this->block_len) * sizeof(float));
  }
  delete[] neighbor_data;
  delete[] merged_data;
}
```

### Performance

| Number of Nodes | Number of Tasks | Number Count | Execution Time  | Speedup     |
| --------------- | --------------- | ------------ | --------------- | ----------- |
| 1               | 1               | 100000000    | 12511.503000 ms | 1.          |
| 1               | 2               | 100000000    | 6816.121000 ms  | 1.83557525  |
| 1               | 4               | 100000000    | 3940.129000 ms  | 3.17540441  |
| 1               | 8               | 100000000    | 2236.570000 ms  | 5.59405831  |
| 1               | 16              | 100000000    | 1402.828000 ms  | 8.91877194  |
| 2               | 16              | 100000000    | 969.002000 ms   | 12.91174115 |

## v2

Loop `nprocs` times + Blocking Communication + Optimized Merge

### Optimized Merge

左侧 Worker 从左向右 Merge, 右侧 Worker 从右向左 Merge.

### Source Code

```c++
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
  float* merged_data = new float[this->block_len];
  MPI_Status status;
  for (int i = 0; i < nprocs; ++i) {
    if ((i & 1) ^ ((this->rank) & 1)) {
      if (right_block_len == 0)
        // no neighbor
        continue;
      MPI_Sendrecv(/*sendbuf=*/this->data + (this->block_len - 1),
                   /*sendcount=*/1,
                   /*sendtype=*/MPI_FLOAT, /*dest=*/right_rank,
                   /*sendtag=*/i, /*recvbuf=*/neighbor_data, /*recvcount=*/1,
                   /*recvtype=*/MPI_FLOAT, /*source=*/right_rank,
                   /*recvtag=*/i, /*comm=*/MPI_COMM_WORLD, /*status=*/&status);
      if (this->data[this->block_len - 1] <= neighbor_data[0])
        // no need to merge
        continue;
      MPI_Sendrecv(/*sendbuf=*/this->data, /*sendcount=*/this->block_len,
                   /*sendtype=*/MPI_FLOAT, /*dest=*/right_rank,
                   /*sendtag=*/i,
                   /*recvbuf=*/neighbor_data, /*recvcount=*/right_block_len,
                   /*recvtype=*/MPI_FLOAT, /*source=*/right_rank,
                   /*recvtag=*/i, /*comm=*/MPI_COMM_WORLD, /*status=*/&status);
      size_t j = 0, k = 0, l = 0;
      while (j < this->block_len && k < right_block_len &&
             l < this->block_len) {
        if (this->data[j] < neighbor_data[k]) {
          merged_data[l++] = this->data[j++];
        } else {
          merged_data[l++] = neighbor_data[k++];
        }
      }
      while (j < this->block_len && l < this->block_len)
        merged_data[l++] = this->data[j++];
      while (k < right_block_len && l < this->block_len)
        merged_data[l++] = neighbor_data[k++];
    } else {
      if (left_block_len == 0)
        // no neighbor
        continue;
      MPI_Sendrecv(/*sendbuf=*/this->data,
                   /*sendcount=*/1,
                   /*sendtype=*/MPI_FLOAT, /*dest=*/left_rank,
                   /*sendtag=*/i, /*recvbuf=*/neighbor_data, /*recvcount=*/1,
                   /*recvtype=*/MPI_FLOAT, /*source=*/left_rank,
                   /*recvtag=*/i, /*comm=*/MPI_COMM_WORLD, /*status=*/&status);
      if (neighbor_data[0] <= this->data[0])
        // no need to merge
        continue;
      MPI_Sendrecv(/*sendbuf=*/this->data, /*sendcount=*/this->block_len,
                   /*sendtype=*/MPI_FLOAT, /*dest=*/left_rank,
                   /*sendtag=*/i,
                   /*recvbuf=*/neighbor_data, /*recvcount=*/left_block_len,
                   /*recvtype=*/MPI_FLOAT, /*source=*/left_rank,
                   /*recvtag=*/i, /*comm=*/MPI_COMM_WORLD, /*status=*/&status);
      int j = left_block_len - 1, k = this->block_len - 1,
          l = this->block_len - 1;
      while (j > -1 && k > -1 && l > -1) {
        if (neighbor_data[j] < this->data[k]) {
          merged_data[l--] = this->data[k--];
        } else {
          merged_data[l--] = neighbor_data[j--];
        }
      }
      while (j > -1 && l > -1) merged_data[l--] = neighbor_data[j--];
      while (k > -1 && l > -1) merged_data[l--] = this->data[k--];
    }
    std::memcpy(this->data, merged_data, (this->block_len) * sizeof(float));
  }
  delete[] neighbor_data;
  delete[] merged_data;
}
```

### Performace

| Number of Nodes | Number of Tasks | Number Count | Execution Time  | Speedup     |
| --------------- | --------------- | ------------ | --------------- | ----------- |
| 1               | 1               | 100000000    | 12513.623000 ms | 1.          |
| 1               | 2               | 100000000    | 6665.909000 ms  | 1.8772568   |
| 1               | 4               | 100000000    | 3712.588000 ms  | 3.37059297  |
| 1               | 8               | 100000000    | 2073.758000 ms  | 6.03427353  |
| 1               | 16              | 100000000    | 1295.445000 ms  | 9.65970998  |
| 2               | 32              | 100000000    | 851.194000 ms   | 14.70125847 |

## v3

Loop `nprocs` times + Non-Blocking Communication + Optimized Merge + Lazy Copy

### Lazy Copy

不必每次 Merge 后都将 Merge 的结果拷贝到原 `data` 中, 而是可以使用两个数组交替作为 "旧数据" 和 "新数据", 只需交换指针即可.

### Source Code

```c++
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
  float* data_copy = new float[this->block_len];
  MPI_Request requests[2];
  for (int i = 0; i < nprocs; ++i) {
    if ((i & 1) ^ ((this->rank) & 1)) {
      if (right_block_len == 0)
        // no neighbor
        continue;
      MPI_Sendrecv(/*sendbuf=*/&(this->data[this->block_len - 1]),
                   /*sendcount=*/1, /*sendtype=*/MPI_FLOAT, /*dest=*/right_rank,
                   /*sendtag=*/i, /*recvbuf=*/&(neighbor_data[0]),
                   /*recvcount=*/1, /*recvtype=*/MPI_FLOAT,
                   /*source=*/right_rank, /*recvtag=*/i,
                   /*comm=*/MPI_COMM_WORLD, /*status=*/MPI_STATUS_IGNORE);
      if (this->data[this->block_len - 1] <= neighbor_data[0]) {
        // no need to merge
        continue;
      }
      MPI_Isend(/*buf=*/this->data, /*count=*/this->block_len,
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
        if (this->data[j] < neighbor_data[k]) {
          data_copy[l++] = this->data[j++];
        } else {
          data_copy[l++] = neighbor_data[k++];
        }
      }
      while (j < this->block_len && l < this->block_len)
        data_copy[l++] = this->data[j++];
      while (k < right_block_len && l < this->block_len)
        data_copy[l++] = neighbor_data[k++];
    } else {
      if (left_block_len == 0)
        // no neighbor
        continue;
      MPI_Sendrecv(/*sendbuf=*/&(this->data[0]),
                   /*sendcount=*/1, /*sendtype=*/MPI_FLOAT, /*dest=*/left_rank,
                   /*sendtag=*/i,
                   /*recvbuf=*/&(neighbor_data[left_block_len - 1]),
                   /*recvcount=*/1, /*recvtype=*/MPI_FLOAT,
                   /*source=*/left_rank, /*recvtag=*/i,
                   /*comm=*/MPI_COMM_WORLD, /*status=*/MPI_STATUS_IGNORE);
      if (neighbor_data[left_block_len - 1] <= this->data[0])
        // no need to merge
        continue;
      MPI_Isend(/*buf=*/this->data, /*count=*/this->block_len,
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
        if (neighbor_data[j] < this->data[k]) {
          data_copy[l--] = this->data[k--];
        } else {
          data_copy[l--] = neighbor_data[j--];
        }
      }
      while (j > -1 && l > -1) data_copy[l--] = neighbor_data[j--];
      while (k > -1 && l > -1) data_copy[l--] = this->data[k--];
    }
    MPI_Wait(/*request=*/&requests[0], /*status=*/MPI_STATUS_IGNORE);
    std::memcpy(this->data, data_copy, (this->block_len) * sizeof(float));
  }
  delete[] neighbor_data;
  delete[] data_copy;
}
```

### Performance

| Number of Nodes | Number of Tasks | Number Count | Execution Time  | Speedup     |
| --------------- | --------------- | ------------ | --------------- | ----------- |
| 1               | 1               | 100000000    | 12509.441000 ms | 1.          |
| 1               | 2               | 100000000    | 6715.725000 ms  | 1.86270894  |
| 1               | 4               | 100000000    | 3682.796000 ms  | 3.39672385  |
| 1               | 8               | 100000000    | 2033.162000 ms  | 6.15270254  |
| 1               | 16              | 100000000    | 1212.278000 ms  | 10.31895407 |
| 2               | 32              | 100000000    | 797.821000 ms   | 15.67950831 |

## v4

采用类似 Buffered Stream 方式边计算边发送, 即每计算 `chunk_size` 个 `float` 后就进行一次 `Isend`, 使得通信能够尽可能与计算重叠. 其中, `send_right_buffer` 和 `recv_left_buffer` 采取逆向存储, 以符合逆向归并的读写顺序. 为了尽可能缩短阻塞通信的时长, 仅在将要使用某一个 buffer 时才对该 buffer 进行 `Waitall` 确保上一轮通信已完成.

### Bandwidth

```bash
$ srun -N 1 -n 2 osu_bw
# OSU MPI Bandwidth Test v5.6.3
# Size      Bandwidth (MB/s)
1                       9.47
2                      18.54
4                      38.20
8                      76.28
16                    152.38
32                    286.08
64                    598.25
128                   382.26
256                   764.46
512                  1518.65
1024                 2471.40
2048                 3541.91
4096                 4986.57
8192                 6506.46
16384                5481.32
32768                7418.92
65536               10191.21
131072              12052.71
262144              12183.31
524288              11554.38
1048576             11871.98
2097152             12109.49
4194304             12458.68
```

选取 `131072` 作为 `chunk_size`.

### Source Code

```c++
// odd_even_sort.cpp
#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <queue>

#include "worker.h"

#ifdef DEBUG
// void PrintData(const float* data, const int count) {
//   for (int i = 0; i < count; ++i) printf("%f, ", data[i]);
// }
#endif

template <class T = float>
class MPIStream {
 public:
  MPIStream(const int capacity = 0, MPI_Datatype datatype = MPI_FLOAT,
            MPI_Comm comm = MPI_COMM_WORLD)
      : datatype_(datatype), comm_(comm), target_(-1) {
    this->chunk_count_ = CalcChunkCount(capacity);
    this->data_ = new T[this->chunk_count_ << kLogChunkSize];
    this->requests_ = new MPI_Request[this->chunk_count_];
    std::fill(this->requests_, this->requests_ + this->chunk_count_,
              MPI_REQUEST_NULL);
  }

  ~MPIStream() {
    delete[] this->data_;
    delete[] this->requests_;
  }

  virtual T Get(const int pos = 0) { return this->data_[pos]; }

  virtual T ReverseGet(const int pos = 0) {
    return this->Get(this->size_ - pos - 1);
  }

  virtual void Put(const T& value, const int pos = 0) {
    this->data_[pos] = value;
  }

  virtual void ReversePut(const T& value, const int pos = 0) {
    this->Put(value, this->size_ - pos - 1);
  }

  T* Data() { return this->data_; }

  void Connect(int target, const int size = 0) {
    this->target_ = target;
    this->size_ = size;
    this->chunk_count_ = CalcChunkCount(size);
  }

  void CancelAll() {
    for (int i = 0; i < this->chunk_count_; ++i)
      MPI_Cancel(&(this->requests_[i]));
  }

  int Waitall() {
    return MPI_Waitall(
        /*count=*/this->chunk_count_,
        /*array_of_requests=*/this->requests_,
        /*array_of_statuses=*/MPI_STATUSES_IGNORE);
  }

#ifdef DEBUG
 public:
#else
 protected:
#endif
  static int CalcChunkCount(const int size) {
    return ((size + kChunkSizeMask) >> kLogChunkSize);
  }
  static constexpr int kLogChunkSize = 17;
  static constexpr int kChunkSize = (1 << kLogChunkSize);
  static constexpr int kChunkSizeMask = (kChunkSize - 1);

#ifdef DEBUG
 public:
#else
 protected:
#endif
  T* data_;
  MPI_Datatype datatype_;
  MPI_Comm comm_;
  MPI_Request* requests_;
  int size_, chunk_count_;
  int target_;
};

template <class T = float>
class MPIInStream : public MPIStream<T> {
 public:
  MPIInStream(const int capacity = 0, MPI_Datatype datatype = MPI_FLOAT,
              MPI_Comm comm = MPI_COMM_WORLD)
      : MPIStream<T>(capacity, datatype, comm) {}

  virtual T Get(const int pos) override {
    if (this->target_ == -1) return this->data_[pos];
    int chunk_id = (pos >> kLogChunkSize);
    MPI_Wait(/*request=*/&(this->requests_[chunk_id]),
             /*status=*/MPI_STATUS_IGNORE);
    return this->data_[pos];
  }

  void Irecv() {
    if (this->target_ == -1) return;
    for (int i = 0; i < this->chunk_count_; ++i) {
#ifdef DEBUG
      // int rank;
      // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      // printf(
      //     "Irecv rank = %d, buf_pos = %d, count = %d, source = %d, tag =
      //     %d\n", rank, i << kLogChunkSize, kChunkSize, this->target_, i);
#endif
      MPI_Irecv(/*buf=*/this->data_ + (i << kLogChunkSize),
                /*count=*/kChunkSize,
                /*datatype=*/this->datatype_,
                /*source=*/this->target_,
                /*tag=*/i,
                /*comm=*/this->comm_,
                /*request=*/&(this->requests_[i]));
    }
  }

  void ConnectSource(int source, const int size = 0) {
    this->Connect(/*target=*/source, /*size=*/size);
  }

#ifdef DEBUG
 public:
#else
 protected:
#endif
  using MPIStream<T>::kLogChunkSize;
  using MPIStream<T>::kChunkSize;
  using MPIStream<T>::kChunkSizeMask;
};

template <class T = float>
class MPIOutStream : public MPIStream<T> {
 public:
  MPIOutStream(const int capacity = 0, MPI_Datatype datatype = MPI_FLOAT,
               MPI_Comm comm = MPI_COMM_WORLD)
      : MPIStream<T>(capacity, datatype, comm) {}

  void Load(const T* buffer, int count) {
    std::memcpy(this->data_, buffer, count * sizeof(T));
  }

  void Reverse() { std::reverse(this->data_, this->data_ + this->size_); }

  void Isend() {
    if (this->target_ == -1) return;
    for (int i = 0; i < this->chunk_count_; ++i) {
#ifdef DEBUG
      // int rank;
      // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      // printf("Isend rank = %d, buf_pos = %d, count = %d, dest = %d, tag =
      // %d\n",
      //        rank, i << kLogChunkSize, kChunkSize, this->target_, i);
#endif
      MPI_Isend(
          /*buf=*/this->data_ + (i << kLogChunkSize),
          /*count=*/kChunkSize,
          /*datatype=*/this->datatype_,
          /*dest=*/this->target_,
          /*tag=*/i,
          /*comm=*/this->comm_,
          /*request=*/&(this->requests_[i]));
    }
  }

  virtual void Put(const T& value, const int pos = 0) override {
    if (this->target_ == -1) return this->MPIStream<T>::Put(value, pos);
    this->data_[pos] = value;
    if ((((pos + 1) & kChunkSizeMask) == 0) || (pos == this->size_ - 1)) {
      int chunk_id = (pos >> kLogChunkSize);
#ifdef DEBUG
      // int rank;
      // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      // printf("Isend rank = %d, buf_pos = %d, count = %d, dest = %d, tag =
      // %d\n",
      //        rank, chunk_id << kLogChunkSize, kChunkSize, this->target_,
      //        chunk_id);
#endif
      MPI_Isend(
          /*buf=*/this->data_ + (chunk_id << kLogChunkSize),
          /*count=*/kChunkSize,
          /*datatype=*/this->datatype_,
          /*dest=*/this->target_,
          /*tag=*/chunk_id,
          /*comm=*/this->comm_,
          /*request=*/&(this->requests_[chunk_id]));
    }
  }

  void ConnectDest(int dest, const int size = 0) {
    this->Connect(/*target=*/dest, /*size=*/size);
  }

#ifdef DEBUG
 public:
#else
 protected:
#endif
  using MPIStream<T>::kLogChunkSize;
  using MPIStream<T>::kChunkSize;
  using MPIStream<T>::kChunkSizeMask;
};

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
#ifdef DEBUG
    // printf("[%d] begin: ", this->rank);
    // for (size_t i = 0; i < this->block_len; ++i) printf("%f, ",
    // this->data[i]); printf("\n");
#endif
  int left_rank = ((this->rank == 0) ? (-1) : (this->rank - 1));
  int right_rank = ((this->rank + 1 == this->nprocs) ? (-1) : (this->rank + 1));
  size_t left_block_len = CalcBlockLen(this->n, this->nprocs, left_rank);
  if (left_block_len == 0) left_rank = -1;
  size_t right_block_len = CalcBlockLen(this->n, this->nprocs, right_rank);
  if (right_block_len == 0) right_rank = -1;
  auto send_left_buffer = MPIOutStream<>(/*capacity=*/this->block_len);
  send_left_buffer.ConnectDest(/*dest=*/left_rank, /*size=*/this->block_len);
  send_left_buffer.Load(this->data, this->block_len);
  auto send_right_buffer = MPIOutStream<>(/*capacity=*/this->block_len);
  send_right_buffer.ConnectDest(/*dest=*/right_rank, /*size=*/this->block_len);
  send_right_buffer.Load(this->data, this->block_len);
  send_right_buffer.Reverse();
  auto recv_left_buffer = MPIInStream<>(/*capacity=*/left_block_len);
  recv_left_buffer.ConnectSource(/*source=*/left_rank, /*size=*/left_block_len);
  auto recv_right_buffer = MPIInStream<>(/*capacity=*/right_block_len);
  recv_right_buffer.ConnectSource(/*source=*/right_rank,
                                  /*size=*/right_block_len);
  if (this->rank & 1) {
    send_right_buffer.Isend();
  } else {
    send_left_buffer.Isend();
  }
  for (int i = 0; i < this->nprocs; ++i) {
    // printf("!!! Rank = %d, i = %d\n", this->rank, i);
    if ((i ^ this->rank) & 1) {
      // recv from right, send to left
      send_left_buffer.Waitall();
      if (i == this->nprocs - 1) send_left_buffer.ConnectDest(/*dest=*/-1);
      if (right_block_len == 0) {
        send_left_buffer.Load(send_right_buffer.Data(), this->block_len);
        send_left_buffer.Reverse();
        send_left_buffer.Isend();
      } else {
        recv_right_buffer.Waitall();
        recv_right_buffer.Irecv();
        size_t j = 0, k = 0, l = 0;
        while (j < this->block_len && k < right_block_len &&
               l < this->block_len) {
          if (send_right_buffer.ReverseGet(j) < recv_right_buffer.Get(k)) {
            send_left_buffer.Put(send_right_buffer.ReverseGet(j++), l++);
          } else {
            send_left_buffer.Put(recv_right_buffer.Get(k++), l++);
          }
        }
        while (j < this->block_len && l < this->block_len)
          send_left_buffer.Put(send_right_buffer.ReverseGet(j++), l++);
        while (k < right_block_len && l < this->block_len)
          send_left_buffer.Put(recv_right_buffer.Get(k++), l++);
      }
    } else {
      // recv from left, send to right
      send_right_buffer.Waitall();
      if (i == this->nprocs - 1) send_right_buffer.ConnectDest(/*dest=*/-1);
      if (left_block_len == 0) {
        send_right_buffer.Load(send_left_buffer.Data(), this->block_len);
        send_right_buffer.Reverse();
        send_right_buffer.Isend();
      } else {
        recv_left_buffer.Waitall();
        recv_left_buffer.Irecv();
        size_t j = 0, k = 0, l = 0;
        while (j < this->block_len && k < left_block_len &&
               l < this->block_len) {
          if (send_left_buffer.ReverseGet(j) > recv_left_buffer.Get(k)) {
            send_right_buffer.Put(send_left_buffer.ReverseGet(j++), l++);
          } else {
            send_right_buffer.Put(recv_left_buffer.Get(k++), l++);
          }
        }
        while (j < this->block_len && l < this->block_len)
          send_right_buffer.Put(send_left_buffer.ReverseGet(j++), l++);
        while (k < left_block_len && l < this->block_len)
          send_right_buffer.Put(recv_left_buffer.Get(k++), l++);
      }
    }
  }
  if ((this->rank ^ this->nprocs) & 1) {
    std::memcpy(this->data, send_right_buffer.Data(),
                this->block_len * sizeof(float));
    std::reverse(this->data, this->data + this->block_len);
  } else {
    std::memcpy(this->data, send_left_buffer.Data(),
                this->block_len * sizeof(float));
  }
  send_left_buffer.Waitall();
  send_right_buffer.Waitall();
  recv_left_buffer.Waitall();
  recv_right_buffer.Waitall();
#ifdef DEBUG
  // printf("[%d] end send_left: ", this->rank);
  // PrintData(send_left_buffer.Data(), this->block_len);
  // printf("\n");
  // printf("[%d] end send_right: ", this->rank);
  // PrintData(send_right_buffer.Data(), this->block_len);
  // printf("\n");
  // printf("[%d] end recv_left: ", this->rank);
  // PrintData(recv_left_buffer.Data(), left_block_len);
  // printf("\n");
  // printf("[%d] end recv_right: ", this->rank);
  // PrintData(recv_right_buffer.Data(), right_block_len);
  // printf("\n");
#endif
}
```

### Performance

由于逻辑变得复杂, 效果非常不理想.

| Number of Nodes | Number of Tasks | Number Count | Execution Time  | Speedup    |
| --------------- | --------------- | ------------ | --------------- | ---------- |
| 1               | 1               | 100000000    | 12483.309000 ms | 1.         |
| 1               | 2               | 100000000    | 7433.843000 ms  | 1.67925379 |
| 1               | 4               | 100000000    | 5337.546000 ms  | 2.3387731  |
| 1               | 8               | 100000000    | 3785.481000 ms  | 3.29768106 |
| 1               | 16              | 100000000    | 2839.709000 ms  | 4.39598177 |
| 2               | 32              | 100000000    | 2490.944000 ms  | 5.01147717 |
