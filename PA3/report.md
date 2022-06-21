---
title: "PA3: 全源最短路"
category: "Course Work"
tags:
  - "Introduction to High Performance Computing"
  - "HPC"
  - "CUDA"
---

# PA3: 全源最短路

## Environment

P100 GPU 最大支持每个 SM 64KB shared memory，但每个 thread block 最多只支持 48KB

## Method

使用 [实验三 - 高性能计算导论实验文档 (tsinghua.edu.cn)](https://lab.cs.tsinghua.edu.cn/hpc/doc/exp/3.apsp/) 中的分块方法. 一个 thread block 处理一个矩阵分块. 每个 thread block 所需使用的数据全部拷贝到 shared memory 中. 在 `threadIdx` 的基础上偏移 `i_start`, `j_start` 或 `center_block_start` 即可将 shared memory 中的坐标映射到 global memory 中的不同矩阵分块.

### Phase 1

```
k = [ p * b, (p + 1) * b )
```

对于每个 thread block, 访问范围包括 `k * k` 共 `b * b` 个 `int`, 也即需要 `b * b * sizeof(int)` 大小的 shared memory.

对于每个 `p`, 仅需一个 thread block 即可完成任务. 但是很浪费.

### Phase 2

#### Horizontal

```
k = [ p * b          , (p + 1) * b          )
i = [ p * b          , (p + 1) * b          )
j = [ blockIdx.x * b , (blockIdx.x + 1) * b )
```

特别的, 若 `j` 的范围恰好在 center block 后, 即 `blockIdx.x * b >= center_block_start` 时, 则需额外偏移 `b`.

对于每个 thread block, 访问范围包括 `i * j`, `i * k`, `k * j`, 其中 `i * j` 和 `k * j` 重合, 因此共 `2 * b * b` 个 `int`.

共需 `(ceil(n / p) - 1) * 1` 个 thread block.

#### Vertical

```
k = [ p * b          , (p + 1) * b          )
i = [ blockIdx.y * b , (blockIdx.y + 1) * b )
j = [ p * b          , (p + 1) * b          )
```

特别的, 若 `i` 的范围恰好在 center block 后, 即 `blockIdx.y * b >= center_block_start` 时, 则需额外偏移 `b`.

对于每个 thread block, 访问范围包括 `i x j`, `i x k`, `k x j`, 其中 `i x j` 和 `i x k` 重合, 因此共 `2 * b * b` 个 `int`.

共需 `1 * (ceil(n / p) - 1)` 个 thread block.

### Phase 3

```
k = [ p * b          , (p + 1) * b          )
i = [ blockIdx.y * b , (blockIdx.y + 1) * b )
i = [ blockIdx.x * b , (blockIdx.x + 1) * b )
```

特别的, 若 `i` 或 `j` 的范围恰好在 center block 后, 即 `blockIdx * b >= center_block_start` 时, 则需额外偏移 `b`.

对于每个 thread block, 访问范围包括 `i x j`, `i x k`, `k x j`, 均不重合, 共 `3 * b * b` 个 `int`.

共需 `(ceil(n / p) - 1) * (ceil(n / p) - 1)` 个 thread block.

综合考虑, 取 `b = 32`, 每个 thread block 共 `32 x 32` 个 thread, 既不会超出 shared memory 限制, 又能够避免 bank conflict.

## Performance

| n     | `apspRef()` (ms) | `apsp()` (ms) | Speedup     |
| ----- | ---------------- | ------------- | ----------- |
| 1000  | 14.814903        | 2.969371      | 4.98923947  |
| 2500  | 377.148402       | 37.660415     | 10.01445157 |
| 5000  | 2972.073596      | 260.960028    | 11.38899938 |
| 7500  | 10016.146987     | 872.866804    | 11.47500047 |
| 10000 | 22632.211686     | 2060.573817   | 10.98345107 |

在 `n = 1000` 下进行 profiling.

### `nvprof` Events

```
Invocations                                Event Name         Min         Max         Avg       Total
Device "Tesla P100-PCIE-16GB (0)"
    Kernel: _GLOBAL__N__51_tmpxft_000981f1_00000000_20_apsp_compute_61_cpp1_ii_034c69fe::Phase2KernelHorizontal(int, int*, int, int)
         96                   shared_ld_bank_conflict           0           0           0           0
         96                   shared_st_bank_conflict           0           0           0           0
    Kernel: _GLOBAL__N__51_tmpxft_000981f1_00000000_20_apsp_compute_61_cpp1_ii_034c69fe::Phase2KernelVertical(int, int*, int, int)
         96                   shared_ld_bank_conflict           0           0           0           0
         96                   shared_st_bank_conflict           0           0           0           0
    Kernel: _GLOBAL__N__51_tmpxft_000981f1_00000000_20_apsp_compute_61_cpp1_ii_034c69fe::Phase1Kernel(int, int*, int, int)
         96                   shared_ld_bank_conflict           0           0           0           0
         96                   shared_st_bank_conflict           0           0           0           0
    Kernel: _GLOBAL__N__51_tmpxft_000981f1_00000000_20_apsp_compute_61_cpp1_ii_034c69fe::Phase3Kernel(int, int*, int, int)
         96                   shared_ld_bank_conflict           0           0           0           0
         96                   shared_st_bank_conflict           0           0           0           0
```

没有出现 bank conflict.

### `nvprof` Metrics

```
Invocations                               Metric Name                         Metric Description         Min         Max         Avg
Device "Tesla P100-PCIE-16GB (0)"
    Kernel: _GLOBAL__N__51_tmpxft_000981f1_00000000_20_apsp_compute_61_cpp1_ii_034c69fe::Phase2KernelHorizontal(int, int*, int, int)
         96                         branch_efficiency                          Branch Efficiency     100.00%     100.00%     100.00%
         96                 warp_execution_efficiency                  Warp Execution Efficiency      97.97%     100.00%      98.03%
         96         warp_nonpred_execution_efficiency   Warp Non-Predicated Execution Efficiency      81.95%      95.89%      95.26%
         96                            gld_efficiency              Global Memory Load Efficiency     100.00%     100.00%     100.00%
         96                            gst_efficiency             Global Memory Store Efficiency     100.00%     100.00%     100.00%
         96                         shared_efficiency                   Shared Memory Efficiency      67.38%      69.64%      67.45%
    Kernel: _GLOBAL__N__51_tmpxft_000981f1_00000000_20_apsp_compute_61_cpp1_ii_034c69fe::Phase2KernelVertical(int, int*, int, int)
         96                         branch_efficiency                          Branch Efficiency     100.00%     100.00%     100.00%
         96                 warp_execution_efficiency                  Warp Execution Efficiency      51.59%     100.00%      98.49%
         96         warp_nonpred_execution_efficiency   Warp Non-Predicated Execution Efficiency      41.59%      97.89%      95.93%
         96                            gld_efficiency              Global Memory Load Efficiency     100.00%     100.00%     100.00%
         96                            gst_efficiency             Global Memory Store Efficiency     100.00%     100.00%     100.00%
         96                         shared_efficiency                   Shared Memory Efficiency      18.58%      69.01%      67.43%
    Kernel: _GLOBAL__N__51_tmpxft_000981f1_00000000_20_apsp_compute_61_cpp1_ii_034c69fe::Phase1Kernel(int, int*, int, int)
         96                         branch_efficiency                          Branch Efficiency     100.00%     100.00%     100.00%
         96                 warp_execution_efficiency                  Warp Execution Efficiency      47.13%     100.00%      98.35%
         96         warp_nonpred_execution_efficiency   Warp Non-Predicated Execution Efficiency      46.31%      97.76%      96.15%
         96                            gld_efficiency              Global Memory Load Efficiency     100.00%     100.00%     100.00%
         96                            gst_efficiency             Global Memory Store Efficiency     100.00%     100.00%     100.00%
         96                         shared_efficiency                   Shared Memory Efficiency      18.52%      68.69%      67.12%
    Kernel: _GLOBAL__N__51_tmpxft_000981f1_00000000_20_apsp_compute_61_cpp1_ii_034c69fe::Phase3Kernel(int, int*, int, int)
         96                         branch_efficiency                          Branch Efficiency     100.00%     100.00%     100.00%
         96                 warp_execution_efficiency                  Warp Execution Efficiency      98.12%     100.00%      98.18%
         96         warp_nonpred_execution_efficiency   Warp Non-Predicated Execution Efficiency      84.89%      95.93%      95.23%
         96                            gld_efficiency              Global Memory Load Efficiency     100.00%     100.00%     100.00%
         96                            gst_efficiency             Global Memory Store Efficiency     100.00%     100.00%     100.00%
         96                         shared_efficiency                   Shared Memory Efficiency      67.69%      69.91%      67.75%
```

可以看出各项指标的利用率都较充分, 但 shared memory 利用率较低.