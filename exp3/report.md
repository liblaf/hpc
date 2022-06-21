---
title: "exp3: 测量 OpenMP 并行 for 循环不同调度策略的性能"
category: "Course Work"
tags:
  - "Introduction to High Performance Computing"
  - "HPC"
---

# exp3: 测量 OpenMP 并行 for 循环不同调度策略的性能

| 调度策略           | static     | dynamic    | guided     |
| ------------------ | ---------- | ---------- | ---------- |
| Sort uniform parts | 70.4433 ms | 94.0649 ms | 69.919 ms  |
| Sort random parts  | 1436.51 ms | 1376.06 ms | 1443.13 ms |

## Sort Uniform Parts

由于每个 part 的长度相同, 每次迭代的开销基本均衡. `nUniformParts = 100000`, 总迭代次数较 **多**.

因此, 分配过程耗时较长的 `dynamic` 调度开销显著高于 `static` 和 `guided`. 

考虑到每次迭代的开销波动, 使用 `static` 调度的负载并不完全均衡. 而使用 `guided` 调度能够通过 **引入较小的开销优化由于波动产生的负载不均**.

## Sort Random Parts

每个 part 的长度较为不均, 每次迭代的开销极不均衡. `nRandomParts = 100`, 总迭代次数较 **少**.

因此, 使用 `static` 和 `guided` 将导致负载的极不均衡. 而 `dynamic` 虽然引入了分配开销, 但总迭代次数少, **负载均衡带来的收益** 能够弥补 **分配** 带来的额外开销.