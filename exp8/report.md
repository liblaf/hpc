---
title: "exp8: 单机性能优化"
category: "Course Work"
tags:
  - "Introduction to High Performance Computing"
  - "HPC"
---

# exp8: 单机性能优化

## Task 0

### Performance

| Option  | Elapsed Time / seconds | Performance / GFlops |
| ------- | ---------------------- | -------------------- |
| `-O0`   | 1.0072                 | 0.2665               |
| `-O1`   | 0.3461                 | 0.7755               |
| `-O2`   | 0.3332                 | 0.8057               |
| `-O3`   | 0.0496                 | 5.4081               |
| `-fast` | 0.0386                 | 6.9524               |

## Task 1

### Performance

| `UNROLL_N` | Elapsed Time / seconds | Performance / GFlops |
| ---------- | ---------------------- | -------------------- |
| 1          | 2.0814                 | 15.7431              |
| 2          | 1.9311                 | 16.9688              |
| 4          | 1.8048                 | 18.1562              |
| 8          | 1.7787                 | 18.4227              |
| 16         | 1.8276                 | 17.9297              |

## 回答问题

##### Question 1

请参考 [ICC 手册](https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/compiler-options/alphabetical-list-of-compiler-options.html) 并简述参数 (`-O0`, `-O1`, `-O2`, `-O3`, `-fast`) 分别进行了哪些编译优化。每种参数罗列几个优化技术即可。

##### Answer 1

###### `-O0`

禁用所有优化

###### `-O1`

- data-flow analysis
- code motion
- strength reduction and test replacement
- split-lifetime analysis
- instruction scheduling

###### `-O2`

- Vectorization
- Inlining of intrinsics
- inlining
- constant propagation
- forward substitution
- routine attribute propagation
- variable address-taken analysis
- dead static function elimination
- removal of unreferenced variables

###### `-O3`

- Fusion
- Block-Unroll-and-Jam
- collapsing `if` statements

###### `-fast`

- interprocedural optimization between files
- optimization of floating-point divides that give slightly less precise results than full IEEE division
- link all libraries statically
- generate instructions for the highest instruction set available

##### Question 2

请简述任务一中循环展开带来的好处。

##### Answer 2

- 可以减少循环变量的比较次数和分支跳转次数
- 减少数据依赖, 增加并发, 充分利用 CPU 流水线