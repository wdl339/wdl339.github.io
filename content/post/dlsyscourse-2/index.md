---
author : "wdl"
title : "《CMU 10-414 Deep Learning System》课程学习笔记（下篇）"
date : "2025-01-24"
description : "深度学习系统全栈入门"
tags : [
    "AI",
    "system",
    "自学课程"
]
categories : [
    "SelfStudy"
]
math: true
---



（内容持续更新中）

## GPU Acceleration

### GPU编程

在本节我们主要讨论CUDA，尽管还有OpenCL（ARM GPU）、Metal（Apple设备）这些编程模型。

CPU是一种通用处理器，每个核都有独立的控制器，可以灵活地处理不同的任务。GPU擅长处理大量的重复任务（例如在图形渲染中，对每个像素都进行相同的处理），大量的核可以批量执行同一指令。

![](index.assets/image-20250124162401204.png)

SIMT是NVIDIA CUDA 架构的核心概念。SIMT允许多个线程同时执行相同的指令，但在不同的数据上操作。线程被分组为block，每个block共享内存。block被分组为grid，一个kernel执行一个grid。当线程遇到分支（如 if 语句）时，SIMT 架构会处理分支发散，即不同线程可能执行不同的路径。

例1：CPU和GPU上的向量加法：

```
void VecAddCPU(float* A, float *B, float* C, int n) {
    for (int i = 0; i < n; ++i) {
        C[i] = A[i] + B[i];
    }
}

__global__ void VecAddKernel(float* A, float *B, float* C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
```

- `blockIdx.x` 是线程块的索引
- `blockDim.x` 是每个线程块中的线程数
- `threadIdx.x` 是线程在线程块中的索引

每个线程计算向量 `a` 和 `b` 的一个元素之和，并将结果存储在向量 `c` 中。如果数据之间的依赖关系较强的话，可能就没办法并行。

为了执行上述GPU的向量加法，在主机端要执行以下内容：

```
void VecAddCUDA(float *Acpu, float *Bcpu, float *Ccpu, int n) {
    float *dA, *dB, *dC;
    cudaMalloc(&dA, n * sizeof(float));
    cudaMalloc(&dB, n * sizeof(float));
    cudaMalloc(&dC, n * sizeof(float));

    cudaMemcpy(dA, Acpu, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, Bcpu, n * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 512;
    int nblocks = (n + threads_per_block - 1) / threads_per_block;
    VecAddKernel<<<nblocks, threads_per_block>>>(dA, dB, dC, n);

    cudaMemcpy(Ccpu, dC, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}
```

首先在GPU上分配显存，将两个加数拷贝到设备中，根据数据的规模确定要启用的block数量，然后执行GPU代码，最后将结果拷贝回CPU内存，释放相应显存。

内存拷贝非常耗时，因此我们希望将数据一直保留在GPU显存中进行计算，而非在CPU和GPU之间频繁地拷贝。

![](index.assets/image-20250124165119346.png)

例2：window sum是一种权重全为1的卷积，可以这么写：

```
#define RADIUS 2

__global__ void WindowSumSimpleKernel(float* A, float *B, int n) {
    int out_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (out_idx < n) {
        float sum = 0;
        for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
            sum += A[dx + out_idx + RADIUS];
        }
        B[out_idx] = sum;
    }
}
```

![](index.assets/image-20250124165431328.png)

但这个算法并不高效，会重复访问数据。这时候可以引入共享内存进行优化，将一个block内会用到的数据全部读取到共享内存中。数据加载的任务可以分给每个线程并行完成。

```
__global__ void WindowSumSharedKernel(float* A, float* B, int n) {
    __shared__ float temp[THREADS_PER_BLOCK + 2 * RADIUS];
    int base = blockDim.x * blockIdx.x;
    int out_idx = base + threadIdx.x;
    if (base + threadIdx.x < n) {
        temp[threadIdx.x] = A[base + threadIdx.x];
    }
    if (threadIdx.x < 2 * RADIUS && base + THREADS_PER_BLOCK + threadIdx.x < n) {
        temp[threadIdx.x + THREADS_PER_BLOCK] = A[base + THREADS_PER_BLOCK + threadIdx.x];
    }
    __syncthreads();
    if (out_idx < n) {
        float sum = 0;
        for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
            sum += temp[threadIdx.x + dx + RADIUS];
        }
        B[out_idx] = sum;
    }
}
```

通过`__syncthreads`同步，确保所有线程都将数据加载完毕。

### 样例学习：GPU上的矩阵乘

从线程的细粒度来说，我们可以在GPU上实现一个寄存器分块版本的矩阵乘法：

```
__global__ void mm(float A[N][N], float B[N][N], float C[N][N]) {
    int ybase = blockIdx.y * blockDim.y + threadIdx.y;
    int xbase = blockIdx.x * blockDim.x + threadIdx.x;

    float c[V][V] = {0};
    float a[V], b[V];
    for (int k = 0; k < N; ++k) {
        a[:] = A[k, ybase*V : ybase*V + V];
        b[:] = B[k, xbase*V : xbase*V + V];
        for (int y = 0; y < V; ++y) {
            for (int x = 0; x < V; ++x) {
                c[y][x] += a[y] * b[x];
            }
        }
    }
    C[ybase * V : ybase * V + V, xbase * V : xbase * V + V] = c[:,:];
}
```

每个线程负责计算一个分块的结果。

![](index.assets/image-20250124170022891.png)

还可以将计算一块的任务交给一个block，这样就可以使用共享内存技术由block内的线程共同加载要用到的数据。

```
__global__ void mm(float A[N][N], float B[N][N], float C[N][N]) {
    __shared__ float sA[S][L], sB[S][L];
    float c[V][V] = {0};
    float a[V], b[V];
    int yblock = blockIdx.y;
    int xblock = blockIdx.x;

    for (int ko = 0; ko < N; ko += S) {
        __syncthreads();
        // needs to be implemented by thread cooperative fetching
        sA[:, :] = A[ko + S, yblock * L : yblock * L + L];
        sB[:, :] = B[ko + S, xblock * L : xblock * L + L];
        __syncthreads();
        for (int ki = 0; ki < S; ++ki) {
            a[:] = sA[ki, threadIdx.x * V + V];
            b[:] = sB[ki, threadIdx.x * V + V];
            for (int y = 0; y < V; ++y) {
                for (int x = 0; x < V; ++x) {
                    c[y][x] += a[y] * b[x];
                }
            }
        }
    }
    int ybase = blockIdx.y * blockDim.y + threadIdx.y;
    int xbase = blockIdx.x * blockDim.x + threadIdx.x;
    C[ybase * V : ybase * V + V, xbase * V : xbase * V + V] = c[:, :];
}
```

上述代码从全部内存到共享内存的加载过程被复用L次（计算每个分块矩阵都要读取L次AB的行列向量），从共享内存到寄存器被复用V次（在分块矩阵中按照长度V进行了二次分块计算）![image.png](https://pics.zhouxin.space/202407261448550.png?x-oss-process=image/quality,q_90/format,webp)各线程读取数据到共享内存的过程为：

```
sA[:, :] = A[k : k + S, yblock * L : yblock * L + L];


int nthreads = blockDim.y * blockDim.x;
int tid = threadIdx.y * blockDim.x + threadIdx.x;
for(int j = 0; j < L * S / nthreads; ++j) {
    int y = (j * nthreads + tid) / L;
    int x = (j * nthreads + tid) % L;
    s[y, x] = A[k + y, yblock * L + x];
}
```

### 更多GPU优化技术

- Global memory continuous read（尽量使内存访问是连续的）
- Shared memory bank conflict（共享内存被分成多个“银行”，每个银行可以同时处理一个访问请求。如果多个线程同时访问同一个银行的不同地址，就会发生冲突，导致访问延迟）
- Software pipelining（通过重排指令顺序，使得计算和内存访问可以重叠进行）
- Warp level optimizations （warp 是一个更小的执行单元，通常包含 32 个线程）
- Tensor Core（NVIDIA GPU 中专门用于加速矩阵乘法和深度学习计算的硬件单元）









## 参考

1. [《CMU 10-414 deep learning system》学习笔记](https://www.zhouxin.space/notes/notes-on-cmu-10-414-deep-learning-system/#lecture-3-manual-neural-networks)
2. [深度学习系统作业 - 知乎 (zhihu.com)](https://www.zhihu.com/column/c_1582462878204063744)





