---
title: "Matrix Multiplication with CUDA: A Beginner's Guide"
date: 2025-02-16
draft: false
---
# Matrix Multiplication with CUDA: A Beginner's Guide

Matrix multiplication is a fundamental operation in scientific computing, machine learning, and computer graphics. Leveraging the parallel processing power of GPUs can significantly speed up matrix operations. In this blog, we'll break down a CUDA-based matrix multiplication program step by step, explaining it in an intuitive manner.

## Why Use CUDA for Matrix Multiplication?

Matrix multiplication involves many repeated calculations that can be performed in parallel. CPUs process computations sequentially for the most part, whereas GPUs excel at handling thousands of parallel operations. CUDA allows us to write programs that run efficiently on NVIDIA GPUs, leveraging their parallel computing capabilities.

## Understanding the Code Step by Step

This C++ program with CUDA performs matrix multiplication using GPU acceleration. Let's break it down logically.

### 1. Including Required Headers

```cpp
#include <iostream>
#include <cuda_runtime.h>
```

We include `<iostream>` for input-output operations and `<cuda_runtime.h>` to use CUDA functions that facilitate GPU computations.

### 2. CUDA Kernel for Matrix Multiplication

```cpp
__global__
void matMulKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x; // Row index
    int j = threadIdx.y + blockDim.y * blockIdx.y; // Column index
    float sum = 0;
    if(i<n && j<n) {
        for (int k = 0; k < n; k++)
            sum += A[i * n + k] * B[k * n + j];
        C[i * n + j] = sum;
    }
}
```

This kernel function is executed on the GPU. Each thread in CUDA is responsible for computing one element of the result matrix `C`.

### Breaking Down the CUDA Kernel with an Example

CUDA kernels execute in parallel, meaning that each thread operates on a specific element of the output matrix `C`. Let's break it down using an example.

#### Example Matrix Multiplication

Suppose we have two small matrices:

```
A = | 1 2 |    B = | 3 4 |
    | 3 4 |        | 5 6 |
```

The expected result `C = A × B` is:

```
C = | (1×3 + 2×5)  (1×4 + 2×6) |  =  | 13 16 |
    | (3×3 + 4×5)  (3×4 + 4×6) |     | 29 36 |
```

### How CUDA Threads Process This Computation

Each thread computes a single element of `C`. Let's assume a 2×2 grid of threads.

- **Thread (0,0)** computes `C[0][0]`:
  - `i = 0` (row 0), `j = 0` (column 0)
  - Calculation: `1×3 + 2×5 = 13`
- **Thread (0,1)** computes `C[0][1]`:
  - `i = 0` (row 0), `j = 1` (column 1)
  - Calculation: `1×4 + 2×6 = 16`
- **Thread (1,0)** computes `C[1][0]`:
  - `i = 1` (row 1), `j = 0` (column 0)
  - Calculation: `3×3 + 4×5 = 29`
- **Thread (1,1)** computes `C[1][1]`:
  - `i = 1` (row 1), `j = 1` (column 1)
  - Calculation: `3×4 + 4×6 = 36`

Each thread retrieves the corresponding row from `A` and column from `B` and performs the dot product calculation independently. Since all threads execute in parallel, the computation completes much faster than sequential execution on a CPU.

### Explanation of CUDA Thread Indexing

```cpp
int i = threadIdx.x + blockDim.x * blockIdx.x; // i represents row index
int j = threadIdx.y + blockDim.y * blockIdx.y; // j represents column index
```

Each thread gets a unique `(i, j)` pair, which determines which element of `C` it calculates.

#### Understanding Index Computation

- `A[i * n + k]` retrieves the element at row `i` and column `k` in `A`.
- `B[k * n + j]` retrieves the element at row `k` and column `j` in `B`.
- The final computed `C[i * n + j]` is stored in the correct position in `C`.

For example, for `i = 0`, `j = 1`:

- The kernel fetches `A[0 * 2 + k]` for `k = 0, 1`, which corresponds to row `0` in `A`.
- It fetches `B[k * 2 + 1]`, which corresponds to column `1` in `B`.
- The sum of the dot product is stored in `C[0 * 2 + 1]`, which is `C[0][1]`.

This indexing system allows us to scale up the matrix multiplication to larger sizes by distributing computations across multiple threads and blocks.

### 3. Main Function: Memory Allocation and Execution

```cpp
int main() {
    const int n = 10;
    float A[n][n], B[n][n], C[n][n];
```

We define the size of matrices and allocate space in host (CPU) memory.

#### Initializing Matrices

```cpp
for(int i=0; i<n; i++) {
    for(int j=0; j<n; j++) {
        A[i][j] = 1.0f;
        B[i][j] = 2.0f;
    }
}
```

We initialize matrix `A` with `1.0` and matrix `B` with `2.0`. This ensures predictable output, making debugging easier.

#### Copying Data to GPU

```cpp
cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
```

Using `cudaMemcpy`, we transfer the matrix data from CPU to GPU memory.

### 4. Launching the CUDA Kernel

```cpp
dim3 blocksPerGrid((2*n-1)/n, (2*n-1)/n);
dim3 threadsPerBlock(16, 16);
matMulKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, n);
```

### 5. Copying Results Back to CPU and Printing

```cpp
cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
```

### 6. Freeing Memory

```cpp
cudaFree(A_d);
cudaFree(B_d);
cudaFree(C_d);
```

## Conclusion

This program demonstrates how to implement matrix multiplication using CUDA for parallel execution on the GPU. Understanding thread indexing, memory allocation, and kernel execution is key to writing efficient CUDA programs. By experimenting with different optimizations, we can further enhance performance and scalability.

