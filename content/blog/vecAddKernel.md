---
title: "Understanding CUDA: A Simple Vector Addition Example"
date: 2025-02-15
draft: false
---
# Understanding CUDA: A Simple Vector Addition Example

CUDA (Compute Unified Device Architecture) is a parallel computing platform by NVIDIA that allows developers to leverage the massive computational power of GPUs. In this blog, we'll break down a simple CUDA program that performs vector addition using GPU acceleration.

## The Code
Let's analyze the given CUDA program, which adds two vectors element-wise using parallel processing:

```cpp
#include <iostream>
#include <cuda_runtime.h>

__global__ 
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int n = 256;
    float A[n], B[n], C[n];

    for(int i=1; i<=n; i++) {
        A[i-1] = i;
        B[i-1] = i;
    }

    float *A_d, *B_d, *C_d;
    int size = n * sizeof(float);

    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);

    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    for(int i=0; i<n; i++) {
        printf("%.2f\n", C[i]);
    }

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
```

## Breaking It Down
### 1. The Kernel Function
The function `vecAddKernel` is a CUDA kernel, which means it runs on the GPU. It follows this format:

```cpp
__global__ 
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
```

- `__global__` indicates this function runs on the GPU and is called from the CPU.
- Each thread calculates an index `i` based on its thread ID (`threadIdx.x`), block ID (`blockIdx.x`), and the number of threads per block (`blockDim.x`).
- If `i` is within the valid range, it performs element-wise addition of vectors A and B.

### 2. Memory Allocation on GPU
Before calling the kernel, we allocate memory on the GPU for vectors A, B, and C using `cudaMalloc`:

```cpp
cudaMalloc((void **) &A_d, size);
cudaMalloc((void **) &B_d, size);
cudaMalloc((void **) &C_d, size);
```

The `cudaMalloc` function reserves GPU memory, and `size` is computed as `n * sizeof(float)` to allocate space for `n` floating-point numbers.

#### Why Do We Cast to `(void **)`?
The address of the pointer variable should be cast to `(void **)` because the function expects a generic pointer. The memory allocation function is a generic function that is not restricted to any particular type of object. Since `cudaMalloc` takes a `void**` as its first argument, casting ensures compatibility regardless of the data type being allocated.

### 3. Copying Data Between Host and Device
Since GPU memory is separate from CPU memory, we need to copy data to the GPU before running the kernel:

```cpp
cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
```

Similarly, after computation, we copy the result back to the host:

```cpp
cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
```

### 4. Kernel Launch
The kernel is launched using:

```cpp
vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);
```

- `<<<ceil(n/256.0), 256>>>` defines the grid and block size.
- We use `256` threads per block, and the number of blocks is `ceil(n/256.0)`, ensuring all elements are processed.

### 5. Cleaning Up
Once processing is complete, we free the allocated GPU memory:

```cpp
cudaFree(A_d);
cudaFree(B_d);
cudaFree(C_d);
```

## Running the Program
To compile and run this CUDA program, use:

```sh
nvcc vector_add.cu -o vector_add
./vector_add
```

This will output the sum of the two vectors.

## Conclusion
This simple CUDA example demonstrates parallel computation with vector addition. Understanding memory allocation, kernel launching, and data transfer is crucial for more advanced GPU programming. Happy coding!

