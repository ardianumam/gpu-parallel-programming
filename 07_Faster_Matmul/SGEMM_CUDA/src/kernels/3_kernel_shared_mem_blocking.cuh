#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  // the output block that we want to compute in this threadblock --> OK
  const uint cRow = blockIdx.x; // cRow is from ceil(M/BLOCKSIZE), assiged as blockIdx.x in runner.cu
  const uint cCol = blockIdx.y; // cRow is from ceil(N/BLOCKSIZE), assiged as blockIdx.y in runner.cu

  // allocate buffer for current block in fast shared mem --> OK
  // shared mem is shared between all threads in a block
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // the inner row & col that we're accessing in this thread --> OK
  const uint threadCol = threadIdx.x % BLOCKSIZE; // --> make threadCol in the consecutive threadIdx.x 
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  // advance pointers to the starting positions of the next nRow, nCol --> OK
  A += cRow * BLOCKSIZE * K; // row=cRow, col=0 --> to go to the next element row, we need shift K. In each nRow, we need to go to the next BLOCKSIZE element row, from the current cRow. So, we need to advance cRow*BLOCKSIZE*K.
  B += cCol * BLOCKSIZE; // row=0, col=cCol
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // Have each thread load one of the elements in A & B. In CUDA, As and Bs below are loaded per element by each thread on their execution --> OK
    // Make the threadCol (=threadIdx.x) the consecutive index --> OK
    // to allow global memory access coalescing
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // block threads in this block until cache is fully populated
    __syncthreads(); // this synch in the BLOCK-LEVEL. So, all threads in the current (each) block will be be synchronized. 
    // advance pointers to the starting position of the next block --> OK 
    A += BLOCKSIZE;
    B += BLOCKSIZE * N; // to shift to the next row, we need shift N threads. So, we shift BLOCKSIZE*N since we want to shift the next BLOCKSIZE row

    // execute the dotproduct on the currently cached block --> OK
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }
  C[threadRow * N + threadCol] = // -- >OK
      alpha * tmp + beta * C[threadRow * N + threadCol];
}