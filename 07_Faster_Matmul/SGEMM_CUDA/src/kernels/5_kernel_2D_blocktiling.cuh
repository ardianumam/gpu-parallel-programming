#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {
  
  /* Important notes:
     Each thread handles one 2D tiles with size TMxTN. SMEM data load is in As and Bs with size As=BMxBK and Bs=BKxBN, so the corresponding sub-block matrix C size will be BMxBN. As a result, the number of needed thread per block to finish this BMxBN sub-matrix C is BNxBN/TM/TN, which defines the blockDim.
  */
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint totalResultsBlocktile = BM * BN;
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN); // number of threads per 2D tile

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile == blockDim.x); // make sure the number threads in each blocks equals to the number of thread in each 2D tile

  // each thread handles 2D tile of TMxTN. So, the treadCol indexes this i-th tile in the BN axis. So does threadRow.
  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const uint strideA = numThreadsBlocktile / BK; // if TM=TN=BK=8, strideA=8*8/8=8
  const uint innerRowB = threadIdx.x / BN; // ranges [0,0]
  const uint innerColB = threadIdx.x % BN; // ranges [0,63]
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const uint strideB = numThreadsBlocktile / BN; // 1

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  // register caches for As and Bs
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate GMEM to the SMEM stored in As and Bs, with size As=BMxBK, Bs=BKxBN, to compute sub-matrix C with size BMxBN. The number of thread per block is defined as BMxBN/TM/TN, as per thread handles a TMxTN tile. These number thread per block is smaller than the size of As and Bs, while the whole As and Bs are needed to compute sub-matrix C with size of BMxBN. Therefore, every thread will load multiple entries in As, Bs, using loop as shown below. Specifically, we load in column-major order to maximize the GMEM coalescing.
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) { // 
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) { // THICKY!! Here, we also flip the loop order as in 1D tiling --> corresponds to inner loop in original loop order
      // block into registers
      for (uint i = 0; i < TM; ++i) { // store "1 col with TM size of sub-block As" to register
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) { // store "1 row with TN size of sub-block Bs" to register
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) { // row loop for the TMxTN grid result --> outer loop in original loop order
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) { // col loop for the TMxTN grid result --> outer loop in original loop order
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the results --> populate threadResults to the matrix C
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
          alpha * threadResults[resIdxM * TN + resIdxN] +
          beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
    }
  }
}