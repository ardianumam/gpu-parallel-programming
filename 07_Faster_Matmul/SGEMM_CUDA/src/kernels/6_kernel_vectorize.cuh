#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmVectorize(int M, int N, int K, float alpha, float *A,
                               float *B, float beta, float *C) {
  /*
  Important note: 
  The driver program (runner.cu) provide two different blockDim. This kernel only valid for the first dim, with hyoerparameters specified below. 
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN)) = (256);
  */
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN); // range = [0, 15]
  const int threadRow = threadIdx.x / (BN / TN); // range = [0, 15]

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK]; // size = [1024]
  __shared__ float Bs[BK * BN]; // size = [1024]

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / bit_per_float = 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4); // 256 / (8/4) -> range = [0, 127]
  const uint innerColA = threadIdx.x % (BK / 4); // 256 % (8/4) -> range = [0, 1]
  const uint innerRowB = threadIdx.x / (BN / 4); // 256 / (128/4) -> range = [0, 7]
  const uint innerColB = threadIdx.x % (BN / 4); // 256 % (128/4) -> range = [0, 31]

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // load chunk data from GMEM to SMEM. Let's start from B to Bs that is the easier one
    reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
    reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    // Bs = BK x BN = 8 x 128. innerRowB ranges [0, 7], populating the BK dimension. innerColB ranges [0, 31], each represent vector of 4 floats, thus representing 32x4 = 128 elements to populate BN dimension of Bs
    
    // to load the A to As, it is similar to "B to BS", with additional of transpose operation to enable coalescing in regM by aligning it in column-major order
    float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0]; // As = BM x BK = 128 x 8. innerRowB ranges [0, 127], populating the BM dimension. innerColB ranges [0, 1], each represent vector of 4 floats, thus representing 2x4 = 8 elements to populate BK dimension of Bs
    // below reassignment operation by modifying the index for transpose operation
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results 
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) { // 
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      // load C vector into registers
      float4 tmp = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      // perform GEMM update in reg
      tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
      tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
      tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
      tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
      // write back
      reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
          tmp;
    }
  }
}