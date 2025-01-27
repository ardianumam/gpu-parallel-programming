#include <stdio.h>

__global__ void myKernel()
{
  printf("Hello from the GPU!\n");
}

int main()
{
  myKernel<<<1, 10>>>(); // Launch kernel with 1 block of 10 threads
  cudaDeviceSynchronize();
}