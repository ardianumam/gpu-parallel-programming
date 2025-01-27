#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void vectorAddKernel(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

float runSingleStream(int N, size_t size, float* h_A, float* h_B, float* h_C, float* d_A, float* d_B, float* d_C) {
    // Create CUDA events for time measurement
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Transfer data to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel on the device
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Transfer the result back to the host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Record the stop event
    cudaEventRecord(stop, 0);

    // Synchronize to make sure all operations are completed
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

float runMultiStream(int N, size_t size, float* h_A, float* h_B, float* h_C, float* d_A, float* d_B, float* d_C) {
    // Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Create CUDA events for time measurement
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Transfer data to GPU asynchronously
    cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream2);

    // Create an event to synchronize streams
    cudaEvent_t event;
    cudaEventCreate(&event);

    // Record an event in stream2 after the data transfer
    cudaEventRecord(event, stream2);

    // Wait for the event in stream1 before launching the kernel
    cudaStreamWaitEvent(stream1, event, 0);

    // Kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel on the device asynchronously
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_A, d_B, d_C, N);

    // Transfer the result back to the host asynchronously
    cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream1);

    // Record the stop event
    cudaEventRecord(stop, 0);

    // Synchronize the streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(event);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return milliseconds;
}

int main() {
    int N = 1000000; // Size of vectors
    size_t size = N * sizeof(float);

    // Host vectors
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // Initialize host vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // Device vectors
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    int iterations = 50;
    float totalSingleStreamTime = 0;
    float totalMultiStreamTime = 0;

    for (int i = 0; i < iterations; i++) {
        totalSingleStreamTime += runSingleStream(N, size, h_A, h_B, h_C, d_A, d_B, d_C);
        totalMultiStreamTime += runMultiStream(N, size, h_A, h_B, h_C, d_A, d_B, d_C);
    }

    float averageSingleStreamTime = totalSingleStreamTime / iterations;
    float averageMultiStreamTime = totalMultiStreamTime / iterations;

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    std::cout << "Average elapsed time (Single Stream): " << averageSingleStreamTime << " ms" << std::endl;
    std::cout << "Average elapsed time (Multi Stream): " << averageMultiStreamTime << " ms" << std::endl;

    return 0;
}
