#include <stdio.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <chrono>

#define BATCH_SIZE 32
#define CHANNELS 3
#define HEIGHT 64
#define WIDTH 64

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUDNN(call) { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN error in %s:%d: %s\n", __FILE__, __LINE__, cudnnGetErrorString(status)); \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    // Initialize cuDNN
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // Create tensor descriptors
    cudnnTensorDescriptor_t in_desc, out_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&in_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&out_desc));

    // Set tensor dimensions and data type
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BATCH_SIZE, CHANNELS, HEIGHT, WIDTH));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BATCH_SIZE, CHANNELS, HEIGHT, WIDTH));

    // Create activation descriptor
    cudnnActivationDescriptor_t activation_desc;
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&activation_desc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_TANH, CUDNN_NOT_PROPAGATE_NAN, 0.0));

    // Allocate device memory
    size_t tensor_size = BATCH_SIZE * CHANNELS * HEIGHT * WIDTH * sizeof(float);
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, tensor_size));
    CHECK_CUDA(cudaMalloc(&d_output, tensor_size));

    // Initialize input with random values
    float *h_input = new float[BATCH_SIZE * CHANNELS * HEIGHT * WIDTH];
    for (int i = 0; i < BATCH_SIZE * CHANNELS * HEIGHT * WIDTH; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    CHECK_CUDA(cudaMemcpy(d_input, h_input, tensor_size, cudaMemcpyHostToDevice));

    // Perform activation forward pass multiple times and measure time
    const float alpha = 1.0f, beta = 0.0f;
    int num_runs = 50;
    double total_time = 0.0;

    for (int i = 0; i < num_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        CHECK_CUDNN(cudnnActivationForward(cudnn, activation_desc, &alpha, in_desc, d_input, &beta, out_desc, d_output));
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        total_time += elapsed.count();
    }

    // Calculate average time
    double avg_time = total_time / num_runs;
    std::cout << "Average time for " << num_runs << " runs: " << avg_time << " ms" << std::endl;

    // Clean up
    delete[] h_input;
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDNN(cudnnDestroyActivationDescriptor(activation_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(in_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(out_desc));
    CHECK_CUDNN(cudnnDestroy(cudnn));

    return 0;
}
