#include <stdio.h>
#include <cuda.h>

__global__ void vector_addition(int *A, int *B, int *C, int size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size)
        C[tid] = A[tid] + B[tid];
}

float single_stream(int *A_Host, int *B_Host, int *C_Host, int size, int DimBlock, int DimGrid) {
    int *A_GPU, *B_GPU, *C_GPU;
    cudaMalloc(&A_GPU, sizeof(int) * size);
    cudaMalloc(&B_GPU, sizeof(int) * size);
    cudaMalloc(&C_GPU, sizeof(int) * size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMemcpy(A_GPU, A_Host, sizeof(int) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_GPU, B_Host, sizeof(int) * size, cudaMemcpyHostToDevice);
    vector_addition<<<DimGrid, DimBlock>>>(A_GPU, B_GPU, C_GPU, size);
    cudaMemcpy(C_Host, C_GPU, sizeof(int) * size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaFree(A_GPU);
    cudaFree(B_GPU);
    cudaFree(C_GPU);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

float two_streams_data_transfer(int *A_Host, int *B_Host, int *C_Host, int size, int DimBlock, int DimGrid) {
    int *A_GPU, *B_GPU, *C_GPU;
    cudaMalloc(&A_GPU, sizeof(int) * size);
    cudaMalloc(&B_GPU, sizeof(int) * size);
    cudaMalloc(&C_GPU, sizeof(int) * size);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaEvent_t start, stop, event1, event2;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    cudaEventRecord(start);
    cudaMemcpyAsync(A_GPU, A_Host, sizeof(int) * size, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(B_GPU, B_Host, sizeof(int) * size, cudaMemcpyHostToDevice, stream2);
    cudaEventRecord(event1, stream1);
    cudaEventRecord(event2, stream2);
    cudaStreamWaitEvent(0, event1, 0);
    cudaStreamWaitEvent(0, event2, 0);
    
    vector_addition<<<DimGrid, DimBlock>>>(A_GPU, B_GPU, C_GPU, size);
    cudaMemcpy(C_Host, C_GPU, sizeof(int) * size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaFree(A_GPU);
    cudaFree(B_GPU);
    cudaFree(C_GPU);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(event1);
    cudaEventDestroy(event2);

    return ms;
}

float two_streams_full_pipeline(int *A_Host, int *B_Host, int *C_Host, int size, int DimBlock, int DimGrid) {
    int halfSize = size / 2;
    int *A1_GPU, *B1_GPU, *C1_GPU, *A2_GPU, *B2_GPU, *C2_GPU;
    cudaMalloc(&A1_GPU, sizeof(int) * halfSize);
    cudaMalloc(&B1_GPU, sizeof(int) * halfSize);
    cudaMalloc(&C1_GPU, sizeof(int) * halfSize);
    cudaMalloc(&A2_GPU, sizeof(int) * halfSize);
    cudaMalloc(&B2_GPU, sizeof(int) * halfSize);
    cudaMalloc(&C2_GPU, sizeof(int) * halfSize);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMemcpyAsync(A1_GPU, A_Host, sizeof(int) * halfSize, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(B1_GPU, B_Host, sizeof(int) * halfSize, cudaMemcpyHostToDevice, stream1);
    vector_addition<<<DimGrid, DimBlock, 0, stream1>>>(A1_GPU, B1_GPU, C1_GPU, halfSize);
    cudaMemcpyAsync(C_Host, C1_GPU, sizeof(int) * halfSize, cudaMemcpyDeviceToHost, stream1);

    cudaMemcpyAsync(A2_GPU, A_Host + halfSize, sizeof(int) * halfSize, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(B2_GPU, B_Host + halfSize, sizeof(int) * halfSize, cudaMemcpyHostToDevice, stream2);
    vector_addition<<<DimGrid, DimBlock, 0, stream2>>>(A2_GPU, B2_GPU, C2_GPU, halfSize);
    cudaMemcpyAsync(C_Host + halfSize, C2_GPU, sizeof(int) * halfSize, cudaMemcpyDeviceToHost, stream2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaFree(A1_GPU);
    cudaFree(B1_GPU);
    cudaFree(C1_GPU);
    cudaFree(A2_GPU);
    cudaFree(B2_GPU);
    cudaFree(C2_GPU);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

float three_streams_full_pipeline(int *A_Host, int *B_Host, int *C_Host, int size, int DimBlock, int DimGrid) {
    int thirdSize = size / 3;
    int *A1_GPU, *B1_GPU, *C1_GPU, *A2_GPU, *B2_GPU, *C2_GPU, *A3_GPU, *B3_GPU, *C3_GPU;
    cudaMalloc(&A1_GPU, sizeof(int) * thirdSize);
    cudaMalloc(&B1_GPU, sizeof(int) * thirdSize);
    cudaMalloc(&C1_GPU, sizeof(int) * thirdSize);
    cudaMalloc(&A2_GPU, sizeof(int) * thirdSize);
    cudaMalloc(&B2_GPU, sizeof(int) * thirdSize);
    cudaMalloc(&C2_GPU, sizeof(int) * thirdSize);
    cudaMalloc(&A3_GPU, sizeof(int) * thirdSize);
    cudaMalloc(&B3_GPU, sizeof(int) * thirdSize);
    cudaMalloc(&C3_GPU, sizeof(int) * thirdSize);

    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMemcpyAsync(A1_GPU, A_Host, sizeof(int) * thirdSize, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(B1_GPU, B_Host, sizeof(int) * thirdSize, cudaMemcpyHostToDevice, stream1);
    vector_addition<<<DimGrid, DimBlock, 0, stream1>>>(A1_GPU, B1_GPU, C1_GPU, thirdSize);
    cudaMemcpyAsync(C_Host, C1_GPU, sizeof(int) * thirdSize, cudaMemcpyDeviceToHost, stream1);

    cudaMemcpyAsync(A2_GPU, A_Host + thirdSize, sizeof(int) * thirdSize, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(B2_GPU, B_Host + thirdSize, sizeof(int) * thirdSize, cudaMemcpyHostToDevice, stream2);
    vector_addition<<<DimGrid, DimBlock, 0, stream2>>>(A2_GPU, B2_GPU, C2_GPU, thirdSize);
    cudaMemcpyAsync(C_Host + thirdSize, C2_GPU, sizeof(int) * thirdSize, cudaMemcpyDeviceToHost, stream2);

    cudaMemcpyAsync(A3_GPU, A_Host + 2 * thirdSize, sizeof(int) * thirdSize, cudaMemcpyHostToDevice, stream3);
    cudaMemcpyAsync(B3_GPU, B_Host + 2 * thirdSize, sizeof(int) * thirdSize, cudaMemcpyHostToDevice, stream3);
    vector_addition<<<DimGrid, DimBlock, 0, stream3>>>(A3_GPU, B3_GPU, C3_GPU, thirdSize);
    cudaMemcpyAsync(C_Host + 2 * thirdSize, C3_GPU, sizeof(int) * thirdSize, cudaMemcpyDeviceToHost, stream3);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaFree(A1_GPU);
    cudaFree(B1_GPU);
    cudaFree(C1_GPU);
    cudaFree(A2_GPU);
    cudaFree(B2_GPU);
    cudaFree(C2_GPU);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

int main() {
    int size = 1000002; // The size of the arrays
    int ThreadPerBlock = 256; // The size of the blocks
    int BlockPerGrid; 
    int *A_Host, *B_Host, *C_Host;


    cudaMallocHost((void**)&A_Host, sizeof(int) * size);
    cudaMallocHost((void**)&B_Host, sizeof(int) * size);
    cudaMallocHost((void**)&C_Host, sizeof(int) * size);
    
    for (int i = 0; i < size; i++) {
        A_Host[i] = i + 1;
        B_Host[i] = 0;
    }

    int iterations = 50;
    float totalSingleStreamTime = 0;
    float totalTwoStreamDataTransferTime = 0;
    float totalTwoStreamFullPipelineTime = 0;
    float totalThreeStreamFullPipelineTime = 0;

    for (int i = 0; i < iterations; i++) {
        BlockPerGrid = ((size / 1) + ThreadPerBlock - 1) / ThreadPerBlock;
        totalSingleStreamTime += single_stream(A_Host, B_Host, C_Host, size, ThreadPerBlock, BlockPerGrid);

        BlockPerGrid = ((size / 1) + ThreadPerBlock - 1) / ThreadPerBlock;
        totalTwoStreamDataTransferTime += two_streams_data_transfer(A_Host, B_Host, C_Host, size, ThreadPerBlock, BlockPerGrid);
        
        BlockPerGrid = ((size / 2) + ThreadPerBlock - 1) / ThreadPerBlock;
        totalTwoStreamFullPipelineTime += two_streams_full_pipeline(A_Host, B_Host, C_Host, size, ThreadPerBlock, BlockPerGrid);
        
        BlockPerGrid = ((size / 3) + ThreadPerBlock - 1) / ThreadPerBlock;
        totalThreeStreamFullPipelineTime += three_streams_full_pipeline(A_Host, B_Host, C_Host, size, ThreadPerBlock, BlockPerGrid);
    }

    float averageSingleStreamTime = totalSingleStreamTime / iterations;
    float averageTwoStreamDataTransferTime = totalTwoStreamDataTransferTime / iterations;
    float averageTwoStreamFullPipelineTime = totalTwoStreamFullPipelineTime / iterations;
    float averageThreeStreamFullPipelineTime = totalThreeStreamFullPipelineTime / iterations;

    printf("Average Time (Single Stream): %f ms\n", averageSingleStreamTime);
    printf("Average Time (Two Streams for Data Transfer): %f ms\n", averageTwoStreamDataTransferTime);
    printf("Average Time (Two Streams Full Pipeline): %f ms\n", averageTwoStreamFullPipelineTime);
    printf("Average Time (Three Streams Full Pipeline): %f ms\n", averageThreeStreamFullPipelineTime);

    cudaFreeHost(A_Host);
    cudaFreeHost(B_Host);
    cudaFreeHost(C_Host);

    return 0;
}
