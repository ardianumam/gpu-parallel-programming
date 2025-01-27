#include <stdio.h>

__global__ void vector_addition(int *A,int *B,int *C,int size)//CUDA kernel
{
	int tid = blockDim.x*blockIdx.x+threadIdx.x;//Global thread id
	if(tid<size)
		C[tid] = A[tid] + B[tid];//Vector Addition performs
}

int main()
{
	int size = 10000002;//The size of the arrays  (is the multiples of 3)
	int ThreadPerBlock = 1024;//The size of the blocks (The maximum value that we can set (hardware limitation) )
	int BlockPerGrid = ((size/3)-1)/ThreadPerBlock+1;//The number of blocks
	int *A_Host,*B_Host,*C_Host;

	cudaMallocHost((void**)&A_Host, sizeof(int)*size);//is allocated in the heap region of the memory on the CPU (Pinned)
	cudaMallocHost((void**)&B_Host, sizeof(int)*size);//is allocated in the heap region of the memory on the CPU (Pinned)
	cudaMallocHost((void**)&C_Host, sizeof(int)*size);//is allocated in the heap region of the memory on the CPU (Pinned)
	
	for(int i=1;i<=size;i++)//The values are assigned to the arrays
	{
		A_Host[i-1] = i;
		B_Host[i-1] = 0;
	}

	int *A1_GPU,*B1_GPU,*C1_GPU;
	int *A2_GPU,*B2_GPU,*C2_GPU;
	int *A3_GPU,*B3_GPU,*C3_GPU;
	cudaMalloc(&A1_GPU,sizeof(int)*size/3);//is allocated on the global memory of the GPU
	cudaMalloc(&B1_GPU,sizeof(int)*size/3);//is allocated on the global memory of the GPU
	cudaMalloc(&C1_GPU,sizeof(int)*size/3);//is allocated on the global memory of the GPU
	cudaMalloc(&A2_GPU,sizeof(int)*size/3);//is allocated on the global memory of the GPU
	cudaMalloc(&B2_GPU,sizeof(int)*size/3);//is allocated on the global memory of the GPU
	cudaMalloc(&C2_GPU,sizeof(int)*size/3);//is allocated on the global memory of the GPU
	cudaMalloc(&A3_GPU,sizeof(int)*size/3);//is allocated on the global memory of the GPU
	cudaMalloc(&B3_GPU,sizeof(int)*size/3);//is allocated on the global memory of the GPU
	cudaMalloc(&C3_GPU,sizeof(int)*size/3);//is allocated on the global memory of the GPU

	dim3 DimBlock(ThreadPerBlock);//The number of threads in a block
	dim3 DimGrid(BlockPerGrid);//The number of blocks in the grid

	cudaEvent_t start, stop;//Variables for the time
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float totaltime;

	cudaStream_t stream[3];
	cudaStreamCreate(&stream[0]);//First stream is created
	cudaStreamCreate(&stream[1]);//Second stream is created
	cudaStreamCreate(&stream[2]);//Third stream is created

	cudaEventRecord(start);//Time is started

  //The operations of each part are overlapped by using CUDA Streams
	cudaMemcpyAsync(A1_GPU,A_Host+0*(size/3),sizeof(int)*size/3,cudaMemcpyHostToDevice,stream[0]);//Copying data from CPU to GPU
	cudaMemcpyAsync(B1_GPU,B_Host+0*(size/3),sizeof(int)*size/3,cudaMemcpyHostToDevice,stream[0]);//Copying data from CPU to GPU
	vector_addition<<<DimGrid,DimBlock,0,stream[0]>>>(A1_GPU,B1_GPU,C1_GPU,size/3);//CUDA kernel is executed
	cudaMemcpyAsync(C_Host+0*(size/3),C1_GPU,sizeof(int)*size/3,cudaMemcpyDeviceToHost,stream[0]);//Copying data from GPU to CPU

	cudaMemcpyAsync(A2_GPU,A_Host+1*(size/3),sizeof(int)*size/3,cudaMemcpyHostToDevice,stream[1]);//Copying data from CPU to GPU
	cudaMemcpyAsync(B2_GPU,B_Host+1*(size/3),sizeof(int)*size/3,cudaMemcpyHostToDevice,stream[1]);//Copying data from CPU to GPU
	vector_addition<<<DimGrid,DimBlock,0,stream[1]>>>(A2_GPU,B2_GPU,C2_GPU,size/3);//CUDA kernel is executed
	cudaMemcpyAsync(C_Host+1*(size/3),C2_GPU,sizeof(int)*size/3,cudaMemcpyDeviceToHost,stream[1]);//Copying data from GPU to CPU

	cudaMemcpyAsync(A3_GPU,A_Host+2*(size/3),sizeof(int)*size/3,cudaMemcpyHostToDevice,stream[2]);//Copying data from CPU to GPU
	cudaMemcpyAsync(B3_GPU,B_Host+2*(size/3),sizeof(int)*size/3,cudaMemcpyHostToDevice,stream[2]);//Copying data from CPU to GPU
	vector_addition<<<DimGrid,DimBlock,0,stream[2]>>>(A3_GPU,B3_GPU,C3_GPU,size/3);//CUDA kernel is executed
	cudaMemcpyAsync(C_Host+2*(size/3),C3_GPU,sizeof(int)*size/3,cudaMemcpyDeviceToHost,stream[2]);//Copying data from GPU to CPU

	cudaEventRecord(stop);//Time is stopped
  cudaEventSynchronize(stop);//The program waits here until all the operations of the events completed
	cudaEventElapsedTime(&totaltime, start, stop);//The execution time is calculated
	printf("Total Execution Time = %f ms\n",totaltime);
	printf("C[size-1] = %d\n",C_Host[size-1]);

	cudaFreeHost(A_Host);//Array on the memory of the CPU is freed
	cudaFreeHost(B_Host);//Array on the memory of the CPU is freed
	cudaFreeHost(C_Host);//Array on the memory of the CPU is freed

	cudaFree(A1_GPU);//Array on the memory of the GPU is freed
	cudaFree(B1_GPU);//Array on the memory of the GPU is freed
	cudaFree(C1_GPU);//Array on the memory of the GPU is freed

	cudaFree(A2_GPU);//Array on the memory of the GPU is freed
	cudaFree(B2_GPU);//Array on the memory of the GPU is freed
	cudaFree(C2_GPU);//Array on the memory of the GPU is freed

	cudaFree(A3_GPU);//Array on the memory of the GPU is freed
	cudaFree(B3_GPU);//Array on the memory of the GPU is freed
	cudaFree(C3_GPU);//Array on the memory of the GPU is freed

	cudaError_t err = cudaGetLastError();//Catchs the latest error occured on the GPU
	if ( err != cudaSuccess )
		printf("CUDA Error: %s\n",cudaGetErrorString(err));
}
