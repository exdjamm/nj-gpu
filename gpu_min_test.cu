#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "utils.cuh"
#include "nj_read.cuh"

#include "time_analisys.cuh"

#define TPB 32
#define MAX_F 1000

const int N = 8*1024; // 1M de elementos ou 1<<20

__global__ void getMinQReductionInterPair2D(Matrix Q, float* result);
__global__ void minQReductionNeighbor2D(Matrix Q, float* result);
__device__ Matrix getSubMatrix(Matrix A, int row, int col);

__host__ float gpu_min_neighbor(Matrix Q);
__host__ float gpu_min_interpair(Matrix Q);

float gpu_min_neighbor(Matrix h_Q){
    Matrix d_Q;
    d_Q.stride = d_Q.width = d_Q.height = d_Q.N = h_Q.N;

    cudaMalloc(&d_Q.elements, N*N*sizeof(float));
    cudaMemcpy(d_Q.elements, h_Q.elements, N*N*sizeof(float), cudaMemcpyHostToDevice);

    int gridX, gridY, iter;
    iter = d_Q.N;
    gridX = (iter+TPB -1)/(TPB);
    gridY = (iter+TPB -1)/(TPB);

    int minq_size_array;
    minq_size_array = sizeof(float)*gridX*gridY;
    
    float* h_minq_array;
    float* d_minq_array;
    float min_q = 1e20;

    dim3 dimGrid(gridX, gridY);
    dim3 dimThread(TPB, TPB); // 2D

    h_minq_array = (float*) malloc(minq_size_array);
    cudaMalloc(&d_minq_array, minq_size_array);

    time_start();
    minQReductionNeighbor2D<<<dimGrid, dimThread>>>(d_Q, d_minq_array);
    cudaDeviceSynchronize();
    time_end();
    printf("Time GPU QMinReduceNeighbor:\t%6.3f ms\n", elapsed_time);

    cudaMemcpy(h_minq_array, d_minq_array, minq_size_array, cudaMemcpyDeviceToHost);

    for(int i = 0; i < gridX*gridY; i++){
        float element = h_minq_array[i];
        if(element < min_q)
            min_q = element;
    }
    
    free(h_minq_array);
    cudaFree(d_minq_array);
    cudaFree(d_Q.elements);

    return min_q;
}

float gpu_min_interpair(Matrix h_Q){
    Matrix d_Q;
    d_Q.stride = d_Q.width = d_Q.height = d_Q.N = h_Q.N;

    cudaMalloc(&d_Q.elements, N*N*sizeof(float));
    cudaMemcpy(d_Q.elements, h_Q.elements, N*N*sizeof(float), cudaMemcpyHostToDevice);

    int gridX, gridY, iter;
    iter = d_Q.N;
    gridX = (iter+TPB -1)/(TPB);
    gridY = (iter+TPB -1)/(TPB);

    int minq_size_array;
    minq_size_array = sizeof(float)*gridX*gridY;
    
    float* h_minq_array;
    float* d_minq_array;
    float min_q = 1e20;

    dim3 dimGrid(gridX, gridY);
    dim3 dimThread(TPB, TPB); // 2D

    h_minq_array = (float*) malloc(minq_size_array);
    cudaMalloc(&d_minq_array, minq_size_array);

    time_start();
    getMinQReductionInterPair2D<<<dimGrid, dimThread>>>(d_Q, d_minq_array);
    cudaDeviceSynchronize();
    time_end();
    printf("Time GPU QMinReduceNeighbor:\t%6.3f ms\n", elapsed_time);

    cudaMemcpy(h_minq_array, d_minq_array, minq_size_array, cudaMemcpyDeviceToHost);

    for(int i = 0; i < gridX*gridY; i++){
        float element = h_minq_array[i];
        if(element < min_q)
            min_q = element;
    }
    
    free(h_minq_array);
    cudaFree(d_minq_array);
    cudaFree(d_Q.elements);

    return min_q;
}

int main(){
    Matrix A;
    A.stride = A.width = A.height = A.N = N;
    A.elements = (float*) calloc(N*N, sizeof(float));

    float minQNeighbor, minQInterPair;
    float min_q = 1e20;

    for(int i = 0; i < N*N; i++){
        float max_float = (float) RAND_MAX;
        A.elements[i] = ((float) (rand()/max_float))*MAX_F;

        if(A.elements[i] < min_q)
            min_q = A.elements[i];
    }

    minQNeighbor = gpu_min_neighbor(A);
    minQInterPair = gpu_min_interpair(A);

    if(minQNeighbor - min_q > 0.0001)
        printf("Neighbor Method FAILED\n");
    else
        printf("Neighbor Method OKAY %.2f\n", minQNeighbor);

    if(minQInterPair - min_q > 0.0001)
        printf("InterPair Method FAILED\n");
    else
        printf("InterPair Method OKAY %.2f\n", minQInterPair);

    free(A.elements);
    return 0;
}

__device__ Matrix getSubMatrix(Matrix A, int row, int col){
    Matrix Asub;
    Asub.height = TPB;
    Asub.width = TPB;
    Asub.stride = A.stride;
    Asub.N = A.N;
    Asub.elements = &A.elements[TPB*row*Asub.stride + TPB*col];

    return Asub;
}

__global__ void minQReductionNeighbor2D(Matrix Q, float* result){
    const int tidRow = threadIdx.y;
    const int tidCol = threadIdx.x;

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int idxRow = blockRow*blockDim.y + tidRow;
    int idxCol = blockCol*blockDim.x + tidCol;

    int r_idx = gridDim.y*blockRow + blockCol;
    
    __shared__ float smin[TPB*TPB];
    
    smin[tidRow*TPB + tidCol] = 1e20; // Inicializa para poder acessar posições invalidas na matrix
    __syncthreads();

    Matrix Qsub = getSubMatrix(Q, blockRow, blockCol); // Sub matrix do bloco

    // Talvez o teste de barreira nao faça sentido aqui
    if(idxRow < Q.N && idxCol < Q.N) {
        // Cada thread carrega para a shared memory um elemento do matrix-bloco
        smin[tidRow*TPB + tidCol] = get_matrix_element(Qsub, tidRow, tidCol);    
    }
    
    __syncthreads();
    
    for(int strideCol = 1; strideCol < TPB*TPB; strideCol *= 2){
        int threadId = tidRow*blockDim.y + tidCol;
        int index = 2*strideCol*threadId;

        if((index < TPB*TPB)){
            if(smin[index] > smin[index + strideCol])
                smin[index] = smin[index + strideCol];
        }
        __syncthreads();
    }
    
    if(tidRow == 0 && tidCol == 0){
        result[r_idx] = smin[0];
    }

}

__global__ void getMinQReductionInterPair2D(Matrix Q, float* result){

    int idxRow = blockIdx.y*blockDim.y + threadIdx.y;
    int idxCol = blockIdx.x*blockDim.x + threadIdx.x;

    int threadId = threadIdx.y*blockDim.y + threadIdx.x;

    int r_idx = gridDim.y*blockIdx.y + blockIdx.x;
    
    __shared__ float smin[TPB*TPB]; // Valores da Bloco para a reducao
    
    
    smin[threadIdx.y*TPB + threadIdx.x] = 1e20; // Inicializa para poder acessar posições invalidas na matrix
    __syncthreads();

    Matrix Qsub = getSubMatrix(Q, blockIdx.y, blockIdx.x);// Sub matrix do bloco

    if(idxRow < Q.N && idxCol < Q.N) {
        // Cada thread carrega para a shared memory um elemento do matrix-bloco
        smin[threadIdx.y*TPB + threadIdx.x] = get_matrix_element(Qsub, threadIdx.y, threadIdx.x);
    }
    __syncthreads();

    for(int stride = (TPB*TPB)/2; stride > 0 ; stride >>= 1){
        if(threadId < stride){
            if(smin[threadId] > smin[threadId + stride]){
                smin[threadId] = smin[threadId + stride];
            }
        }
        __syncthreads();
    }
    
    if(threadIdx.y == 0 && threadIdx.x == 0){
        result[r_idx] = smin[0];
    }
}
