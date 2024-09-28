#ifndef _H_KERNELS_OTU_ARRAY
#define _H_KERNELS_OTU_ARRAY

#include "utils.cuh"
#include "kernels_elements.cuh"
#include "kernels_otu.cuh"

#ifndef HEAP_BLOCK
#define HEAP_BLOCK 16
#endif

#ifndef TPB_1D
#define TPB_1D 32
#endif

#ifndef TPB_2D
#define TPB_2D 32
#endif

struct heap_block_array{
    int N;
    int heap_size;
    float* elements;
    int* position;
};
typedef struct heap_block_array heap_block_array_t;

__device__ MatrixOtuArray getSubMatrixOtuArray(MatrixOtuArray A, int row, int col);
__global__ void getQMatrixOtuArray(Matrix D, Vector S, MatrixOtuArray Q); // 2D

__global__ void getMinQReductionNeighbor2DOtuArray(MatrixOtuArray Q, QPos* result);

__device__ void buildMinHeapOtusArray(heap_block_array_t h, int start_end);
__device__ void heapsortHeapOtusArray(heap_block_array_t h, int i, int start_end);

__device__ void buildMinHeapBlockArray(heap_block_array_t* h, int size);
__device__ void heapsortHeapBlockArray(heap_block_array_t* h, int i, int size);

__device__ MatrixOtuArray getSubMatrixOtuLinearArray(MatrixOtuArray A, int idx);

__global__ void buildHeapArray(heap_block_array_t* result, int gridDimMax);
__global__ void getMinQOtuHeapArray(MatrixOtuArray Q, heap_block_array_t* result, int gridDimMax);// PODE SER INTERESSANTE CRIAR OUTRA DIMENSAO PARA ELE

__device__ MatrixOtuArray getSubMatrixOtuArray(MatrixOtuArray A, int row, int col){
    MatrixOtuArray Asub;
    Asub.height = TPB_2D;
    Asub.width = TPB_2D;
    Asub.stride = A.stride;
    Asub.N = A.N;
    Asub.elements = &A.elements[TPB_2D*row*Asub.stride + TPB_2D*col];
    Asub.position = &A.position[TPB_2D*row*Asub.stride + TPB_2D*col];

    return Asub;
}

__global__ void getQMatrixOtuArray(Matrix D, Vector S, MatrixOtuArray Q){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float value, d_rc;
    if(row >= D.N || col >= D.N) return;

    if(row == col){
        // SETA OS VALORES DA DIAGONAL PARA ABSURDOS PARA IMPEDIR LIXO E SELECAO INDEVIDA
        set_matrix_otu_array(Q, row, col, 1e20);
        return;
    }

    d_rc = get_matrix_element(D, row, col);
    value = (D.N-2)*d_rc - S.elements[row] - S.elements[col];

    set_matrix_otu_array(Q, row, col, value);
}

__global__ void getMinQReductionNeighbor2DOtuArray(MatrixOtuArray Q, QPos* result){
    int idxRow = blockIdx.y*blockDim.y + threadIdx.y;
    int idxCol = blockIdx.x*blockDim.x + threadIdx.x;

    int threadId = threadIdx.y*blockDim.y + threadIdx.x;

    int r_idx = gridDim.y*blockIdx.y + blockIdx.x;
    
    __shared__ float smin[TPB_2D*TPB_2D]; // Valores da Bloco para a reducao
    __shared__ int smin_index[TPB_2D*TPB_2D]; // Index do bloco para reducao
    
    smin[threadIdx.y*TPB_2D + threadIdx.x] = 1e20; // Inicializa para poder acessar posições invalidas na matrix
    __syncthreads();

    MatrixOtuArray Qsub = getSubMatrixOtuArray(Q, blockIdx.y, blockIdx.x);// Sub matrix do bloco

    if(idxRow < Q.N && idxCol < Q.N) {
        // Cada thread carrega para a shared memory um elemento do matrix-bloco
        smin[threadIdx.y*TPB_2D + threadIdx.x] = get_matrix_otu_array(Qsub, threadIdx.y, threadIdx.x);
        smin_index[threadIdx.y*TPB_2D + threadIdx.x] = threadIdx.y*TPB_2D + threadIdx.x;        
    }
    __syncthreads();

    for(int stride = 1; stride < TPB_2D*TPB_2D; stride *= 2){
        int index = 2*stride*threadId;

        if((index < TPB_2D*TPB_2D)){
            if(smin[index] > smin[index + stride]){
                smin[index] = smin[index + stride];
                smin_index[index] = smin_index[index+stride];
            }   
        }
        __syncthreads();
    }
    
    if(threadIdx.y == 0 && threadIdx.x == 0){
        int i = blockIdx.y*blockDim.y;
        int j = blockIdx.x*blockDim.x;
        int temp;
        
        QPos qPos;
        qPos.i = i + (smin_index[0]/TPB_2D); 
        qPos.j = j + (smin_index[0]%TPB_2D); 
        qPos.value = smin[0]; 
        
        if(qPos.i > qPos.j){
            temp = qPos.j;
            qPos.j = qPos.i;
            qPos.i = temp;
        }

        result[r_idx] = qPos;
    }

}

__device__ MatrixOtuArray getSubMatrixOtuLinearArray(MatrixOtuArray A, int idx){
    MatrixOtuArray Asub;
    Asub.height = HEAP_BLOCK;
    Asub.width = HEAP_BLOCK;
    Asub.stride = A.stride;
    Asub.N = A.N;
    Asub.elements = &A.elements[idx*HEAP_BLOCK*HEAP_BLOCK];
    Asub.position = &A.position[idx*HEAP_BLOCK*HEAP_BLOCK];

    return Asub;
}

__global__ void getMinQOtuHeapArray(MatrixOtuArray Q, heap_block_array_t* result, int gridDimMax){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    int start_end = idx*HEAP_BLOCK*HEAP_BLOCK;

    if(start_end >= Q.N*Q.N) return;

    result[idx].N = Q.N;
    result[idx].heap_size = 0;
    result[idx].elements = NULL;

    if(idx >= gridDimMax) return;

    MatrixOtuArray Qsub = getSubMatrixOtuLinearArray(Q, idx);
    result[idx].N = Qsub.N;
    result[idx].heap_size = Qsub.height*Qsub.width;
    result[idx].elements = Qsub.elements;

    buildMinHeapOtusArray(result[idx], start_end);
}

__global__ void buildHeapArray(heap_block_array_t* result, int gridDimMax){
    buildMinHeapBlockArray(result, gridDimMax);
}

__device__ void buildMinHeapOtusArray(heap_block_array_t h, int start_end){
    for (int i = h.heap_size/2; i >= 0; i--){
        heapsortHeapOtusArray(h, i, start_end);
    }
}

__device__ void heapsortHeapOtusArray(heap_block_array_t h, int i, int start_end){
    int min_pos = i;
    
    int childL = getChildL(i);
    int childR = getChildR(i);

    if(i + start_end >= h.N*h.N || childL + start_end >= h.N*h.N) return;

    if(childL < h.heap_size && h.elements[childL] <= h.elements[i]){
        min_pos = childL;
    }
    
    if(childR + start_end < h.N*h.N){    
        if( childR < h.heap_size && h.elements[childR] <= h.elements[min_pos]){
            min_pos = childR;
        }
    }

    if(min_pos != i){
        float temp = h.elements[i];
        int temp_positon = h.position[i];

        h.elements[i] = h.elements[min_pos];
        h.elements[min_pos] =  temp;

        h.position[i] = h.position[min_pos];
        h.position[min_pos] = temp_positon;

        heapsortHeapOtusArray(h, min_pos, start_end);
    }

}

__device__ void buildMinHeapBlockArray(heap_block_array_t* h, int size){
    for (int i = size/2; i >= 0; i--){
        heapsortHeapBlockArray(h, i, size);
    }
}

__device__ void heapsortHeapBlockArray(heap_block_array_t* h, int i, int size){
    int min_pos = i;
    
    int childL = getChildL(i);
    int childR = getChildR(i);
    
    if(childL < size && h[childL].elements != NULL){
        if(h[childL].elements[0] <= h[i].elements[0])
            min_pos = childL;
    }

    if( childR < size && h[childR].elements != NULL){
        if(h[childR].elements[0] <= h[min_pos].elements[0])
            min_pos = childR;
    }

    if(min_pos != i){
        heap_block_array_t temp = h[i];
        h[i] = h[min_pos];
        h[min_pos] =  temp;

        heapsortHeapBlockArray(h, min_pos, size);
    }
}


#endif