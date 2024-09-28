#ifndef _H_KERNELS_OTU
#define _H_KERNELS_OTU

#include "utils.cuh"

#ifndef HEAP_BLOCK
#define HEAP_BLOCK 16
#endif

#ifndef TPB_1D
#define TPB_1D 32
#endif

#ifndef TPB_2D
#define TPB_2D 32
#endif

struct heap_block{
    int N;
    int heap_size;
    otu_t* elements;
};
typedef struct heap_block heap_block_t;

__device__ int getChildL(int index);
__device__ int getChildR(int index);

__device__ void buildMinHeapOtus(heap_block_t h, int start_end);
__device__ void heapsortHeapOtus(heap_block_t h, int i, int start_end);

__device__ void buildMinHeapBlock(heap_block_t* h, int size);
__device__ void heapsortHeapBlock(heap_block_t* h, int i, int size);

__device__ MatrixOtu getSubMatrixOtuLinear(MatrixOtu A, int idx);

__global__ void buildHeap(heap_block_t* result, int gridDimMax);
__global__ void getMinQOtuHeap(MatrixOtu Q, heap_block_t* result, int gridDimMax);// PODE SER INTERESSANTE CRIAR OUTRA DIMENSAO PARA ELE

__device__ MatrixOtu getSubMatrixOtu(MatrixOtu A, int row, int col);

__global__ void getMinQOtuReductionNeighbor2D(MatrixOtu Q, QPos* result);
__global__ void getMinQOtuReductionInterPair2D(MatrixOtu Q, QPos* result);

__global__ void getMinQOtuReductionInterPair2D_v2(MatrixOtu D, Vector S, QPos* result);

__global__ void getQMatrixOtu(MatrixOtu D, Vector S, MatrixOtu Q); // 2D
__global__ void resetQMatrixOtu(MatrixOtu Q); // 2D
__global__ void updateMatrixOtuD(MatrixOtu D, Vector S, QPos qPosMin); // 1D
__global__ void resizeMatrixOtuD(MatrixOtu D, Vector S, QPos qPosMin); // 1D

__global__ void resizeMatrixOtuD(MatrixOtu D, Vector S, QPos qPosMin){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    
    if(idx >= D.N) return;
    if(qPosMin.j == (D.N-1)) return;
    
    if(idx == 0){
          S.elements[qPosMin.j] = S.elements[D.N-1];  
    }

    if(idx == qPosMin.j) return;
          
    float value = get_matrix_otu(D, idx, D.N-1);
    set_matrix_otu(D, idx, qPosMin.j, value);
    set_matrix_otu(D, qPosMin.j, idx, value);       
}

__global__ void updateMatrixOtuD(MatrixOtu D, Vector S, QPos qPosMin){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx >= D.N) return;

    const float d_ij = get_matrix_otu(D, qPosMin.i, qPosMin.j);

    if(idx == 0){
        S.elements[qPosMin.i] = (S.elements[qPosMin.i]+S.elements[qPosMin.j] - D.N*d_ij)/2;
    }

    if(idx == qPosMin.i || idx == qPosMin.j) return;

    float new_distance_uk = (
        get_matrix_otu(D, qPosMin.i, idx) +
        get_matrix_otu(D, qPosMin.j, idx) -
        d_ij )/2;

    S.elements[idx] = new_distance_uk - (new_distance_uk*2 + d_ij) + S.elements[idx];
    //S.elements[idx] = S.elements[idx] + new_distance_uk - get_matrix_element(D, qPosMin.i, idx) - get_matrix_element(D, qPosMin.j, idx) ;

    set_matrix_otu(D, qPosMin.i, idx, new_distance_uk);
    set_matrix_otu(D, idx, qPosMin.i, new_distance_uk);
}

__global__ void getQMatrixOtu(MatrixOtu D, Vector S, MatrixOtu Q){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float value, d_rc;
    if(row >= D.N || col >= D.N) return;

    if(row == col){
        // SETA OS VALORES DA DIAGONAL PARA ABSURDOS PARA IMPEDIR LIXO E SELECAO INDEVIDA
        set_matrix_otu(Q, row, col, 1e20);
        return;
    }

    d_rc = get_matrix_otu(D, row, col);
    value = (D.N-2)*d_rc - S.elements[row] - S.elements[col];

    set_matrix_otu(Q, row, col, value);
}

__device__ MatrixOtu getSubMatrixOtu(MatrixOtu A, int row, int col){
    MatrixOtu Asub;
    Asub.height = TPB_2D;
    Asub.width = TPB_2D;
    Asub.stride = A.stride;
    Asub.N = A.N;
    Asub.elements = &A.elements[TPB_2D*row*Asub.stride + TPB_2D*col];

    return Asub;
}

__global__ void getMinQOtuReductionNeighbor2D(MatrixOtu Q, QPos* result){
    int idxRow = blockIdx.y*blockDim.y + threadIdx.y;
    int idxCol = blockIdx.x*blockDim.x + threadIdx.x;

    int threadId = threadIdx.y*blockDim.y + threadIdx.x;

    int r_idx = gridDim.y*blockIdx.y + blockIdx.x;
    
    __shared__ otu_t smin[TPB_2D*TPB_2D]; // Valores da Bloco para a reducao
    
    smin[threadIdx.y*TPB_2D + threadIdx.x].value = 1e20; // Inicializa para poder acessar posições invalidas na matrix
    __syncthreads();

    MatrixOtu Qsub = getSubMatrixOtu(Q, blockIdx.y, blockIdx.x);// Sub matrix do bloco

    if(idxRow < Q.N && idxCol < Q.N) {
        // Cada thread carrega para a shared memory um elemento do matrix-bloco
        smin[threadIdx.y*TPB_2D + threadIdx.x] = get_matrix_otu_all(Qsub, threadIdx.y, threadIdx.x);
    }
    __syncthreads();

    for(int stride = 1; stride < TPB_2D*TPB_2D; stride *= 2){
        int index = 2*stride*threadId;

        if((index < TPB_2D*TPB_2D)){
            if(smin[index].value > smin[index + stride].value){
                smin[index] = smin[index + stride];
            }   
        }
        __syncthreads();
    }
    
    if(threadIdx.y == 0 && threadIdx.x == 0){
        int temp;
        
        QPos qPos = smin[0];
        
        if(qPos.i > qPos.j){
            temp = qPos.j;
            qPos.j = qPos.i;
            qPos.i = temp;
        }

        result[r_idx] = qPos;
    }

}

__global__ void resetQMatrixOtu(MatrixOtu Q){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float value, d_rc;
    if(row >= Q.N || col >= Q.N) return;
    set_matrix_otu(Q, row, col, 1e20);
}

__device__ int getChildL(int index){
    return index*2 + 1;
}
__device__ int getChildR(int index){
    return index*2 + 2;
}

__device__ MatrixOtu getSubMatrixOtuLinear(MatrixOtu A, int idx){
    MatrixOtu Asub;
    Asub.height = HEAP_BLOCK;
    Asub.width = HEAP_BLOCK;
    Asub.stride = A.stride;
    Asub.N = A.N;
    Asub.elements = &A.elements[idx*HEAP_BLOCK*HEAP_BLOCK];

    return Asub;
}

__global__ void getMinQOtuHeap(MatrixOtu Q, heap_block_t* result, int gridDimMax){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    int start_end = idx*HEAP_BLOCK*HEAP_BLOCK;

    result[idx].N = Q.N;
    result[idx].heap_size = 0;
    result[idx].elements = NULL;

    if(start_end >= Q.N*Q.N) return;

    if(idx >= gridDimMax) return;

    MatrixOtu Qsub = getSubMatrixOtuLinear(Q, idx);
    result[idx].N = Qsub.N;
    result[idx].heap_size = Qsub.height*Qsub.width;
    result[idx].elements = Qsub.elements;

    buildMinHeapOtus(result[idx], start_end);
}

__global__ void buildHeap(heap_block_t* result, int gridDimMax){
    buildMinHeapBlock(result, gridDimMax);
}

__device__ void buildMinHeapOtus(heap_block_t h, int start_end){
    for (int i = h.heap_size/2; i >= 0; i--){
        heapsortHeapOtus(h, i, start_end);
    }
}

__device__ void heapsortHeapOtus(heap_block_t h, int i, int start_end){
    int min_pos = i;
    
    int childL = getChildL(i);
    int childR = getChildR(i);

    if(i + start_end >= h.N*h.N || childL + start_end >= h.N*h.N) return;

    if(childL < h.heap_size && h.elements[childL].value <= h.elements[i].value){
        min_pos = childL;
    }
    
    if(childR + start_end < h.N*h.N){    
        if( childR < h.heap_size && h.elements[childR].value <= h.elements[min_pos].value){
            min_pos = childR;
        }
    }

    if(min_pos != i){
        otu_t temp = h.elements[i];
        h.elements[i] = h.elements[min_pos];
        h.elements[min_pos] =  temp;

        heapsortHeapOtus(h, min_pos, start_end);
    }

}

__device__ void buildMinHeapBlock(heap_block_t* h, int size){
    for (int i = size/2; i >= 0; i--){
        heapsortHeapBlock(h, i, size);
    }
}

__device__ void heapsortHeapBlock(heap_block_t* h, int i, int size){
    int min_pos = i;
    
    int childL = getChildL(i);
    int childR = getChildR(i);
    
    if(childL < size && h[childL].elements != NULL){
        if(h[childL].elements[0].value <= h[i].elements[0].value)
            min_pos = childL;
    }

    if( childR < size && h[childR].elements != NULL){
        if(h[childR].elements[0].value <= h[min_pos].elements[0].value)
            min_pos = childR;
    }

    if(min_pos != i){
        heap_block_t temp = h[i];
        h[i] = h[min_pos];
        h[min_pos] =  temp;

        heapsortHeapBlock(h, min_pos, size);
    }
}

#endif