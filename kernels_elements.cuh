#ifndef _H_KERNELS_ELEMENTS
#define _H_KERNELS_ELEMENTS

#include "utils.cuh"

#ifndef TPB_1D
#define TPB_1D 32
#endif

#ifndef TPB_2D
#define TPB_2D 32
#endif

__device__ Matrix getSubMatrix(Matrix A, int row, int col);

__global__ void getMinQReductionNeighbor2D(Matrix Q, QPos* result);
__global__ void getMinQReductionInterPair2D(Matrix Q, QPos* result);

__global__ void getMinQReductionInterPair2D_v2(Matrix D, Vector S, QPos* result);

__global__ void getQMatrix(Matrix D, Vector S, Matrix Q); // 2D
__global__ void updateMatrixD(Matrix D, Vector S, QPos qPosMin); // 1D
__global__ void resizeMatrixD(Matrix D, Vector S, QPos qPosMin); // 1D

__global__ void resizeMatrixD(Matrix D, Vector S, QPos qPosMin){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    
    if(idx >= D.N) return;
    if(qPosMin.j == (D.N-1)) return;
    
    if(idx == 0){
          S.elements[qPosMin.j] = S.elements[D.N-1];  
    }

    if(idx == qPosMin.j) return;
          
    float value = get_matrix_element(D, idx, D.N-1);
    set_matrix_element(D, idx, qPosMin.j, value);
    set_matrix_element(D, qPosMin.j, idx, value);       
}

__global__ void updateMatrixD(Matrix D, Vector S, QPos qPosMin){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx >= D.N) return;

    const float d_ij = get_matrix_element(D, qPosMin.i, qPosMin.j);

    if(idx == 0){
        S.elements[qPosMin.i] = (S.elements[qPosMin.i]+S.elements[qPosMin.j] - D.N*d_ij)/2;
    }

    if(idx == qPosMin.i || idx == qPosMin.j) return;

    float new_distance_uk = (
        get_matrix_element(D, qPosMin.i, idx) +
        get_matrix_element(D, qPosMin.j, idx) -
        d_ij )/2;

    S.elements[idx] = new_distance_uk - (new_distance_uk*2 + d_ij) + S.elements[idx];
    //S.elements[idx] = S.elements[idx] + new_distance_uk - get_matrix_element(D, qPosMin.i, idx) - get_matrix_element(D, qPosMin.j, idx) ;

    set_matrix_element(D, qPosMin.i, idx, new_distance_uk);
    set_matrix_element(D, idx, qPosMin.i, new_distance_uk);
}

__global__ void getQMatrix(Matrix D, Vector S, Matrix Q){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float value, d_rc;
    if(row >= D.N || col >= D.N) return;

    if(row == col){
        // SETA OS VALORES DA DIAGONAL PARA ABSURDOS PARA IMPEDIR LIXO E SELECAO INDEVIDA
        set_matrix_element(Q, row, col, 1e20);
        return;
    }

    d_rc = get_matrix_element(D, row, col);
    value = (D.N-2)*d_rc - S.elements[row] - S.elements[col];

    set_matrix_element(Q, row, col, value);
}

__device__ Matrix getSubMatrix(Matrix A, int row, int col){
    Matrix Asub;
    Asub.height = TPB_2D;
    Asub.width = TPB_2D;
    Asub.stride = A.stride;
    Asub.N = A.N;
    Asub.elements = &A.elements[TPB_2D*row*Asub.stride + TPB_2D*col];

    return Asub;
}

__global__ void getMinQReductionNeighbor2D(Matrix Q, QPos* result){
    int idxRow = blockIdx.y*blockDim.y + threadIdx.y;
    int idxCol = blockIdx.x*blockDim.x + threadIdx.x;

    int threadId = threadIdx.y*blockDim.y + threadIdx.x;

    int r_idx = gridDim.y*blockIdx.y + blockIdx.x;
    
    __shared__ float smin[TPB_2D*TPB_2D]; // Valores da Bloco para a reducao
    __shared__ int smin_index[TPB_2D*TPB_2D]; // Index do bloco para reducao
    
    smin[threadIdx.y*TPB_2D + threadIdx.x] = 1e20; // Inicializa para poder acessar posições invalidas na matrix
    __syncthreads();

    Matrix Qsub = getSubMatrix(Q, blockIdx.y, blockIdx.x);// Sub matrix do bloco

    if(idxRow < Q.N && idxCol < Q.N) {
        // Cada thread carrega para a shared memory um elemento do matrix-bloco
        smin[threadIdx.y*TPB_2D + threadIdx.x] = get_matrix_element(Qsub, threadIdx.y, threadIdx.x);
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

__global__ void getMinQReductionInterPair2D(Matrix Q, QPos* result){
    int idxRow = blockIdx.y*blockDim.y + threadIdx.y;
    int idxCol = blockIdx.x*blockDim.x + threadIdx.x;

    int threadId = threadIdx.y*blockDim.y + threadIdx.x;

    int r_idx = gridDim.y*blockIdx.y + blockIdx.x;
    
    __shared__ float smin[TPB_2D*TPB_2D]; // Valores da Bloco para a reducao
    __shared__ int smin_index[TPB_2D*TPB_2D]; // Index do bloco para reducao
    
    smin[threadIdx.y*TPB_2D + threadIdx.x] = 1e20; // Inicializa para poder acessar posições invalidas na matrix
    __syncthreads();

    Matrix Qsub = getSubMatrix(Q, blockIdx.y, blockIdx.x);// Sub matrix do bloco

    if(idxRow < Q.N && idxCol < Q.N) {
        // Cada thread carrega para a shared memory um elemento do matrix-bloco
        smin[threadIdx.y*TPB_2D + threadIdx.x] = get_matrix_element(Qsub, threadIdx.y, threadIdx.x);
        smin_index[threadIdx.y*TPB_2D + threadIdx.x] = threadIdx.y*TPB_2D + threadIdx.x;        
    }
    __syncthreads();

    for(int stride = (TPB_2D*TPB_2D)/2; stride > 0 ; stride >>= 1){
        
        if(threadId < stride){
            if(smin[threadId] > smin[threadId + stride]){
                smin[threadId] = smin[threadId + stride];
                smin_index[threadId] = smin_index[threadId+stride];
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

__global__ void getMinQReductionInterPair2D_v2(Matrix D, Vector S, QPos* result){
    int idxRow = blockIdx.y*blockDim.y + threadIdx.y;
    int idxCol = blockIdx.x*blockDim.x + threadIdx.x;

    int threadId = threadIdx.y*blockDim.y + threadIdx.x;

    int r_idx = gridDim.y*blockIdx.y + blockIdx.x;
    float value;
    __shared__ float smin[TPB_2D*TPB_2D]; // Valores da Bloco para a reducao
    __shared__ int smin_index[TPB_2D*TPB_2D]; // Index do bloco para reducao
    
    smin[threadIdx.y*TPB_2D + threadIdx.x] = 1e20; // Inicializa para poder acessar posições invalidas na matrix
    __syncthreads();

    Matrix Dsub = getSubMatrix(D, blockIdx.y, blockIdx.x);// Sub matrix do bloco

    if(idxRow < D.N && idxCol < D.N) {
        // Cada thread carrega para a shared memory um elemento do matrix-bloco
        value = (D.N-2)*get_matrix_element(Dsub, threadIdx.y, threadIdx.x) - S.elements[idxRow] - S.elements[idxCol];
        smin[threadIdx.y*TPB_2D + threadIdx.x] = value;
        smin_index[threadIdx.y*TPB_2D + threadIdx.x] = threadIdx.y*TPB_2D + threadIdx.x;        
    }
    __syncthreads();

    for(int stride = (TPB_2D*TPB_2D)/2; stride > 0 ; stride >>= 1){
        
        if(threadId < stride){
            if(smin[threadId] > smin[threadId + stride]){
                smin[threadId] = smin[threadId + stride];
                smin_index[threadId] = smin_index[threadId+stride];
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

#endif