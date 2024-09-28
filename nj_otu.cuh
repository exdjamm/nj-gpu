#ifndef _H_NJ_OTU
#define _H_NJ_OTU

#ifndef HEAP_BLOCK
#define HEAP_BLOCK 16
#endif

#ifndef TPB_1D
#define TPB_1D 32
#endif

#ifndef TPB_2D
#define TPB_2D 32
#endif

#include "kernels_otu.cuh"

__host__ void nj_loop_otu(MatrixOtu d_D, Vector d_S, MatrixOtu d_Q);

__host__ void nj_loop_otu(MatrixOtu d_D, Vector d_S, MatrixOtu d_Q){
    int iteration, gridX2D, gridY2D, grid1D;
    iteration = d_D.N;

    dim3 dimThread2D(TPB_2D, TPB_2D);
    dim3 dimThread1D(TPB_1D);
    // tem a posibilidade de deixa o i e j contralado pelo kernel e deixar a execucao 
    // N*(n+1)/2
    gridX2D = (iteration+dimThread2D.x-1)/(dimThread2D.x);
    gridY2D = (iteration+dimThread2D.y-1)/(dimThread2D.y);
    grid1D  = (iteration+dimThread1D.x-1)/(dimThread1D.x);

    int gridYheap, gridXheap;
    gridYheap = (iteration+HEAP_BLOCK-1)/(HEAP_BLOCK);    
    gridXheap = (iteration+HEAP_BLOCK-1)/(HEAP_BLOCK);

    QPos* h_mins_Q;
    QPos* d_mins_Q;

    size_t minsq_size = sizeof(QPos)*gridX2D*gridY2D;
    h_mins_Q = (QPos*) malloc(minsq_size);
    if(h_mins_Q == NULL){
        return;
    }

    cudaMalloc(&d_mins_Q, minsq_size);

    heap_block_t * d_heap_blocks;
    heap_block_t h_heap_blocks;
    
    size_t minsq_size_heap = sizeof(heap_block_t)*gridX2D*gridY2D;
    // h_heap_blocks = (heap_block_t*) malloc(minsq_size_heap);
    // if(h_heap_blocks == NULL){
    //     return;
    // }
    
    cudaMalloc(&d_heap_blocks, minsq_size_heap);

    QPos h_qPosMin;
    new_distances new_d;
    float product, d_ij, difSums;

    while(iteration > 2){
        product = (2*(iteration-2));
    
        dim3 dimGrid2D(gridX2D, gridY2D);
        dim3 dimGrid1D(grid1D);
        dim3 dimGridHeap( (gridXheap*gridYheap + dimThread1D.x - 1)/dimThread1D.x );
        
        getQMatrixOtu<<<dimGrid2D, dimThread2D>>>(d_D, d_S, d_Q);

        #ifndef HEAP
            {
                getMinQOtuReductionNeighbor2D<<<dimGrid2D, dimThread2D>>>(d_Q, d_mins_Q);
                cudaMemcpy(h_mins_Q, d_mins_Q, minsq_size, cudaMemcpyDeviceToHost);
                h_qPosMin = h_mins_Q[0];

                for(int i = 0; i <  gridX2D*gridY2D; i++){
                    QPos qPosI = h_mins_Q[i];
                    if(qPosI.value < h_qPosMin.value){
                        h_qPosMin = qPosI;
                    }
                }
            }
        #else
            {
                getMinQOtuHeap<<<dimGridHeap, dimThread1D>>>(d_Q, d_heap_blocks, gridYheap*gridXheap);
                buildHeap<<<1, 1>>>(d_heap_blocks, gridYheap*gridXheap);
                cudaMemcpy(&h_heap_blocks, d_heap_blocks, sizeof(heap_block_t), cudaMemcpyDeviceToHost);
                cudaMemcpy(&h_qPosMin, h_heap_blocks.elements, sizeof(otu_t), cudaMemcpyDeviceToHost);
            }
        #endif    

        #ifdef Q_VALUE
            printf("%.2f\n", h_qPosMin.value);
        #endif
        /* new_d.i = h_qPosMin.i; new_d.j = h_qPosMin.j;
        difSums = S.elements[new_d.i] - S.elements[new_d.j];

        d_ij = get_matrix_element(D, new_d.i, new_d.j);
        new_d.new_di = d_ij/2  + difSums/product;
        new_d.new_dj = d_ij - new_d.new_di; */

        // FUNCAO QUE ADICIONA O NO

        updateMatrixOtuD<<<dimGrid1D, dimThread1D>>>(d_D, d_S, h_qPosMin);
        resizeMatrixOtuD<<<dimGrid1D, dimThread1D>>>(d_D, d_S, h_qPosMin);
        cudaDeviceSynchronize();

        //resetQMatrixOtu<<<dimGrid2D, dimThread2D>>>(d_Q);

        d_D.N       = d_D.N         - 1;
        d_D.height  = d_D.height    - 1;
        d_D.width   = d_D.width     - 1;
        //d_D.stride  = d_D.stride    - 1;

        d_Q.N       = d_Q.N         - 1;
        d_Q.height  = d_Q.height    - 1;
        d_Q.width   = d_Q.width     - 1;
        d_Q.stride  = d_Q.stride    - 1;

        d_S.size    = d_S.size      - 1;

        iteration--;

        // VARIAVEL E MAIS EFICIENTE POIS PERMITE A DIMINUICAO DA EXECUCAO AO LONGO DOS PASSOS
        gridX2D = (iteration+dimThread2D.x-1)/(dimThread2D.x);
        gridY2D = (iteration+dimThread2D.y-1)/(dimThread2D.y);
        grid1D  = (iteration+dimThread1D.x-1)/(dimThread1D.x);

        gridYheap = (iteration+HEAP_BLOCK-1)/(HEAP_BLOCK);        
        gridXheap = (iteration+HEAP_BLOCK-1)/(HEAP_BLOCK);
    }

    cudaFree(d_mins_Q);
    cudaFree(d_heap_blocks);
    free(h_mins_Q);
}

#endif