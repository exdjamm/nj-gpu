#ifndef _H_NJ_ELEMENTS
#define _H_NJ_ELEMENTS

#ifndef TPB_1D
#define TPB_1D 32
#endif

#ifndef TPB_2D
#define TPB_2D 32
#endif

#include "kernels_elements.cuh"

__host__ void nj_loop_elements(Matrix d_D, Vector d_S, Matrix d_Q);

__host__ void nj_loop_elements(Matrix d_D, Vector d_S, Matrix d_Q){
    int iteration, gridX2D, gridY2D, grid1D;
    iteration = d_D.N;

    dim3 dimThread2D(TPB_2D, TPB_2D);
    dim3 dimThread1D(TPB_1D);
    // tem a posibilidade de deixa o i e j contralado pelo kernel e deixar a execucao 
    // N*(n+1)/2
    gridX2D = (iteration+dimThread2D.x-1)/(dimThread2D.x);
    gridY2D = (iteration+dimThread2D.y-1)/(dimThread2D.y);
    grid1D  = (iteration+dimThread1D.x-1)/(dimThread1D.x);

    QPos* h_mins_Q;
    QPos* d_mins_Q;

    size_t minsq_size = sizeof(QPos)*gridX2D*gridY2D;
    h_mins_Q = (QPos*) malloc(minsq_size);
    if(h_mins_Q == NULL){
        return;
    }

    cudaMalloc(&d_mins_Q, minsq_size);

    QPos h_qPosMin;
    new_distances new_d;
    float product, d_ij, difSums;

    while(iteration > 2){
        product = (2*(iteration-2));
    
        dim3 dimGrid2D(gridX2D, gridY2D);
        dim3 dimGrid1D(grid1D);
        
        if(1){
            getQMatrix<<<dimGrid2D, dimThread2D>>>(d_D, d_S, d_Q);
            getMinQReductionNeighbor2D<<<dimGrid2D, dimThread2D>>>(d_Q, d_mins_Q);
            //getMinQReductionInterPair2D<<<dimGrid2D, dimThread2D>>>(d_Q, d_mins_Q);
        }else{
            getMinQReductionInterPair2D_v2<<<dimGrid2D, dimThread2D>>>(d_D, d_S, d_mins_Q);
        }

        cudaMemcpy(h_mins_Q, d_mins_Q, minsq_size, cudaMemcpyDeviceToHost);
        h_qPosMin = h_mins_Q[0];

        for(int i = 0; i <  gridX2D*gridY2D; i++){
            QPos qPosI = h_mins_Q[i];
            if(qPosI.value < h_qPosMin.value){
                h_qPosMin = qPosI;
            }
        }

        #ifdef Q_VALUE
            printf("%.2f\n", h_qPosMin.value);
        #endif
        /* new_d.i = h_qPosMin.i; new_d.j = h_qPosMin.j;
        difSums = S.elements[new_d.i] - S.elements[new_d.j];

        d_ij = get_matrix_element(D, new_d.i, new_d.j);
        new_d.new_di = d_ij/2  + difSums/product;
        new_d.new_dj = d_ij - new_d.new_di; */

        // FUNCAO QUE ADICIONA O NO

        updateMatrixD<<<dimGrid1D, dimThread1D>>>(d_D, d_S, h_qPosMin);
        resizeMatrixD<<<dimGrid1D, dimThread1D>>>(d_D, d_S, h_qPosMin);
        cudaDeviceSynchronize();
        
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
    }

    cudaFree(d_mins_Q);
    free(h_mins_Q);
}

#endif