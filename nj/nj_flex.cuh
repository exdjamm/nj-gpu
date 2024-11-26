#ifndef _H_NJ_FLEX
#define _H_NJ_FLEX

#include "nj_cuda.cuh"

void nj_flex(nj_data_t d, int threads_per_block);

__global__ void initIgnorePositions(int* ignore_positions);

void nj_flex(nj_data_t d, int threads_per_block){
    int size_array, run;
    int gridMatrix, gridArray;

    size_t sMemSize;

    float* d_values_min;
    int* d_positions_min;
    // int* d_ignore_positions;

    float* values_min;
    int* position_min;
    int* ignore_positions;
    
    float min_value;
    int min_position;
    int gather;

    size_array = d.N*(d.N-1)/2;

    gridMatrix = (size_array+threads_per_block-1)/threads_per_block;
    gridArray = (d.N+threads_per_block-1)/threads_per_block;

    values_min          = ( float * )    malloc(gridMatrix * sizeof(float));
    position_min        = ( int   * )    malloc(gridMatrix * sizeof(int));
    ignore_positions    = ( int   * )    malloc(d.N*d.p    * sizeof(int));

    cudaMalloc(&d_values_min, sizeof(float)*gridMatrix);
    cudaMalloc(&d_positions_min, sizeof(int)*gridMatrix);
    // cudaMalloc(&d_ignore_positions, sizeof(int)*d.N*d.p);

    sMemSize = threads_per_block*sizeof(float) + threads_per_block*sizeof(int);

    run = d.N > 3;

    while(run){
        // initIgnorePositions<<<gridArray, threads_per_block>>>(d_ignore_positions);

        for (int i = 0; i < d.N*d.p; i++)
            ignore_positions[i] = -1;
        
        for ( gather = 0; gather < d.N*d.p; gather++) 
        // VERIFICAR SE IGNORE POSITIONS Q GARANTE UNICIDADE ENTRE ITERACOES
        {
            buildQ<<<gridMatrix, threads_per_block>>>(d);
            for (int i = 0; i < d.N*d.p; i++)
                ignorePositionsQ<<<gridArray, threads_per_block>>>(d, ignore_positions[i]);
            
            reduceQ<<<gridMatrix, threads_per_block, sMemSize>>>(d, d_values_min, d_positions_min);

            cudaMemcpy(values_min, d_values_min, gridMatrix*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(position_min, d_positions_min, gridMatrix*sizeof(int), cudaMemcpyDeviceToHost);    

            min_value = FLT_MAX;
            for (int i = 0; i < gridMatrix; i++)
            {
                if(values_min[i] < min_value){
                    min_value = values_min[i];
                    min_position = position_min[i];
                }
            }

            if(min_value == FLT_MAX){
                exit(1);
            }

            ignore_positions[gather] = min_position;
        }
        
        // TODO: Loop for each n*p pair
        updateD<<< gridArray, threads_per_block >>>(d, min_position);
        resizeD<<< gridArray, threads_per_block >>>(d, min_position);

        cudaDeviceSynchronize();

        d.N -= 1; //
        size_array = d.N*(d.N - 1)/2;

        gridMatrix = (size_array+threads_per_block-1)/threads_per_block;
        gridArray = (d.N+threads_per_block-1)/threads_per_block;

        run = d.N > 3;
    }
}

#endif