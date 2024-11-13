#ifndef _H_NJ_NORMAL
#define _H_NJ_NORMAL

#include "nj_cuda.cuh"

void nj_normal(nj_data_t d, int threads_per_block);

void nj_normal(nj_data_t d, int threads_per_block){
    int size_array, run;
    int gridMatrix, gridArray;

    size_t sMemSize;

    float* d_values_min;
    int* d_positions_min;

    float* values_min;
    int* position_min;
    
    float min_value;
    int min_position;

    size_array = d.N*(d.N)/2;

    gridMatrix = (size_array+threads_per_block-1)/threads_per_block;
    gridArray = (d.N+threads_per_block-1)/threads_per_block;

    values_min = (float*) malloc(gridMatrix * sizeof(float));
    position_min = (int*) malloc(gridMatrix * sizeof(int));

    cudaMalloc(&d_values_min, sizeof(float)*gridMatrix);
    cudaMalloc(&d_positions_min, sizeof(int)*gridMatrix);

    sMemSize = threads_per_block*sizeof(float) + threads_per_block*sizeof(int);

    run = d.N >= 3;

    while(run){

        buildQ<<<gridMatrix, threads_per_block>>>(d);
        reduceQ<<<gridMatrix, threads_per_block, sMemSize>>>(d, d_values_min, d_positions_min);

        cudaMemcpy(values_min, d_values_min, gridMatrix*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(position_min, d_positions_min, gridMatrix*sizeof(int), cudaMemcpyDeviceToHost);

        min_value = FLT_MAX;

        for (int i = 0; i < gridMatrix; i++)
        {
            if(values_min[i] < min_value)
            {
                min_value = values_min[i];
                min_position = position_min[i];
            }
        }
        
        updateD<<< gridArray, threads_per_block >>>(d, min_position);
        resizeD<<< gridArray, threads_per_block >>>(d, min_position);

        cudaDeviceSynchronize();

        d.N -= 1;
        size_array = d.N*(d.N)/2;

        gridMatrix = (size_array+threads_per_block-1)/threads_per_block;
        gridArray = (d.N+threads_per_block-1)/threads_per_block;

        run = d.N >= 3;
    }

    free(values_min);
    free(position_min);
}

#endif