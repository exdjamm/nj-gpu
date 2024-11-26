#ifndef _H_NJ_FLEX
#define _H_NJ_FLEX

#include "nj_cuda.cuh"

void nj_flex(nj_data_t d, int threads_per_block);

__global__ void initIgnorePositions(int *ignore_positions);

void nj_flex(nj_data_t d, int threads_per_block)
{
    int size_array, run, k_pairs, pairs_numbers;
    int gridMatrix, gridArray;

    size_t sMemSize;

    float *d_values_min;
    int *d_positions_min;
    // int* d_ignore_positions;

    float *values_min;
    int *position_min;
    int *ignore_positions;

    float min_value;
    int min_position;
    int gather;

    size_array = d.N * (d.N) / 2;

    gridMatrix = (size_array + threads_per_block - 1) / threads_per_block;
    gridArray = (d.N + threads_per_block - 1) / threads_per_block;

    values_min = (float *)malloc(gridMatrix * sizeof(float));
    position_min = (int *)malloc(gridMatrix * sizeof(int));
    ignore_positions = (int *)malloc(d.N * d.p * sizeof(int));

    cudaMalloc(&d_values_min, sizeof(float) * gridMatrix);
    cudaMalloc(&d_positions_min, sizeof(int) * gridMatrix);
    // cudaMalloc(&d_ignore_positions, sizeof(int)*d.N*d.p);

    sMemSize = threads_per_block * sizeof(float) + threads_per_block * sizeof(int);

    run = d.N >= 3;

    while (run)
    {
        k_pairs = d.N * d.p;
        if (k_pairs <= 0)
            k_pairs = 1;

        pairs_numbers = 0;

        for (int i = 0; i < k_pairs; i++)
        {
            ignore_positions[i] = -1;
        }

        buildQ<<<gridMatrix, threads_per_block>>>(d);

        for (pairs_numbers = 0; pairs_numbers < k_pairs; pairs_numbers++)
        {

            reduceQ<<<gridMatrix, threads_per_block, sMemSize>>>(d, d_values_min, d_positions_min);

            cudaMemcpy(values_min, d_values_min, gridMatrix * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(position_min, d_positions_min, gridMatrix * sizeof(int), cudaMemcpyDeviceToHost);

            min_value = FLT_MAX;

            for (int i = 0; i < gridMatrix; i++)
            {
                if (values_min[i] < min_value)
                {
                    min_value = values_min[i];
                    min_position = position_min[i];
                }
            }

            ignore_positions[pairs_numbers] = min_position;
            ignorePositionsQ<<<gridArray, threads_per_block>>>(d, min_position);
        }

        for (int i = 0; i < k_pairs; i++)
        {
            if (ignore_positions[i] == -1)
                continue;

            updateD<<<gridArray, threads_per_block>>>(d, ignore_positions[i]);
            resizeD<<<gridArray, threads_per_block>>>(d, ignore_positions[i]);
        }

        cudaDeviceSynchronize();

        d.N -= k_pairs; //
        size_array = d.N * (d.N) / 2;

        gridMatrix = (size_array + threads_per_block - 1) / threads_per_block;
        gridArray = (d.N + threads_per_block - 1) / threads_per_block;

        run = d.N >= 3;
    }

    free(values_min);
    free(position_min);
    free(ignore_positions);
}

#endif