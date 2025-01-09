#ifndef _H_NJ_FLEX_HEAP
#define _H_NJ_FLEX_HEAP

#include "../heap/uheap.cuh"

#include "nj_cuda.cuh"
#include "nj_flex_cuda.cuh"

void nj_normal(nj_data_t d, int threads_per_block);

void nj_normal(nj_data_t d, int threads_per_block)
{
    int size_array, run;
    int gridMatrix, gridArray;

    size_t sMemSize;

    int *d_result;

    int *d_positions;
    int h_collect_number;
    int *d_collected_number;

    int batchNum = 512 * 1024;
    int batchSize = 1024;
    int sMemSize;

    int pair_number = d.N * d.p;

    UHeap<float, int> h_heap(batchNum, batchSize, FLT_MAX, -1);
    UHeap<float, int> *d_heap;

    cudaMalloc((void **)&d_heap, sizeof(UHeap<float, int>));
    cudaMemcpy(d_heap, &h_heap, sizeof(UHeap<float, int>), cudaMemcpyHostToDevice);

    cudaMalloc(&d_result, sizeof(int) * batchSize);

    cudaMalloc(&d_positions, sizeof(int) * pair_number);
    cudaMalloc(&d_collected_number, sizeof(int));

    size_array = d.N * (d.N) / 2;

    gridMatrix = (size_array + threads_per_block - 1) / threads_per_block;
    gridArray = (d.N + threads_per_block - 1) / threads_per_block;

    sMemSize = batchSize * 4 * sizeof(float) + batchSize * 4 * sizeof(int);
    sMemSize += (threads_per_block + 1) * sizeof(int) + 2 * batchSize * sizeof(float) + 2 * batchSize * sizeof(int);

    run = d.N >= 3;

    while (run)
    {
        pair_number = d.N * d.p;
        if (pair_number == 0)
            pair_number = 1;

        initPositionsData<<<1, threads_per_block>>>(d_positions, d_collected_number, pair_number);
        cudaMemcpy(&h_collect_number, d_collected_number, sizeof(int), cudaMemcpyDeviceToHost);

        buildQUHeap<<<32, threads_per_block, sMemSize>>>(d, d_heap, batchSize);

        while (h_collect_number < pair_number)
        {
            getPositionsBatch<<<1, threads_per_block, sMemSize>>>(d_heap, d_result, d.N, batchSize);
            consolidationOfPositions<<<1, threads_per_block>>>(d_positions, d_result, d_collected_number, pair_number, batchSize, d.N);
            cudaMemcpy(&h_collect_number, d_collected_number, sizeof(int), cudaMemcpyDeviceToHost);
        }

        updateDK<<<1, 1>>>(d, d_positions, pair_number);

        for (int i = 0; i < pair_number; i++)
        {
            resizeDFlex2<<<gridArray, threads_per_block>>>(d, d_positions, d.N - i - 1, i);
        }

        cudaDeviceSynchronize();

        d.N -= (int)(pair_number);
        size_array = d.N * (d.N) / 2;

        gridMatrix = (size_array + threads_per_block - 1) / threads_per_block;
        gridArray = (d.N + threads_per_block - 1) / threads_per_block;

        run = d.N >= 3;
    }

    cudaFree(d_heap);
    cudaFree(d_collected_number);
    cudaFree(d_positions);
    cudaFree(d_result);
}

#endif