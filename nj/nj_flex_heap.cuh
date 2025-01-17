#ifndef _H_NJ_FLEX_HEAP
#define _H_NJ_FLEX_HEAP

#include "../heap/uheap.cuh"

#include "nj_cuda.cuh"
#include "nj_flex_cuda.cuh"

void nj_flex_heap(nj_data_t d, int threads_per_block);

void nj_flex_heap(nj_data_t d, int threads_per_block)
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

    int pair_number = d.N * d.p;

    UHeap<float, int> h_heap(batchNum, batchSize, FLT_MAX, -1);
    UHeap<float, int> *d_heap;

    cudaMalloc((void **)&d_heap, sizeof(UHeap<float, int>));
    cudaMemcpy(d_heap, &h_heap, sizeof(UHeap<float, int>), cudaMemcpyHostToDevice);

    cudaMalloc(&d_result, sizeof(int) * batchSize);

    cudaMalloc(&d_positions, sizeof(int) * pair_number);
    cudaMalloc(&d_collected_number, sizeof(int));

#ifdef DEGUB
    int *h_result = (int *)malloc(sizeof(int) * batchSize);
#endif

    size_array = d.N * (d.N) / 2;

    gridMatrix = (size_array + threads_per_block - 1) / threads_per_block;
    gridArray = (d.N + threads_per_block - 1) / threads_per_block;

    sMemSize = batchSize * 4 * sizeof(float) + batchSize * 4 * sizeof(int);
    sMemSize += /* (threads_per_block + 1) * sizeof(int) + */ 2 * batchSize * sizeof(float) + 2 * batchSize * sizeof(int);

    run = d.N >= 3;

    while (run)
    {
        pair_number = d.N * d.p;
        if (pair_number == 0)
            pair_number = 1;

        h_heap.reset();
        d_ResetHeap<<<32, threads_per_block>>>(d_heap);
        gpuErrchk(cudaPeekAtLastError());

        initPositionsData<<<1, threads_per_block>>>(d_positions, d_collected_number, pair_number);
        gpuErrchk(cudaPeekAtLastError());

        cudaMemcpy(&h_collect_number, d_collected_number, sizeof(int), cudaMemcpyDeviceToHost);

        buildQUHeap<<<32, threads_per_block, sMemSize>>>(d, d_heap, batchSize);
        gpuErrchk(cudaPeekAtLastError());

        while (h_collect_number < pair_number)
        {
            getPositionsBatch<<<1, threads_per_block, sMemSize>>>(d_heap, d_result, d.N, batchSize);
            gpuErrchk(cudaPeekAtLastError());

#ifdef DEBUG

            cudaMemcpy(h_result, d_result, sizeof(int) * batchSize, cudaMemcpyDeviceToHost);

            printf("[");
            for (int index_result_loop = 0; index_result_loop < batchSize; index_result_loop++)
            {
                printf("\t %d, (%d, %d),  ", h_result[index_result_loop], h_result[index_result_loop] / d.N, h_result[index_result_loop] % d.N);
            }
            printf("]\n");

#endif

            consolidationOfPositions<<<1, threads_per_block>>>(d_positions, d_result, d_collected_number, pair_number, batchSize, d.N);
            gpuErrchk(cudaPeekAtLastError());

            cudaMemcpy(&h_collect_number, d_collected_number, sizeof(int), cudaMemcpyDeviceToHost);
        }

        updateDK<<<1, threads_per_block>>>(d, d_positions, pair_number);
        gpuErrchk(cudaPeekAtLastError());

        for (int i = 0; i < pair_number; i++)
        {
            resizeDFlex2<<<gridArray, threads_per_block>>>(d, d_positions, d.N - i - 1, i);
            gpuErrchk(cudaPeekAtLastError());
        }

        gpuErrchk(cudaDeviceSynchronize());

        d.N -= (int)(pair_number);
        size_array = d.N * (d.N) / 2;

        gridMatrix = (size_array + threads_per_block - 1) / threads_per_block;
        gridArray = (d.N + threads_per_block - 1) / threads_per_block;

        run = d.N >= 3;
    }

#ifdef DEBUG
    free(h_result);
#endif

    cudaFree(d_heap);
    cudaFree(d_collected_number);
    cudaFree(d_positions);
    cudaFree(d_result);
}

#endif