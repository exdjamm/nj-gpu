#ifndef _H_NJ_FLEX_HEAP
#define _H_NJ_FLEX_HEAP

#include "../heap/uheap.cuh"

#include "nj_cuda.cuh"
#include "nj_flex_cuda.cuh"

void nj_flex_heap(nj_data_t d, int threads_per_block);

void nj_flex_heap(nj_data_t d, int threads_per_block)
{
    int size_array, run;
    int pair_number = d.N * d.p;
    if (pair_number == 0)
        pair_number = 1;
    int gridMatrix, gridArray, gridPairNumber;

    int *h_result;
    int *h_positions, *d_positions;
    int h_collect_number, *d_collected_number;
    float *d_batchQ;
    int *d_batchPositions;

    int batchNum, batchSize;

    size_t sMemSize;

    size_array = d.N * (d.N) / 2;
    batchSize = 1024;
    batchNum = 1024 * 512;

    sMemSize = batchSize * 4 * sizeof(float) + batchSize * 4 * sizeof(int);
    sMemSize += /* (threads_per_block + 1) * sizeof(int) + */ 2 * batchSize * sizeof(float) + 2 * batchSize * sizeof(int);

    gridMatrix = (size_array + threads_per_block - 1) / threads_per_block;
    gridArray = (d.N + threads_per_block - 1) / threads_per_block;
    gridPairNumber = (pair_number + threads_per_block - 1) / threads_per_block;

    run = d.N >= 3;

    UHeap<float, int> h_heap(batchNum, batchSize, FLT_MAX, -1);
    UHeap<float, int> *d_heap;

    h_result = (int *)calloc(sizeof(int), batchSize);
    h_positions = (int *)calloc(sizeof(int), pair_number);

    cudaMalloc((void **)&d_heap, sizeof(UHeap<float, int>));
    cudaMemcpy(d_heap, &h_heap, sizeof(UHeap<float, int>), cudaMemcpyHostToDevice);

    cudaMalloc(&d_batchQ, sizeof(float) * batchSize);
    cudaMalloc(&d_batchPositions, sizeof(int) * batchSize);

    cudaMalloc(&d_positions, sizeof(int) * pair_number);
    cudaMalloc(&d_collected_number, sizeof(int));

    while (run)
    {
        pair_number = d.N * d.p;
        if (pair_number == 0)
            pair_number = 1;

        h_collect_number = 0;

        h_heap.reset();
        d_ResetHeap<<<32, threads_per_block>>>(d_heap);
        gpuErrchk(cudaPeekAtLastError());

        initPositionsData<<<gridPairNumber, threads_per_block>>>(d_positions, d_collected_number, pair_number);
        gpuErrchk(cudaPeekAtLastError());

        buildQUHeap<<<32, threads_per_block, sMemSize>>>(d, d_heap, batchSize);
        gpuErrchk(cudaPeekAtLastError());

        while (h_collect_number < pair_number)
        {

            getPositionsBatch<<<1, threads_per_block, sMemSize>>>(d_heap,
                                                                  d_batchQ, d_batchPositions,
                                                                  d.N, batchSize);
            gpuErrchk(cudaPeekAtLastError());

            cudaMemcpy(h_result, d_batchPositions, sizeof(int) * batchSize, cudaMemcpyDeviceToHost);

            for (int i = 0; i < batchSize; i++)
                for (int j = i + 1; j < batchSize; j++)
                    if (hasIntersection(h_result[i], h_result[j], d.N))
                        if (h_positions[i] != -1 && h_positions[j] != -1)
                            h_result[j] = -1;

            cudaMemcpy(d_batchPositions, h_result, sizeof(int) * batchSize, cudaMemcpyHostToDevice);

            consolidationOfPositions<<<1, threads_per_block>>>(d_positions, d_batchPositions, d_collected_number, pair_number, batchSize, d.N);
            gpuErrchk(cudaPeekAtLastError());

            cudaMemcpy(&h_collect_number, d_collected_number, sizeof(int), cudaMemcpyDeviceToHost);
        }

        // updateDK<<<1, threads_per_block>>>(d, d_positions, pair_number);

        cudaMemcpy(h_positions, d_positions, sizeof(int) * pair_number, cudaMemcpyDeviceToHost);

        for (int i = 0; i < pair_number; i++)
        {
            updateD<<<gridArray, threads_per_block>>>(d, h_positions[i]);
            gpuErrchk(cudaPeekAtLastError());
        }

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

    free(h_result);
    free(h_positions);

    cudaFree(d_heap);
    cudaFree(d_collected_number);
    cudaFree(d_positions);
    cudaFree(d_batchQ);
    cudaFree(d_batchPositions);
}

#endif