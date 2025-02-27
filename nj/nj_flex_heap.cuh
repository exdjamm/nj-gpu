#ifndef _H_NJ_FLEX_HEAP
#define _H_NJ_FLEX_HEAP

#include "../heap/uheap.cuh"

#include "nj_cuda.cuh"
#include "nj_flex_cuda.cuh"

#include <time_debug.cuh>

void nj_flex_heap(nj_data_t *d, int threads_per_block, int N_STOP);

void nj_flex_heap(nj_data_t *d, int threads_per_block, int N_STOP)
{
    int size_array, run;
    int pair_number = d->N * d->p;
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

    size_array = d->N * (d->N) / 2;
    batchSize = 1024;
    batchNum = 1024 * 512;

    sMemSize = batchSize * 4 * sizeof(float) + batchSize * 4 * sizeof(int);
    sMemSize += /* (threads_per_block + 1) * sizeof(int) + */ 2 * batchSize * sizeof(float) + 2 * batchSize * sizeof(int);

    gridMatrix = (size_array + threads_per_block - 1) / threads_per_block;
    gridArray = (d->N + threads_per_block - 1) / threads_per_block;
    gridPairNumber = (pair_number + threads_per_block - 1) / threads_per_block;

    run = d->N >= N_STOP;
    i_time("HEAP ALLOC", 2, 3);
    UHeap<float, int> h_heap(batchNum, batchSize, FLT_MAX, -1);
    UHeap<float, int> *d_heap;

    cudaMalloc((void **)&d_heap, sizeof(UHeap<float, int>));
    cudaMemcpy(d_heap, &h_heap, sizeof(UHeap<float, int>), cudaMemcpyHostToDevice);

    // Does it need at least once?
    d_ResetHeap<<<32, threads_per_block>>>(d_heap);
    gpuErrchk(cudaPeekAtLastError());

    f_time(3);

    h_result = (int *)calloc(sizeof(int), batchSize);
    h_positions = (int *)calloc(sizeof(int), pair_number);

    cudaMalloc(&d_batchQ, sizeof(float) * batchSize);
    cudaMalloc(&d_batchPositions, sizeof(int) * batchSize);

    cudaMalloc(&d_positions, sizeof(int) * pair_number);
    cudaMalloc(&d_collected_number, sizeof(int));

    i_time("LOOP", 2, 4);
    while (run)
    {
        pair_number = d->N * d->p;
        if (pair_number == 0)
            pair_number = 1;

        h_collect_number = 0;

        i_time("HEAP RESET", 4, 5);
        h_heap.reset();

#ifdef RESET_HEAP
        d_ResetHeap<<<32, threads_per_block>>>(d_heap);
        gpuErrchk(cudaPeekAtLastError());
#endif
        f_time(5);

        i_time("CLEAR POSITIONS", 4, 6);
        initPositionsData<<<gridPairNumber, threads_per_block>>>(d_positions, d_collected_number, pair_number);
        gpuErrchk(cudaPeekAtLastError());
        f_time(6);

        i_time("BUILD Q HEAP", 4, 7);
        buildQUHeap<<<32, threads_per_block, sMemSize>>>(*d, d_heap, batchSize);
        gpuErrchk(cudaPeekAtLastError());
        f_time(7);

        i_time("FILTER&COLLECT", 4, 8);
        while (h_collect_number < pair_number)
        {
            i_time("GET POSITON HOST", 8, 10);
            getPositionsBatch<<<1, threads_per_block, sMemSize>>>(d_heap,
                                                                  d_batchQ, d_batchPositions,
                                                                  d->N, batchSize);
            gpuErrchk(cudaPeekAtLastError());

            f_time(10);
            cudaMemcpy(h_result, d_batchPositions, sizeof(int) * batchSize, cudaMemcpyDeviceToHost);

            i_time("CLEAN POS DEV", 8, 11);
            // clearBatchPositions<<<1, threads_per_block>>>(d_batchPositions, batchSize, d.N);
            for (int i = 0; i < batchSize; i++)
                for (int j = i + 1; j < batchSize; j++)
                    if (h_result[i] != -1 && h_result[j] != -1)
                        if (hasIntersection(h_result[i], h_result[j], d->N))
                            h_result[j] = -1;

            cudaMemcpy(d_batchPositions, h_result, sizeof(int) * batchSize, cudaMemcpyHostToDevice);
            f_time(11);

            i_time("CONSOLIDATION", 8, 12);
            consolidationOfPositions<<<1, threads_per_block>>>(d_positions, d_batchPositions, d_collected_number, pair_number, batchSize, d->N);
            gpuErrchk(cudaPeekAtLastError());

            cudaMemcpy(&h_collect_number, d_collected_number, sizeof(int), cudaMemcpyDeviceToHost);
            f_time(12);
        }
        f_time(8);

        // updateDK<<<1, threads_per_block>>>(d, d_positions, pair_number);

        cudaMemcpy(h_positions, d_positions, sizeof(int) * pair_number, cudaMemcpyDeviceToHost);

        i_time("UPDATE&RESIZE", 4, 9);
        for (int i = 0; i < pair_number; i++)
        {
            printf("%d, %d\n", h_positions[i] / d->N, h_positions[i] % d->N);
            updateD<<<gridArray, threads_per_block>>>(*d, h_positions[i]);
            gpuErrchk(cudaPeekAtLastError());
        }

        for (int i = 0; i < pair_number; i++)
        {
            resizeDFlex2<<<gridArray, threads_per_block>>>(*d, d_positions, d->N - i - 1, i);
            gpuErrchk(cudaPeekAtLastError());
        }

        gpuErrchk(cudaDeviceSynchronize());
        f_time(9);

        d->N -= (int)(pair_number);
        size_array = d->N * (d->N) / 2;

        gridMatrix = (size_array + threads_per_block - 1) / threads_per_block;
        gridArray = (d->N + threads_per_block - 1) / threads_per_block;

        run = d->N >= N_STOP;
    }
    f_time(4);

    free(h_result);
    free(h_positions);

    cudaFree(d_heap);
    cudaFree(d_collected_number);
    cudaFree(d_positions);
    cudaFree(d_batchQ);
    cudaFree(d_batchPositions);
}

#endif