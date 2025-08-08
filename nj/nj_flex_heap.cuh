#ifndef _H_NJ_FLEX_HEAP
#define _H_NJ_FLEX_HEAP

#include "../heap/heap.cuh"

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
    float *qValues;
    int *h_positions, *d_positions;
    int h_collect_number, *d_collected_number;
    float *d_batchQ;
    int *d_batchPositions;

    int batchNum, batchSize;

    size_t sMemSize;

    size_array = d->N * (d->N) / 2;
    batchSize = 1024;
    batchNum = (size_array + batchSize - 1) / batchSize;

    sMemSize = batchSize * 4 * sizeof(float) + batchSize * 4 * sizeof(int);
    sMemSize += /* (threads_per_block + 1) * sizeof(int) + */ 2 * batchSize * sizeof(float) + 2 * batchSize * sizeof(int);

    gridMatrix = (size_array + threads_per_block - 1) / threads_per_block;
    gridArray = (d->N + threads_per_block - 1) / threads_per_block;
    gridPairNumber = (pair_number + threads_per_block - 1) / threads_per_block;

    run = d->N >= N_STOP;
    TIME_POINT("HEAP ALLOC", 2, 3);
    KAuxHeap<float, int> h_heap(batchNum, batchSize, FLT_MAX, -1);
    KAuxHeap<float, int> *d_heap;

    cudaMalloc((void **)&d_heap, sizeof(KAuxHeap<float, int>));
    cudaMemcpy(d_heap, &h_heap, sizeof(KAuxHeap<float, int>), cudaMemcpyHostToDevice);

    // Does it need at least once?
    int blockSizeHeapExec = ((batchNum + 1) * batchSize + threads_per_block - 1) / threads_per_block;
    int upB, downB;
    upB = 0;
    downB = (batchNum + 1) * batchSize;
    // d_ResetHeap<<<blockSizeHeapExec, threads_per_block>>>(d_heap, upB, downB);
    gpuErrchk(cudaPeekAtLastError());

    TIME_POINT_END(3);

    h_result = (int *)calloc(sizeof(int), batchSize);
    qValues = (float *)calloc(sizeof(float), batchSize);
    h_positions = (int *)calloc(sizeof(int), pair_number);

    cudaMalloc(&d_batchQ, sizeof(float) * batchSize);
    cudaMalloc(&d_batchPositions, sizeof(int) * batchSize);

    cudaMalloc(&d_positions, sizeof(int) * batchSize);
    cudaMalloc(&d_collected_number, sizeof(int));

    TIME_POINT("LOOP", 2, 4);
    while (run)
    {
        pair_number = d->N * d->p;
        if (pair_number == 0)
            pair_number = 1;

        h_collect_number = 0;

        TIME_POINT("HEAP RESET", 4, 5);
        // h_heap.reset();

#ifdef RESET_HEAP
        // d_ResetHeap<<<blockSizeHeapExec, threads_per_block>>>(d_heap, upB, downB);
        gpuErrchk(cudaPeekAtLastError());
#endif
        TIME_POINT_END(5);

        TIME_POINT("CLEAR POSITIONS", 4, 6);
        initPositionsData<<<gridPairNumber, threads_per_block>>>(d_positions, d_collected_number, pair_number);
        gpuErrchk(cudaPeekAtLastError());
        TIME_POINT_END(6);

        TIME_POINT("BUILD Q HEAP", 4, 7);
        buildQUHeap<<<32, threads_per_block, sMemSize>>>(*d, d_heap, batchSize);
        gpuErrchk(cudaPeekAtLastError());
        TIME_POINT_END(7);

        TIME_POINT("FILTER&COLLECT", 4, 8);
        while (h_collect_number < pair_number)
        {
            TIME_POINT("GET POSITON HOST", 8, 10);
            getPositionsBatch<<<1, threads_per_block, sMemSize>>>(d_heap,
                                                                  d_batchQ, d_batchPositions,
                                                                  d->N, batchSize);
            gpuErrchk(cudaPeekAtLastError());

            TIME_POINT_END(10);

            // cudaMemcpy(h_result, d_batchPositions, sizeof(int) * batchSize, cudaMemcpyDeviceToHost);

            TIME_POINT("CLEAN POS DEV", 8, 11);
            // // clearBatchPositions<<<1, threads_per_block>>>(d_batchPositions, batchSize, d.N);
            // for (int i = 0; i < batchSize; i++)
            //     for (int j = i + 1; j < batchSize; j++)
            //         if (h_result[i] != -1 && h_result[j] != -1)
            //             if (hasIntersection(h_result[i], h_result[j], d->N))
            //                 h_result[j] = -1;

            // cudaMemcpy(qValues, d_batchQ, batchSize * sizeof(float), cudaMemcpyDeviceToHost);

            // for (int i = 0; i < batchSize; i++)
            // {
            //     if (h_result[i] == -1)
            //         continue;
            //     printf("%d: (%d, %d) -> %.2f\n", d->N, h_result[i] / d->N, h_result[i] % d->N, qValues[i]);
            // }

            // cudaMemcpy(d_batchPositions, h_result, sizeof(int) * batchSize, cudaMemcpyHostToDevice);
            int blocks = (batchSize + threads_per_block - 1) / threads_per_block;
            eliminateInjuctions<<<blocks, threads_per_block>>>(d_batchPositions, batchSize, d->N, d_positions);
            gpuErrchk(cudaPeekAtLastError());

            cleanPositions<<<blocks, threads_per_block>>>(d_batchPositions, d_positions, batchSize);
            TIME_POINT_END(11);
            gpuErrchk(cudaPeekAtLastError());
            TIME_POINT("CONSOLIDATION", 8, 12);
            consolidationOfPositions<<<1, threads_per_block>>>(d_positions, d_batchPositions, d_collected_number, pair_number, batchSize, d->N);
            gpuErrchk(cudaPeekAtLastError());

            cudaMemcpy(&h_collect_number, d_collected_number, sizeof(int), cudaMemcpyDeviceToHost);
            TIME_POINT_END(12);
            gpuErrchk(cudaPeekAtLastError());
        }
        TIME_POINT_END(8);

        // updateDK<<<1, threads_per_block>>>(d, d_positions, pair_number);

        cudaMemcpy(h_positions, d_positions, sizeof(int) * pair_number, cudaMemcpyDeviceToHost);
        gpuErrchk(cudaPeekAtLastError());
        TIME_POINT("UPDATE&RESIZE", 4, 9);
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
        TIME_POINT_END(9);

        downB = batchSize * (((size_array + batchSize - 1) / batchSize) + 1);

        d->N -= (int)(pair_number);
        size_array = d->N * (d->N) / 2;

        upB = batchSize * (((size_array + batchSize - 1) / batchSize) + 1);

        gridMatrix = (size_array + threads_per_block - 1) / threads_per_block;
        gridArray = (d->N + threads_per_block - 1) / threads_per_block;

        run = d->N >= N_STOP;
    }
    TIME_POINT_END(4);

    free(h_result);
    free(h_positions);

    cudaFree(d_heap);
    cudaFree(d_collected_number);
    cudaFree(d_positions);
    cudaFree(d_batchQ);
    cudaFree(d_batchPositions);
}

#endif