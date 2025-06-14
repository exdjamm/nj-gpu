#ifndef _H_NJ_FLEX_CUDA
#define _H_NJ_FLEX_CUDA

#include <float.h>

#include "../nj_read/nj_utils.cuh"
#include "../heap/uheap.cuh"

__device__ int atomicIncFixed(int *address, int limit)
{
    int old = *address;
    int assumed, val;

    do
    {
        assumed = old;
        val = assumed + 1;

        val = val < limit ? val : limit;

        old = atomicCAS(address, assumed, val);
    } while (old != assumed);

    return old;
}

__device__ float calc_q(nj_data_t d, int position);
__device__ void buildQPositions(nj_data_t d, float *Q, int *positions, int start, int batchSize, int d_size);

__device__ __host__ int hasIntersection(int position_1, int position_2, int N);
__device__ void nonUniqueFilter(int *positions, int N, int size, int smOffset);
__device__ void pushEliminetedToEnd(float *Q, int *positions, int N, int size);

__device__ void pushEliminetedToEndPositions(int *positions, int size);

__global__ void initPositionsData(int *positions, int *collect_number, int size);

__global__ void buildQUHeap(nj_data_t d, UHeap<float, int> *heap, int batchSize);
__global__ void getPositionsBatch(UHeap<float, int> *heap, float *batchQ, int *batchPositions, int N, int batchSize);

__global__ void eliminateInjuctions(int *positions, int size, int N, int *r);

__global__ void cleanPositions(int *positions, int *results, int size);

__global__ void cleanPositions(int *positions, int *results, int size)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x; // Max 1023

    if (results[idx] == -1)
        positions[idx] = -1;
}

__global__ void eliminateInjuctions(int *positions, int size, int N, int *r)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x; // Max 1023
    const int pos = positions[idx];

    __shared__ int sm_pos[1024];

    int index, result;

    sm_pos[idx] = positions[idx];
    r[idx] = 1;
    __syncthreads();

    for (index = 0; index < idx; index += 1)
    {
        // #pragma unroll
        // "for (size_t i = 0; i < 32; i++)
        // {

        // }

        result = hasIntersection(positions[idx], sm_pos[index], N);
        if (result)
        {
            r[idx] = -1;
            break;
        }
    }
    __syncthreads();
}
/*
 Compare positions already selected with a new positions batch. Those it was not filtered,
 it will be add to selected positions.
*/
__global__ void consolidationOfPositions(int *positions, int *new_positions, int *collected_number, int positions_size, int batchSize, int N);

__device__ void update_d_2nodes(nj_data_t d, int position_1, int positoin_2);

__global__ void updateDK(nj_data_t d, int *positions, int k);
__global__ void resizeDFlex2(nj_data_t d, int *positions, int end_position, int index);

__global__ void consolidationOfPositions(int *positions, int *new_positions, int *collected_number, int positions_size, int batchSize, int N)
{
    // SOMENTE UM BLOCO!
    for (int index = threadIdx.x; index < batchSize; index += blockDim.x)
    {
        for (int i = 0; i < *collected_number; i++)
        {
            if (hasIntersection(positions[i], new_positions[index], N))
            {
                new_positions[index] = -1;
                break;
            }
        }
    }
    __syncthreads();

    for (int index = threadIdx.x; index < batchSize; index += blockDim.x)
    {
        if (new_positions[index] != -1)
        {
            int oldIndex = atomicIncFixed(collected_number, positions_size);
            if (oldIndex < positions_size)
                positions[oldIndex] = new_positions[index];
        }
        __syncthreads();
    }
    __syncthreads();
}

// utilizar shared mem é melhor se quiser realizar varias entradas em blocos diferentes, usando a global so pode um!
__global__ void buildQUHeap(nj_data_t d, UHeap<float, int> *heap, int batchSize)
{
    extern __shared__ int sm[];
    float *batchQ = (float *)&sm[0];
    int *batchPositions = (int *)&batchQ[batchSize];
    int smOffset = 2 * batchSize;

    int d_size = d.N * (d.N) / 2;
    int batchNeeded = (d_size + batchSize - 1) / batchSize;

    for (int idxInsertion = blockIdx.x; idxInsertion < batchNeeded; idxInsertion += gridDim.x)
    {
        buildQPositions(d, batchQ, batchPositions, idxInsertion * batchSize, batchSize, d_size);
        heap->insertion(batchQ, batchPositions, batchSize, smOffset);
        __syncthreads();
    }
}

__global__ void getPositionsBatch(UHeap<float, int> *heap, float *batchQ, int *batchPositions, int N, int batchSize)
{
    int batchNeeded = 1;
    int outSize = 0;

    for (int i = blockIdx.x; i < batchNeeded; i += gridDim.x)
    {

        // delete items from heap
        if (heap->deleteRoot(batchQ, batchPositions, outSize) == true)
        {
            __syncthreads();

            heap->deleteUpdate(0);
        }
        __syncthreads();
    }

    // NON-UNIQUE FILTER, DUPLICATION OF UNIQUE THROW OUT THE ARRAY

    // nonUniqueFilter(positions, N, batchSize, smOffset);
    // __syncthreads();
    // pushEliminetedToEnd(Q, positions, N, batchSize);
    // __syncthreads();

    __syncthreads();
}

__global__ void updateDK(nj_data_t d, int *positions, int k)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    for (int index = idx; index < d.N * k; index += blockDim.x * gridDim.x)
    {
        int index_update = index / k;
        int position_index = index % k;

        int position = positions[position_index];

        int i_pos = position / d.N;
        int j_pos = position % d.N;

        float d_ij, new_duk;
        d_ij = d_get_D_position(d, i_pos, j_pos);

        if (index_update == 0)
        {
            float sum_k, sum_update_i, sum_update_j;
            int k_i, k_j;

            sum_k = 0;
            sum_update_i = 0;
            sum_update_j = 0;

            for (int k_pos = 0; k_pos < k; k_pos++)
            {
                if (k_pos == position_index)
                    continue;

                k_i = positions[k_pos] / d.N;
                k_j = positions[k_pos] % d.N;

                sum_k += d_get_D_position(d, k_i, k_j);
                sum_update_i += (d_get_D_position(d, k_i, i_pos) +
                                 d_get_D_position(d, k_j, i_pos) -
                                 d_get_D_position(d, k_i, k_j)) /
                                2;
                sum_update_j += (d_get_D_position(d, k_i, j_pos) +
                                 d_get_D_position(d, k_j, j_pos) -
                                 d_get_D_position(d, k_i, k_j)) /
                                2;
                // Oque deve fazer
                /*
                    sum_update_k_i +=
                    sum_update_k_j +=
                    sum_duu +=
                */
            }
            // O que deve acontecer com N?
            d.S[i_pos] = ((d.S[i_pos] + d.S[j_pos] - (d.N - position_index) * d_ij - sum_update_i - sum_update_j) / 2) - sum_k;
            // d.S[i_pos] = (d.S[i_pos] + d.S[j_pos] - d.N  * d_ij) - sum_update_k_i sum_update_k_j + sum_duu;
            // d.S[i_pos] = (d.S[i_pos] + d.S[j_pos] - (d.N -position_index) * d_ij) - sum_update_k_i sum_update_k_j + sum_duu;
        }

        if (index_update == 0)
        {
            for (int pos_idx = position_index; pos_idx < k; pos_idx++)
            {
                update_d_2nodes(d, position, positions[pos_idx]);
            }
        }
        int run = 1;

        for (int pos_idx = 0; pos_idx < k; pos_idx++)
        {
            int i_vrf = positions[pos_idx] / d.N;
            int j_vrf = positions[pos_idx] % d.N;

            if (i_vrf == index_update || j_vrf == index_update)
                run = 0;
        }

        // S'(k) = S(k) - d(a, k) - d(b, k) + d(u, k)
        // d.S[index_update] = new_duk - (new_duk * 2 + d_ij) + d.S[index_update];
        if (run)
        {
            new_duk = (d_get_D_position(d, i_pos, index_update) +
                       d_get_D_position(d, j_pos, index_update) - d_ij) /
                      2;
            atomicAdd(&d.S[index_update], new_duk - (new_duk * 2 + d_ij));
        }
        __syncthreads();

        if (run)
            d_set_D_position(d, index_update, i_pos, new_duk);
    }
}

__global__ void resizeDFlex2(nj_data_t d, int *positions, int end_position, int index)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= d.N)
        return;

    // int position_i = position / d.N;
    if (positions[index] == -1)
        return;

    int position_j = positions[index] % d.N;

    float d_n_minus_value = d_get_D_position(d, idx, end_position);

    if (position_j == (end_position))
        return;

    if (idx == 0)
        d.S[position_j] = d.S[end_position];

    if (idx == position_j)
        return;

    d_set_D_position(d, idx, position_j, d_n_minus_value);
}

// WELP FUNC

__device__ void update_d_2nodes(nj_data_t d, int position_1, int position_2)
{
    int a_ = position_1 / d.N;
    int a = position_2 / d.N;

    int b_ = position_1 % d.N;
    int b = position_2 % d.N;

    float d_uv = d_get_D_position(d, a_, a) + d_get_D_position(d, a_, b) +
                 d_get_D_position(d, b_, a) + d_get_D_position(d, b_, b) -
                 2 * d_get_D_position(d, a, b) - 2 * d_get_D_position(d, a_, b_);

    d_uv = d_uv / 4;

    d_set_D_position(d, a, a, d_uv);
}

__device__ __host__ int hasIntersection(int position_1, int position_2, int N)
{
    int i_pos1, j_pos1;
    int i_pos2, j_pos2;

    i_pos1 = position_1 / N;
    j_pos1 = position_1 % N;

    i_pos2 = position_2 / N;
    j_pos2 = position_2 % N;

    int result = i_pos1 == i_pos2 || i_pos1 == j_pos2;
    result = result || j_pos1 == i_pos2 || j_pos1 == j_pos2;

    return result;
}

__device__ void pushEliminetedToEnd(float *Q, int *positions, int N, int size)
{
    float temp;

    for (int k = 2; k <= size; k <<= 1)
    {
        for (int j = k / 2; j > 0; j >>= 1)
        {
            for (int i = threadIdx.x; i < size; i += blockDim.x)
            {
                int ixj = i ^ j;
                if (ixj > i)
                {
                    if ((i & k) == 0)
                    {
                        if (positions[i] == -1 && positions[ixj] != -1)
                        {
                            positions[i] = positions[ixj];
                            positions[ixj] = -1;

                            temp = Q[i];
                            Q[i] = Q[ixj];
                            Q[ixj] = temp;
                        }
                    }
                    else
                    {
                        if (positions[i] != -1 && positions[ixj] == -1)
                        {
                            positions[ixj] = positions[i];
                            positions[i] = -1;

                            temp = Q[ixj];
                            Q[ixj] = Q[i];
                            Q[i] = temp;
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
}

__device__ void nonUniqueFilter(int *positions, int N, int size, int smOffset)
{
    __shared__ extern int sm[];
    int *excludePositions = (int *)&sm[smOffset];

    for (int i = threadIdx.x; i < size; i += blockDim.x)
    {
        excludePositions[i] = 0;
    }
    __syncthreads();

    for (int k = 2; k <= size; k <<= 1)
    {
        for (int j = k / 2; j > 0; j >>= 1)
        {
            for (int i = threadIdx.x; i < size; i += blockDim.x)
            {
                int ixj = i ^ j;
                if (ixj > i)
                {

                    if (hasIntersection(positions[i], positions[ixj], N))
                    {
                        excludePositions[ixj] = -1;

                        if (excludePositions[i] == -1)
                        {
                            excludePositions[ixj] = -1;
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < size; i += blockDim.x)
    {
        if (excludePositions[i] == -1)
        {
            positions[i] = -1;
        }
    }

    __syncthreads();
}

__device__ void pushEliminetedToEndPositions(int *positions, int size)
{
    for (int k = 2; k <= size; k <<= 1)
    {
        for (int j = k / 2; j > 0; j >>= 1)
        {
            for (int i = threadIdx.x; i < size; i += blockDim.x)
            {
                int ixj = i ^ j;
                if (ixj > i)
                {
                    if ((i & k) == 0)
                    {
                        if (positions[i] == -1 && positions[ixj] != -1)
                        {
                            positions[i] = positions[ixj];
                            positions[ixj] = -1;
                        }
                    }
                    else
                    {
                        if (positions[i] != -1 && positions[ixj] == -1)
                        {
                            positions[ixj] = positions[i];
                            positions[i] = -1;
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
}

__device__ void buildQPositions(nj_data_t d, float *Q, int *positions, int start, int batchSize, int d_size)
{
    int position;

    for (int index = threadIdx.x; index < batchSize; index += blockDim.x)
    {
        int idx = start + index;

        Q[index] = FLT_MAX;
        positions[index] = -1;

        if (idx < d_size)
        {
            position = rect_pos_to_otu_pos(idx, d.N);
            Q[index] = calc_q(d, position);
            positions[index] = position;
        }
    }
    __syncthreads();
}

__inline__ __device__ float calc_q(nj_data_t d, int position)
{
    float d_rc, value;
    int i = position / d.N;
    int j = position % d.N;

    d_rc = d_get_D_position(d, i, j);
    value = (d.N - 2) * d_rc - d.S[i] - d.S[j];

    return value;
}

__global__ void initPositionsData(int *positions, int *collect_number, int size)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += blockDim.x * gridDim.x)
    {
        positions[i] = -1;
    }

    if (threadIdx.x + blockDim.x * blockIdx.x == 0)
        *collect_number = 0;
}

#endif