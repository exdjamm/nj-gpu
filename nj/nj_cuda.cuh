#ifndef _H_NJ_CUDA
#define _H_NJ_CUDA

#include <float.h>

#include "../nj_read/nj_utils.cuh"

__global__ void buildQ(nj_data_t d);
__global__ void buildQBatch(nj_data_t d, int batch);
__global__ void buildQ2D(nj_data_t d);

__global__ void reduceQ(nj_data_t d, float *values_result, int *position_result);

__global__ void updateD(nj_data_t d, int position);
__global__ void resizeD(nj_data_t d, int position);
__global__ void resizeDFlex(nj_data_t d, int position, int end_position);

__global__ void ignorePositionsQ(nj_data_t d, int position);

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__global__ void buildQ(nj_data_t d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value, d_rc;
    int pos, i, j;

    if (idx >= d.N * (d.N) / 2)
        return;

    pos = rect_pos_to_otu_pos(idx, d.N);
    i = pos / d.N;
    j = pos % d.N;

    d_rc = d_get_D_position(d, i, j);
    value = (d.N - 2) * d_rc - d.S[i] - d.S[j];

    d_set_Q_position(d, i, j, value);
}

__global__ void buildQBatch(nj_data_t d, int batch)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x * batch;
    float value, d_rc;
    int pos, i, j;

    if (idx >= d.N * (d.N) / 2)
        return;

    for (int k = 0; k < batch; k++)
        if (idx + k < d.N * (d.N) / 2)
        {
            pos = rect_pos_to_otu_pos(idx + k, d.N);
            i = pos / d.N;
            j = pos % d.N;

            d_rc = d_get_D_position(d, i, j);
            value = (d.N - 2) * d_rc - d.S[i] - d.S[j];

            d_set_Q_position(d, i, j, value);
        }
}

__global__ void ignorePositionsQ(nj_data_t d, int position)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i, j;

    if (position == -1)
        return;

    if (idx >= d.N)
        return;

    i = position / d.N;
    j = position % d.N;

    printf("ignPositions: %d %d %d\n", i, j, idx);

    d_set_Q_position(d, idx, j, FLT_MAX);
    d_set_Q_position(d, idx, i, FLT_MAX);
}

__global__ void reduceQ(nj_data_t d, float *values_result, int *position_result)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int r_idx = blockIdx.x;
    int thread_per_block = blockDim.x;

    int i, j;
    int pos = rect_pos_to_otu_pos(idx, d.N);

    i = pos / d.N;
    j = pos % d.N;

    extern __shared__ int array[]; // 2*thread_per_block

    float *values_min = (float *)&array[0];
    int *positions_min = (int *)&values_min[thread_per_block];

    values_min[threadIdx.x] = FLT_MAX;
    __syncthreads();

    if (idx < d.N * (d.N) / 2)
    {
        values_min[threadIdx.x] = d_get_Q_position(d, i, j);
        positions_min[threadIdx.x] = pos;
    }
    __syncthreads();

    for (int stride = 1; stride < thread_per_block; stride *= 2)
    {
        int index = 2 * stride * threadIdx.x;

        if ((index < thread_per_block))
        {
            if (values_min[index] > values_min[index + stride])
            {
                values_min[index] = values_min[index + stride];
                positions_min[index] = positions_min[index + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        values_result[r_idx] = values_min[0];
        position_result[r_idx] = positions_min[0];
    }
}

__global__ void updateD(nj_data_t d, int position)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= d.N)
        return;

    int position_i = position / d.N;
    int position_j = position % d.N;

    float d_ij;
    d_ij = d_get_D_position(d, position_i, position_j);

    if (idx == 0)
    {

        d.S[position_i] = (d.S[position_i] + d.S[position_j] - d.N * d_ij) / 2;
    }

    if (idx == position_i || idx == position_j)
        return;

    float new_d_uk = (d_get_D_position(d, position_i, idx) +
                      d_get_D_position(d, position_j, idx) - d_ij) /
                     2;

    d.S[idx] = new_d_uk - (new_d_uk * 2 + d_ij) + d.S[idx];
    d_set_D_position(d, position_i, idx, new_d_uk);
}

__global__ void resizeD(nj_data_t d, int position)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= d.N)
        return;

    // int position_i = position / d.N;
    int position_j = position % d.N;

    float d_n_minus_value = d_get_D_position(d, idx, d.N - 1);

    if (position_j == (d.N - 1))
        return;

    if (idx == 0)
        d.S[position_j] = d.S[d.N - 1];

    if (idx == position_j)
        return;

    d_set_D_position(d, idx, position_j, d_n_minus_value);
}

__global__ void resizeDFlex(nj_data_t d, int position, int end_position)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= d.N)
        return;

    // int position_i = position / d.N;
    int position_j = position % d.N;

    float d_n_minus_value = d_get_D_position(d, idx, end_position);

    if (position_j == (end_position))
        return;

    if (idx == 0)
        d.S[position_j] = d.S[end_position];

    if (idx == position_j)
        return;

    d_set_D_position(d, idx, position_j, d_n_minus_value);
}

#endif