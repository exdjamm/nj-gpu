#ifndef _H_NJ_UTILS
#define _H_NJ_UTILS

#include <float.h>

struct nj_read
{
    float *D;
    float *S;
    int N;

    int error;
};
typedef struct nj_read nj_read_t;

struct nj_data
{
    float *D;
    float *Q;
    float *S;
    int *positions;

    int N;
    int stride;
    float p;
    int k;
};
typedef struct nj_data nj_data_t;

/*
As matrizes em memoria estao no formato triangular em um vetor linear,
com a posicao do vetor dada por i*(i-1)/2 + j, para i > j.

O espaço de kernel cuda (thread group x (thread x k)), k sendo o numero de posições
executadas em uma thread, estão no formado retangular N por N/2
para o caso de execução utilizando as matrizes.

Para cada posição do espaço de kernel é estraido as coordenadas, convertidas
nas coordenadas para uma matrix N X N e então usadas para consulta na matriz.

Abaixo as definições de tais funções de transformação.
*/

/*
Converte as coordenadas do espaço de OTUs para o espaço do array em memoria.
 */
__device__ __host__ int otu_to_mem_position(int i, int j);

/*
Converte as coordenadas do espaço de exec GPU para o espaço de OTUs.
 */
__device__ __host__ int rect_pos_to_otu_pos(int s_idx, int N);

/*
Set value in host Matrix D struct nj read array.
*/
__host__ void set_array_position(nj_read_t r, int i, int j, float value);

/*
Set value in device Matrix D struct nj_data array.
*/
__device__ void d_set_D_position(nj_data_t d, int i, int j, float value);

/*
Get value in device Matrix D struct nj_data array.
*/
__device__ float d_get_D_position(nj_data_t d, int i, int j);

/*
Set value in device Matrix Q struct nj_data array.
*/
__device__ void d_set_Q_position(nj_data_t d, int i, int j, float value);
/*
Get value in device Matrix Q struct nj_data array.
*/
__device__ float d_get_Q_position(nj_data_t d, int i, int j);

__device__ __host__ int otu_to_mem_position(int i, int j)
{
    int pos;

    pos = i * (i - 1) / 2 + j;
    if (i < j)
        pos = j * (j - 1) / 2 + i;

    return pos;
}

__device__ __host__ int rect_pos_to_otu_pos(int s_idx, int N)
{
    int pos, i_matrix, j_matrix;
    i_matrix = s_idx / N;
    j_matrix = s_idx % N;

    pos = (i_matrix + 1) * N + j_matrix;

    if (j_matrix > i_matrix)
        pos = (N - (i_matrix + 1)) * N + j_matrix - i_matrix - 1;

    return pos;
}

/* __device__ __host__ int otu_to_matrix_position(int s_i, int s_j, int N)
{
    int pos, i, j, temp;
    temp = s_i;
    if (s_i < s_j)
    {
        s_i = s_j;
        s_j = temp;
    }

    i = s_i - 1;
    j = s_j;

    if (s_i > N / 2)
    {
        i = N - 1 - s_i;
        j = s_j + i + 1;
    }

    pos = i * (N) + j;

    return pos;
} */

/* __device__ __host__ int matrix_to_otu_position(int s_idx, int N, int stride)
{
    int pos, i_matrix, j_matrix;
    i_matrix = s_idx / N;
    j_matrix = s_idx % N;

    pos = (i_matrix + 1) * stride + j_matrix;

    if (j_matrix > i_matrix)
        pos = (N - (i_matrix + 1)) * stride + j_matrix - i_matrix - 1;

    return pos;
} */

__host__ void set_array_position(nj_read_t r, int i_otu, int j_otu, float value)
{
    int pos = otu_to_mem_position(i_otu, j_otu);
    r.D[pos] = value;
}

__device__ void d_set_D_position(nj_data_t d, int i_otu, int j_otu, float value)
{
    int pos = otu_to_mem_position(i_otu, j_otu);

    if (i_otu >= d.N || j_otu >= d.N)
        return;
    if (i_otu < 0 || j_otu < 0)
        return;
    if (i_otu == j_otu)
        return;

    d.D[pos] = value;
}
__device__ float d_get_D_position(nj_data_t d, int i_otu, int j_otu)
{
    int pos = otu_to_mem_position(i_otu, j_otu);

    if (i_otu >= d.N || j_otu >= d.N)
        return 0;
    if (i_otu < 0 || j_otu < 0)
        return 0;
    if (i_otu == j_otu)
        return 0;

    return d.D[pos];
}

__device__ void d_set_Q_position(nj_data_t d, int i_otu, int j_otu, float value)
{
    int pos = otu_to_mem_position(i_otu, j_otu);

    d.Q[pos] = value;
}
__device__ float d_get_Q_position(nj_data_t d, int i_otu, int j_otu)
{
    if (i_otu >= d.N || j_otu >= d.N)
        return FLT_MAX;
    if (i_otu < 0 || j_otu < 0)
        return FLT_MAX;

    int pos = otu_to_mem_position(i_otu, j_otu);

    return d.Q[pos];
}

#endif