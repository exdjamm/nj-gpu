#ifndef _H_NJ_UTILS
#define _H_NJ_UTILS

struct nj_read
{
    float* D;
    float* S;
    int N;

    int error;
};
typedef struct nj_read nj_read_t;

struct nj_data
{
    float* D;
    float* Q;
    float* S;
    int* positions;

    int N;
    int stride;
    float p;
    int k;
};
typedef struct nj_data nj_data_t;

/* 
Converte as coordenadas do espaço de OTUs para o espaço do array em memoria.
 */
__device__ __host__ int otu_to_matrix_position(int s_i, int s_j, int N);

/* 
Converte as coordenadas do espaço do array em memoria para o espaço de OTUs.
 */
__device__ __host__ int matrix_to_otu_position(int s_idx, int N, int stride);

__host__ void set_array_position(nj_read_t r, int i, int j, float value);

__device__ void d_set_D_position(nj_data_t d, int i, int j, float value);
__device__ float d_get_D_position(nj_data_t d, int i, int j);

__device__ void d_set_Q_position(nj_data_t d, int i, int j, float value);
__device__ float d_get_Q_position(nj_data_t d, int i, int j);

__device__ __host__ int otu_to_matrix_position(int s_i, int s_j, int N){
    int pos, i, j, temp;
    temp = s_i;
    if(s_i < s_j){
        s_i = s_j;
        s_j = temp;
    }

    i = s_i-1;
    j = s_j;

    if( s_i > N/2 ){
        i = N - 1 - s_i;
        j = s_j + i + 1;
    }

    pos = i*(N) + j;

    return pos;
}

__device__ __host__ int matrix_to_otu_position(int s_idx, int N, int stride){
    int pos, i_matrix, j_matrix;
    i_matrix = s_idx / N;
    j_matrix = s_idx % N;

    pos = (i_matrix+1)*stride + j_matrix;

    if( j_matrix > i_matrix)
        pos = (N - (i_matrix+1))*stride + j_matrix - i_matrix - 1;

    return pos;

}

__host__ void set_array_position(nj_read_t r, int i_otu, int j_otu, float value){
    int pos = otu_to_matrix_position(i_otu, j_otu, r.N);
    r.D[pos] = value;
}

__device__ void d_set_D_position(nj_data_t d, int i_otu, int j_otu, float value){
    int pos = otu_to_matrix_position(i_otu, j_otu, d.N);

    d.D[pos] = value;
}
__device__ float d_get_D_position(nj_data_t d, int i_otu, int j_otu){
    int pos = otu_to_matrix_position(i_otu, j_otu, d.N);

    return d.D[pos];
}

__device__ void d_set_Q_position(nj_data_t d, int i_otu, int j_otu, float value){
    int pos = otu_to_matrix_position(i_otu, j_otu, d.N);

    d.Q[pos] = value;
}
__device__ float d_get_Q_position(nj_data_t d, int i_otu, int j_otu){
    int pos = otu_to_matrix_position(i_otu, j_otu, d.N);

    return d.Q[pos];
}

#endif