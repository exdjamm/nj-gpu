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
    float p;
    int k;
};
typedef struct nj_data nj_data_t;

__host__ void set_array_position(nj_read_t r, int i, int j, float value);

__device__ void d_set_D_position(nj_data_t d, int i, int j, float value);
__device__ float d_get_D_position(nj_data_t d, int i, int j);

__device__ void d_set_Q_position(nj_data_t d, int i, int j, float value);
__device__ float d_get_Q_position(nj_data_t d, int i, int j);

__host__ void set_array_position(nj_read_t r, int i, int j, float value){
    int pos;
    pos = (i*(i-1)/2) + j;
    if( j > i){
        pos = j*(j-1)/2 + i;
    }

    r.D[pos] = value;
}

__device__ void d_set_D_position(nj_data_t d, int i, int j, float value){
    int pos;
    pos = (i*(i-1)/2) + j;
    if( j > i){
        pos = j*(j-1)/2 + i;
    }

    d.D[pos] = value;
}
__device__ float d_get_D_position(nj_data_t d, int i, int j){
    int pos;
    pos = (i*(i-1)/2) + j;
    if( j > i){
        pos = j*(j-1)/2 + i;
    }

    return d.D[pos];
}

__device__ void d_set_Q_position(nj_data_t d, int i, int j, float value){
    int pos;
    pos = (i*(i-1)/2) + j;
    if( j > i){
        pos = j*(j-1)/2 + i;
    }

    d.Q[pos] = value;
}
__device__ float d_get_Q_position(nj_data_t d, int i, int j){
    int pos;
    pos = (i*(i-1)/2) + j;
    if( j > i){
        pos = j*(j-1)/2 + i;
    }

    return d.Q[pos];
}

#endif