#ifndef _H_UTILS_NJ_
#define _H_UTILS_NJ_

typedef struct {
    int i;
    int j;
    float new_di;
    float new_dj;
} new_distances;

#ifdef OTU_ALIGN
struct __align__(16) otu{
    int i, j;
    float value;
};
#else
struct otu{
    int i, j;
    float value;
};
#endif

typedef struct otu otu_t;

typedef struct{
    int size;
    float *elements;
} Vector;

typedef struct{
    unsigned int height;
    unsigned int width;
    unsigned int stride;

    unsigned int N;
    unsigned int midpoint;

    otu_t *elements;
} MatrixOtu;

typedef struct{
    unsigned int height;
    unsigned int width;
    unsigned int stride;

    unsigned int N;
    unsigned int midpoint;

    float *elements;
    int *position;
} MatrixOtuArray;

typedef struct{
    unsigned int height;
    unsigned int width;
    unsigned int stride;

    unsigned int N;
    unsigned int midpoint;

    float *elements;
} Matrix;

typedef otu_t QPos;

__device__ __host__ void set_matrix_element(Matrix A, int row, int col, float value);
__device__ __host__ float get_matrix_element(Matrix A, int row, int col);

__device__ __host__ void set_matrix_otu(MatrixOtu A, int row, int col, float value);
__device__ __host__ float get_matrix_otu(MatrixOtu A, int row, int col);
__device__ __host__ otu_t get_matrix_otu_all(MatrixOtu A, int row, int col);

__device__ __host__ void set_matrix_otu_array(MatrixOtuArray A, int row, int col, float value);
__device__ __host__ float get_matrix_otu_array(MatrixOtuArray A, int row, int col);
__device__ __host__ int get_matrix_otu_array_pos(MatrixOtuArray A, int row, int col);

__device__ __host__ void set_matrix_element(Matrix A, int row, int col, float value){
    int idx;
    idx = row*A.stride + col;

    A.elements[idx] = value;
}

__device__ __host__ float get_matrix_element(Matrix A, int row, int col){
    int idx;
    idx = row*A.stride + col;

    return A.elements[idx];
}

__device__ __host__ void set_matrix_otu(MatrixOtu A, int row, int col, float value){
    int idx;
    otu_t temp;
    otu_t* elements = A.elements;

    idx = row*A.stride + col;

    temp.i = row;
    temp.j = col;
    temp.value = value;

    elements[idx] = temp;
}

__device__ __host__ float get_matrix_otu(MatrixOtu A, int row, int col){
    int idx;
    idx = row*A.stride + col;
    //otu_t* elements = A.elements;

    return (A.elements)[idx].value;
}

__device__ __host__ otu_t get_matrix_otu_all(MatrixOtu A, int row, int col){
    int idx;
    idx = row*A.stride + col;
    otu_t* elements = A.elements;

    return elements[idx];
}

__device__ void swap_matrix_otus(MatrixOtu A, int s_i, int s_j, int d_i, int d_j){
    int idx_s, idx_d;
    otu_t temp, *elements;

    idx_s = s_i*A.stride + s_j;
    idx_d = d_i*A.stride + d_j;
    elements = A.elements;

    temp = elements[idx_s];
    elements[idx_s] = elements[idx_d];
    elements[idx_d] = temp;
}

__device__ __host__ void set_matrix_otu_array(MatrixOtuArray A, int row, int col, float value){
    int idx;
    idx = row*A.stride + col;

    A.elements[idx] = value;
    A.position[idx] = idx;
}

__device__ __host__ float get_matrix_otu_array(MatrixOtuArray A, int row, int col){
    int idx;
    idx = row*A.stride + col;

    return A.elements[idx];
}

__device__ __host__ int get_matrix_otu_array_pos(MatrixOtuArray A, int row, int col){
    int idx;
    idx = row*A.stride + col;

    return A.position[idx];
}

__device__ __host__ float swap_matrix_otu_array(MatrixOtuArray A, int row_s, int col_s, int row_d, int col_d){
    int idx_s, idx_d;
    idx_s = row_s*A.stride + col_s;
    idx_d = row_d*A.stride + col_d;

    float temp_value;
    int temp_position;

    temp_position = A.position[idx_s];
    A.position[idx_s] = A.position[idx_d];
    A.position[idx_d] = temp_position;
    
    temp_value = A.elements[idx_s];
    A.elements[idx_s] = A.elements[idx_d];
    A.elements[idx_d] = temp_value;
}

#endif
