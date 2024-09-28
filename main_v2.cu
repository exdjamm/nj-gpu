#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuda_runtime.h>

#ifndef TPB_1D
#define TPB_1D 32
#endif

#ifndef TPB_2D
#define TPB_2D 32
#endif

#include "nj_read.cuh"
#include "time_analisys.cuh"

#include "nj_otu.cuh"
#include "nj_elements.cuh"
#include "nj_otu_array.cuh"

__host__ void nj_gpu(nj_read MatrixRead, int is_otu, int is_otu_array);

__host__ void nj_gpu(nj_read MatrixRead, int is_otu, int is_otu_array){
    Matrix D = MatrixRead.distances;
    MatrixOtu DOtu = MatrixRead.distances_otu;
    Vector S = MatrixRead.SUM;

    Matrix d_D, d_Q;
    MatrixOtu d_DOtu, d_QOtu;
    MatrixOtuArray d_QOtuArray;
    Vector d_S;

    size_t SIZE_ELEMENTS;
    if (is_otu)
        SIZE_ELEMENTS = sizeof(otu_t);
    else if(is_otu_array)
        SIZE_ELEMENTS = sizeof(float);
    else
        SIZE_ELEMENTS = sizeof(float);

    // PROPS
    d_D.N = d_Q.N = d_S.size = D.N; 
    d_D.midpoint = d_Q.midpoint = D.midpoint;
    d_D.stride = d_Q.stride = D.stride;
    d_D.height = d_Q.height = D.height;
    d_D.width = d_Q.width = D.width;

    d_DOtu.N = d_QOtu.N = d_S.size = D.N; 
    d_DOtu.midpoint = d_QOtu.midpoint = D.midpoint;
    d_DOtu.stride = d_QOtu.stride = D.stride;
    d_DOtu.height = d_QOtu.height = D.height;
    d_DOtu.width = d_QOtu.width = D.width;

    d_QOtuArray.N           = D.N       ;
    d_QOtuArray.midpoint    = D.midpoint;
    d_QOtuArray.stride      = D.stride  ;
    d_QOtuArray.height      = D.height  ;
    d_QOtuArray.width       = D.width   ;

    /* // EXTERNO
    const float d_ij = get_matrix_element(D, qPosMin.i, qPosMin.j);
    const float difSums = S.elements[qPosMin.i] - S.elements[qPosMin.j];
    const float product = (2*(D.N-2));

    float d_iu = d_ij/2  + difSums/product;
    float d_ju = d_ij - d_iu;
    // FIM EXTERNO */
    if(is_otu){
        cudaMalloc(&d_DOtu.elements, d_D.height*d_D.width*SIZE_ELEMENTS);
        cudaMalloc(&d_QOtu.elements, d_Q.height*d_Q.width*SIZE_ELEMENTS);    
        
        cudaMemcpy(d_DOtu.elements, DOtu.elements, DOtu.height*DOtu.width*SIZE_ELEMENTS, cudaMemcpyHostToDevice);
        
    }else if(is_otu_array){
        cudaMalloc(&d_D.elements, d_D.height*d_D.width*SIZE_ELEMENTS);

        cudaMalloc(&d_QOtuArray.elements, d_Q.height*d_Q.width*SIZE_ELEMENTS);    
        cudaMalloc(&d_QOtuArray.position, d_Q.height*d_Q.width*sizeof(int));    

        cudaMemcpy(d_D.elements, D.elements, D.height*D.width*SIZE_ELEMENTS, cudaMemcpyHostToDevice);
    }
    else{
        cudaMalloc(&d_D.elements, d_D.height*d_D.width*SIZE_ELEMENTS);
        cudaMalloc(&d_Q.elements, d_Q.height*d_Q.width*SIZE_ELEMENTS);    
        
        cudaMemcpy(d_D.elements, D.elements, D.height*D.width*SIZE_ELEMENTS, cudaMemcpyHostToDevice);
    }
    
    cudaMalloc(&d_S.elements, S.size*sizeof(float));
    cudaMemcpy(d_S.elements, S.elements, S.size*sizeof(float), cudaMemcpyHostToDevice);

    time_start();

        if(is_otu)
            nj_loop_otu(d_DOtu, d_S, d_QOtu);
        else if(is_otu_array)
            nj_loop_otu_array(d_D, d_S, d_QOtuArray);
        else
            nj_loop_elements(d_D, d_S, d_Q);

    time_end();
    
    if(is_otu){
        cudaFree(d_DOtu.elements);
        cudaFree(d_QOtu.elements);
    }else if(is_otu_array){
        cudaFree(d_D.elements);
        cudaFree(d_QOtuArray.elements);
        cudaFree(d_QOtuArray.position);
    }
    else{
        cudaFree(d_D.elements);
        cudaFree(d_Q.elements);
    }

    cudaFree(d_S.elements);
    
}

int main(int argc, char const *argv[]){
    
    const char *dir = "/content/drive/MyDrive/colab-data/nj-data";
    const char *filename = "test5.ent";
    char *dir_filename;
    int dir_filename_size;
    
    if(argc != 1){
        if(argc == 3){
            dir = argv[1];
            filename = argv[2];
        }else{
            return 1;
        }
    }
    
    dir_filename_size = strlen(dir)+strlen(filename) + 2;
    dir_filename = (char*) calloc(sizeof(char), dir_filename_size);

    dir_filename = strcat(dir_filename, dir);
    dir_filename = strcat(dir_filename, "/");
    dir_filename = strcat(dir_filename, filename);

    nj_read matrix_read;
    float start, time_cpu;

    #ifndef OTU
    int is_otu = 0;
    #else
      int is_otu = 1;
    #endif

    #ifndef OTU_ARRAY
    int is_otu_array = 0;
    #else
      int is_otu_array = 1;
    #endif
    
    matrix_read = read_matrix(dir_filename, is_otu);    

    if(matrix_read.error) goto EXIT;

    #ifdef DEBUG_VALUES_D
        for(int i = 0; i <matrix_read.N; i++){
        for (int j = 0; j < i; j++){
            float value;
            if(is_otu)
                value = get_matrix_otu(matrix_read.distances_otu, i, j);
            else
                value = get_matrix_element(matrix_read.distances, i, j);

            printf("%.2f ", value);
        }
            printf("\n");
        }
    #endif

    start = clock();
    nj_gpu(matrix_read, is_otu, is_otu_array);
    start = clock() - start;
    
    time_cpu = 1000*( (float) start) / CLOCKS_PER_SEC;
    //printf("CPU-Time NJ;%.3f;\n", time_cpu);
    printf("%.3f;", elapsed_time);
    //printf("GPU-Time NJ;%.3f;\n", elapsed_time);

    free(dir_filename);
    free_nj_read(matrix_read);

    return 0;

    EXIT:
    free(dir_filename);
    free_nj_read(matrix_read);

    return 1;
}
