#ifndef _H_NJ_READ_
#define _H_NJ_READ_

#include <stdio.h>
#include <stdlib.h>

#include "utils.cuh"

typedef struct {
    Matrix distances;
    MatrixOtu distances_otu;
    Vector SUM;

    size_t size_D;
    size_t size_SUM;
    
    int N; 
    
    int error;
} nj_read ;

nj_read read_matrix(const char* filename, int is_otu);
void free_nj_read(nj_read read);

nj_read read_matrix(const char* filename, int is_otu){
    int N;
    float value, sum_i, sum_j;
    size_t size_D, size_S;

    nj_read result;
    result.error = 0;

    FILE* f = fopen(filename, "r");
    if(!f) goto EXIT;

    if (fscanf(f,"%d ",&N) != 1) goto READERRO;

    result.N = N;

    result.distances_otu.elements = NULL;
    result.distances.elements = NULL;

    result.distances.height = N;
    result.distances_otu.height = N;

    result.distances.width = N;
    result.distances_otu.width = N;

    result.distances.stride = N;
    result.distances_otu.stride = N;
    
    result.distances.N = N;
    result.distances_otu.N = N;

    result.SUM.size = N;
    
    if(is_otu){
        size_D = (N*N)*sizeof(otu_t);
        result.distances_otu.elements = (otu_t*) malloc(size_D);
        if(result.distances_otu.elements == NULL) goto EXIT;
    }
    else{
        size_D = (N*N)*sizeof(float);
        result.distances.elements = (float*) malloc(size_D);    
        if(result.distances.elements == NULL) goto EXIT;
    }

    size_S = N*sizeof(float);
    result.SUM.elements = (float*) calloc(N, sizeof(float));
    if(result.SUM.elements == NULL) goto EXIT;

    result.size_SUM = size_S;
    result.size_D = size_D;

    // Distancias
    for(int i = 0; i < N; ++i)
    {
        for (int j = 0; j < i; ++j)
        {
            if(fscanf(f, "%f;", &value) != 1) goto READERRO;
            if(is_otu){
                set_matrix_otu(result.distances_otu, i, j, value);
                set_matrix_otu(result.distances_otu, j, i, value);
            }else{
                set_matrix_element(result.distances, i, j, value);
                set_matrix_element(result.distances, j, i, value);
            }
        
            sum_i = result.SUM.elements[i];
            sum_j = result.SUM.elements[j];

            result.SUM.elements[i] = sum_i + value;
            result.SUM.elements[j] = sum_j + value;
            // printf("%f, %d-%d\n", value, i, j);
        }
        if(is_otu)
            set_matrix_otu(result.distances_otu, i, i, 0.0);
        else
            set_matrix_element(result.distances, i, i, 0.0);
    }

    fclose(f);
    
    return result;

    READERRO:
    printf("Erro de leitura");
    goto EXIT;

    EXIT:
    fclose(f);
    free_nj_read(result);
    result.error = 1;
    return result;
}    

void free_nj_read(nj_read read){
    free(read.distances.elements);
    free(read.distances_otu.elements);
    free(read.SUM.elements);
}


#endif

