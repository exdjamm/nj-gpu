#ifndef _H_NJ_READ
#define _H_NJ_READ

#include <stdio.h>
#include <cuda_runtime.h>

#include "nj_utils.cuh"

void nj_read_init(nj_read_t *r);

/*
Recebe um nj_read_t e os parametros p e k, alloca e copia para memoria global do device.
 */
nj_data_t nj_data_to_device(nj_read_t r, float p, int k);

/*
 Copia os valores das variaveis de device para as variaveis de host.
*/
nj_data_host_t nj_data_to_host_pointer(nj_data_t d);

void nj_read_init(nj_read_t *r)
{
    r->D = r->S = NULL;
    r->N = 0;

    r->error = 0;
}

void nj_read_file(nj_read_t *r, const char *file)
{
    FILE *f = fopen(file, "r");

    if (f == NULL)
    {
        r->error = -1;
        return;
    }
    int N;
    float value;

    fscanf(f, "%d ", &N);

    // TODO: Colocar para criar ou ler ID das OTUs para criacao de string de arvore

    r->N = N;
    r->D = (float *)calloc(N * (N) / 2, sizeof(float));
    r->S = (float *)calloc(N, sizeof(float));

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < i; j++)
        {
            fscanf(f, "%f;", &value);
            set_array_position(*r, i, j, value);

            r->S[i] += value;
            r->S[j] += value;
        }
    }

    fclose(f);
}

void free_nj_read(nj_read_t r)
{
    free(r.D);
    free(r.S);
}

nj_data_t nj_data_to_device(nj_read_t r, float p, int k)
{
    nj_data_t d_data;
    size_t size_matrix = r.N * (r.N) / 2;
    size_t size_select_otus = p * r.N;

    d_data.N = r.N;
    d_data.p = p;
    d_data.k = k;
    d_data.stride = r.N;

    d_data.positions = 0;

    cudaMalloc(&(d_data.D), sizeof(float) * size_matrix);
    cudaMalloc(&(d_data.Q), sizeof(float) * size_matrix);
    cudaMalloc(&(d_data.S), sizeof(float) * r.N);

    // if(flex){
    //     cudaMalloc(&(d_data.positions), sizeof(int)*size_matrix);
    // }

    cudaMemcpy(d_data.D, r.D, sizeof(float) * size_matrix, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data.S, r.S, sizeof(float) * r.N, cudaMemcpyHostToDevice);

    return d_data;
}

nj_data_host_t nj_data_to_host_pointer(nj_data_t d)
{
    nj_data_host_t d_data;
    size_t size_matrix = d.N * (d.N) / 2;
    size_t size_select_otus = d.p * d.N;

    d_data.N = d.N;
    d_data.p = d.p;
    d_data.k = d.k;
    d_data.stride = d.N;

    d_data.D = (float *)calloc(size_matrix, sizeof(float));
    d_data.Q = (float *)calloc(size_matrix, sizeof(float));
    d_data.S = (float *)calloc(d.N, sizeof(float));
    d_data.positions = NULL;

    cudaMemcpy(d_data.D, d.D, sizeof(float) * size_matrix, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_data.S, d.S, sizeof(float) * d.N, cudaMemcpyDeviceToHost);

    return d_data;
}

void free_nj_data_device(nj_data_t d)
{
    cudaFree(d.D);
    cudaFree(d.S);
    cudaFree(d.Q);
    cudaFree(d.positions);
}

void free_nj_data_host(nj_data_host_t d)
{
    free(d.D);
    free(d.S);
    free(d.Q);
    free(d.positions);
}

#endif