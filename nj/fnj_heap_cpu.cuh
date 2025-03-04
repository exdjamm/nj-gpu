#ifndef _H_FNJ_HEAP_CPU
#define _H_FNJ_HEAP_CPU

#include "../nj_read/nj_utils.cuh"
#include <normal_heap.cuh>

struct otu
{
    size_t i;
    size_t j;
};
typedef struct otu otu_t;

void fnj_heap_cpu(nj_data_t data);

void update_d(nj_data_t data, otu_t pos);
void resize_d(nj_data_t data, otu_t pos, int end_pos);

void fnj_heap_cpu(nj_data_t data)
{
    float value_q, d_ij;
    int size_matrix = (data.N / 2) * (data.N - 1);

    int pairs_size, collected;
    otu_t *pairs;
    float out_heap;
    int out_aux;

    pairs_size = data.N * data.p;
    if (pairs_size <= 0)
        pairs_size = 1;

    collected = 0;
    pairs = (otu_t *)calloc(pairs_size + 1, sizeof(otu_t));

    // Initialize Q-Matrix to max values
    /* for (int idx = 0; idx < size_matrix; idx++)
        data.Q[idx] = FLT_MAX; */

    int run = data.N >= 3;

    while (run)
    {
        // Set Q-Matrix
        for (int i = 0; i < data.N; i++)
        {
            for (int j = 0; j < i; j++)
            {
                int pos = i * (i - 1) / 2 + j;
                d_ij = d_get_D_position(data, i, j);
                value_q = d_ij * (data.N - 2) - data.S[i] - data.S[j];

                data.Q[pos] = value_q;
                data.positions[pos] = i * data.N + j;
            }
        }

        size_matrix = (data.N / 2) * (data.N - 1);

        pairs_size = data.N * data.p;
        if (pairs_size <= 0)
            pairs_size = 1;

        collected = 0;

        heapify(data.Q, data.positions, size_matrix);

        while (collected < pairs_size)
        {
            int non_disjunct = 0;

            pop(data.Q, data.positions, &size_matrix, &out_heap, &out_aux);

            // printf("\tTEST DISJUNCT (%d, %d)\n", out_aux / data.N, out_aux % data.N);
            for (int i = 0; i < collected; i++)
            {
                int i_pos, j_pos;
                int i_pos_, j_pos_;

                i_pos = pairs[i].i;
                j_pos = pairs[i].j;

                i_pos_ = out_aux / data.N;
                j_pos_ = out_aux % data.N;

                int result = (i_pos == i_pos_) || (i_pos == j_pos_);
                result = result || (j_pos == i_pos_) || (j_pos == j_pos_);

                // printf("\t\t%d (%d): (%d, %d) - (%d, %d) -> %d\n", i, collected, i_pos, j_pos, i_pos_, j_pos_, result);

                if (result)
                {
                    non_disjunct = 1;
                    break;
                }
            }

            if (non_disjunct)
                continue;

            printf("%d: (%d, %d) -> %.2f\n", data.N, out_aux / data.N, out_aux % data.N, out_heap);

            pairs[collected].i = out_aux / data.N;
            pairs[collected].j = out_aux % data.N;
            collected++;
        }
        printf("\n");
        for (size_t i = 0; i < collected; i++)
        {
            update_d(data, pairs[i]);
            resize_d(data, pairs[i], data.N - 1);
            data.N -= 1;

            printf("======== (%d, %d), %d\n", pairs[i].i, pairs[i].j, data.N + 1);

            // AQUI FOI ALTERADO E ESSA ALTERACAO CAUSOU MUDANCA
            for (int j = 0; j < collected; j++)
            {
                if (pairs[j].j == data.N)
                {
                    pairs[j].j = pairs[i].j;
                }
            }
        }

        for (size_t i = 0; i < collected; i++)
        {
        }

        // data.N -= pairs_size;
        run = data.N >= 3;
        if (!run)
            break;
    }

    free(pairs);
}

void update_d(nj_data_t data, otu_t pos)
{
    int i_pos = pos.i;
    int j_pos = pos.j;

    float d_ij = d_get_D_position(data, i_pos, j_pos);
    data.S[i_pos] = (data.S[i_pos] + data.S[j_pos] - data.N * d_ij) / 2;

    float new_duk;

    for (int i = 0; i < data.N; i++)
    {
        if (i == i_pos || i == j_pos)
            continue;

        new_duk = (d_get_D_position(data, i_pos, i) + d_get_D_position(data, j_pos, i) - d_ij) / 2;

        data.S[i] = data.S[i] + new_duk - (new_duk * 2 + d_ij);
        d_set_D_position(data, i_pos, i, new_duk);
    }
}

void resize_d(nj_data_t data, otu_t pos, int end_pos)
{
    int j_pos = pos.j;

    if (j_pos == end_pos)
        return;

    data.S[j_pos] = data.S[end_pos];

    for (int idx = 0; idx < data.N; idx++)
    {
        if (idx == j_pos)
            continue;

        float d_end = d_get_D_position(data, idx, end_pos);
        d_set_D_position(data, idx, j_pos, d_end);
    }
}

#endif