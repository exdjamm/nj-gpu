#include <time_debug.cuh>
#include <cuda_runtime.h>

time_i info;

void dies_from_saddess()
{
    printf("ERRO NO DEBUG!!! ALGO ATINGIU O TAMANHO MAXIM!!!\n");
    exit(1);
}

void init_time()
{
    info.size = 0;

    for (int i = 0; i < 256; i++)
    {
        info.child_size[i] = 0;
        info.stime[i] = 0;
    }
}

void i_time(const char *name, int father, int id)
{
    int idx = info.size;
    int idx_child;
    int is_child = 0;

    info.size += 1;

    strncpy(info.names[id], name, 15);
    cudaEventCreate(&info.start[id]);   // # Irá marcar o inicio da execucao
    cudaEventCreate(&info.end[id]);     // # Irá  marcar o final da execucao
    cudaEventRecord(info.start[id], 0); // # insere na fila

    if (father == -1)
        return;

    idx_child = info.child_size[father];

    for (int i = 0; i < idx_child; i++)
    {
        if (info.child[father][i] == id)
        {
            is_child = 1;
            break;
        }
    }

    if (!is_child)
    {
        info.child_size[father] += 1;
        info.child[father][idx_child] = id;
    }
}

void f_time(int id)
{
    float elapsed_time;
    cudaEventRecord(info.end[id], 0);                                  // # insere na fila
    cudaEventSynchronize(info.end[id]);                                // # espera terminar
    cudaEventElapsedTime(&elapsed_time, info.start[id], info.end[id]); // # calcula
    info.stime[id] += elapsed_time;
}

void time_print(int id, int t)
{

    for (int i = 0; i < t; i++)
    {
        printf("\t");
    }

    printf("%s - %.2f\n", info.names[id], info.stime[id]);

    for (int i = 0; i < info.child_size[id]; i++)
    {
        time_print(info.child[id][i], t + 1);
    }
}
