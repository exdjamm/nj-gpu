#include <stdio.h>

#include <cuda_runtime.h>

#define NJ

#include <time_debug.cuh>

#include "./nj_read/nj_read.cuh"

#include "./nj/nj_flex_heap.cuh"
#include "./nj/nj_flex.cuh"
#include "./nj/nj_normal.cuh"
#include "./nj/fnj_heap_cpu.cuh"

#include "./time_analisys.cuh"

int main(int argc, char const *argv[])
{
    i_time("MAIN", -1, 0);
    if (argc != 5)
    {
        printf("Arguments in the form: [file] [type] [p] [TPB]\n");
        return 1;
    }

    const char *file = argv[1];
    int type = atoi(argv[2]);
    float p_value = atof(argv[3]);
    // int k_number = atoi(argv[4]);
    int TPB = atoi(argv[4]);

    printf("file; execution type; p_value; n otus; time (ms)\n");
    printf("%s; %d; %.3f;", file, type, p_value);

    nj_read_t read;
    nj_data_t data;
    i_time("READFILE&DEVICE", 0, 1);
    nj_read_init(&read);

    nj_read_file(&read, file);

    data = nj_data_to_device(read, p_value, 0);
    f_time(1);

    time_start();
    i_time("EXEC", 0, 2);
    if (type == 0) // NJ
    {
        nj_normal(data, TPB);
    }
    else if (type == 1) // FNJ - Reduce
    {
        // TODO: Function call
        nj_flex(data, TPB);
    }
    else if (type == 2) // FNJ - kHeap
    {
        nj_flex_heap(data, TPB, 3);
    }
    else if (type == 3)
    {
        nj_flex_heap(data, TPB, 128);
        nj_data_t data_host = nj_data_to_host_pointer(data);
        i_time("CPU FNJ", 2, 15);
        fnj_heap_cpu(data_host);
        f_time(15);
        free_nj_data_host(data_host);
    }
    else if (type == 4)
    {
        nj_data_t data_host = nj_data_to_host_pointer(data);
        fnj_heap_cpu(data_host);
        free_nj_data_host(data_host);
    }
    f_time(2);
    time_end();

    printf("%d; %.4f;\n", read.N, elapsed_time);

    f_time(0);

    time_print(0, 0);

    free_nj_data_device(data);
    free_nj_read(read);

    return 0;
}
