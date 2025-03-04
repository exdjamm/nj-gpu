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
    TIME_POINT("MAIN", -1, 0);
    if (argc != 6)
    {
        printf("Arguments in the form: [file] [type] [p] [TPB] [HEAVISIDE FNJ_CPU]\n");
        return 1;
    }

    const char *file = argv[1];
    int type = atoi(argv[2]);
    float p_value = atof(argv[3]);
    // int k_number = atoi(argv[4]);
    int TPB = atoi(argv[4]);
    int hpoint_fnj_cpu = atoi(argv[5]);

    printf("file; execution type; p_value; TPB; heaviside point; n otus; time (ms)\n");
    printf("%s; %d; %.3f; %d; %d;", file, type, p_value, TPB, hpoint_fnj_cpu);

    nj_read_t read;
    nj_data_t data, data_host;
    TIME_POINT("READFILE&DEVICE", 0, 1);
    nj_read_init(&read);

    nj_read_file(&read, file);

    if (type != 4)
    {
        data = nj_data_to_device(read, p_value, 0);
    }
    else
    {
        data_host = nj_data_init_host(read, p_value, 0);
    }

    TIME_POINT_END(1);

    time_start();
    TIME_POINT("EXEC", 0, 2);
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
        nj_flex_heap(&data, TPB, 3);
    }
    else if (type == 3)
    {
        nj_flex_heap(&data, TPB, hpoint_fnj_cpu);
        data_host = nj_data_to_host_pointer(data);
        TIME_POINT("CPU FNJ", 2, 15);
        TIME_POINT_END(15);
    }
    else if (type == 4)
    {
        // data_host = nj_data_to_host_pointer(data);
        fnj_heap_cpu(data_host);
    }
    TIME_POINT_END(2);
    time_end();

    printf(" %d; %.4f;\n", read.N, elapsed_time);

    TIME_POINT_END(0);

    time_print(0, 0);

    free_nj_data_device(data);
    free_nj_data_host(data_host);
    free_nj_read(read);

    return 0;
}
