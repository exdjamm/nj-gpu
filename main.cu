#include <stdio.h>

#include <cuda_runtime.h>

#include "./nj_read/nj_read.cuh"

#include "./nj/nj_flex.cuh"
#include "./nj/nj_normal.cuh"

#include "./time_analisys.cuh"

int main(int argc, char const *argv[])
{

    if (argc != 6)
    {
        printf("Arguments in the form: [file] [type] [p] [k] [TPB]\n");
        return 1;
    }

    const char *file = argv[1];
    int type = atoi(argv[2]);
    float p_value = atof(argv[3]);
    int k_number = atoi(argv[4]);
    int TPB = atoi(argv[5]);

    printf("%s, %d, %.3f, %d\n", file, type, p_value, k_number);

    nj_read_t read;
    nj_data_t data;
    nj_read_init(&read);

    nj_read_file(&read, file);

    data = nj_data_to_device(read, p_value, k_number);

    time_start();

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
        // TODO: Function call
    }

    time_end();

    printf("%d; %.4f;\n", read.N, elapsed_time);

    free_nj_data_device(data);
    free_nj_read(read);

    return 0;
}
