#include "options.cuh"

#ifndef TESTE
#define TESTE 12
#endif

int main(int argc, char const *argv[])
{
    options_t op = read_op(argc, argv);

    printf("%s %s\n",argv[1], argv[2]);

    printf("otu: \t%d\n",         op.otu);
    printf("heap: \t%d\n",        op.heap);
    printf("print_q: \t%d\n",     op.print_q_value);
    printf("heap_size: \t%d\n",   op.heap_size);
    printf("tpb_size2d: \t%d\n",  op.TPB_2D);
    printf("tpb_size1d: \t%d\n",  op.TPB_1D);
    printf("file: \t%s\n",  op.dir_filename);
    printf("error: \t%d\n",  op.error);

    printf("\nTESTE: \t%d\n",  TESTE);

    if(op.error){
        return 1;
    }

    return 0;
}
