#ifndef _H_OPTIONS
#define _H_OPTIONS

#include <stdio.h>
#include <string.h>

enum op {
    OTU,
    PRINT_Q_VALUE,
    HEAP,
    HEAP_SIZE,
    THREAD_PER_BLOCK_2D,
    THREAD_PER_BLOCK_1D,
    FILE_NJ,
    NONE
};

struct options{
    int otu;
    int print_q_value;
    int heap;
    int heap_size;

    int TPB_2D;
    int TPB_1D;

    char* dir_filename;

    int error;
};

typedef struct options options_t;

enum op text_to_op(const char* t);
void set_camp_op(options_t* op,  enum op OP_CAMP, int value, char * char_value);

options_t initialize_op();
options_t read_op(const int arvc, const char* arvs[]);

void validate_options(options_t *op);

options_t initialize_op(){
    options_t op;
    const char *dir_filename = "/content/drive/MyDrive/colab-data/nj-data/test5.ent\0";
    const int dir_filename_size = strlen(dir_filename);

    op.otu = 0;
    op.heap = 0;
    op.heap_size = 16;
    op.print_q_value = 0;

    op.TPB_2D = 32;
    op.TPB_1D = 32;

    op.dir_filename = (char*) calloc(sizeof(char), dir_filename_size);
    strcpy(op.dir_filename, dir_filename);

    op.error = 0;

    return op;
}

void validate_options(options_t *op){
    if(op->heap && !op->otu){
        printf("HEAP CAN NOT BE USE WITHOUT OTU IMPLEMENTATION! SETTING OTU TO 1.\n");
        op->otu = 1;
    }

    if(op->TPB_2D*op->TPB_2D > 1024){
        printf("MAXIMUM THREAD PER BLOCK IS 1024! TPB 2D %dx%d=%d IS GREATER THEN.\n", op->TPB_2D, op->TPB_2D, op->TPB_2D*op->TPB_2D);

        op->error = 1;
    }

    if(op->TPB_1D > 1024){
        printf("MAXIMUM THREAD PER BLOCK IS 1024! TPB 1D %d IS GREATER THEN.\n", op->TPB_1D);
        op->error = 1;
    }
}

options_t read_op(const int argc, const char* argv[]){
    options_t op = initialize_op();

    const char* arg;

    char op_text[256];
    char op_value_char[1024]; 
    int op_value_int, r_scan;

    for (int i = 1; i < argc; i++){
        arg = argv[i];    
        r_scan = sscanf(arg, "--%[^=]=%s", op_text, op_value_char);
        sscanf(op_value_char, "%d", &op_value_int);

        if(r_scan == 0) continue;

        //printf("%s -> \t%d, %d\n", op_text, op_value, r_scan);
        enum op OP_CAMP = text_to_op(op_text);

        if(OP_CAMP == NONE){
            printf("REVISE SEU COMANDO! --%s nao e uma opcao.\n", op_text);
            op.error = 1;
            break;
        }
        else
            set_camp_op(&op, OP_CAMP, op_value_int, op_value_char);
    }

    validate_options(&op);

    return op;
}

enum op text_to_op(const char* t){
    if(strcmp("OTU\0", t) == 0){
        return OTU;
    }
    else if(strcmp("HEAP\0", t) == 0){
        return HEAP;
    }
    else if(strcmp("HEAP_SIZE\0", t) == 0){
        return HEAP_SIZE;
    }
    else if(strcmp("TPB_2D\0", t) == 0){
        return THREAD_PER_BLOCK_2D;
    }
    else if(strcmp("TPB_1D\0", t) == 0){
        return THREAD_PER_BLOCK_1D;
    }
    else if(strcmp("Q_VALUE\0", t) == 0){
        return PRINT_Q_VALUE;
    }
    else if(strcmp("FILE\0", t) == 0){
        return FILE_NJ;
    } 

    return NONE;
}

void set_camp_op(options_t* op,  enum op OP_CAMP, int value, char* char_value){

    switch ( OP_CAMP ) {
        case OTU:
            op->otu = value;
            break;

        case HEAP:
            op->heap = value;
            break;

        case PRINT_Q_VALUE:
            op->print_q_value = value;
            break;

        case HEAP_SIZE:
            op->heap_size = value;
            break;
        
        case THREAD_PER_BLOCK_2D:
            op->TPB_2D = value;
            break;
        
        case THREAD_PER_BLOCK_1D:
            op->TPB_1D = value;
            break;
        
        case FILE_NJ:
            op->dir_filename = (char*) realloc(op->dir_filename, sizeof(char)*strlen(char_value) + 1);
            strcpy(op->dir_filename, char_value);
            break;

        default:
            printf("Error!\n");
            break;
    }
}

#endif
