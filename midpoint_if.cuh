__device__ __host__ void set_matrix_element(Matrix A, int row, int col, float value){
    int idx, x, y, temp;
    y = row;
    x = col;
    temp = y;

    if(x >= y){
        y = x;
        x = temp;
    }

    if(x == y) return;

    if(y > (A.midpoint - 1)){
        x = (A.N - 1 - x);
        y = (A.N - 1 - y);
    }

    idx = y*A.stride + x;
    A.elements[idx] = value;
}

__device__ __host__ float get_matrix_element(Matrix A, int row, int col){
    int idx, x, y, temp;
    y = row;
    x = col;
    temp = y;

    if(x >= y){
        y = x;
        x = temp;
    }

    if(x == y) return 0.0;

    if(y > (A.midpoint - 1)){
        x = (A.N - 1 - x);
        y = (A.N - 1 - y);
    }

    idx = y*A.stride + x;

    return A.elements[idx];
}

nj_read read_matrix(const char* filename){
    int N, midpoint;
    float value, sum_i, sum_j;
    size_t size_D, size_S;

    nj_read result;
    result.error = 0;

    FILE* f = fopen(filename, "r");
    if(!f) goto EXIT;

    if (fscanf(f,"%d ",&N) != 1) goto READERRO;
    midpoint = (N+1)/2;

    result.N = N;

    result.D.height = midpoint;
    result.D.width = N;
    result.D.stride = N;
    result.D.N = N;
    result.D.midpoint = midpoint;

    result.SUM.size = N;
    
    size_D = (N*midpoint)*sizeof(float);
    result.D.elements = (float*) malloc(size_D);
    if(result.D.elements == NULL) goto EXIT;

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
            set_matrix_element(result.D, i, j, value);

            sum_i = result.SUM.elements[i];
            sum_j = result.SUM.elements[j];

            result.SUM.elements[i] = sum_i + value;
            result.SUM.elements[j] = sum_j + value;
            // printf("%f, %d-%d\n", value, i, j);
        }
        set_matrix_pos(result.D, i, i, 0.0);
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
