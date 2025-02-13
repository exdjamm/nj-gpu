#ifndef _H_NORMAL_HEAP
#define _H_NORMAL_HEAP

void heapify(float *heap, int *aux, int size);
void pop(float *heap, int *aux, int size, float *out_heap, int *out_aux);

void tb_update(float *heap, int *aux, int node, int size);

void heapify(float *heap, int *aux, int size)
{
    for (int node = size / 2; node >= 0; node--)
    {
        tb_update(heap, aux, node, size);
    }
}

void pop(float *heap, int *aux, int *size, float *out_heap, int *out_aux)
{
    *out_heap = heap[0];
    *out_aux = aux[0];
    *size -= 1;

    heap[0] = heap[*size];
    aux[0] = aux[*size];

    tb_update(heap, aux, 0, *size);
}

void tb_update(float *heap, int *aux, int node, int size)
{
    int index, left, right;
    int lesser, has_less;
    float heap_temp;
    int aux_temp;

    index = node;
    left = 2 * index + 1;
    right = 2 * index + 2;

    if (left >= size)
        return;

    if (right >= size && heap[left] < heap[index])
    {
        heap_temp = heap[index];
        heap[index] = heap[left];
        heap[left] = heap_temp;

        aux_temp = aux[index];
        aux[index] = aux[left];
        aux[left] = aux_temp;
    }

    has_less = heap[right] < heap[index] || heap[left] < heap[index];

    while (has_less)
    {
        if (left >= size)
            break;

        if (right >= size && heap[left] < heap[index])
        {
            heap_temp = heap[index];
            heap[index] = heap[left];
            heap[left] = heap_temp;

            aux_temp = aux[index];
            aux[index] = aux[left];
            aux[left] = aux_temp;
            break;
        }

        lesser = right;
        if (heap[left] < heap[right])
            lesser = left;

        heap_temp = heap[index];
        heap[index] = heap[lesser];
        heap[lesser] = heap_temp;

        aux_temp = aux[index];
        aux[index] = aux[lesser];
        aux[lesser] = aux_temp;

        index = lesser;
        left = 2 * index + 1;
        right = 2 * index + 2;

        if (left >= size)
            break;

        if (right >= size && heap[left] < heap[index])
        {
            heap_temp = heap[index];
            heap[index] = heap[left];
            heap[left] = heap_temp;

            aux_temp = aux[index];
            aux[index] = aux[left];
            aux[left] = aux_temp;
            break;
        }

        has_less = heap[right] < heap[index] || heap[left] < heap[index];
    }
}

#endif
