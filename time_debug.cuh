#ifndef _H_TIME_DEBUG
#define _H_TIME_DEBUG

#include <stdlib.h>
#include <stdio.h>

#include <string.h>
#include <time.h>

struct time_info
{
    char names[256][16];
    float stime[256];
    cudaEvent_t start[256];
    cudaEvent_t end[256];
    int count[256];
    int child[256][16];
    int child_size[256];
    int size;
};
typedef struct time_info time_i;

void init_time();
void time_print(int id, int t);

void i_time(const char *name, int father, int id);
void f_time(int id);

#ifdef NO_TIME
#define TIME_POINT(name, parent, id) \
    {                                \
    }
#define TIME_POINT_END(ID) \
    {                      \
    }
#else
#define TIME_POINT(name, parent, id) \
    {                                \
        i_time(name, parent, id);    \
    }
#define TIME_POINT_END(ID) \
    {                      \
        f_time(ID);        \
    }
#endif

#endif