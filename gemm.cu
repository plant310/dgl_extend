#include<mma.h>

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void wmma_example(hafl *a, half *b, float *c, )