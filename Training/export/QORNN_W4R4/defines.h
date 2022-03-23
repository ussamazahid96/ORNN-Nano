#pragma once
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define INPUT_BW 4
#define BUFFER_BW 16
#define RECURRENT_BW 8
#define IN_DIM 32
#define REC_DIM 128
#define NUM_CLASSES 16
#define I_SIMD 1
#define I_PE 4
#define R_SIMD I_PE
#define O_SIMD I_PE
#define O_PE 4
#define I_BIAS true
#define R_BIAS true
#define O_BIAS false
#define MAX_BUFFER_SIZE MAX(I_SIMD*INPUT_BW, R_SIMD*RECURRENT_BW)
