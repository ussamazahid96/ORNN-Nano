#pragma once
#include"input_weight.h"
#include"recurrent_weight.h"
#include"output_weight.h"
#include"relu_bias.h"
typedef ac_fixed<RECURRENT_BW,4,true,AC_RND,AC_SAT> recurrent_type;
typedef ac_fixed<INPUT_BW,1,true,AC_RND,AC_SAT> input_type;
typedef ac_fixed<BUFFER_BW,6,true, AC_RND, AC_SAT> accumulator_type;
typedef accumulator_type output_type;
