#pragma once

#ifndef RNN_H
#define RNN_H

template
<
unsigned int INPUT_DIM,
unsigned int RECURRENT_DIM,
unsigned int OUTPUT_DIM,
unsigned int ACT_FUNC,

unsigned int INPUT_SIMD,
unsigned int RECURRENT_SIMD,
unsigned int INPUT_PE,

unsigned int OUTPUT_SIMD,
unsigned int OUTPUT_PE,

bool INPUT_BIAS,
bool RECURRENT_BIAS,
bool OUTPUT_BIAS,

typename R_TYPE,
typename I_TYPE,
typename ACC_BUFFER_TYPE,

typename I_W_TYPE, 
typename I_B_TYPE,
typename I_NORM_W_TYPE, 
typename I_NORM_B_TYPE,

typename R_W_TYPE, 
typename R_B_TYPE,
typename R_NORM_W_TYPE, 
typename R_NORM_B_TYPE,

typename O_W_TYPE, 
typename O_B_TYPE,
typename O_NORM_W_TYPE, 
typename O_NORM_B_TYPE,


int in_stream_width, 
int out_stream_width,
int OUT_DEPTH,
int IN_DEPTH
>
void RNN_Layer
(
ihc::stream<ac_int<out_stream_width, false>, ihc::buffer<OUT_DEPTH>> &output_stream, 
ihc::stream<ac_int<in_stream_width, false>, ihc::buffer<IN_DEPTH>> &input_stream, 

I_W_TYPE input_weight, 
I_B_TYPE input_bias,
I_NORM_W_TYPE input_norm_weight, 
I_NORM_B_TYPE input_norm_bias,

R_W_TYPE recurrent_weight, 
R_B_TYPE recurrent_bias,
R_NORM_W_TYPE recurrent_norm_weight, 
R_NORM_B_TYPE recurrent_norm_bias,

O_W_TYPE output_weight, 
O_B_TYPE output_bias,
O_NORM_W_TYPE output_norm_weight, 
O_NORM_B_TYPE output_norm_bias,

unsigned int seq_len
)
{
    R_TYPE state_buffer[RECURRENT_DIM];
    #pragma unroll RECURRENT_DIM
    for(auto i=0; i<RECURRENT_DIM; i++)
        {state_buffer[i] = 0;}

    adjust_stream_width<(INPUT_DIM*I_TYPE::width)/in_stream_width>(wa_input_stream, input_stream, seq_len);
    
    // #pragma disable_loop_pipelining
    for (auto i=0; i<seq_len; i++)
    {
        Linear<INPUT_DIM, RECURRENT_DIM, INPUT_SIMD, INPUT_PE, INPUT_BIAS, ACC_BUFFER_TYPE, I_TYPE>
        (ig_to_act_stream, wa_input_stream, 
         input_weight, input_bias, 
         input_norm_weight, input_norm_bias);

        Linear_Buffer<RECURRENT_DIM, RECURRENT_DIM, RECURRENT_SIMD, INPUT_PE, 
        RECURRENT_BIAS, ACC_BUFFER_TYPE, R_TYPE>
        (rg_to_act_stream, state_buffer,
         recurrent_weight, recurrent_bias, 
         recurrent_norm_weight, recurrent_norm_bias);

        TanH<RECURRENT_DIM, INPUT_PE, R_TYPE, ACC_BUFFER_TYPE>
        (rec_to_out_stream, state_buffer, ig_to_act_stream, rg_to_act_stream);

        Linear<RECURRENT_DIM, OUTPUT_DIM, OUTPUT_SIMD, OUTPUT_PE, OUTPUT_BIAS, ACC_BUFFER_TYPE, R_TYPE>
        (output_layer_stream, rec_to_out_stream, 
         output_weight, output_bias, 
         output_norm_weight, output_norm_bias);
    }
    adjust_stream_width<OUTPUT_DIM/O_PE>(output_stream, output_layer_stream, seq_len);
}


#endif
