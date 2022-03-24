#pragma once

#ifndef RNN_H
#define RNN_H

template
<
unsigned int INPUT_DIM,
unsigned int RECURRENT_DIM,
unsigned int OUTPUT_DIM,

unsigned int INPUT_SIMD,
unsigned int RECURRENT_SIMD,
unsigned int INPUT_PE,

unsigned int OUTPUT_SIMD,
unsigned int OUTPUT_PE,

typename R_TYPE,
typename I_TYPE,
typename ACC_BUFFER_TYPE,

typename I_W_TYPE, 
typename R_W_TYPE,
typename R_B_TYPE, 
typename O_W_TYPE, 

int in_stream_width, 
int out_stream_width,
int OUT_DEPTH,
int IN_DEPTH
>
void RNN_Layer_NBNB
(
ihc::stream<ac_int<out_stream_width, false>, ihc::buffer<OUT_DEPTH>> &output_stream, 
ihc::stream<ac_int<in_stream_width, false>, ihc::buffer<IN_DEPTH>> &input_stream, 

I_W_TYPE input_weight, 
R_W_TYPE recurrent_weight,
R_B_TYPE relu_bias, 
O_W_TYPE output_weight, 

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
        Linear<INPUT_DIM, RECURRENT_DIM, INPUT_SIMD, INPUT_PE, ACC_BUFFER_TYPE, I_TYPE>
        (ig_to_act_stream, wa_input_stream, input_weight);
       
        Linear_Buffer<RECURRENT_DIM, RECURRENT_DIM, RECURRENT_SIMD, INPUT_PE, 
        ACC_BUFFER_TYPE, R_TYPE>
        (rg_to_act_stream, state_buffer,
         recurrent_weight);

        Mod_ReLU<RECURRENT_DIM, INPUT_PE, R_TYPE, ACC_BUFFER_TYPE>
        (rec_to_out_stream, state_buffer, ig_to_act_stream, rg_to_act_stream, relu_bias);

        Linear<RECURRENT_DIM, OUTPUT_DIM, OUTPUT_SIMD, OUTPUT_PE, ACC_BUFFER_TYPE, R_TYPE>
        (output_layer_stream, rec_to_out_stream, 
         output_weight);
    }
    adjust_stream_width<OUTPUT_DIM/OUTPUT_PE>(output_stream, output_layer_stream, seq_len);
}


#endif
