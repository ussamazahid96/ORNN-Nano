#pragma once

#ifndef LINEAR_H
#define LINEAR_H
#include "HLS/math_dsp_control.h"

template
<
unsigned int INPUT_DIM, 
unsigned int OUT_DIM,
unsigned int INPUT_SIMD,
unsigned int INPUT_PE, 
bool BIAS, 
typename O_TYPE, 
typename I_TYPE, 
typename W_TYPE, 
typename B_TYPE,
typename NORM_W_TYPE, 
typename NORM_B_TYPE,
int in_stream_width, 
int out_stream_width,
int OUT_DEPTH,
int IN_DEPTH
>
void Linear
(
ihc::stream<ac_int<out_stream_width, false>, ihc::buffer<OUT_DEPTH>> &output_stream, 
ihc::stream<ac_int<in_stream_width, false>, ihc::buffer<IN_DEPTH>> &input_stream, 
W_TYPE weight, 
B_TYPE bias,
NORM_W_TYPE norm_weight, 
NORM_B_TYPE norm_bias
)
{
    ASSERT(INPUT_SIMD*I_TYPE::width <= MAX_BUFFER_SIZE, "Buffer Overflow.");

    // output and input tiling
    const unsigned int OT = OUT_DIM / INPUT_PE;
    const unsigned int IT = INPUT_DIM / INPUT_SIMD;
    const unsigned int TOTAL_FOLD = OT*IT;
    unsigned int ot=0, it=0;

    ac_int<in_stream_width, false> inputBuf[IT];
    O_TYPE accumulator[INPUT_PE], n_accumulator[INPUT_PE];

    #pragma ii 1
    for(auto f=0; f < TOTAL_FOLD; f++)
    {
        ac_int<MAX_BUFFER_SIZE, false> inElem;
        if(ot == 0)
        {
            inElem = input_stream.read();
            inputBuf[it] = inElem;
        }
        else
            {inElem = inputBuf[it];}

        if(it == 0)
        {
            #pragma unroll INPUT_PE
            for(auto i = 0; i<INPUT_PE; i++)
            {
                accumulator[i] = BIAS ? bias[ot*INPUT_PE+i] : 0;
                n_accumulator[i] = 0;
            }    
        }

        #pragma unroll INPUT_PE
        for(auto j=0; j<INPUT_PE; j++)
        {

            O_TYPE res = 0;
            #pragma unroll INPUT_SIMD
            for (auto i=0; i<INPUT_SIMD; i++)
            {
                // ihc::math_dsp_control<ihc::Preference::DSP>([&]{ 
                ac_int<I_TYPE::width, false> input_temp = inElem.slc<I_TYPE::width>(I_TYPE::width*i);
                I_TYPE input = *reinterpret_cast<I_TYPE*>(&input_temp);
                auto dot_p = input*weight[ot*INPUT_PE+j][it*INPUT_SIMD+i];
                res += dot_p;
                // });
            }
            accumulator[j] += res;
        }

        if(++it == IT)
        {
            #pragma unroll INPUT_PE
            for(auto i=0; i< INPUT_PE; i++)
            {
                // ihc::math_dsp_control<ihc::Preference::DSP>([&]{ 
                O_TYPE temp = accumulator[i]*norm_weight[0];
                n_accumulator[i] = norm_bias[0]+temp;
                // });
            }
            ac_int<out_stream_width> acc_packed = *reinterpret_cast<ac_int<out_stream_width,false>*>(&n_accumulator);
            output_stream.write(acc_packed);
            it = 0;
            if(++ot == OT)
                {ot = 0;}
        }
    }
}


template
<
unsigned int INPUT_DIM, 
unsigned int OUT_DIM,
unsigned int INPUT_SIMD,
unsigned int INPUT_PE, 
bool BIAS, 
typename O_TYPE, 
typename I_TYPE, 
typename W_TYPE, 
typename B_TYPE,
typename NORM_W_TYPE, 
typename NORM_B_TYPE,
typename S_BUFFER_TYPE,
int out_stream_width,
int OUT_DEPTH
>
void Linear_Buffer
(
ihc::stream<ac_int<out_stream_width, false>, ihc::buffer<OUT_DEPTH>> &output_stream, 
S_BUFFER_TYPE &state_buffer,
W_TYPE weight, 
B_TYPE bias,
NORM_W_TYPE norm_weight, 
NORM_B_TYPE norm_bias
)
{
    ASSERT(INPUT_SIMD*I_TYPE::width <= MAX_BUFFER_SIZE, "Buffer Overflow.");

    // output and input tiling
    const unsigned int OT = OUT_DIM / INPUT_PE;
    const unsigned int IT = INPUT_DIM / INPUT_SIMD;
    const unsigned int TOTAL_FOLD = OT*IT;
    unsigned int ot=0, it=0;

    ac_int<INPUT_SIMD*I_TYPE::width, false> inputBuf[IT];
    O_TYPE accumulator[INPUT_PE], n_accumulator[INPUT_PE];

    #pragma ii 1
    for(auto f=0; f < TOTAL_FOLD; f++)
    {
        I_TYPE inElem[INPUT_SIMD];
        #pragma unroll
        for(auto k=0;k<INPUT_SIMD;k++)
            {inElem[k] = state_buffer[it*INPUT_SIMD+k];}
        
        if(it == 0)
        {
            #pragma unroll INPUT_PE
            for(auto i = 0; i<INPUT_PE; i++)
            {
                accumulator[i] = BIAS ? bias[ot*INPUT_PE+i] : 0;
                n_accumulator[i] = 0;
            }    
        }

        #pragma unroll INPUT_PE
        for(auto j=0; j<INPUT_PE; j++)
        {

            O_TYPE res = 0;
            #pragma unroll INPUT_SIMD
            for (auto i=0; i<INPUT_SIMD; i++)
            {
                // ihc::math_dsp_control<ihc::Preference::DSP>([&]{ 
                res += (inElem[i]*weight[ot*INPUT_PE+j][it*INPUT_SIMD+i]);
                // });
            }
            accumulator[j] += res;
        }

        if(++it == IT)
        {
            #pragma unroll INPUT_PE
            #pragma ivdep
            for(auto i=0; i< INPUT_PE; i++)
            {
                // ihc::math_dsp_control<ihc::Preference::Softlogic>([&]{ 
                O_TYPE temp = accumulator[i]*norm_weight[0];
                n_accumulator[i] = norm_bias[0]+temp;
                // }); 
            }
            ac_int<out_stream_width> acc_packed = *reinterpret_cast<ac_int<out_stream_width,false>*>(&n_accumulator);
            output_stream.write(acc_packed);
            it = 0;
            if(++ot == OT)
                {ot = 0;}
        }
    }
}

#endif

