#pragma once


#ifndef ACT_H
#define ACT_H

template
<
unsigned int INPUT_DIM,
unsigned int PE,
typename O_TYPE,
typename I_TYPE,
typename S_BUFFER_TYPE,
int out_stream_width, 
int in_stream_width,
int OUT_DEPTH, 
int IN_DEPTH
>
void TanH
(
ihc::stream<ac_int<out_stream_width, false>, ihc::buffer<OUT_DEPTH>> &output_stream, 
S_BUFFER_TYPE &state_buffer, 
ihc::stream<ac_int<in_stream_width, false>, ihc::buffer<IN_DEPTH>> &input_stream1, 
ihc::stream<ac_int<in_stream_width, false>, ihc::buffer<IN_DEPTH>> &input_stream2
)
{
	const unsigned int TOTAL_ITER = INPUT_DIM/PE;
	#pragma ii 1
	for(auto i=0; i<TOTAL_ITER; i++)
	{
		O_TYPE act[PE];
		ac_int<in_stream_width, false> in1 = input_stream1.read();
		ac_int<in_stream_width, false> in2 = input_stream2.read();
		
		I_TYPE *infx1 = reinterpret_cast<I_TYPE*>(&in1);
		I_TYPE *infx2 = reinterpret_cast<I_TYPE*>(&in2);

		#pragma unroll PE
		for(auto j=0; j<PE; j++)
		{
			I_TYPE sum = infx1[j]+infx2[j];
			act[j] = tanh_lut<O_TYPE, I_TYPE>(sum);
		    state_buffer[i*PE+j] = act[j];
		}
        ac_int<out_stream_width> act_packed = *reinterpret_cast<ac_int<out_stream_width,false>*>(act);
        output_stream.write(act_packed);

	}
}
#endif