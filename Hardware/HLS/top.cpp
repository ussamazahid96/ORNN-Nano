#include "QORNN_W4R4/defines.h"

#include "HLS/hls.h"
#include "HLS/ac_int.h"
#include "HLS/ac_fixed.h"

// intermediate streams need to be declared outside the component i.e., globally!
ihc::stream<ac_int<64, false>, ihc::buffer<(IN_DIM*IN_DIM*INPUT_BW)/64>>   input_stream;
ihc::stream<ac_int<I_SIMD*INPUT_BW,   false>, ihc::buffer<(IN_DIM*IN_DIM)/I_SIMD>> wa_input_stream;
ihc::stream<ac_int<I_PE*BUFFER_BW, false>, ihc::buffer<REC_DIM/I_PE>>  ig_to_act_stream;
ihc::stream<ac_int<I_PE*BUFFER_BW, false>, ihc::buffer<REC_DIM/I_PE>>  rg_to_act_stream;
ihc::stream<ac_int<I_PE*RECURRENT_BW, false>, ihc::buffer<REC_DIM/I_PE>>  rec_to_out_stream;
ihc::stream<ac_int<O_PE*BUFFER_BW, false>, ihc::buffer<(NUM_CLASSES*IN_DIM)/O_PE>>  output_layer_stream;
ihc::stream<ac_int<64, false>, ihc::buffer<(NUM_CLASSES*IN_DIM*BUFFER_BW)/64>> output_stream;

#include "QORNN.h"
#include "QORNN_W4R4/include_params.h"

hls_avalon_agent_component component
void ACCL_TOP(
                ihc::mm_master<
                                ac_int<64, false>, 
                                ihc::aspace<1>,
                                ihc::awidth<32>, 
                                ihc::dwidth<8*sizeof(ac_int<64, false>)>, 
                                ihc::align<sizeof(ac_int<64, false>)>,
                                ihc::latency<0>, 
                                ihc::maxburst<8>,
                                ihc::waitrequest<true>
                                >  hls_avalon_agent_register_argument &input_buffer,
                ihc::mm_master<
                                ac_int<64, false>, 
                                ihc::aspace<1>,
                                ihc::awidth<32>, 
                                ihc::dwidth<8*sizeof(ac_int<64, false>)>, 
                                ihc::align<sizeof(ac_int<64, false>)>,
                                ihc::latency<0>, 
                                ihc::maxburst<8>,
                                ihc::waitrequest<true>
                                >  hls_avalon_agent_register_argument &output_buffer,

                hls_avalon_agent_register_argument unsigned int seq_len
            ) 
{

    ddr_to_stream<64, (IN_DIM*INPUT_BW)/8>(input_stream, input_buffer, seq_len);

    RNN_Layer
    <
    IN_DIM, REC_DIM, NUM_CLASSES,0, I_SIMD, R_SIMD, I_PE, O_SIMD, O_PE, I_BIAS, R_BIAS, O_BIAS, 
    recurrent_type, input_type, accumulator_type>
    (output_stream, input_stream, 
    input_weight, input_bias, input_norm_weight, input_norm_bias,
    recurrent_weight, recurrent_bias, recurrent_norm_weight, recurrent_norm_bias,
    output_weight, output_bias, 
    output_norm_weight, output_norm_bias,
    seq_len);

    stream_to_ddr<64, (NUM_CLASSES*BUFFER_BW)/8> (output_buffer, output_stream, seq_len);
}