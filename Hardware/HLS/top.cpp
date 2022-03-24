#include "QORNN_W4R4/defines.h"

#include "HLS/hls.h"
#include "HLS/ac_int.h"
#include "HLS/ac_fixed.h"

// intermediate streams need to be declared outside the component i.e., globally!
// interface streams
ihc::stream<ac_int<64, false>, ihc::buffer<(IN_DIM*INPUT_BW*35)/64>>   input_stream;
ihc::stream<ac_int<64, false>, ihc::buffer<(NUM_CLASSES*35*BUFFER_BW)/64>> output_stream;

// top internal streams
ihc::stream<ac_int<IN_DIM*INPUT_BW, false>, ihc::buffer<35>>   column_stream;
ihc::stream<ac_int<I_SIMD*INPUT_BW, false>, ihc::buffer<IN_DIM*35/I_SIMD>> wa_input_stream;

// recurrent cell streams
ihc::stream<ac_int<I_PE*BUFFER_BW, false>, ihc::buffer<REC_DIM/I_PE>>  ig_to_act_stream;
ihc::stream<ac_int<I_PE*BUFFER_BW, false>, ihc::buffer<REC_DIM/I_PE>>  rg_to_act_stream;
ihc::stream<ac_int<I_PE*RECURRENT_BW, false>, ihc::buffer<REC_DIM/I_PE>>  rec_to_out_stream;

// output layer stream
ihc::stream<ac_int<O_PE*BUFFER_BW, false>, ihc::buffer<(NUM_CLASSES*35)/O_PE>>  output_layer_stream;


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
    adjust_stream_width<IN_DIM*INPUT_BW/64>(column_stream, input_stream, seq_len);

    // RNN layer No Bias No Batchnorm
    RNN_Layer_NBNB
    <
    IN_DIM, REC_DIM, NUM_CLASSES, I_SIMD, R_SIMD, I_PE, O_SIMD, O_PE, 
    recurrent_type, input_type, accumulator_type>
    (output_stream, column_stream, 
    input_weight, recurrent_weight, relu_bias, output_weight,
    seq_len);

    stream_to_ddr<64, (NUM_CLASSES*BUFFER_BW)/8> (output_buffer, output_stream, seq_len);
}