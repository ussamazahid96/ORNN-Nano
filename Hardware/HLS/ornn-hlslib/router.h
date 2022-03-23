#pragma once

#ifndef ROUTER_H
#define ROUTER_H

template
<
unsigned int DATAWIDTH,
unsigned int NUM_BYTES,
int DEPTH
>
void ddr_to_stream
(
ihc::stream<ac_int<DATAWIDTH, false>, ihc::buffer<DEPTH>> &output_stream,
ac_int<DATAWIDTH, false> *input_buffer,
unsigned int seq_len
)
{
    const unsigned int NUM_WORDS = NUM_BYTES / sizeof(ac_int<64, false>);
    #pragma ii 1
    for(auto j=0; j< NUM_WORDS*seq_len; j++)
    {
        ac_int<DATAWIDTH, false> temp = input_buffer[j];
        output_stream.write(temp); 
    }
}


template
<
unsigned int DATAWIDTH,
unsigned int NUM_BYTES,
int DEPTH
>
void stream_to_ddr
(
ac_int<DATAWIDTH, false> *output_buffer,
ihc::stream<ac_int<DATAWIDTH, false>, ihc::buffer<DEPTH>> &input_stream,
unsigned int seq_len
)
{
    const unsigned int NUM_WORDS = NUM_BYTES / sizeof(ac_int<64, false>);
    #pragma ii 1
    for(auto j=0; j< NUM_WORDS*seq_len; j++)
    {
        ac_int<DATAWIDTH, false> temp = input_stream.read();
        output_buffer[j] = temp; 
    }
}

#endif