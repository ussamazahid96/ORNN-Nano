#pragma once

#ifndef STREAM_UTILS_H
#define STREAM_UTILS_H

template
<
unsigned int NUM_IN_WORDS,
int OUT_WIDTH, 
int IN_WIDTH,
int OUT_DEPTH,
int IN_DEPTH
>
void adjust_stream_width
(
ihc::stream<ac_int<OUT_WIDTH, false>, ihc::buffer<OUT_DEPTH>> &output_stream, 
ihc::stream<ac_int<IN_WIDTH, false>, ihc::buffer<IN_DEPTH>>  &input_stream, 
unsigned int seq_len
)
{
    if(IN_WIDTH > OUT_WIDTH)
    {
        unsigned int counter = 0;
        const unsigned int OUT_PER_IN = IN_WIDTH/OUT_WIDTH;
        ac_int<IN_WIDTH, false> element_in = 0;
        #pragma ii 1
        for(auto i=0; i < OUT_PER_IN*NUM_IN_WORDS*seq_len; i++)
        {   
            if(counter == 0)
                element_in = input_stream.read();
            ac_int<OUT_WIDTH, false> element_out = element_in;
            element_in = element_in >> OUT_WIDTH;
            output_stream.write(element_out);
            if(++counter==OUT_PER_IN)
                counter = 0;
        }

    }
    else if (IN_WIDTH == OUT_WIDTH)
    {
        #pragma ii 1
        for(auto i=0; i<NUM_IN_WORDS*seq_len; i++)
        {
            ac_int<IN_WIDTH, false> element_in = input_stream.read();
            output_stream.write(element_in);
        }
    }
    else
    {
        const unsigned int IN_PER_OUT = OUT_WIDTH/IN_WIDTH;
        unsigned int counter = 0;
        ac_int<OUT_WIDTH, false> element_out = 0;
        #pragma ii 1
        for(auto i=0; i<NUM_IN_WORDS*seq_len; i++)
        {
            ac_int<OUT_WIDTH, false> element_in = input_stream.read();
            element_out = element_out >> IN_WIDTH;
            element_out = element_out | (element_in << (OUT_WIDTH-IN_WIDTH));
            if(++counter == IN_PER_OUT)
            {
                counter = 0;
                output_stream.write(element_out);
            }
        }


    }
}

#endif