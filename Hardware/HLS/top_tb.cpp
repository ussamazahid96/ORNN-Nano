#include <iostream>
#include "assert.h"
#include <fstream>
#include "top.cpp"

using namespace std;

int main() 
{
    string test;
    ifstream in_file("../../Training/export/QORNN_W4R4/test_image.txt", ios::in);
    ac_fixed<8, 1> arr[560];
    for(unsigned int i=0; i < 560; i++)
    {
        getline(in_file, test);
        arr[i] = atof(test.c_str());
    } 
    int seq_len = 560/16;   
    ac_int<64, false> *packed = reinterpret_cast<ac_int<64, false>*>(arr);
    ac_int<64, false> output[NUM_CLASSES*seq_len*BUFFER_BW/64];
    for(auto i=0; i<NUM_CLASSES*seq_len*BUFFER_BW/64;i++)
    {
        output[i] = 0;
    }


    ihc::mm_master<
                   ac_int<64, false>, 
                   ihc::aspace<1>,
                   ihc::awidth<32>, 
                   ihc::dwidth<8*sizeof(ac_int<64, false>)>, 
                   ihc::align<sizeof(ac_int<64, false>)>,
                   ihc::latency<0>, 
                   ihc::maxburst<8>,
                   ihc::waitrequest<true>
                  >  mm_input(packed, sizeof(ac_int<64, false>)*560*8/64);
    ihc::mm_master<
                   ac_int<64, false>, 
                   ihc::aspace<1>,
                   ihc::awidth<32>, 
                   ihc::dwidth<8*sizeof(ac_int<64, false>)>, 
                   ihc::align<sizeof(ac_int<64, false>)>,
                   ihc::latency<0>, 
                   ihc::maxburst<8>,
                   ihc::waitrequest<true>
                  >  mm_output(output, sizeof(ac_int<64, false>)*NUM_CLASSES*seq_len*BUFFER_BW/64); 

    ACCL_TOP(mm_input, mm_output, seq_len);

    output_type *packed_out = reinterpret_cast<output_type*>(output);
    
    int max_val;
    for(auto i=0; i < seq_len; i++)
    {
        max_val = 0;
        output_type max_score = packed_out[i*NUM_CLASSES];
        for(auto j=0; j<NUM_CLASSES; j++)
        {
            if(j < 10)
            {
                if (packed_out[i*NUM_CLASSES+j] >= max_score)
                {
                    max_score  = packed_out[i*NUM_CLASSES+j];
                    max_val = j;
                }
            }
        }
    }
    std::cout << "Pred = " << max_val << '\n';
    return 0;
}





