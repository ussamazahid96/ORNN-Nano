#include <iostream>
#include "assert.h"
#include "top.cpp"
#include "mnist_parser.h"

#define SEQ_LEN 32

int main() 
{
    const std::string images_path = "utils/mnist_data/t10k-images.idx3-ubyte";
    const std::string labels_path = "utils/mnist_data/t10k-labels.idx1-ubyte";
    std::vector<vec_t> images;
    std::vector<label_t> labels;
    parse_mnist_images(images_path, &images, -1.0, 1.0, 2, 2);
    parse_mnist_labels(labels_path, &labels);
    int image_no = 0;
    vec_t one_test_image = images[image_no];
    label_t one_test_label = labels[image_no];
    

    ac_int<64, false> input_image[IN_DIM*IN_DIM*INPUT_BW/64];
    ac_int<64, false> output_image[NUM_CLASSES*IN_DIM*BUFFER_BW/64];
    int index = 0;
    ac_int<64, false> packed = 0;
    for(auto i=0; i< IN_DIM*IN_DIM; i++)
    {
        ac_fixed<4, 1, true, AC_RND, AC_SAT> val = one_test_image[i];
        ac_int<4, true> val_int = *reinterpret_cast<ac_int<4, true>*>(&val);
        packed.set_slc((i*4)%64, val_int);
        if((i+1)%16 == 0)
        {
            input_image[index] = packed;
            index++;
        }
    }   

    // return 0;
    ihc::mm_master<
                   ac_int<64, false>, 
                   ihc::aspace<1>,
                   ihc::awidth<32>, 
                   ihc::dwidth<8*sizeof(ac_int<64, false>)>, 
                   ihc::align<sizeof(ac_int<64, false>)>,
                   ihc::latency<0>, 
                   ihc::maxburst<8>,
                   ihc::waitrequest<true>
                  >  mm_input(input_image, sizeof(ac_int<64, false>)*IN_DIM*IN_DIM*INPUT_BW/64); 
    ihc::mm_master<
                   ac_int<64, false>, 
                   ihc::aspace<1>,
                   ihc::awidth<32>, 
                   ihc::dwidth<8*sizeof(ac_int<64, false>)>, 
                   ihc::align<sizeof(ac_int<64, false>)>,
                   ihc::latency<0>, 
                   ihc::maxburst<8>,
                   ihc::waitrequest<true>
                  >  mm_output(output_image, sizeof(ac_int<64, false>)*NUM_CLASSES*IN_DIM*BUFFER_BW/64); 

    ACCL_TOP(mm_input, mm_output, SEQ_LEN);

    output_type *output = reinterpret_cast<output_type*>(output_image);
        int max_val;
    for(auto i=0; i < IN_DIM; i++)
    {
        max_val = 0;
        output_type max_score = output[i*NUM_CLASSES];
        for(auto j=0; j<NUM_CLASSES; j++)
        {
            if(j < 10)
            {
                if (output[i*NUM_CLASSES+j] >= max_score)
                {
                    max_score  = output[i*NUM_CLASSES+j];
                    max_val = j;
                }
            }
        }
    }
    std::cout << "Label = " << one_test_label << "\tPred = " << max_val << '\n';
    return 0;
}





