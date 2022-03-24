import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import geotorch
from geotorch.so import torus_init_

from utils import *

BIAS = False

class QExpRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, config):
        super(QExpRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_kernel = Quantized_Linear(config.wb, input_size, hidden_size, bias=BIAS)
        self.recurrent_kernel = Quantized_Linear(config.rb, hidden_size, hidden_size, bias=BIAS)
        self.nonlinearity = qmodrelu(hidden_size, config.ab)

        # Make recurrent_kernel orthogonal
        geotorch.orthogonal(self.recurrent_kernel, "weight")
        self.reset_parameters()

    def reset_parameters(self):

        nn.init.uniform_(self.input_kernel.weight, -1, 1)
        def init_(x):
            x.uniform_(0.0, math.pi / 2.0)
            c = torch.cos(x.data)
            x.data = -torch.sqrt((1.0 - c) / (1.0 + c))
        K = self.recurrent_kernel
        K.weight = torus_init_(K.weight, init_=init_)

    def default_hidden(self, input_):
        return input_.new_zeros(input_.size(0), self.hidden_size, requires_grad=False)

    def forward(self, input_, hidden):
        input_ = self.input_kernel(input_)
        hidden_states = []
        with geotorch.parametrize.cached():
            for inputs in torch.unbind(input_, dim=1):
                hidden = self.recurrent_kernel(hidden)
                hidden = inputs + hidden
                hidden = self.nonlinearity(hidden)
                hidden_states.append(hidden)
        out = torch.stack(hidden_states, axis=1)
        return out


class QORNN_Model(nn.Module):
    def __init__(self, input_size, sequence_length, hidden_size, output_size, config):
        super(QORNN_Model, self).__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn = QExpRNNCell(input_size, hidden_size, config)
        self.lin = Quantized_Linear(config.wb, hidden_size, output_size, bias=BIAS)
        self.ibw = config.ib
        self.config = config

    def forward(self, inputs):
        inputs = Quantize(inputs, self.ibw)
        hidden_init = self.rnn.default_hidden(torch.zeros(inputs.size(0), self.hidden_size).type(inputs.type()))
        out_rnn = self.rnn(inputs, hidden_init)
        out = self.lin(out_rnn[:,-1,:])
        return out


    def correct(self, pred, target):
        pred = torch.argmax(pred, dim=1)
        acc = torch.mean((target == pred).float())
        return acc

    def export(self, path):
        params = {}
        
        # input weight
        input_weights = Quantize(self.rnn.input_kernel.weight.cpu().detach(), self.config.wb).numpy()
        input_weights = np.concatenate((input_weights, np.zeros((input_weights.shape[0], 3))), axis=-1)
        params[f"input_weight0{self.config.wb}"] = input_weights
        if self.rnn.input_kernel.bias is not None:
            params[f"input_bias0{max(B_MAX_BW, self.config.wb)}"] = Quantize(
                self.rnn.input_kernel.bias.cpu().detach(), max(B_MAX_BW, self.config.wb)).numpy()

        # recurrent weight
        params[f"recurrent_weight0{self.config.rb}"] = Quantize(
            self.rnn.recurrent_kernel.weight.cpu().detach(), self.config.rb).numpy()
        if self.rnn.recurrent_kernel.bias is not None:
            params[f"recurrent_bias0{max(B_MAX_BW, self.config.rb)}"] = Quantize(
                self.rnn.recurrent_kernel.bias.cpu().detach(), max(B_MAX_BW, self.config.rb)).numpy()
        
        # output weight
        output_weights = Quantize(self.lin.weight.cpu().detach(), self.config.wb).numpy()
        zeros = np.zeros((6,output_weights.shape[1]))
        output_weights = np.append(output_weights, zeros, axis=0)
        params[f"output_weight0{self.config.wb}"] = output_weights
        if self.lin.bias is not None:
            params[f"output_bias0{max(B_MAX_BW, self.config.wb)}"] = Quantize(
                self.lin.bias.cpu().detach(), max(B_MAX_BW, self.config.rb)).numpy()
        
        # bias of mode relu
        if isinstance(self.rnn.nonlinearity, qmodrelu):
            params[f"relu_bias0{self.config.ab}"] = Quantize(
                self.rnn.nonlinearity.b.cpu().detach(), self.config.ab).numpy()

        incl_param_string = "#pragma once\n"
        # write weights
        for key in params.keys():
            c_array = self.numpy_to_c(str(key[:-2]), int(key[-2:]), params[key])
            with open(path + "/" + str(key[:-2]) + ".h", 'w') as f:
                f.write(c_array)
                incl_param_string += "#include\"{}\"\n".format(str(key[:-2]) + ".h")
        
        # write defines.h
        with open(path + "/" + "defines.h", "w") as f:
            f.write("#pragma once\n")
            f.write("#define MAX(x, y) (((x) > (y)) ? (x) : (y))\n")
            f.write("#define INPUT_BW {}\n".format(self.config.ib))
            f.write("#define BUFFER_BW {}\n".format(16))
            f.write("#define RECURRENT_BW {}\n".format(self.config.ab))
            f.write("#define IN_DIM {}\n".format(self.input_size+3))
            f.write("#define REC_DIM {}\n".format(self.hidden_size))
            f.write("#define NUM_CLASSES {}\n".format(16))
            f.write("#define I_SIMD {}\n".format(1))
            f.write("#define I_PE {}\n".format(4))
            f.write("#define R_SIMD {}\n".format(8))
            f.write("#define O_SIMD I_PE\n")
            f.write("#define O_PE {}\n".format(4))          
            f.write("#define MAX_BUFFER_SIZE MAX(I_SIMD*INPUT_BW, R_SIMD*RECURRENT_BW)\n")

        # write params_include files
        incl_param_string += "typedef ac_fixed<RECURRENT_BW,4,true,AC_RND,AC_SAT> recurrent_type;\n"
        incl_param_string += "typedef ac_fixed<INPUT_BW,1,true,AC_RND,AC_SAT> input_type;\n"
        incl_param_string += "typedef ac_fixed<BUFFER_BW,6,true, AC_RND, AC_SAT> accumulator_type;\n"
        incl_param_string += "typedef accumulator_type output_type;\n"
        with open(path + "/" + "include_params.h", "w") as f:
            f.write(incl_param_string)

    def numpy_to_c(self, var_name, bit_width, np_array):
        if np_array is None:
            c_array = "const ac_fixed<4,1,true,AC_RND,AC_SAT> {}[1] = {{0}};".format(var_name)
            return c_array   
        max_val = max(0.5, np.abs(np_array).max())
        i_bw = max(math.floor(math.log2(max_val))+1, 1)
        signedness = "true" if (np_array < 0).any() else "false"
        c_array = "const ac_fixed<{},{},{},AC_RND,AC_SAT> {}".format(bit_width, i_bw, signedness, var_name)
        c_array += ''.join(["[{}]".format(np_array.shape[i]) for i in range(np_array.ndim)])
        string_list = [" = "]
        self.matrix_to_string_list(np_array, np_array.ndim - 1, string_list)
        c_array += ''.join(string_list)
        c_array +=";\n"
        return c_array 

    def matrix_to_string_list(self, numpy_array, dims, string_list):
        string_list.append("{ ")
        for i in range(numpy_array.shape[0]):
            if dims > 0:
                self.matrix_to_string_list(numpy_array[i], dims - 1, string_list)
            else:
                string_list.append(str(numpy_array[i]))
            if i < numpy_array.shape[0] - 1:
                string_list.append(", \n")
            else:
                string_list.append("\n")
        string_list.append("}\n")
        return

