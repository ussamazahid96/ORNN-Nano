import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import geotorch
from geotorch.so import torus_init_

from utils import *
from layernorm import *


class QExpRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, config):
        super(QExpRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_kernel = Quantized_Linear(config.rb, hidden_size, hidden_size, bias=True)
        self.input_kernel = Quantized_Linear(config.wb, input_size, hidden_size, bias=True)
        self.norm_input = LayerNorm()
        self.norm_recurrent = LayerNorm()
        self.dropout = nn.Dropout(0.1)
        self.nonlinearity = nn.Tanh()

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
        hidden = self.recurrent_kernel(hidden)
        hidden = self.dropout(hidden)
        input_ = self.norm_input(input_)
        hidden = self.norm_recurrent(hidden)
        new_state = input_ + hidden
        new_state = self.nonlinearity(new_state)
        return new_state


class QORNN_Model(nn.Module):
    def __init__(self, input_size, sequence_length, hidden_size, output_size, config):
        super(QORNN_Model, self).__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn = QExpRNNCell(input_size, hidden_size, config)
        self.lin = Quantized_Linear(config.wb, hidden_size, output_size, bias=False)
        self.norm = LayerNorm()
        self.ibw = config.ib
        self.config = config

    def forward(self, inputs):
        inputs = inputs.view(-1, self.sequence_length, self.input_size)
        inputs = Quantize(inputs, self.ibw)
        out_rnn = self.rnn.default_hidden(torch.zeros(inputs.size(0), self.hidden_size).type(inputs.type()))
        with geotorch.parametrize.cached():
            for input in torch.unbind(inputs, dim=1):
                out_rnn = self.rnn(input, out_rnn)
                out = self.lin(out_rnn)
                out = self.norm(out)
        return out


    def correct(self, pred, target):
        pred = pred.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        acc = correct/target.size(0)
        return acc

    def export(self, path):
        params = {}
        params[f"input_weight0{self.config.wb}"] = Quantize(self.rnn.input_kernel.weight.cpu().detach(), self.config.wb).numpy()
        if self.rnn.input_kernel.bias is not None:
            params[f"input_bias0{max(4, self.config.wb)}"] = Quantize(self.rnn.input_kernel.bias.cpu().detach(), max(4, self.config.wb)).numpy()
        else:
            params[f"input_bias0{max(4, self.config.wb)}"] = None
        mean = self.rnn.norm_input.running_mean.numpy()
        inv_std = np.array(1./np.sqrt(self.rnn.norm_input.running_var + self.rnn.norm_input.eps))
        weight = self.rnn.norm_input.weight.cpu().detach().numpy()
        bias = self.rnn.norm_input.bias.cpu().detach().numpy()

        params[f"input_norm_weight"+str(16)] = weight*inv_std
        params[f"input_norm_bias"+str(16)]   = (bias-mean*(weight*inv_std))

        params[f"recurrent_weight0{self.config.rb}"] = Quantize(self.rnn.recurrent_kernel.weight.cpu().detach(), self.config.rb).numpy()
        if self.rnn.recurrent_kernel.bias is not None:
            params[f"recurrent_bias0{max(4, self.config.rb)}"] = Quantize(self.rnn.recurrent_kernel.bias.cpu().detach(), max(4, self.config.rb)).numpy()
        else:
            params[f"recurrent_bias0{max(4, self.config.rb)}"] = None
        mean = self.rnn.norm_recurrent.running_mean.numpy()
        inv_std = np.array(1./np.sqrt(self.rnn.norm_recurrent.running_var + self.rnn.norm_recurrent.eps))
        weight = self.rnn.norm_recurrent.weight.cpu().detach().numpy()
        bias = self.rnn.norm_recurrent.bias.cpu().detach().numpy()

        params[f"recurrent_norm_weight"+str(16)] = weight*inv_std
        params[f"recurrent_norm_bias"+str(16)]   = (bias-mean*(weight*inv_std))

        output_weights = Quantize(self.lin.weight.cpu().detach(), self.config.wb).numpy()
        zeros = np.zeros((6,output_weights.shape[1]))
        output_weights = np.append(output_weights, zeros, axis=0)
        params[f"output_weight0{self.config.wb}"] = output_weights
        if self.lin.bias is not None:
            params[f"output_bias0{max(4, self.config.wb)}"] = Quantize(self.lin.bias.cpu().detach(), max(4, self.config.rb)).numpy()
        else:
            params[f"output_bias0{max(4, self.config.wb)}"] = None
        
        mean = self.norm.running_mean.numpy()
        inv_std = np.array(1./np.sqrt(self.norm.running_var + self.norm.eps))
        weight = self.norm.weight.cpu().detach().numpy()
        bias = self.norm.bias.cpu().detach().numpy()

        params[f"output_norm_weight"+str(16)] = weight*inv_std
        params[f"output_norm_bias"+str(16)]   = (bias-mean*(weight*inv_std))
        

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
            f.write("#define RECURRENT_BW {}\n".format(8))
            f.write("#define IN_DIM {}\n".format(self.input_size))
            f.write("#define REC_DIM {}\n".format(self.hidden_size))
            f.write("#define NUM_CLASSES {}\n".format(16))
            f.write("#define I_SIMD {}\n".format(1))
            f.write("#define I_PE {}\n".format(4))
            f.write("#define R_SIMD I_PE\n")
            f.write("#define O_SIMD I_PE\n")
            f.write("#define O_PE {}\n".format(4))
            input_bias = "true" if (self.rnn.input_kernel.bias is not None) else "false"
            rec_bias = "true" if (self.rnn.recurrent_kernel.bias is not None) else "false"
            output_bias = "true" if (self.lin.bias is not None) else "false"
            f.write("#define I_BIAS {}\n".format(input_bias))            
            f.write("#define R_BIAS {}\n".format(rec_bias))            
            f.write("#define O_BIAS {}\n".format(output_bias))            
            f.write("#define MAX_BUFFER_SIZE MAX(I_SIMD*INPUT_BW, R_SIMD*RECURRENT_BW)\n")

        # write params_include files
        incl_param_string += "typedef ac_fixed<RECURRENT_BW,1,true,AC_RND,AC_SAT> recurrent_type;\n"
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
        i_bw = math.floor(math.log2(max_val))+1
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

