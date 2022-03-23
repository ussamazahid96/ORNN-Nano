import torch
import torch.nn as nn

def Quantize(tensor, numbits):
    if numbits == 1:
        tensor = torch.where(tensor >= 0, 1.0, -1.0)
        return tensor
    elif numbits == 2:
        return torch.floor(tensor + 0.5)
    elif numbits == 4:
        clipped = torch.clamp(tensor, -1, 1-2**-3)
        quantized = torch.round(clipped * 2.**3)/2.**3        
        return quantized
    elif numbits == 8:
        clipped = torch.clamp(tensor, -1, 1)
        quantized = torch.round(clipped * 2.**7)/2.**7
        return quantized
    else:
        return tensor


class Quantized_Linear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        self.wb = kargs[0]
        kargs = kargs[1:]
        super(Quantized_Linear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if not hasattr(self.weight,'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = Quantize(self.weight.org, self.wb)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            if not hasattr(self.bias,'org'):
                self.bias.org=self.bias.data.clone()
            self.bias.data = Quantize(self.bias.org, max(4, self.wb))            
            out += self.bias.view(1, -1).expand_as(out)
        return out

