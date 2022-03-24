import torch
import torch.nn as nn 
B_MAX_BW = 8

def save_sets(path, train_set, test_set, test_size):
    train_name = path + "/train_" + str(1-test_size) + ".txt"
    test_name = path + "/test_" + str(test_size) + ".txt"
    with open(train_name, 'w') as f:
        for elem in train_set:
            f.write(elem + "\n")
    with open(test_name, 'w') as f:
        for elem in test_set:
            f.write(elem + "\n")


def load_sets(path, test_size):
    train_files, test_files = [], []
    
    train_name = path + "/train_" + str(1-test_size) + ".txt"
    test_name = path + "/test_" + str(test_size) + ".txt"
    with open(train_name, 'r') as f:
        names = f.readlines()
        for n in names:
            train_files.append(n[:-1])
    with open(test_name, 'r') as f:
        names = f.readlines()
        for n in names:
            test_files.append(n[:-1])
    return train_files, test_files


def collate_fn(batch):
    """Collects together sequences into a single batch, arranged in descending length order."""
    batch_size = len(batch)

    # Sort the (sequence, label) pairs in descending order of duration
    batch.sort(key=(lambda x: len(x[0])), reverse=True)
    # Shape: list(tuple(tensor(TxD), int))

    # Create list of sequences, and tensors for lengths and labels
    sequences, filenames, lengths, labels = [], [], torch.zeros(batch_size, dtype=torch.long), torch.zeros(batch_size, dtype=torch.long)
    for i, (file, sequence, label) in enumerate(batch):
        lengths[i], labels[i] = len(sequence), label
        sequences.append(sequence)
        filenames.append(file)

    # Combine sequences into a padded matrix
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    # Shape: (B x T_max x D)

    return padded_sequences, filenames, lengths, labels
    # Shapes: (B x T_max x D), (B,), (B,)

class QuantizeAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, numbits):
        ctx.save_for_backward(input)
        if numbits == 1:
            return input.sign()
        elif numbits == 2:
            return torch.floor(input + 0.5)
        elif numbits == 8:
            # clipped = torch.clamp(input, 0, 1)
            clipped = torch.round(input * 2.**6)/2.**6
            return clipped
        else:
            return input
   
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class Quantizer(nn.Module):
    def __init__(self, numbits):
        super(Quantizer, self).__init__()
        self.numbits=numbits

    def forward(self, input):
        return QuantizeAct.apply(input, self.numbits)

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
        clipped = torch.clamp(tensor, -1, 1-2**-7)
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
            self.bias.data = Quantize(self.bias.org, max(B_MAX_BW, self.wb))            
            out += self.bias.view(1, -1).expand_as(out)
        return out

class qmodrelu(nn.Module):
    def __init__(self, features, numbits):
        super(qmodrelu, self).__init__()
        self.features = features
        self.b = nn.Parameter(torch.Tensor(self.features))
        self.reset_parameters()
        self.ab = numbits
        self.quantize = Quantizer(numbits)

    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)

    def forward(self, inputs):
        norm = torch.abs(inputs)
        if not hasattr(self.b,'org'):
            self.b.org = self.b.data.clone()
        self.b.data = Quantize(self.b.org, self.ab)
        biased_norm = norm + self.b
        magnitude = nn.functional.relu6(biased_norm)
        magnitude = self.quantize(magnitude)
        phase = torch.sign(inputs)
        out = phase * magnitude
        return out



        