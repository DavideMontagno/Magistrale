# https://github.com/spro/char-rnn.pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model="gru", n_layers=1,cuda_option=False):
        super(CharRNN, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        if self.model == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers,dropout=0.1)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers,dropout=0.1)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def forward2(self, input, hidden):
        encoded = self.encoder(input.view(1, -1))
        output, hidden = self.rnn(encoded.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self, batch_size, cuda_option):
        if self.model == "lstm":
            
            if(cuda_option):
                V1 = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).to("cuda")
                V2 = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).to("cuda")
                
               
            else:
                V1 = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).to("cpu")
                V2 = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).to("cpu")
               
                return (V1, V2)
        else:
            if(cuda_option):
                V1 = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).to("cuda")
                
                return V1
            else:
                V1 = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).to("cpu")
               
                return V1

