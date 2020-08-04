#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import os
import argparse
import matplotlib.pyplot as plt
from helpers import *
from model import *
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import rcParams
import math
import pickle
 



k=3


        

def normalize(values, max, min, cuda):
    normalized = []
    for i in range(len(values)):
        value_tmp = (values[i]-min)/(max-min)
        if(value_tmp <= 1 and value_tmp >= 0):
            normalized.append(value_tmp)
    if(cuda):
        return torch.FloatTensor(normalized).cuda()
        
    else: 
        return normalized
    

def split_word(word): 
    return [char for char in word]  

def generate(decoder, prime_str='A', predict_len=100, temperature=0.8, cuda=False,val="test",folder="../result/Pytorch/",divide=1,is_train=0):
    for i in range(10,62):
        prime_str = all_characters[i]
        hidden = decoder.init_hidden(1,cuda)
        prime_input = Variable(char_tensor(prime_str).unsqueeze(0))
        if cuda:
        
            prime_input = prime_input.cuda()
        predicted = prime_str

        # Use priming string to "build up" hidden state
        for p in range(len(prime_str) - 1):
            _, hidden = decoder(prime_input[:,p], hidden)
            
        inp = prime_input[:,-1]
        if(cuda):
            inp = inp.cuda()
        text = []
        data = []
        annotation = []
        for p in range(predict_len):
            output, hidden = decoder(inp, hidden)
            
            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            output_dist = normalize(output_dist,output_dist.max().item(),output_dist.min().item(),cuda)
            best_characters, best_index = torch.topk(output_dist,k)
            letters = [all_characters[i] for i in best_index]
            text.append(all_characters[best_index[0]])
            predicted+=all_characters[best_index[0]]
            annotation.append(letters)
            data.append(best_characters.to("cpu").tolist())
            #list_predicted_value.append(list(best_characters.to("cpu").numpy()))
            inp = Variable(char_tensor(all_characters[best_index[0]]).unsqueeze(0))
            if cuda:
                inp = inp.cuda()
       
        append_write = 'w'
        
        if os.path.exists(folder):
                append_write = 'a' # append if already exists
        else:
                append_write = 'w' # make a new file if not
        f= open(folder+"/all_text.txt",append_write)
        f.write(predicted+"\n")
        f.close()
        print(predicted)
    


# Run as standalone script
if __name__ == '__main__':

# Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename', type=str)
    argparser.add_argument('folder',type=str)
    argparser.add_argument('-p', '--prime_str', type=str, default='A')
    argparser.add_argument('-l', '--predict_len', type=int, default=300)
  
    argparser.add_argument('--cuda', action='store_true')
    args = argparser.parse_args()
    
    decoder = torch.load(args.filename)
    del args.filename
    generate(decoder, **vars(args))

