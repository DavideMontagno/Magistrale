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

def plot_data(text, data, annotation, elem_image,val,folder):
    len_text=len(text)
    n_top_char=len(data)
    data=np.array(data).transpose()
    annotation=np.array(annotation).transpose()
    #labels = (np.asarray(["{0}\n{1:.2f}".format(annotation,data) for annotation, data in zip(annotation.flatten(), data.flatten())])).reshape(n_top_char,len_text)
  
    n_images=math.ceil(len_text/elem_image)
    for i in range(0,n_images):
        f, ax = plt.subplots(figsize=(25, 3))
        ax.xaxis.set_tick_params(labeltop='on')
        ax.xaxis.set_tick_params(labelbottom='')
 
        partial_text=text[i*elem_image:(i*elem_image)+elem_image]
        partial_data=[data[0:(i*elem_image)+elem_image]]
        x_axis_labels = list(partial_text)
        
        heat_map = sns.heatmap(partial_data,  fmt='', xticklabels=x_axis_labels, yticklabels=False,  annot_kws={"size": 8})
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        plt.savefig(str(folder)+'/plots_final_'+str(val)+'.jpg',dpi=300)
        plt.close()
        

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
            best_characters, best_index = torch.topk(output_dist,1)
            letters = [all_characters[i] for i in best_index]
            text.append(all_characters[best_index[0]])
            predicted+=all_characters[best_index[0]]
            annotation.append(letters)
            temp = normalize(hidden[1][0],hidden[1][0].max().item(),hidden[1][0].min().item(),cuda)
            
            data.append(temp)
            

            #list_predicted_value.append(list(best_characters.to("cpu").numpy()))
            inp = Variable(char_tensor(all_characters[best_index[0]]).unsqueeze(0))
            if cuda:
                inp = inp.cuda()
        
        for i in range(len(data[0])):
            print(i)
            lst2 = [item[i].item() for item in data]
            plot_data(text,lst2,annotation,divide,val+"_Hidden"+str(i)+"_"+str(prime_str),folder)
        append_write = 'w'
        
        if os.path.exists(folder):
                append_write = 'a' # append if already exists
        else:
                append_write = 'w' # make a new file if not
        f= open(folder+"/test.txt",append_write)
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
    argparser.add_argument('-l', '--predict_len', type=int, default=100)
    argparser.add_argument('-d', '--divide', type=int, default=1)
    argparser.add_argument('--cuda', action='store_true')
    args = argparser.parse_args()
    
    decoder = torch.load(args.filename)
    del args.filename
    generate(decoder, **vars(args))

