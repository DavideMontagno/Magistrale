import os


model = "gru"
epochs = [1400]
learning_rates = [ 0.005]
batch_sizes = [192]
hidden_sizes = [300]
n_layers = [2]
chunk_lens = [200]

for epoch in epochs:
    for learning_rate in learning_rates:
        for batch_size in batch_sizes:
            for hidden_size in hidden_sizes:
                for n_layer in n_layers:
                    for chunk_len in chunk_lens:
                        print("python train.py TRUMP ..\dataset\dataset-cleaned\Trump\single_effects.txt --model {} --n_epochs {} --print_every 200 --hidden_size {} --n_layers {} --chunk_len {} --learning_rate {} --batch_size {} --cuda".format(model,epoch,hidden_size,n_layer,chunk_len,learning_rate,batch_size))
                        os.system("python train.py TRUMP ..\dataset\dataset-cleaned\Trump\single_effects.txt --model {} --n_epochs {} --print_every 200 --hidden_size {} --n_layers {} --chunk_len {} --learning_rate {} --batch_size {} --cuda".format(model,epoch,hidden_size,n_layer,chunk_len,learning_rate,batch_size))


model = "gru"
epochs = [800]
learning_rates = [ 0.001]
batch_sizes = [192]
hidden_sizes = [300]
n_layers = [2]
chunk_lens = [200]


for epoch in epochs:
    for learning_rate in learning_rates:
        for batch_size in batch_sizes:
            for hidden_size in hidden_sizes:
                for n_layer in n_layers:
                    for chunk_len in chunk_lens:
                        print("python train.py CLINTON ..\dataset\dataset-cleaned\Clinton\single_effects.txt --model {} --n_epochs {} --print_every 200 --hidden_size {} --n_layers {} --chunk_len {} --learning_rate {} --batch_size {} --cuda".format(model,epoch,hidden_size,n_layer,chunk_len,learning_rate,batch_size))
                        os.system("python train.py CLINTON ..\dataset\dataset-cleaned\Clinton\single_effects.txt --model {} --n_epochs {} --print_every 200 --hidden_size {} --n_layers {} --chunk_len {} --learning_rate {} --batch_size {} --cuda".format(model,epoch,hidden_size,n_layer,chunk_len,learning_rate,batch_size))




