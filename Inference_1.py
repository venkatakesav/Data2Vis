# %%
# Import the Libraries
import numpy as np
import pandas as pd

# Pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import json

# %% [markdown]
# ## Load the Data

# %%
import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)    

# Read Line by Line
with open('./OneDrive_1_11-11-2023/train.sources', 'r') as f:
    train_sources = [line.strip() for line in f.readlines()]

with open('./OneDrive_1_11-11-2023/train.targets', 'r') as f:
    train_targets = [line.strip() for line in f.readlines()]

# Create a Vocabulary for the Source Language
source_vocab = {'<sos>': 0, '<eos>': 1, '<pad>': 2}
source_index = 3

# Create a dictionary for the source language
for line in train_sources:
    for char in line:
        if char not in source_vocab:
            source_vocab[char] = source_index
            source_index += 1

print(source_index)

# Create a Vocabulary for the Target Language
target_vocab = {'<sos>': 0, '<eos>': 1, '<pad>': 2}
target_index = 3

# Create a dictionary for the target language
for line in train_targets:
    for char in line:
        if char not in target_vocab:
            target_vocab[char] = target_index
            target_index += 1

print(target_index)

# Define a function to convert the source sentences into indices
def source_sentence_to_index(sentence):
    return [source_vocab[char] for char in sentence]

# Define a function to convert the target sentences into indices
def target_sentence_to_index(sentence):
    return [target_vocab[char] for char in sentence]

# %%
# Define the Dataset Class
class CustomDataset(Dataset):
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        return self.source[idx], self.target[idx]
    
# Create the Dataset
dataset = CustomDataset(train_sources, train_targets)

# Create the Dataloader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# %% [markdown]
# ### Model

# %%
class Encoder(nn.Module):
    def __init__(self, dims = 512, hidden_size = 512,num_layers = 2, max_src = 500, max_tgt = 500):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(len(source_vocab),dims)
        self.input_size = dims
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_src = max_src
        self.max_tgt = max_tgt
        self.drop = nn.Dropout(p = 0.5)
        self.encoder = nn.LSTM(
            dims, hidden_size, num_layers, batch_first = True,bidirectional=True, dropout = 0.5
        )
    
    def forward(self, x):
        encoded = x
        embedded = self.embedding(encoded)
        embedded = self.drop(embedded)
        arg1 = 2*self.num_layers
        arg2 = embedded.shape[0]
        arg3 = self.hidden_size
        hidden = torch.zeros(arg1, arg2, arg3)
        # shifting to cuda
        hidden = hidden.cuda()
        arg1 = 2*self.num_layers
        arg2 = embedded.shape[0]
        arg3 = self.hidden_size
        cell = torch.zeros(arg1, arg2, arg3)
        # shifting to cuda
        cell = cell.cuda()
        arg1 = embedded
        arg_tuple = (hidden, cell)
        out_tuple = self.encoder(arg1, arg_tuple)
        out = out_tuple[0]
        return out

    def inference(self, x):
        self.max_src = len(x)


class Decoder(nn.Module):
    def __init__(self,dims = 512, hidden_size = 512,num_layers = 2, max_src = 500, max_tgt = 500):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(len(target_vocab),dims)
        self.input_size = dims
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_src = max_src
        self.max_tgt = max_tgt
        self.drop = nn.Dropout(p = 0.5)
        self.dec_cells = nn.ModuleList([nn.LSTMCell(2*hidden_size+dims, hidden_size), nn.LSTMCell(hidden_size, hidden_size)])
        self.linear = nn.Linear(hidden_size,len(target_vocab))
    
    def attention(self, timestep, query):
        arg1 = (query, query)
        mod_query = torch.cat(arg1, dim=1)
        arg2 = self.context
        arg2 = arg2.permute(1,0,2)
        mod_ctxt = arg2
        arg3 = mod_ctxt * mod_query
        arg3 = torch.sum(arg3, dim=2)
        scores = arg3.permute(1,0)
        weights = nn.Softmax(dim = 1)(scores)
        weights = weights.unsqueeze(2)
        temp_arg = self.context 
        arg4 = weights * temp_arg
        alignment = torch.sum(arg4, dim=1)
        return alignment
        
    def forward(self, context, target_,teacher_ratio):
        self.context = context
        target_seq = self.embedding(target_)

        initial_hidden1 = torch.rand(target_seq.shape[0], self.hidden_size).cuda()
        initial_cell1 = torch.rand(target_seq.shape[0], self.hidden_size).cuda()
        initial_hidden2 = torch.rand(target_seq.shape[0], self.hidden_size).cuda()
        initial_cell2 = torch.rand(target_seq.shape[0], self.hidden_size).cuda()
        
        outputs = []
        hidden_states = []
        cell_states = []
        query = [initial_hidden2]
        steps = 0
        steps = self.max_tgt
        all_steps = []
        for i in range(steps):
            all_steps.append(i)
        for step in all_steps:
            if (step):
                random_num = torch.rand(1).item()
                if (random_num < teacher_ratio):
                    input = target_seq[:, step]
                else:
                    embed_args = torch.argmax(nn.Softmax(dim=1)(outputs[-1][:, 0]), dim=1)
                    input = self.embedding(embed_args)
                    arg1 = torch.cat((input, self.attention(step, query[-1])), dim=1)
                    arg2 = (hidden_states[-1][0], cell_states[-1][0])
                    arg1 = self.drop(arg1)
                    temp_tuple = self.dec_cells[0](arg1, arg2)
                    h_t1 = temp_tuple[0]
                    c_t1 = temp_tuple[1]
                    arg1 = self.drop(h_t1)
                    arg2 = (hidden_states[-1][1], cell_states[-1][1])
                    temp_tuple = self.dec_cells[1](arg1, arg2)
                    h_t2 = temp_tuple[0]
                    c_t2 = temp_tuple[1]
            else:
                arg1 = torch.cat((target_seq[:, step], self.attention(0, query[-1])), dim=1)
                arg2 = (initial_hidden1, initial_cell1)
                arg1 = self.drop(arg1)
                temp_tuple = self.dec_cells[0](arg1, arg2)
                h_t1 = temp_tuple[0]
                c_t1 = temp_tuple[1]
                arg1 = self.drop(h_t1)
                arg2 = (initial_hidden2, initial_cell2)
                temp_tuple = self.dec_cells[1](arg1, arg2)
                h_t2 = temp_tuple[0]
                c_t2 = temp_tuple[1]
            hidden_states.append([h_t1, h_t2])
            query.append(h_t2)
            cell_states.append([c_t1, c_t2])
            out = self.linear(h_t2)
            outputs.append(out.unsqueeze(1))
        output_prob = torch.cat(outputs, dim=1)
        final_out = nn.LogSoftmax(dim=2)(output_prob)
        return final_out, target_

# %%
# Define the Hyperparameters
learning_rate = 0.0001

# Define the Optimizer
# Define the Loss Function
pad_idx = target_vocab['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# Define the Epochs
epochs = 10

# Define the Clip
clip = 1

# Define the Best Validation Loss
best_valid_loss = float('inf')

encoder = Encoder().cuda()
decoder = Decoder().cuda()

# %%
# Load Both Encoder and Decoder Models

encoder.load_state_dict(torch.load('./encoder_0.5.pth'))
decoder.load_state_dict(torch.load('./decoder_0.5.pth'))

# %%
# Run Inferencing on the Test Set

# Read Line by Line
with open('./OneDrive_1_11-11-2023/test.sources', 'r') as f:
    test_sources = [line.strip() for line in f.readlines()]

with open('./OneDrive_1_11-11-2023/test.targets', 'r') as f:
    test_targets = [line.strip() for line in f.readlines()]

# Convert the source sentences into indices
test_sources = [source_sentence_to_index(line) for line in test_sources]

# Prepend the <sos> token to the source sentences, and append the <eos> token to the source sentences
test_sources = [[source_vocab['<sos>']] + line + [source_vocab['<eos>']] for line in test_sources]

# Convert the target sentences into indices
test_targets = [target_sentence_to_index(line) for line in test_targets]

# Prepend the <sos> token to the target sentences, and append the <eos> token to the target sentences
test_targets = [[target_vocab['<sos>']] + line + [target_vocab['<eos>']] for line in test_targets]

# Pad So that the maximum length is 500 Characters
test_sources = [line + [source_vocab['<pad>']] * (500 - len(line)) for line in test_sources]
test_targets = [line + [target_vocab['<pad>']] * (500 - len(line)) for line in test_targets]

# Convert to numpy arrays
test_sources = np.array(test_sources)
test_targets = np.array(test_targets)

# Convert to Pytorch Tensors
test_sources = torch.from_numpy(test_sources).cuda()
test_targets = torch.from_numpy(test_targets).cuda()

# Create the Dataset
test_dataset = CustomDataset(test_sources, test_targets)

# Create the Dataloader
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# %%
# Define a function to revert back to the original sentence
def revert_to_sentence(indices, vocab):
    sentence = ''
    for index in indices:
        if index == vocab['<eos>']:
            break
        elif index != vocab['<sos>'] and index != vocab['<pad>']:
            for char, char_index in vocab.items():
                if char_index == index:
                    sentence += char
    return sentence

# Create a File
with open("predictions.txt", "w") as f:
    f.write("")

# Define the Inference Function
def Inference(Encoder, Decoder):
    with torch.no_grad():
        Encoder.eval()
        Decoder.eval()
        for batch_idx, (source, target) in enumerate(test_dataloader):
            source = source.cuda()
            target = target.cuda()
            context = Encoder(source)
            print("Encoded!!!!!!!")
            output, target = Decoder(context, target, 0)
            print("Decoded!!!!!!!")
            output = torch.argmax(output, dim=2)
            for i in range(len(output)):
                # print("Source:", revert_to_sentence(source[i], source_vocab))
                # Write to file
                with open("predictions.txt", "a") as f:
                    f.write("Source: " + revert_to_sentence(source[i], source_vocab) + "\n")
                    f.write("Target: " + revert_to_sentence(target[i], target_vocab) + "\n")
                    f.write("Prediction: " + revert_to_sentence(output[i], target_vocab) + "\n")
                    f.write("\n")
            print("Done with Batch:", batch_idx)

beam_width = 15

# Define the Inference Function
def Inference_1(Encoder, Decoder):
    with torch.no_grad():
        Encoder.eval()
        Decoder.eval()
        for batch_idx, (source, target) in enumerate(test_dataloader):
            source = source.cuda()
            target = target.cuda()
            context = Encoder(source)
            print("Encoded!!!!!!!")
            output, target = Decoder(context, target, 0)
            output_loss = output.permute(0, 2, 1)
            print("Decoded!!!!!!!")
            # Beam Search Part
            probabilities = output
            output = torch.argmax(output, dim=2)
            # Beam Search Part
            for i in range(len(probabilities)):
                # Get top-k predictions using beam search
                topk_probabilities, topk_indices = torch.topk(probabilities[i], beam_width, dim=1)
                print(topk_probabilities.shape)
                print(topk_indices.shape)
                for j in range(beam_width):
                    # Extract the j-th sequence and its probability
                    seq = topk_indices[:, j]
                    # Print or save the results
                    print("Source:", revert_to_sentence(source[i], source_vocab))
                    print("The {j}th Best Result is as follows: ", j)
                    print("Target:", revert_to_sentence(target[i], target_vocab))
                    print("Prediction:", revert_to_sentence(seq, target_vocab))
            # Find the Loss
            loss = criterion(output_loss, target)
            print("Loss:", loss.item())


# %%
# Run the Inference Function
Inference(encoder, decoder)
