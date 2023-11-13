class Encoder(nn.Module):
    def _init_(self, dims = 512, hidden_size = 512,num_layers = 2, max_src = 500, max_tgt = 500):
        super(Encoder, self)._init_()
        self.embedding = nn.Embedding(len(vocab_src),dims)
        self.input_size = dims
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_src = max_src
        self.max_tgt = max_tgt
        self.drop = nn.Dropout(p = 0.5)
        self.encoder = nn.LSTM(
            dims, hidden_size, num_layers, batch_first = True,bidirectional=True, dropout = 0.5
        )
        
    def encode_inp(self, x):
        encoded_x = torch.zeros(len(x), self.max_src, dtype = int) + 85
        for i in range(len(x)):
            for j in range(len(x[i])):
                encoded_x[i][j] = vocab_src[x[i][j]]
            encoded_x[i][len(x[i])] = vocab_src["END"]
        return encoded_x.to(device)
    
    def forward(self, x):
        encoded_x = self.encode_inp(x)
        input_seq = self.drop(self.embedding(encoded_x))
        hidden = torch.zeros(2*self.num_layers,input_seq.shape[0],self.hidden_size).to(device)
        cell = torch.zeros(2*self.num_layers,input_seq.shape[0],self.hidden_size).to(device)
        out, _ = self.encoder(input_seq,(hidden, cell))
        return out

    def inference(self, x):
        self.max_src = len(x)



class Decoder(nn.Module):
    def _init_(self,dims = 512, hidden_size = 512,num_layers = 2, max_src = 500, max_tgt = 500):
        super(Decoder, self)._init_()
#         self.embedding = embedding
        self.embedding = nn.Embedding(len(vocab_tgt),dims)
        self.input_size = dims
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_src = max_src
        self.max_tgt = max_tgt
        self.drop = nn.Dropout(p = 0.5)
        self.dec_cells = nn.ModuleList([nn.LSTMCell(2*hidden_size+dims, hidden_size), nn.LSTMCell(hidden_size, hidden_size)])
        self.linear = nn.Linear(hidden_size,len(vocab_tgt)-1)
    
    def encode_inp(self, x):
        encoded_x = torch.zeros(len(x), self.max_tgt+1, dtype = int) + 46
        encoded_x_end = torch.zeros(len(x))
        for i in range(len(x)):
            encoded_x[i][0] = vocab_tgt["STR"]
            for j in range(len(x[i])):
                encoded_x[i][j+1] = vocab_tgt[x[i][j]]
            encoded_x[i][len(x[i])+1] = vocab_tgt["END"]
            encoded_x_end[i] = len(x[i])+1
        return encoded_x.to(device), encoded_x_end
    
    def calcontext(self, timestep, query):
        extended_query = torch.cat((query, query), dim = 1)
        permuted_context = self.context.permute(1,0,2)
#         for encoder_timestep in range(self.context.shape[1]):
#             scores.append(torch.sum(self.context[:,encoder_timestep] * extended_query, dim = 1, keepdims=True))
#         scores = torch.cat(scores, dim = 1)
        scores = torch.sum(permuted_context * extended_query, dim = 2).permute(1,0)
        weights = nn.Softmax(dim = 1)(scores).unsqueeze(2)
        alignment = torch.sum(weights * self.context,  dim = 1)
        
        return alignment
        
    
    def forward(self, context, target_,teacher_ratio):
        self.context = context
        encoded_x, encoded_x_end = self.encode_inp(target_)
        target_seq = self.embedding(encoded_x)
        
        initial_hidden1 = torch.rand(target_seq.shape[0], self.hidden_size).to(device)
        initial_cell1 = torch.rand(target_seq.shape[0], self.hidden_size).to(device)
        initial_hidden2 = torch.rand(target_seq.shape[0], self.hidden_size).to(device)
        initial_cell2 = torch.rand(target_seq.shape[0], self.hidden_size).to(device)
        
        outputs = []
        hidden_states = []
        cell_states = []
        query = [initial_hidden2]
        for timestep in range(self.max_tgt+1):
            if(timestep == 0):
                (h_t1, c_t1) = self.dec_cells[0](self.drop(torch.cat((target_seq[:,timestep],self.calcontext(0,query[-1])),dim=1)), (initial_hidden1, initial_cell1))
                (h_t2, c_t2) = self.dec_cells[1](self.drop(h_t1), (initial_hidden2, initial_cell2))
            else:
                input = []
                if(torch.rand(1).item() < teacher_ratio):
                    input = target_seq[:,timestep]
                else:
                    input = self.embedding(torch.argmax(nn.Softmax(dim = 1)(outputs[-1][:,0]),dim=1))
                    
                (h_t1, c_t1) = self.dec_cells[0](self.drop(torch.cat((input,self.calcontext(timestep,query[-1])),dim=1)), (hidden_states[-1][0], cell_states[-1][0]))
                (h_t2, c_t2) = self.dec_cells[1](self.drop(h_t1), (hidden_states[-1][1], cell_states[-1][1]))
            hidden_states.append([h_t1, h_t2])
            cell_states.append([c_t1, c_t2])
            query.append(h_t2)
            out = self.linear(h_t2)
            outputs.append(out.unsqueeze(1))
    

        output_prob = torch.cat(outputs,dim = 1)
        
        return nn.LogSoftmax(dim = 2)(output_prob), encoded_x

    def inference(self, x):
      self.max_tgt = len(x)