def lookup_dictionaries(words):
    chars = sorted(list(set(''.join(words))))
    s2i = {s:i+1 for i,s in enumerate(chars)} 
    s2i['.'] = 0

    i2s = {i:s for s, i in s2i.items()}

    return s2i, i2s

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

class MakemoreMLP:

    def __init__(self, words, block_size=3, emb_size=5):
        self.words = words
        self.s2i, self.i2s = lookup_dictionaries(self.words)
        self.block_size = block_size
        self.emb_size = emb_size
        self.g = torch.Generator().manual_seed(2147483647)
        
        random.shuffle(self.words)
        train_index = int(len(self.words) * 0.8)
        val_index = int(len(self.words) * 0.9)
        self.X, self.Y = self.__make_dataset(self.words[:train_index])
        self.X_val, self.Y_val = self.__make_dataset(self.words[train_index:val_index])
        self.X_test, self.Y_test = self.__make_dataset(self.words[val_index:])
        

    def __init_parameters(self, hidden_units=100):
        self.C = torch.randn((27, self.emb_size), generator=self.g)
        self.W1 = torch.randn((self.emb_size * self.block_size, hidden_units), generator=self.g)
        self.b1 = torch.randn(hidden_units, generator=self.g)
        self.W2 = torch.randn((hidden_units, 27), generator=self.g)
        self.b2 = torch.randn(27, generator=self.g)
        self.parameters = [self.C, self.W1, self.b1, self.W2, self.b2]

        for p in self.parameters:
            p.requires_grad = True

    def fit(self, epochs=100, hidden_units=100, lr=0.1):
        self.__init_parameters(hidden_units)

        for i in range(0, epochs):

            batch_ix = torch.randint(0, self.X.shape[0], (32,))

            emb = self.C[self.X[batch_ix]]
            h = torch.tanh(emb.view(-1, self.block_size * self.emb_size) @ self.W1 + self.b1)
            logits = h @ self.W2 + self.b2
            loss = F.cross_entropy(logits, self.Y[batch_ix])
            if i % (epochs/10) == 0:
                print(loss.item())
            
            for p in self.parameters:
                p.grad = None
            
            loss.backward()

            for p in self.parameters:
                p.data += -lr * p.grad
        
        emb = self.C[self.X]
        h = torch.tanh(emb.view(-1, self.block_size * self.emb_size) @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        loss = F.cross_entropy(logits, self.Y)
        print('Whole loss', loss.item())
    
    def test(self, type='validation'):
        if type == 'validation':
            X = self.X_val
            Y = self.Y_val
        else:
            X = self.X_test
            Y = self.Y_test
        
        emb = self.C[X]
        h = torch.tanh(emb.view(-1, self.block_size * self.emb_size) @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        loss = F.cross_entropy(logits, Y)
        print('Loss', loss.item())


    def inference(self):
        word = []
        ix = 99
        context = [0] * self.block_size
        
        while ix != 0:
            emb = self.C[torch.tensor([context])]
            h = torch.tanh(emb.view(1, -1) @ self.W1 + self.b1)
            logits = h @ self.W2 + self.b2
            p = F.softmax(logits, dim=1)

            ix = torch.multinomial(p, num_samples=1, generator=self.g).item()
            word.append(self.i2s[ix])
            context = context[1:] + [ix]
        
        return ''.join(w for w in word)




    def __make_dataset(self, words):
        X, Y = [], []
        for word in words:
            context = [0] * self.block_size
            for ch in word + '.':
                ix = self.s2i[ch]
                X.append(context)
                Y.append(ix)

                #print(''.join(self.i2s[cx] for cx in context), '->', self.i2s[ix])
                context = context[1:] + [ix] 
        X = torch.tensor(X)
        Y = torch.tensor(Y)

        return X, Y

        

