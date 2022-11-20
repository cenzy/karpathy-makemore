import torch
import matplotlib.pyplot as plt


class BigramModel:

    def __init__(self, words):
        self.words = words
        self.N = torch.zeros((27, 27)).int()

        self.s2i, self.i2s = self.__lookup_dictionaries()

    def fit(self):
        for w in self.words:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                ix1 = self.s2i[ch1]
                ix2 = self.s2i[ch2]
                self.N[ix1, ix2] += 1

        self.P = self.N.float()
        self.P /= self.P.sum(axis=1, keepdims=True)

    def inference(self, n=5, seed=32):
        g = torch.Generator().manual_seed(seed)
        outs = []
        for i in range(n):
            out = []
            ix = 0
            while True:
                p = self.P[ix]
                ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
                if ix == 0:
                    break
                out.append(self.i2s[ix])
            outs.append(''.join(out))
        return outs


    
    def __lookup_dictionaries(self):
        chars = sorted(list(set(''.join(self.words))))
        s2i = {s:i+1 for i,s in enumerate(chars)} 
        s2i['.'] = 0

        i2s = {i:s for s, i in s2i.items()}

        return s2i, i2s

    def pretty_output(self):
        plt.figure(figsize=(16,16))
        plt.imshow(self.N, cmap='Blues')
        for i in range(27):
            for j in range(27):
                chstr = self.i2s[i] + self.i2s[j]
                plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
                plt.text(j, i, self.N[i, j].item(), ha='center', va='top', color='gray')
            plt.axis('off')

    def loss(self):
        log_likelihood = 0
        n = 0 

        for word in self.words:
            for ch1, ch2 in zip(word, word[1:]):
                p = self.P[self.s2i[ch1], self.s2i[ch2]]
                logp = torch.log(p)
                log_likelihood += logp
                n = n+1
        
        return -log_likelihood/n



    