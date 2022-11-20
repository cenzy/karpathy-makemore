import torch
import matplotlib.pyplot as plt


class TrigramModel:

    def __init__(self, words):
        self.words = words
        self.N = torch.zeros((27, 27, 27)).int()

        self.s2i, self.i2s = self.__lookup_dictionaries()

    def fit(self, smoothing=10):
        for w in self.words:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
                ix1 = self.s2i[ch1]
                ix2 = self.s2i[ch2]
                ix3 = self.s2i[ch3]
                self.N[ix1, ix2, ix3] += 1


        self.P = torch.reshape(self.N+smoothing, (-1, 27)).float()
        self.P /= self.P.sum(axis=1, keepdims=True)

    def inference(self, n=5, seed=32):
        g = torch.Generator().manual_seed(seed)
        outs = []
        for i in range(n):
            out = []
            ix = 0
            ix_s = 0
            while True:
                p = self.P[ix]
                next = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
                ix = ix_s * 27 + next #compute the extact row of the next prob. distrib.
                ix_s = next

                if next == 0:
                    break
                out.append(self.i2s[next])
            outs.append(''.join(out))
        return outs


    
    def __lookup_dictionaries(self):
        chars = sorted(list(set(''.join(self.words))))
        s2i = {s:i+1 for i,s in enumerate(chars)} 
        s2i['.'] = 0

        i2s = {i:s for s, i in s2i.items()}

        return s2i, i2s

    def pretty_output(self):
        pass

    def loss(self):
        log_likelihood = 0
        n = 0 

        for word in self.words:
            for ch1, ch2, ch3 in zip(word, word[1:], word[2:]):
                ix1, ix2, ix3 = self.s2i[ch1], self.s2i[ch2], self.s2i[ch3]
                p = self.P[27*ix1+ix2, ix3]
                logp = torch.log(p)
                log_likelihood += logp
                n = n+1
        
        return -log_likelihood/n



    