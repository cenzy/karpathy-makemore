import torch
import torch.nn.functional as F

def lookup_dictionaries(words):
    chars = sorted(list(set(''.join(words))))
    s2i = {s:i+1 for i,s in enumerate(chars)} 
    s2i['.'] = 0

    i2s = {i:s for s, i in s2i.items()}

    return s2i, i2s

class NNBrigramModel:

    def __init__(self, words, split=True):
        self.words = words
        self.s2i, self.i2s = lookup_dictionaries(self.words)
        self.__make_dataset(split)

        g = torch.Generator().manual_seed(2147483647)
        self.W = torch.randn((27,27), generator=g, requires_grad=True)

    def __make_dataset(self, split=True):
        xs, ys = [], []
        self.n = 0 

        for word in self.words:
            chs = ['.'] + list(word) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                xs.append(self.s2i[ch1])
                ys.append(self.s2i[ch2])
                self.n = self.n+1
            
        xs = torch.tensor(xs)
        ys = torch.tensor(ys)

        self.x = F.one_hot(xs, num_classes=27).float()
        self.y = ys
        
        assert self.x.shape[0] == self.n

        if split == True:
            self.__split()

    def __split(self):
        indices = torch.randperm(len(self.x))

        train_size = int(0.8 * len(self.x))
        val_size = int(0.1 * len(self.x))

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices [train_size + val_size:]

        self.x_train = self.x[train_indices]
        self.x_val = self.x[val_indices]
        self.x_test = self.x[test_indices]

        self.y_train = self.y[train_indices]
        self.y_val = self.y[val_indices]
        self.y_test = self.y[test_indices]


    def train(self, epochs=10, verbose=False, lr=50, smoothing=0, full_dataset=False):
        if full_dataset == True:
            data = self.x
            label = self.y
        else:
            data = self.x_train
            label = self.y_train

        for k in range(epochs):
            logits = data @ self.W ## n x 27 @ 27 x 27 = n x 27
            counts = logits.exp()
            prob = counts/counts.sum(1, keepdims=True) #basically softmax

            #loss = -prob[torch.arange(len(data)), label].log().mean() + smoothing*(self.W**2).mean()
            loss = F.cross_entropy(logits[torch.arange(len(data))], label) + smoothing*(self.W**2).mean()

            if verbose == True:
                if k % 10 == 0:
                    print(loss.item())
        
            self.W.grad = None
            loss.backward()

            self.W.data += -lr * self.W.grad

    def test(self, type='test'):
        if type == 'test':
            data = self.x_test
            label = self.y_test
        else:
            data = self.x_val
            label = self.y_val

        logits = data @ self.W
        counts = logits.exp()
        prob = counts/counts.sum(1, keepdims=True)

        loss = -prob[torch.arange(len(data)), label].log().mean()

        return loss.item()
        

    
    def inference(self, seed=42, n_word=5):
        g = torch.Generator().manual_seed(seed)
        outs = []
        for i in range(n_word):
            out = []
            ix = 0
            while True:
                xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
                logit = xenc @ self.W
                count = logit.exp()
                p = count/count.sum(1, keepdims=True) #probabilities for the next character

                ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
                out.append(self.i2s[ix])
                if ix == 0:
                    break
            outs.append(''.join(out))
        return outs

class NNTrigramModel:

    def __init__(self, words):
        self.words = words
        self.s2i, self.i2s = lookup_dictionaries(self.words)
        self.__make_dataset()

        g = torch.Generator().manual_seed(2147483647)
        self.W = torch.randn((54, 27), generator=g, requires_grad=True)

    def __make_dataset(self):
        xs1, xs2, ys = [], [], []
        self.n = 0 

        for word in self.words:
            chs = ['.'] + list(word) + ['.']
            for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
                xs1.append(self.s2i[ch1])
                xs2.append(self.s2i[ch2])
                ys.append(self.s2i[ch3])
                self.n = self.n+1
            
        xs1 = torch.tensor(xs1)
        xs2 = torch.tensor(xs2)
        ys = torch.tensor(ys)

        x1 = F.one_hot(xs1, num_classes=27).float()
        x2 = F.one_hot(xs2, num_classes=27).float()
        self.x = torch.cat((x1, x2), dim=1)
        self.y = ys
        
        assert self.x.shape[0] == self.n

    def train(self, epochs=10, verbose=False, lr=50, smoothing=0):
        for k in range(epochs):
            logits = self.x @ self.W ## n x 54 @ 54 x 27 = n x 27
            counts = logits.exp()
            prob = counts/counts.sum(1, keepdims=True) #basically softmax

            loss = -prob[torch.arange(self.n), self.y].log().mean() + smoothing*(self.W**2).mean()

            if verbose == True:
                if k % 10 == 0:
                    print(loss.item())
        
            self.W.grad = None
            loss.backward()

            self.W.data += -lr * self.W.grad

    
    def inference(self, seed=42, n_word=5):
        g = torch.Generator().manual_seed(seed)
        outs = []
        for i in range(n_word):
            out = []
            ix = 0
            jx = 0
            while True:
                ix_onehot = F.one_hot(torch.tensor([ix]), num_classes=27).float()
                jx_onehot = F.one_hot(torch.tensor([jx]), num_classes=27).float()
                xenc = torch.concat((ix_onehot, jx_onehot), dim=1)
                logit = xenc @ self.W
                count = logit.exp()
                p = count/count.sum(1, keepdims=True) #probabilities for the next character

                ix = jx
                jx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
                out.append(self.i2s[jx])
                if jx == 0:
                    break
            outs.append(''.join(out))
        return outs