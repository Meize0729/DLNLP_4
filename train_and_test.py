import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import jieba
from tqdm import tqdm


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def __len__(self):
        return len(self.word2idx)

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def get_data(self, paths, batch_size=20):

        tokens = 0
        for path in paths:
            with open(path, 'r', encoding='ansi') as f:
                for line in f.readlines():
                    words = jieba.lcut(line) + ['<eos>']
                    tokens += len(words)
                    for word in words:
                        self.dictionary.add_word(word)


        ids = torch.LongTensor(tokens)
        token = 0
        for path in paths:
            with open(path, 'r', encoding="ansi") as f:
                for line in f.readlines():
                    words = jieba.lcut(line) + ['<eos>']
                    for word in words:
                        ids[token] = self.dictionary.word2idx[word]
                        token += 1


        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches * batch_size]
        ids = ids.view(batch_size, -1)
        return ids

# 定义LSTM模型类
class LSTMmodel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMmodel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        x = self.embed(x)
        out, (h, c) = self.lstm(x, h)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))
        out = self.linear(out)
        return out, (h, c)

def train(model, ids, num_epochs, batch_size, seq_length, learning_rate, device):

    cost = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(num_epochs):
        states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                  torch.zeros(num_layers, batch_size, hidden_size).to(device))

        for i in tqdm(range(0, ids.size(1) - seq_length, seq_length)):
            inputs = ids[:, i:i+seq_length].to(device)
            targets = ids[:, (i+1):(i+1)+seq_length].to(device)

            states = [state.detach() for state in states]
            outputs, states = model(inputs, states)
            loss = cost(outputs, targets.reshape(-1))

            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        train_ppl = evaluate(model, ids, batch_size, seq_length, device, cost)
        print('Epoch [{}/{}], train perplexity: {:.2f}'.format(epoch+1, num_epochs, train_ppl))

def evaluate(model, ids, batch_size, seq_length, device, cost):
    states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
              torch.zeros(num_layers, batch_size, hidden_size).to(device))
    total_loss = 0
    total_words = 0

    with torch.no_grad():
        for i in range(0, ids.size(1) - seq_length, seq_length):
            inputs = ids[:, i:i+seq_length].to(device)
            targets = ids[:, (i+1):(i+1)+seq_length].to(device)

            outputs, states = model(inputs, states)
            loss = cost(outputs, targets.reshape(-1))
            total_loss += loss.item() * seq_length
            total_words += seq_length

        ppl = torch.exp(torch.tensor(total_loss / total_words))


    return ppl

def generate(model, corpus, num_samples, device):
    article = str()
    state = (torch.zeros(num_layers, 1, hidden_size).to(device),
            torch.zeros(num_layers, 1, hidden_size).to(device))

    prob = torch.ones(vocab_size)
    _input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)
    for i in range(num_samples):
        output, state = model(_input, state)
        prob = output.exp()
        word_id = torch.multinomial(prob, num_samples=1).item()
        _input.fill_(word_id)
        word = corpus.dictionary.idx2word[word_id]
        word = '\n' if word == '<eos>' else word
        article += word


    with open("example.txt", "w", encoding="ansi") as file:
        file.write(article)
    file.close()


embed_size = 128
hidden_size = 1024
num_layers = 1
num_epochs = 7
batch_size = 50
seq_length = 30
learning_rate = 0.01
num_samples = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


corpus = Corpus()


ids = corpus.get_data(['/home/b902-r4-02/WMZ/zuoye/data/三十三剑客图.txt', '/home/b902-r4-02/WMZ/zuoye/data/侠客行.txt', '/home/b902-r4-02/WMZ/zuoye/data/书剑恩仇录.txt', '/home/b902-r4-02/WMZ/zuoye/data/天龙八部', '/home/b902-r4-02/WMZ/zuoye/data/鸳鸯刀'], batch_size)

vocab_size = len(corpus.dictionary)

model = LSTMmodel(vocab_size, embed_size, hidden_size, num_layers).to(device)


train(model, ids, num_epochs, batch_size, seq_length, learning_rate, device)


generate(model, corpus, num_samples, device)