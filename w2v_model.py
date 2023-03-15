import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F


def tokenize_corpus(corpus):
    return corpus.lower().split()


# Input layer shape = [1, vocabulary_size]
def get_input_layer(word_idx, vocabulary_size):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x


class W2V(nn.Module):
    def __init__(self, embed_dim: int = 7, vocab_size: int = 10):
        super(W2V, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.fc = nn.Linear(in_features=embed_dim, out_features=vocab_size)
        
    def forward(self, x):
        x = self.embed(x)
        x = x.squeeze(1)
        return self.fc(x)


def train(data: str):
    """
    return: w2v_dict: dict
            - key: string (word)
            - value: np.array (embedding)
    """
    tokenized_corpus = tokenize_corpus(data)
    vocabulary = []
    for token in tokenized_corpus:
        if token not in vocabulary:
            vocabulary.append(token)

    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
    idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

    vocabulary_size = len(vocabulary)

    window_size = 2
    X = []
    y = []
    indices = [word2idx[word] for word in tokenized_corpus]
    for center_word_pos in range(len(indices)):
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            if (context_word_pos < 0) or (context_word_pos >= len(indices)) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            X.append(context_word_idx)
            y.append(indices[center_word_pos])

    # Hidden layer shape = [embeddings_dim, vocabulary size]
    embedding_dims = 7
    # W1 = Variable(torch.randn(embedding_dims, vocabulary_size), requires_grad=True)
    # Output layer
    # W2 = Variable(torch.randn(vocabulary_size, embedding_dims), requires_grad=True)
    EPOCHS = 20
    lr = 10e-2
    X = torch.tensor(X)
    y = torch.tensor(y)
    model = W2V(embedding_dims, vocabulary_size)
    opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    for epoch in range(EPOCHS):
        opt.zero_grad()
        losses = 0
        loss_acc = []
        pred = model(X)
        
        loss = F.nll_loss((pred), y)
        loss.backward()
        opt.step()

    output = {}
    W2 = model.fc._parameters["weight"].detach().numpy()
    for (idx, w) in enumerate(vocabulary):
        output[w] = W2[idx]

    return output
