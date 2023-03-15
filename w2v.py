import numpy as np
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
    idx_pairs = []
    indices = [word2idx[word] for word in tokenized_corpus]
    for center_word_pos in range(len(indices)):
        for w in range(-window_size, window_size+1):
            context_word_pos = center_word_pos + w
            if (context_word_pos < 0) or (context_word_pos >= len(indices)) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            idx_pairs.append((indices[center_word_pos], context_word_idx))
          
    
    # Hidden layer shape = [embeddings_dim, vocabulary size]
    embedding_dims = 7
    W1 = Variable(torch.randn(embedding_dims, vocabulary_size), requires_grad=True)
    # Output layer
    W2 = Variable(torch.randn(vocabulary_size, embedding_dims), requires_grad=True)
    EPOCHS = 20
    lr = 10e-2

    
    for epoch in range(EPOCHS):
      losses = 0
      loss_acc = []
      idx_pairs = np.random.permutation(idx_pairs)
      for data, target in idx_pairs:
        X = Variable(get_input_layer(data, vocabulary_size))
        y = Variable(torch.from_numpy(np.array([target])))
    
        z1 = torch.matmul(W1, X)
        z2 = torch.matmul(W2, z1)
    
        log_softmax = F.log_softmax(z2, dim=0)
    
        loss = F.nll_loss(log_softmax.view(1,-1), y)
        losses += loss.item()
        loss_acc.append(loss.item())
        loss.backward()
        W1.data -= lr * W1.grad.data
        W2.data -= lr * W2.grad.data
    
        W1.grad.data.zero_()
        W2.grad.data.zero_()
     
    output = {}
    W2 = W2.detach().numpy()
    for (idx, w) in enumerate(vocabulary):
        output[w] = W2[idx]

    return output
