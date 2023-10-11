import numpy as np


class Module(object):
    def __init__(self):
        self.output = None
        self.grad_input = None
        self.training = True

    def forward(self, input):
        return self.update_output(input)

    def backward(self, input, grad_output):
        self.update_grad_input(input, grad_output)
        self.acc_grad_parameters(input, grad_output)
        return self.grad_input

    def update_output(self, input):
        pass

    def update_grad_input(self, input, grad_output):
        pass

    def acc_grad_parameters(self, input, grad_output):
        pass

    def zero_grad_parameters(self):
        pass

    def get_parameters(self):
        return []

    def get_grad_parameters(self):
        return []

    def train(self):
        self.training = True

    def evaluate(self):
        self.training = False

    def __repr__(self):
        return "Module"


class Sequential(Module):
    def __init__(self):
        super().__init__()
        self.modules = []

    def add(self, module):
        self.modules.append(module)

    def update_output(self, input):
        tmp_input = input
        tmp_output = None
        for i in range(len(self.modules)):
            tmp_output = self.modules[i].forward(tmp_input)
            tmp_input = tmp_output
        self.output = tmp_output
        return self.output

    def backward(self, input, grad_output):
        tmp_grad_output_in = grad_output
        tmp_grad_output_out = None
        self.modules[-1].grad_input = np.zeros_like(self.modules[-1].output)
        for i in range(len(self.modules) - 1, 0, -1):
            tmp_grad_output_out = self.modules[i].backward(self.modules[i - 1].output, tmp_grad_output_in)
            tmp_grad_output_in = tmp_grad_output_out

        self.grad_input = self.modules[0].backward(input, tmp_grad_output_in)
        return self.grad_input

    def zero_grad_parameters(self):
        for module in self.modules:
            module.zero_grad_parameters()

    def get_parameters(self):
        return [p.get_parameters() for p in self.modules]

    def get_grad_parameters(self):
        return [p.get_grad_parameters() for p in self.modules]

    def __repr__(self):
        name = "".join([str(p) for p in self.modules])

    def __getitem__(self, item):
        return self.modules.__getitem__(item)

    def train(self):
        self.training = True
        for module in self.modules:
            module.train()

    def evaluate(self):
        self.training = False
        for module in self.modules:
            module.evaluate()


class Linear(Module):
    def __init__(self, in_features: int, out_feature: int):
        super().__init__()
        self.in_features = in_features
        self.out_feature = out_feature
        stdv = np.sqrt(6.0 / float(in_features + out_feature))
        self.weight = np.random.uniform(low=-stdv, high=stdv, size=(out_feature, in_features))
        self.bias = np.random.uniform(low=-stdv, high=stdv, size=out_feature)
        self.output = None

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)

    def backward(self, input, grad_output):
        self.update_grad_input(input, grad_output)
        self.acc_grad_parameters(input, grad_output)
        return self.grad_input

    def update_output(self, input):
        self.output = np.dot(input, self.weight.T)
        np.add(
            self.output,
            self.bias[None, :],
            out=self.output
        )
        return self.output

    def update_grad_input(self, input, grad_output):
        self.grad_input = np.dot(grad_output, self.weight)
        return self.grad_input

    def acc_grad_parameters(self, input, grad_output):
        self.grad_weight += np.dot(grad_output.T, input)
        self.grad_bias += np.sum(grad_output, axis=0)

    def zero_grad_parameters(self):
        self.grad_weight.fill(0)
        self.grad_bias.fill(0)

    def get_parameters(self):
        return [self.weight, self.bias]

    def get_grad_parameters(self):
        return [self.grad_weight, self.grad_bias]

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_feature={self.out_feature})"


class LogSoftMax(Module):
    def __init__(self):
        super().__init__()

    def update_output(self, input):
        self.output = np.float64(np.subtract(input, input.max(axis=-1, keepdims=True)))
        exp = np.exp(self.output)
        sums = np.log(np.sum(exp, axis=-1, keepdims=True))
        np.subtract(self.output, sums, out=self.output)
        return self.output

    def update_grad_input(self, input, grad_output):
        batch, dim = self.output.shape

        probs = np.tile(np.exp(self.output)[:, None, :], (1, dim, 1))
        eye = np.tile(np.eye(dim), (batch, 1, 1))
        logsoftmax_grad = eye - probs

        self.grad_input = np.einsum("b i, b i j -> b j", grad_output, logsoftmax_grad)
        return self.grad_input

    def __repr__(self):
        return "LogSoftMax()"


class Loss(object):
    def __init__(self):
        self.output = None
        self.grad_input = None

    def forward(self, input, target):
        return self.update_output(input, target)

    def backward(self, input, target):
        return self.update_grad_input(input, target)

    def update_output(self, input, target):
        return self.output

    def update_grad_input(self, input, target):
        self.grad_input

    def __repr__(self):
        return "Loss()"


class NLL_Loss(Loss):
    def __init__(self):
        super().__init__()

    def update_output(self, input, target):
        # one_hot_target = np.zeros_like(input)
        # one_hot_target[np.arange(target.size), target] = 1
        self.output = -np.sum(input * target) / input.shape[0]
        return self.output

    def update_grad_input(self, input, target):
        # one_hot_target = np.zeros_like(input)
        # one_hot_target[np.arange(target.size), target] = 1
        self.grad_input = -target / input.shape[0]
        return self.grad_input

    def __repr__(self):
        return "NLL_Loss()"


class Optimizer(object):
    def __init__(self, model: Module):
        self.model = model
        self._state = {}

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, model: Module, lr: float, momentum: float = 0.1):
        super().__init__(model)
        self._learining_rate = lr
        self._momentum = momentum

    def step(self):
        variables = self.model.get_parameters()
        gradients = self.model.get_grad_parameters()

        self._state.setdefault("accumulated_grads", {})
        var_index = 0

        for current_layer_vars, current_layer_grad in zip(variables, gradients):
            for current_var, current_grad in zip(current_layer_vars, current_layer_grad):
                old_grad = self._state["accumulated_grads"].setdefault(var_index, np.zeros_like(current_grad))
                np.add(self._momentum * old_grad, current_grad, out=old_grad)
                current_var -= self._learining_rate * old_grad
                var_index += 1


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        stdv = np.sqrt(6.0 / float(num_embeddings + embedding_dim))
        self.weight = np.random.uniform(low=-stdv, high=stdv, size=(num_embeddings, embedding_dim))
        self.output = None

        self.grad_weight = np.zeros_like(self.weight)

    def backward(self, input, grad_output):
        self.update_grad_input(input, grad_output)
        self.acc_grad_parameters(input, grad_output)
        return self.grad_input

    def update_output(self, input):
        self.output = input @ self.weight
        return self.output

    def update_grad_input(self, input, grad_output):
        self.grad_input = np.dot(input.T, grad_output)
        return self.grad_input

    def acc_grad_parameters(self, input, grad_output):
        self.grad_weight += np.dot(input.T, grad_output)

    def zero_grad_parameters(self):
        self.grad_weight.fill(0)

    def get_parameters(self):
        return [self.weight]

    def get_grad_parameters(self):
        return [self.grad_weight]

    def __repr__(self):
        return f"Embedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim})"


def tokenize_corpus(corpus):
    return corpus.lower().split()


def generate_idx(max_range, batch_size):
    r = np.arange(max_range)
    np.random.shuffle(r)
    output = []
    for i in range(0, max_range, batch_size):
        output.append(r[i:i + batch_size])
    return output


def train(data: str):
    """
    return: w2v_dict: dict
            - key: string (word)
            - value: np.array (embedding)
    """
    tokenized_corpus = tokenize_corpus(data)
    vocabulary = set()
    for token in tokenized_corpus:
        vocabulary.add(token)

    vocabulary = list(vocabulary)

    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
    idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

    vocab_size = len(vocabulary)

    window_size = 3
    X = []
    y = []
    indexes = [word2idx[word] for word in tokenized_corpus]
    for c in range(len(indexes)):
        for w in range(-window_size, window_size + 1):
            context_word_pos = c + w
            if (context_word_pos < 0 or context_word_pos >= len(indexes)) or context_word_pos == c:
                continue
            x0 = np.zeros(vocab_size)
            x0[indexes[c]] = 1
            X.append(x0)
            y0 = np.zeros(vocab_size)
            y0[indexes[context_word_pos]] = 1
            y.append(y0)
    X = np.array(X)
    y = np.array(y)

    embedding_dim = 16

    model = Sequential()
    model.add(Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim))
    model.add(Linear(in_features=embedding_dim, out_feature=vocab_size))
    model.add(LogSoftMax())
    EPOCHS = 5
    lr = 0.2
    momentum = 0.9
    batch_size = 32
    opt = SGD(model, lr, momentum)
    loss_fn = NLL_Loss()
    for epoch in range(EPOCHS):
        
        for idx in generate_idx(len(X), batch_size):
            model.zero_grad_parameters()

            preds = model.forward(X[idx])

            loss = loss_fn.forward(preds, y[idx])
            dloss = loss_fn.backward(preds, y[idx])

            model.backward(X[idx], dloss)

            opt.step()

    output = {}
    W = model[0].weight
    for (idx, w) in enumerate(vocabulary):
        output[w] = W[idx]

    return output
