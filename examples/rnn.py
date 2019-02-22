import numpy as np
from os.path import dirname, join
from chainer0 import Function, Variable
import chainer0.functions as F
import chainer0.links as L
from chainer0 import Chain
from chainer0.optimizers import SGD


class RNN(Chain):

    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()

        I, H, O = in_size, hidden_size, out_size
        with self.init_scope():
            self.embed = L.EmbedID(I, H)
            self.mid = L.SimpleRNN(H, H)
            self.out = L.Linear(H, O)


    def reset_state(self):
        self.mid.reset_state()

    def __call__(self, x):
        x = self.embed(x)
        h = self.mid(x)
        y = self.out(h)
        return y


model = RNN(10,10,10)

def compute_loss(x_list):
    loss = 0
    for cur_word, next_word in zip(x_list, x_list[1:]):
        score = model(cur_word)
        loss += F.softmax_cross_entropy(score, next_word)
    return loss


optimizer = SGD(lr=0.1)
optimizer.setup(model)

x_list = [1,2,3,4,0,3,4,2,2,4]
x_list = [Variable(np.array([x])) for x in x_list]

for i in range(10):
    model.reset_state()
    model.cleargrads()
    loss = compute_loss(x_list)
    loss.backward()
    optimizer.update()
    print(loss)



'''

### Dataset setup ##################

def string_to_one_hot(string, maxchar):
    """Converts an ASCII string to a one-of-k encoding."""
    ascii = np.array([ord(c) for c in string]).T
    return np.array(ascii[:,None] == np.arange(maxchar)[None, :], dtype=int)

def one_hot_to_string(one_hot_matrix):
    return "".join([chr(np.argmax(c)) for c in one_hot_matrix])

def build_dataset(filename, sequence_length, alphabet_size, max_lines=-1):
    """Loads a text file, and turns each line into an encoded sequence."""
    with open(filename) as f:
        content = f.readlines()
    content = content[:max_lines]
    content = [line for line in content if len(line) > 2]   # Remove blank lines
    seqs = np.zeros((sequence_length, len(content), alphabet_size))
    for ix, line in enumerate(content):
        padded_line = (line + " " * sequence_length)[:sequence_length]
        seqs[:, ix, :] = string_to_one_hot(padded_line, alphabet_size)
    return seqs



num_chars = 128

# Learn to predict our own source code.
text_filename = join(dirname(__file__), 'rnn.py')
train_inputs = build_dataset(text_filename, sequence_length=30,
                                 alphabet_size=num_chars, max_lines=60)

print(train_inputs.shape)
'''