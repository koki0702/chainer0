import numpy as np
from os.path import dirname, join
from chainer0 import Function, Variable
from chainer0.functions import sigmoid, matmul, log, sum, embed_id


def logistic_predictions(weights, inputs):
    score = matmul(inputs, weights)
    return sigmoid(score)


x = np.array([[1,2,3], [2,3,4])


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