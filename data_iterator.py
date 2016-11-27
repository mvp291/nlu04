import numpy as np


class BatchIterator():
    """
    An iterator over a dataset file, which converts each
    line of the file into an example.

    The option ``'load_line'`` is a function which, given
    a string (a line in the file) outputs an example.
    """

    def __init__(self, source_path, target_path, batch_size=80):
        print source_path
        print target_path
        self.source_path = source_path
        self.target_path = target_path
        self.batch_size = batch_size
        self.source_f = open(self.source_path)
        self.target_f = open(self.target_path)

    def __iter__(self):
        return self

    def load_line(self, source_line, target_line):
        source = np.array(map(int, source_line.split()), dtype=np.int64)
        target = np.array(map(int, target_line.split()), dtype=np.int64)
        return source, target

    def next(self):
        batch_x, batch_y = [], []
        for i in range(self.batch_size):
            source_line = self.source_f.readline()
            target_line = self.target_f.readline()
            if not (source_line or target_line):  # if line is empty
                if i == 0:
                    self.source_f.close()
                    self.target_f.close()
                    self.source_f = open(self.source_path)
                    self.target_f = open(self.target_path)
                    raise StopIteration()
                else:
                    break
            both_lines = self.load_line(source_line, target_line)
            batch_x += [both_lines[0]]
            batch_y += [both_lines[1]]
        return (batch_x, batch_y)


def prepare_data(seqs_x, seqs_y,
                 maxlen=None,
                 n_words_src=30000,
                 n_words=30000):

    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x) + 1
    maxlen_y = np.max(lengths_y) + 1

    x = np.zeros((maxlen_x, n_samples)).astype('int64')
    y = np.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = np.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = np.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        s_x[np.where(s_x >= n_words_src - 1)] = 1
        s_y[np.where(s_y >= n_words - 1)] = 1
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx] + 1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx] + 1, idx] = 1.

    return x, x_mask, y, y_mask


def load_data(train_source_path='./data/source_train_idx',
              train_target_path='./data/target_train_idx',
              validation_source_path='./data/source_val_idx',
              validation_target_path='./data/target_val_idx',
              test_source_path='./data/source_test_idx',
              test_target_path='./data/target_test_idx',
              train_batch_size=80,
              val_batch_size=80,
              test_batch_size=80):

    #############
    # LOAD DATA #
    #############

    print '... initializing data iterators'

    train = BatchIterator(train_source_path,
                          train_target_path, train_batch_size)
    valid = BatchIterator(validation_source_path,
                          validation_target_path, val_batch_size)
    test = BatchIterator(test_source_path,
                         test_target_path, test_batch_size)

    return train, valid, test
