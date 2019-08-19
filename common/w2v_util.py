import re
import numpy as np
import gzip

embedd_dict = None
embedd_dim = -1


def w2v(word: str):
    global embedd_dict
    if embedd_dict is None:
        load_embedding_dict(normalize_digits=True)
    return embedd_dict[word] if word in embedd_dict else embedd_dict['<UNK>']


def create_spec_w2vec(n_dim: int):
    np.random.seed(seed=0)
    spec_tokens = ['<UNK>', '<PAD>', '<BOS>', '<EOS>', '<ROOT>']
    w2vec = {}
    for token in spec_tokens:
        sd = 1 / np.sqrt(n_dim)  # Standard deviation to use
        weights = np.random.normal(0, scale=sd, size=[1, n_dim])
        weights = weights.astype(np.float32)
        w2vec[token] = weights
    return w2vec


def load_embedding_dict(model_path: str, normalize_digits: bool = True):
    """
    load word embeddings from file
    :param normalize_digits:
    :return: embedding dict, embedding dimention, caseless
    """
    global embedd_dict
    global embedd_dim

    if not embedd_dict:
        embedd_dim = -1
        embedd_dict = dict()
        with gzip.open(model_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split()
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim], dtype=np.float32)
                embedd[:] = tokens[1:]
                word = re.sub(br'\d', b'0', tokens[0]) if normalize_digits else tokens[0]
                word = word.decode('utf-8')
                embedd_dict[word] = embedd

        np.random.seed(seed=0)
        spec_tokens = ['<UNK>', '<PAD>', '<BOS>', '<EOS>', '<ROOT>']
        for token in spec_tokens:
            sd = 1 / np.sqrt(embedd_dim)  # Standard deviation to use
            weights = np.random.normal(0, scale=sd, size=[1, embedd_dim])
            weights = weights.astype(np.float32)
            embedd_dict[token] = weights


def release_memory():
    global embedd_dict
    global embedd_dim
    embedd_dict = None
    embedd_dim = -1


def main():
    global embedd_dict
    global embedd_dim
    load_embedding_dict(normalize_digits=True)
    #print(len(embedd_dict))
    print(embedd_dict['university'].shape)
    print(embedd_dict['<UNK>'].shape)
    #print(embedd_dim)
    release_memory()
    return


if __name__ == '__main__':
    main()
