import enum
import torch
import argparse

class Dictionary(object):

    def __init__(self, id2word, word2id, lang):
        assert len(id2word) == len(word2id)
        self.id2word = id2word
        self.word2id = word2id
        self.lang = lang
        self.check_valid()

    def __len__(self):
        """
        Returns the number of words in the dictionary.
        """
        return len(self.id2word)

    def __getitem__(self, i):
        """
        Returns the word of the specified index.
        """
        return self.id2word[i]

    def __contains__(self, w):
        """
        Returns whether a word is in the dictionary.
        """
        return w in self.word2id

    def __eq__(self, y):
        """
        Compare the dictionary with another one.
        """
        self.check_valid()
        y.check_valid()
        if len(self.id2word) != len(y):
            return False
        return self.lang == y.lang and all(self.id2word[i] == y[i] for i in range(len(y)))

    def check_valid(self):
        """
        Check that the dictionary is valid.
        """
        assert len(self.id2word) == len(self.word2id)
        for i in range(len(self.id2word)):
            assert self.word2id[self.id2word[i]] == i

    def index(self, word):
        """
        Returns the index of the specified word.
        """
        return self.word2id[word]

    def prune(self, max_vocab):
        """
        Limit the vocabulary size.
        """
        assert max_vocab >= 1
        self.id2word = {k: v for k, v in self.id2word.items() if k < max_vocab}
        self.word2id = {v: k for k, v in self.id2word.items()}
        self.check_valid()


def parse_args():
    parser = argparse.ArgumentParser(description="Extract the word embeddings from a model.")
    parser.add_argument('--model', type=str, help="model path")
    parser.add_argument('--dict', type=str, help="dictionary path")
    parser.add_argument('--name', type=str, help="emb name")
    parser.add_argument('--dest', type=str, help="path to save emb")
    return parser.parse_args()

def main(args):
    model = torch.load(args.model)
    emb = model["model"]["encoder.embed_tokens.weight"]
    dico = ['<s>', '<pad>', '</s>', '<unk>']
    with open(args.dict, 'r') as f:
        for line in f:
            dico.append(line.split(" ")[0])
    
    id2word = {}
    word2id = {}
    for id, w in enumerate(dico):
        id2word[id] = w
        word2id[w] = id
    dico = Dictionary(id2word, word2id, args.name)
    data = {
        'dico': dico,
        'vectors': emb
    }
    torch.save(data, args.dest)
    


if __name__ == "__main__":
    args = parse_args()
    main(args)
