import torch
import torch.nn as nn
import warnings
import argparse
from src.utils import Dictionary

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="model path")
    parser.add_argument('--emb1', type=str, required=True, help="embedding1 path (aligned emb)")
    parser.add_argument('--emb2', type=str, required=True, help="embedding2 path (original emb)")
    parser.add_argument('--d', type=str, required=True, help="final dict path")
    parser.add_argument('--dest', type=str, required=True, help="path to save model")
    parser.add_argument('--add-mask', action="store_true", help="if add <mask> token")
    parser.add_argument('--add-tag', action="store_true", help="if add __pivot__ token")

    return parser.parse_args()

def main(args):
    model = torch.load(args.model)

    data1 = torch.load(args.emb1)
    dico1, emb1 = data1['dico'], data1['vectors']

    data2 = torch.load(args.emb2)
    dico2, emb2 = data2['dico'], data2['vectors']

    assert torch.equal(model["model"]["encoder.embed_tokens.weight"], emb2)
    assert torch.equal(model["model"]["encoder.embed_tokens.weight"], model["model"]["decoder.embed_tokens.weight"])
    assert emb1.shape[1] == emb2.shape[1]
    assert emb1.shape[1] == emb2.shape[1]
    assert emb1.shape[0] == len(dico1)
    assert emb2.shape[0] == len(dico2)

    emd_dim = emb1.shape[1]

    w_li = ['<s>', '<pad>', '</s>', '<unk>']
    with open(args.d, 'r') as f:
        for line in f:
            w = line.strip().split()[0]
            w_li.append(w)

    if args.add_tag:
        w_li.append('__pivot__')

    if args.add_mask:
        w_li.append('<mask>')


    
    new_emb = torch.empty((len(w_li), emd_dim), dtype=model["model"]["encoder.embed_tokens.weight"].dtype)
    for i, w in enumerate(w_li):
        if w in dico2.word2id.keys():
            new_emb[i] = emb2[dico2.word2id[w]]
        elif w in dico1.word2id.keys():
            new_emb[i] = emb1[dico1.word2id[w]]
        else:
            warnings.warn(f"{w} is neither in the dico1 nor in the dico2, use random init")
            nn.init.normal_(new_emb[i], mean=0, std=emd_dim**-0.5)

    
    model["model"]["encoder.embed_tokens.weight"] = new_emb
    model["model"]["decoder.embed_tokens.weight"] = new_emb
    model["model"]["decoder.output_projection.weight"] = new_emb

    torch.save(model, args.dest)




if __name__ == "__main__":
    args = parse_args()
    main(args)
