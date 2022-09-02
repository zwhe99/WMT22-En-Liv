import torch
import json
import argparse

parser= argparse.ArgumentParser();
parser.add_argument("--model",help="Model path")

if __name__ == '__main__':
    args= parser.parse_args()
    checkpoint= torch.load(args.model)
    par_dict= vars(checkpoint['args'])
    print(json.dumps(par_dict, indent=2, sort_keys=True))