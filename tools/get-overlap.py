import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d1", type=str, help="dictionary one")
    parser.add_argument("--d2", type=str, help="dictionary two")
    return parser.parse_args()

def dict2set(dict_file):
    res = set()
    with open(dict_file, 'r') as f:
        for line in f:
            res.add(line.split()[0])
    return res

def main(args):
    set1 = dict2set(args.d1)
    set2 = dict2set(args.d2)
    print(f"|d1|={len(set1)}", file=sys.stderr)
    print(f"|d2|={len(set2)}", file=sys.stderr)
    print(f"|d1 ∩ d2|={len(set1 & set2)}", file=sys.stderr)
    print(f"|d1 - d2|={len(set1 - set2)}", file=sys.stderr)
    print(f"|d2 - d1|={len(set2 - set1)}", file=sys.stderr)
    print(f"|d2 ∪ d1|={len(set2 | set1)}", file=sys.stderr)
    print(f"|d1 ∩ d2| / |d2 ∪ d1|={len(set1 & set2)/len(set2 | set1)}", file=sys.stderr)
    for w in set1 & set2:
        print(f"{w} {w}")

if __name__ == '__main__':
    args = parse_args()
    main(args)