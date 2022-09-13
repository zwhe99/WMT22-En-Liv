import os
from prettytable import PrettyTable
from sacrebleu.metrics import BLEU, CHRF, TER

SCRIPT_DIR=os.path.dirname(os.path.realpath(__file__))
MAIN_DIR=os.path.join(SCRIPT_DIR, "../")

EVAL_DATA_DIR=os.path.join(MAIN_DIR, "data/eval")
BENCHMARK_SRC_FILE=os.path.join(EVAL_DATA_DIR, "benchmark-test.src")

SUBSETS = ["Full", "Facebook", "livones.net", "dictionary", "trilium", "stalte", "esuka", "satversme"]


def get_start_end_lines(subset):
    assert subset in SUBSETS

    line_no_li = []
    with open(BENCHMARK_SRC_FILE, 'r') as f:
        for idx, line in enumerate(f):
            if line.strip() == subset or subset == "Full":
                line_no_li.append(idx)
    return min(line_no_li), max(line_no_li)

if __name__ == "__main__":
    langs = ["en", "et", "lv", "liv"]
    lang_pairs = [f"{src_lng}-{tgt_lng}" for tgt_lng in langs for src_lng in langs if tgt_lng != src_lng]


    for ref_prefix in ["benchmark-test", "benchmark-test.nfkc"]:
        x = PrettyTable(title=("Before" if ref_prefix == "benchmark-test" else "After") + " normalizing references to NFKC")
        x.field_names = ["subset"] + lang_pairs + ["avg."]

        for subset in SUBSETS:
            start_line, end_line = get_start_end_lines(subset)
            row = [subset]

            bleu_scores = []
            for tgt_lng in ["en", "et", "lv", "liv"]:
                for src_lng in ["en", "et", "lv", "liv"]:

                    if tgt_lng == src_lng:
                        continue

                    with open(os.path.join(SCRIPT_DIR, f"{src_lng}-{tgt_lng}.hyp"), 'r') as f:
                        sys = f.readlines()[start_line: end_line+1]

                    with open(os.path.join(EVAL_DATA_DIR, f"{ref_prefix}.{tgt_lng}"), 'r') as f:
                        ref = [f.readlines()[start_line: end_line+1]]
                    
                    bleu_scores.append(BLEU().corpus_score(sys, ref).score)
            
            row += [f"{v:.2f}" for v in bleu_scores] + [f"{sum(bleu_scores)/len(bleu_scores):.2f}"]
            x.add_row(row)
        print(x)