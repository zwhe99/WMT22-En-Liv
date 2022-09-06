import argparse
import nltk
import unicodedata

def get_freqs(line):
    words = line.split()
    unigrams = list(nltk.ngrams(words, 1))
    bigrams = list(nltk.ngrams(words, 2))
    trigrams = list(nltk.ngrams(words, 3))

    unigram_freqs = nltk.FreqDist(unigrams)
    bigram_freqs = nltk.FreqDist(bigrams)
    trigram_freqs = nltk.FreqDist(trigrams)

    return unigram_freqs, bigram_freqs, trigram_freqs

def get_counts(unigram_freqs, bigram_freqs, trigram_freqs):
    counts = []
    for freqs in [unigram_freqs, bigram_freqs, trigram_freqs]:
        if len(freqs) > 0:
            _, most_common_gram_count =  freqs.most_common(1)[0]
        else:
            most_common_gram_count = 0
        counts.append(most_common_gram_count)
    return counts

def is_repeat(src_line, hyp_line):
    hyp_unigram_freqs, hyp_bigram_freqs, hyp_trigram_freqs = get_freqs(hyp_line)
    hyp_counts = get_counts(hyp_unigram_freqs, hyp_bigram_freqs, hyp_trigram_freqs)

    if max(hyp_counts) <= 3:
        return False
    else:
        src_unigram_freqs, src_bigram_freqs, src_trigram_freqs = get_freqs(src_line)
        src_counts = get_counts(src_unigram_freqs, src_bigram_freqs, src_trigram_freqs)
        if abs(max(src_counts) - max(hyp_counts)) <= 2:
            return False
        else:
            return True

def decimal_point(line):
    res = list(line)
    for idx in range(1, len(line)-1):
        if line[idx] == ',' and line[idx-1].isdigit() and line[idx+1].isdigit():
            res[idx] = '.'
    return "".join(res)

def main(args):
    if args.lang == 'liv' and args.no_repeat_hyp_file is None:
        parser.error("'--lang liv' requires '--no-repeat-hyp-file'.")
    
    if args.lang == 'liv':
        with open(args.src_file) as sf, open(args.hyp_file) as hf , open(args.no_repeat_hyp_file) as nrhf: 
            for (src_line, hyp_line, no_repeat_hyp_line) in zip(sf, hf, nrhf):
                # Apply NFC normalization
                hyp_line = unicodedata.normalize('NFC', hyp_line)
                no_repeat_hyp_line = unicodedata.normalize('NFC', no_repeat_hyp_line)

                # Replace all the "httpshttp" with "https://"
                if 'http' in src_line:
                    hyp_line = hyp_line.replace("httpshttp", "https://")
                    no_repeat_hyp_line = no_repeat_hyp_line.replace("httpshttp", "https://")

                # Replace <unk> with empty string.
                hyp_line = hyp_line.replace("<unk>", "").replace("  ", " ")
                no_repeat_hyp_line = no_repeat_hyp_line.replace("<unk>", "").replace("  ", " ")
                
                # When a comma appears in between two digits, replace it with a decimal point (only for Liv).
                hyp_line = decimal_point(hyp_line)
                no_repeat_hyp_line = decimal_point(no_repeat_hyp_line)

                # Replace the sentences that detected as repetition with no-repeat one (only for Liv).
                if is_repeat(src_line, hyp_line):
                    print(no_repeat_hyp_line, end="")
                else:
                    print(hyp_line, end="")
    elif args.lang == 'en':
        with open(args.src_file) as sf, open(args.hyp_file) as hf: 
            for (src_line, hyp_line) in zip(sf, hf):
                # Apply NFC normalization
                hyp_line = unicodedata.normalize('NFC', hyp_line)

                # Replace all the "httpshttp" with "https://"
                if 'http' in src_line:
                    hyp_line = hyp_line.replace("httpshttp", "https://")

                # Replace <unk> with empty string.
                hyp_line = hyp_line.replace("<unk>", "").replace("  ", " ")
                print(hyp_line, end="")
    else:
        raise ValueError

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-processing for en-liv/liv-en")
    parser.add_argument("--src-file", required=True, type=str, help="source file")
    parser.add_argument("--hyp-file", required=True, type=str, help="model output")
    parser.add_argument("--no-repeat-hyp-file", required=False, type=str, help="model output with no-repeat constraint")
    parser.add_argument("-l", "--lang", required=True, type=str, choices=['en', 'liv'], help="language")
    parser.add_argument("--encoding", default='utf-8', help='character encoding for input/output')
    main(parser.parse_args())