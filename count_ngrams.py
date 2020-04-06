from collections import Counter
import argparse

import nltk
from nltk.util import ngrams

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()

    unique_grams = Counter()

    with open(args.path) as f:
        for line in f.readlines():
            line = line.strip()
            words = nltk.word_tokenize(line)
            grams = ngrams(words, n=3)
            unique_grams.update(grams)

    print('Unique ngrams', len(unique_grams))
    print('Total ngrams', sum(unique_grams.values()))

if __name__ == '__main__':
    main()