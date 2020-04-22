import argparse
import os
import numpy as np

from nltk.translate.bleu_score import sentence_bleu

def evaluate_bleu(args):
    source_file = args.source
    hypo_file = args.hypo

    with open(source_file) as sf, open(hypo_file) as hf:
        refs, hypos = [], []
        line_no = 1
        sentence_scores = []

        for ref, hypo in zip(sf.readlines(), hf.readlines()):
            ref = ref.strip()
            hypo = hypo.strip()

            refs.append(ref)
            hypos.append(hypo)

            if line_no % 3 == 0:
                scores = [sentence_bleu(refs, h) for h in hypos]
                sentence_scores.append(np.mean(scores))

                refs, hypos = [], []

            line_no += 1

    bleu_score = np.mean(sentence_scores)
    print("BLEU score", bleu_score)
            

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--target")
    parser.add_argument("--hypo")

    args = parser.parse_args()

    evaluate_bleu(args)

if __name__ == "__main__":
    main()