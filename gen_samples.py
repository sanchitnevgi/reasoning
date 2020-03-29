import torch
from fairseq.models.bart import BARTModel
import logging

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate():
    bart = BARTModel.from_pretrained(
        'checkpoints/',
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path='cnn_dm-bin'
    )

    bart.cuda()
    bart.eval()

    count = 1
    bsz = 32

    with open('data/val.source') as source, open('data/val.hypo', 'w') as fout:
        sline = source.readline().strip()
        slines = [sline]
        for sline in source:
            if count % bsz == 0:
                with torch.no_grad():
                    hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)

                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + '\n')
                    fout.flush()
                slines = []

            slines.append(sline.strip())
            count += 1
        if slines != []:
            hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()

if __name__ == "__main__":
    evaluate()