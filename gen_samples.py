import torch
from fairseq.models.bart import BARTModel
import logging

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate():
    logger.info('***** Begin Evalution *****')

    bart = BARTModel.from_pretrained(
        'checkpoints/',
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path='bin'
    )

    bart.cuda()
    bart.eval()

    with open('./data/val.source') as source, open('./data/val.hypo', 'w') as fout:
        for sline in source:
            sline = sline.strip().lower()
            with torch.no_grad():
                hypo = bart.sample(sline, beam=5, lenpen=2.0, max_len_b=100, min_len=20, no_repeat_ngram_size=3)
            fout.write(hypo.lower())

if __name__ == "__main__":
    evaluate()
