# python3
import argparse
from collections import defaultdict
import logging

class DictEvaluator():
    def __init__(self, args):
        self.dict_to_eval_filen = args.dict_to_eval
        self.ref_dict_filen = args.ref_dict

    def main(self):
        self.read_reference()
        self.evaluate_dict()

    def read_reference(self):
        self.ref_dict = defaultdict(set)
        with open(self.ref_dict_filen) as dict_tsv_f:
            for line in dict_tsv_f:
                sr, tg = line.strip().split()
                self.ref_dict[sr].add(tg)

    def evaluate_dict(self):
        with open(self.dict_to_eval_filen) as dict_tsv_f:
            step = 1000
            good = 0
            total = 0
            for line in dict_tsv_f:
                if total == step:
                    #print('{}\t{:.2%}'.format(total/1000,good/step))
                    good = 0 
                    total = 0
                sr, tg, score = line.strip().split()
                if sr in self.ref_dict:
                    total += 1
                    if tg in self.ref_dict[sr]:
                        good += 1
                    else:
                        logging.debug('{:.4}\t{}\t{}\t{}'.format(
                            score, sr, ', '.join(self.ref_dict[sr]), tg))

def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('dict_to_eval')
    arg_parser.add_argument('ref_dict')
    return arg_parser.parse_args()

if __name__ == '__main__':
    fmt = "%(asctime)s %(module)s (%(lineno)s) %(levelname)s %(message)s"
    logging.basicConfig(format=fmt, level=logging.DEBUG)
    DictEvaluator(parse_args()).main()
