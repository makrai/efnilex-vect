import argparse
import logging

import numpy
from scipy.spatial.distance import cosine

class DictionaryScorer():
    def __init__(self):
        format_ = "%(asctime)s: %(module)s (%(lineno)s) %(levelname)s %(message)s"
        logging.basicConfig(level=logging.DEBUG, format=format_)
        self.parse_args()
        self.outfile = open(self.args.outfile, mode='w') 
        self.indict = [line.strip().split()
                       for line in open(self.args.indict)]
        if self.args.reverse:
            self.indict = map(lambda p: list(reversed(p)), self.indict)
        logging.info('reading trans mx')
        self.trans_mx = numpy.genfromtxt(self.args.mx)
        sr_voc = set(s for s, _ in self.indict)
        tg_voc = set(t for _, t in self.indict)
        self.sr_embed = self.read_embed(self.args.sr_embed, sr_voc)
        self.tg_embed = self.read_embed(self.args.tg_embed, tg_voc)

    def parse_args(self):
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument('mx')
        arg_parser.add_argument('indict')
        arg_parser.add_argument('sr_embed')
        arg_parser.add_argument('tg_embed')
        arg_parser.add_argument('outfile')
        arg_parser.add_argument(
            '-r', '--reverse', action='store_true', 
            help='source and target language are reversed in the dictionary')
        self.args = arg_parser.parse_args()

    def read_embed(self, filen, voc):
        logging.info('reading embedding from {}'.format(filen))
        embed = {}
        for line in open(filen):
            word, vec_str = line.strip().split(' ', 1)
            if word in voc:
                    embed[word] = numpy.array([float(cell) for cell in
                                                 vec_str.split()])
        return embed

    def main(self):
        logging.info('scoring')
        for srw, tgw in self.indict:
            if srw in self.sr_embed and tgw in self.tg_embed:
                self.outfile.write('{}\n'.format(cosine(self.tg_embed[tgw],
                             self.sr_embed[srw].reshape(1,-1).dot(self.trans_mx))))
            else:
                self.outfile.write('{}\n'.format(2))


if __name__ == '__main__':
    DictionaryScorer().main()
