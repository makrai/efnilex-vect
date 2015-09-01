#coding=utf-8

import argparse
import logging
from itertools import izip
import os

import numpy
from scipy.spatial.distance import cdist
from nearpy import Engine
from nearpy.hashes import PCABinaryProjections

class SenseTranslator():
    """
    Finds translations from the source VSM to the target VSM.
    Input in the typical case:
        * translation matrix
        * a source model with multiple prototypes (i.e. vectors) corresponding
            to different senses of the word forms, and
        * the target model with a single prototype (usual word2vec format). 
    Output: a dictionary containing triplets of
    
        1. the headword (HWD): a source word form, one of whose meanings the
            record corresponds to)
        2. neighbors: source words that are nearest in the model to the actual
            sense of the HWD, and
        3. translations: target words that are the translation of the actual
            meaning of the HWD by the linear translation model.

    A Hungarian to English example:

        jelentés    készített, közlemény        report
        jelentés    értelmezés, tulajdonnév     meaning

    """ 
    def __init__(self):
        format_ = "%(asctime)s %(module)s (%(lineno)s) %(levelname)s %(message)s"
        logging.basicConfig(level=logging.DEBUG, format=format_)
        self.parse_args()
        logging.info(
            'reading translation mx from {}'.format(self.args.mx))
        self.mx = numpy.genfromtxt(self.args.mx)
        self.sr_vocab, self.sr_vecs = self.get_embed(self.args.sr_embed)
        self.tg_vocab, self.tg_vecs = self.get_embed(self.args.tg_embed)
        self.sr_engine = self.get_engine(self.sr_vocab, self.sr_vecs)
        self.tg_engine = self.get_engine(self.tg_vocab, self.tg_vecs)
        self.outfile = open(self.args.outfile, mode='w')

    def parse_args(self):
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument('mx')
        arg_parser.add_argument('sr_embed')
        arg_parser.add_argument('tg_embed')
        arg_parser.add_argument('outfile')
        self.args = arg_parser.parse_args()

    def get_embed(self, filen):
        filenp, ext = os.path.splitext(filen)
        logging.info('getting embedding from {} ...'.format(filen))
        if ext in ['.w2v', '.mpt']:
            infile = open(filen)
            header = infile.readline().strip()
            vocab_size, dim = [int(token) for token in header.split()]
            vocab = [line.strip().split()[0] for line in infile.readlines()]
            assert len(vocab) == vocab_size
            vecs = numpy.genfromtxt( 
                open(filen).readlines()[:300001], skip_header=1,
                usecols=numpy.arange(1,dim+1), comments=None, dtype='float32')
            filen_npz = '{}.npz'.format(filenp)
            if not os.path.isfile(filen_npz):
                logging.info('saving embedding to {}'.format(filen_npz))
                numpy.savez_compressed(filen_npz, vocab, vecs)
        elif ext == '.npz':
            vocab = numpy.load(filen)['arr_0']
            vecs = numpy.load(filen)['arr_1']
        else:
            raise Exception('Unknown embedding extension: {}'.format(ext))
        logging.debug('... with shape {} ({})'.format(vecs.shape, vecs.dtype))
        return vocab, vecs

    def get_engine(self, vocab, vecs):
        hashes = [PCABinaryProjections('ne1v', 1, vecs[:1000,:].T)]
        #                                      ^ number of hasheses
        engine = Engine(
            vecs.shape[1], lshashes=hashes,
            distance=[],
            vector_filters=[])
        for ind, vec in enumerate(vecs):
            if not ind % 100000:                
                logging.debug( 
                    '{} words added to nearpy engine'.format(ind))
            engine.store_vector(vec, ind)
        return engine 

    def near_words(self, engine, vec, vocab):
        near_vecs, near_inds = izip(*engine.neighbours(vec.reshape(-1)))
        distances = cdist(near_vecs, vec.reshape((1,-1)), 'cosine').reshape(-1)
        inds_among_near = numpy.argsort(distances)[:10]
        top_indices_ranked = [near_inds[i] 
                              for i in inds_among_near]
        return [vocab[ind] for ind in top_indices_ranked]

    def main(self):
        logging.info('writing dictionary to {} ...'.format(self.args.outfile))
        towarn = []
        for i, (hwd, sr_vec) in enumerate(izip(self.sr_vocab, self.sr_vecs)):
            if not i % 1000:
                msg = '{} words translated'.format(i)
                if towarn:
                    msg +=', except for {}'.format(', '.join(towarn[:9]))
                logging.debug(msg)
                towarn = []
            try:
                self.outfile.write('{}\t{}\t{}\n'.format(
                    hwd, 
                    ', '.join(self.near_words(self.sr_engine, 
                                              sr_vec, self.sr_vocab)[1:5]),
                    ', '.join(self.near_words(self.tg_engine,
                                              sr_vec.dot(self.mx), self.tg_vocab))))
            except:
                towarn.append(hwd)


if __name__ == "__main__":
    SenseTranslator().main()