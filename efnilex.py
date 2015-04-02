import argparse
import codecs
from collections import defaultdict
from itertools import izip
import logging
import os.path
import pickle
import re

from bidict import bidict
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.linear_model import LinearRegression
from gensim.models import Word2Vec
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.filters import NearestFilter
from nearpy.distances import CosineDistance


class LinearTranslator:
    """
    Collects translational word pairs from neural language models (embeddings)
    for the source and the target language.  The translation model is a linear
    mapping between the two vector spaces following Mikolov et al (2013
    Exploiting...), trained and tested on a seed dictionary.
    """
    def __init__(self, output_dir=None):
        self.args = self.parse_args()
        self.output_dir = output_dir if output_dir else self.args.output_dir
        if not os.path.isdir(output_dir):
            os.mkdir(self.args.output_dir)
        mode_dir = '{}/{}'.format(self.output_dir, self.args.mode)
        if not os.path.isdir(mode_dir):
            os.mkdir(mode_dir)
        self.get_outfilen()
        self.config_logger(self.args.log_to_err)
        if self.args.mode in ['collect', 'score']:
            self.init_biling()

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-m", "--mode", dest='mode', 
            choices=['collect', 'score', 'analogy'], default='collect')
        parser.add_argument(
            '--output-directory', dest=output_dir)
        parser.add_argument(
            '-o', '--output-file-name', dest='outfilen', 
            help='prefix of names of output and log files without path')
        parser.add_argument(
            '-l', '--log-to-screen', action='store_true', dest='log_to_err')
        parser.add_argument(
            "-s", "--source-lang-mod", dest='sr_langm_filen', type=str)
        parser.add_argument(
            "-t", "--target-lang-mod", dest='tg_langm_filen', type=str)
        parser.add_argument(
            "-d", "--seed-dict", type=str, 
            default="wikt2dict",
            dest='seed_filen')
        parser.add_argument(
            "-r", "--reverse", 
            action='store_true',
            help="use if the seed dict contains pair in reverse order")
        parser.add_argument(
            "-v", "--translate-oov", dest='trans_freq_oov', action='store_true', 
            help='Translate frequent words without a seed dictionary translation')
        parser.add_argument(
            '-e', '-exact-neighbor', dest='exact_neighbor',
            action='store_true')
        parser.add_argument(
            '-p', '--pair', default='en_hu', 
            help='language pair (in alphabetical order)')
        parser.add_argument(
            '-c', '--restrict-embed', choices = ['n', 's', 't', 'st'],
            dest='restrict_embed', default='n')
        parser.add_argument(
            "-f", "--forced-stem", dest='forced_stem', action='store_true',
            help='consider only "forced" stems')
        return parser.parse_args()

    def config_logger(self, log_to_err):
        level = logging.DEBUG
        format_ = "%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s"
        if log_to_err:
            logging.basicConfig(level=level, format=format_) 
        else:
            filename = '{}/{}/log/{}'.format(self.output_dir, self.args.mode,
                                            self.args.outfilen)
            if os.path.isfile(filename):
                os.remove(filename)
            logging.basicConfig(filename=filename, level=level,
                                format=format_) 

    def get_outfilen(self):
        if not self.args.outfilen:
            if self.args.mode == 'analogy':
                self.args.outfilen = self.strip_embed_filen(
                    self.args.sr_langm_filen)
            else:
                # assert self.args.mode in ['collect', 'score']
                self.args.outfilen = '{}__{}__{}_{}f_c{}_o{}'.format(
                    self.strip_embed_filen(self.args.sr_langm_filen),
                    self.strip_embed_filen(self.args.tg_langm_filen),
                    self.args.seed_filen.split('/')[-1], 
                    int(self.args.forced_stem),
                    self.args.restrict_embed,
                    int(self.args.trans_freq_oov))

    def strip_embed_filen(self, old_filen):
        path, new_filen = os.path.split(old_filen)
        new_filen, ext = os.path.splitext(new_filen)
        if ext[1:] in ['w2v', 'gensim', 'gz', 'bin', 'pkl', 'txt']:
            return self.strip_embed_filen(new_filen)
        else:
            return old_filen

    def init_biling(self):
        out_dict_filen = self.output_dir + self.args.outfilen
        if os.path.isfile(out_dict_filen): 
            raise Exception(
                'file for collected translation pairs exists {}'.format(
                    out_dict_filen))
        self.outfile = open(out_dict_filen, mode='w')
        self.train_needed = 5000 if self.args.mode == 'collect' else 20000
        self.test_needed= 1000 if self.args.mode == 'collect' else 20000

    def main(self):
        if self.args.mode == 'analogy':
            self.analogy_main()
        else: 
            # This branch is for modes 'collect' and 'score'
            self.biling_main()

    def analogy_main(self):
        self.config_logger(self.args.log_to_err)
        model = self.load_embed(self.args.sr_langm_filen, type_='model')
        lower = False
        for label in ['+l', '1l']:
            if label in self.args.sr_langm_filen:
                lower = True
        return model.accuracy( 
            os.path.expanduser('~') +
            '/project/efnilex/vector/analogy/hu/questions-words.txt',
            lower=lower)

    def biling_main(self):
        """
        This is the main function on the two bilingual tasks, 'collect' and
        'score'.
        """
        self.sr_model = self.load_embed(self.args.sr_langm_filen)
        self.tg_model = self.load_embed(self.args.tg_langm_filen)
        # tg_model.syn0.astype('float32', casting='same_kind', copy=False)
        self.tg_index = bidict(enumerate(self.tg_model.index2word))
        self.read_seed()
        self.get_trans_model()
        self.sr_embed_f = codecs.open(self.args.sr_langm_filen,
                                      encoding='utf-8')
        sr_position, ootg = self.get_training_data()
        logging.debug(
            'out of target embed: {}'.format(
                '; '.join(word.encode('utf8') for word in ootg[:20])))
        if not sr_position:
            raise Exception(
                'Too few training pairs ({})'.format(train_collected))
        logging.info('fitting model')
        self.trans_model.fit(np.array(self.sr_train), np.array(self.tg_train))
        logging.info('testing')
        if self.args.mode == 'collect':
            self.collect_main(sr_position)
        elif self.args.mode == 'score':
            self.score_main()

    def load_embed(self, filen, write_vocab=False, type_='mx_and_bidict'):
        if re.search('gensim', filen):
            return Word2Vec.load(filen)
        elif re.search('polyglot-..\.pkl$', filen):
            model = Word2Vec()
            model.index2word, model.syn0 = pickle.load(
                open(filen, mode='rb'))
            return model
        elif re.search('webcorp/polyglot', filen):
            from polyglot2.polyglot2 import Polyglot
            logging.info('loading lang_mod from {}'.format(filen))
            return Polyglot.load_word2vec_format(filen)
        elif 'hunvec' in filen:
            #mx = nnlm.model.get_params()[0].get_value()
            raise NotImplementedError
        else:
            # The embedding in the format of the original C code 
            return Word2Vec.load_word2vec_format(filen, binary='bin' in filen)

    def read_word_and_vec(self, line):
        cells = line.strip().split(' ')
        word = cells[0]
        vec = np.array([float(coord) for coord in
                        cells[1:]]).astype('float32')
        return word, vec

    def read_seed(self):
        # TODO do we need fallback?
        columns = [1,3] if 'wikt2dict' in self.args.seed_filen else range(2)
        if self.args.reverse:
            columns.reverse()
        filen = self.args.seed_filen
        self.seed_dict = {}
        with codecs.open(filen, encoding='utf-8') as file_:
            for line in file_.readlines():
                separator = '\t' if '\t' in line else ' '
                cells = line.strip().split(separator)
                sr_word, tg_word = [cells[index] for index in columns]
                if sr_word in self.seed_dict:
                    # It is assumed that the first translation is the best.
                    # E.g. in the case of wikt2dict, the first translation is
                    # present in more editions of Wiktionary.
                    continue
                self.seed_dict[sr_word] = tg_word
        logging.info('{} seed pairs e.g. {}'.format(len(self.seed_dict),
                                                    self.seed_dict.items()[:5]))
        if len(self.seed_dict.keys()) < self.train_needed:
                logging.error('too few training pairs')
                raise Exception('too few training pairs')

    def get_trans_model(self):
        self.trans_model = LinearRegression() # TODO parametrize (penalty, ...)
        # TODO self.trans_model = ElasticNet(), Lasso, Ridge, SVR
        # http://stackoverflow.com/questions/19650115/which-scikit-learn-tools-can-handle-multivariate-output
        # TODO crossvalidation
        logging.info(str(self.trans_model))

    def get_training_data(self):
        self.sr_train = []
        self.sr_position = 0
        self.tg_train = []        
        self.sr_freq_not_seed = []
        ootg = []
        train_collected = 0
        #self.sr_embed_f.readline() # The header is skipped.
        for i, (sr_word, sr_vec) in enumerate(izip(self.sr_model.syn0,
                                                   self.sr_model.index2word)):
            if train_collected < self.train_needed:
                if sr_word in self.seed_dict:
                    if not train_collected % 1000:
                        logging.debug(
                            '{} training items collected'.format(train_collected))
                    train_collected += 1
                    tg_word = self.seed_dict[sr_word]
                    if tg_word in self.tg_index.itervalues():
                        if sr_vec is None:
                            # This branch is for mode 'score'
                            sr_vec = self.embeds['sr'][sr_word]
                        self.sr_train.append(sr_vec)
                        self.tg_train.append(self.tg_model.syn0[self.tg_index[:tg_word]])
                    else:
                        ootg.append(tg_word)
                else:
                    self.sr_freq_not_seed.append((sr_word, sr_vec))
            else:
                return i, ootg
        else:
            return None, ootg


    def get_nearpy_engine(self, top_n=10):
        rbps = []
        for _ in xrange(1):
            # TODO 
            #   less or more projections
            #   other types of projections
            rbps.append(RandomBinaryProjections('rbp', 10))
        dim = self.tg_model.syn0.shape[1]
        self.engine = Engine(dim, lshashes=rbps, distance=CosineDistance(),
                             vector_filters=[NearestFilter(top_n)])
        for ind, vec in enumerate(self.tg_model.syn0):
            if not ind % 100000:
                logging.info(
                    '{} target words added to nearpy engine'.format(ind))
            self.engine.store_vector(vec, ind)

    def collect_main(self, sr_position):
        """
        First look for translations with gold data to see precision, and
        compute translations of frequent words without seed data after that.
        """
        self.get_nearpy_engine()
        self.collected = 0
        self.has_seed = 0
        self.score_at_5 = 0
        self.score_at_1 = 0
        self.reved_neighbors = defaultdict(set)
        for sr_word, sr_vec in izip(self.sr_model.syn0,
                                    self.sr_model.index2word)[sr_position:]:
            self.test_item(sr_word, sr_vec)
            if self.has_seed >= self.test_needed:
                # TODO If the goal is not only measuring precision but
                # collecting as many translations as possible, and sr lang_mod
                # is properly chosen, no breaking is needed. 
                break
        else: 
            logging.error(
                'Only {} test pairs with gold data. {} needed.'.format(
                    self.has_seed, self.test_needed))
        if self.has_seed != self.test_needed:
            logging.debug((self.has_seed, self.test_needed))
        logging.info('on {} words, prec@1: {:.2%}\tprec@5: {:.2%}'.format(
            self.has_seed, 
            *[float(score)/self.has_seed 
              for score in [ self.score_at_1, self.score_at_5]]))
        if self.args.trans_freq_oov:
            logging.info(
                'Translating frequent words without seed data...')
            for sr_word, sr_vec in self.sr_freq_not_seed:
                self.test_item(sr_word, sr_vec)
        else:
            logging.info('Frequent words without seed data skipped.')

    def test_item(self, sr_word, sr_vec, prec_at=9):
        self.collected += 1
        if not self.collected % 100:
            logging.debug(
                '{} translations collected, {} have reference translation'.format(
                    self.collected, self.has_seed))
        guessed_vec = self.trans_model.predict(sr_vec.reshape((1,-1))).astype(
            'float32').reshape((-1))
        if self.exact_neighbor:
            gold_tg_word, gold_rank = self.eval_item_with_gold(sr_word,
                                                               guessed_vec=guessed_vec)
        else:
            _, tg_indices_ranked, distances = zip(*self.engine.neighbours(guessed_vec))
            gold_tg_word, gold_rank = self.eval_item_with_gold(sr_word,
                                                               tg_indices_ranked=tg_indices_ranked)
        self.outfile.write(
            '{sr_word}\t{gold_tg_word}\t{gold_rank}\t{cos_dist:.4}\t{tg_words}\n'.format(
                sr_word=sr_word.encode('utf-8'),
                gold_tg_word=gold_tg_word.encode('utf-8'),
                gold_rank=gold_rank,
                cos_dist=distances[0],
                tg_words=' '.join(self.tg_index[ind].encode('utf-8') 
                          for ind in tg_indices_ranked[:prec_at])))

    def eval_item_with_gold(self, sr_word, tg_indices_ranked=None, guessed_vec=None):
        """
        Looks up the gold target word, computes its similarity rank to the
        computed target vector, and books precision.
        # TODO TODO
        tg_norms = np.apply_along_axis(
            np.linalg.norm, 1, self.tg_model.syn0).reshape(-1,1)
        self.tg_model.syn0 /= tg_norms
        self.tg_model.syn0 = self.tg_model.syn0.T
        """
        if sr_word in self.seed_dict:
            gold_tg_word = self.seed_dict[sr_word]
            self.has_seed += 1
            if gold_tg_word in self.tg_index.itervalues():
                if not tg_indices_ranked:
                    sim_row = guessed_vec.dot(self.tg_model.syn0)
                    tg_indices_ranked = np.argsort(-sim_row)
                if self.tg_index[:gold_tg_word] in tg_indices_ranked:
                    gold_rank = tg_indices_ranked.index(
                        self.tg_index[:gold_tg_word])
                    if gold_rank < 5:
                        self.score_at_5 += 1
                        if gold_rank == 0:
                            self.score_at_1 += 1
            else:
                gold_rank = '>10' # TODO
        else:
            gold_tg_word, gold_rank = '', ''
        return gold_tg_word, gold_rank

    def score_main(self):
        raise NotImplementedError
        guessed_vecs = self.trans_model.predict(np.array(sr_vecs))
        for sr_word, tg_word, sim in zip(sr_words, tg_words, sim_mx):
            self.outfile.write('\t'.join([sr_word, tg_word,
                                           str(sim)])+'\n')


if __name__=='__main__':
    output_dir = '/home/makrai/project/efnilex/vector'
    # TODO '/mnt/store/home/makrai/project/efnilex/vector/'
    LinearTranslator(output_dir).main()
