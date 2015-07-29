import argparse
import codecs
import cPickle
#import cProfile
from itertools import izip
import logging
import os.path
import pickle
import re

from bidict import bidict
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from gensim.models import Word2Vec
from nearpy import Engine
from nearpy.hashes import PCABinaryProjections

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
        self.mode_dir = '{}/{}/'.format(self.output_dir, self.args.mode)
        if not os.path.isdir(self.mode_dir):
            os.mkdir(self.mode_dir)
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
            '-l', '--log-to-stderr', action='store_true', dest='log_to_err')
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
            "-v", "--translate-oov", dest='trans_freq_oov',
            action='store_false',
            help='Not translate frequent words that are not covered by the\
            seed')
        parser.add_argument(
            '-e', '-exact-neighbour', dest='exact_neighbour',
            action='store_true',
            help="instead of approximating by nearpy, compute exact neighbours")
        parser.add_argument(
            '-p', '--pair', default='en_hu',
            help='language pair (in alphabetical order)')
        parser.add_argument(
            '-c', '--restrict-embed', choices = ['n', 's', 't', 'st'],
            dest='restrict_embed', default='n',
            help="which of the source and the target embeddings to restrict:\
            *n*one, *s*ource, *t*arget, or *st* both")
            
        parser.add_argument(
            "-f", "--forced-stem", dest='forced_stem', action='store_true',
            help='consider only "forced" stems')
        return parser.parse_args()

    def get_outfilen(self):
        """"
        Output files are named according to the pattern
        source__target__seed__opts
        where
            source describes the source language model
            target describes the target language model
            seed describes the seed dictionary
            opts describes the value of command line options v, f, c,
            and e (mostly for debugging or papameter analysis)
        """
        if not self.args.outfilen:
            if self.args.mode == 'analogy':
                self.args.outfilen = self.strip_embed_filen(
                    self.args.sr_langm_filen)
            else:
                assert self.args.mode in ['collect', 'score']
                self.args.outfilen = '{}__{}__{}__{}v-{}f-c{}-{}e'.format(
                    self.strip_embed_filen(self.args.sr_langm_filen),
                    self.strip_embed_filen(self.args.tg_langm_filen),
                    self.args.seed_filen.split('/')[-1],
                    int(self.args.trans_freq_oov),
                    int(self.args.forced_stem),
                    self.args.restrict_embed,
                    int(self.args.exact_neighbour))
        self.out_dict_filen = self.mode_dir + self.args.outfilen
        if os.path.isfile(self.out_dict_filen):
            raise Exception(
                'file for collected translation pairs exists {}'.format(
                    self.out_dict_filen))

    def strip_embed_filen(self, old_filen):
        path, new_filen = os.path.split(old_filen)
        new_filen, ext = os.path.splitext(new_filen)
        if ext[1:] in ['w2v', 'gensim', 'gz', 'bin', 'pkl', 'txt']:
            return self.strip_embed_filen(new_filen)
        else:
            return old_filen

    def config_logger(self, log_to_err):
        level = logging.DEBUG
        format_ = "%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - \
        %(message)s"
        if log_to_err:
            logging.basicConfig(level=level, format=format_)
        else:
            filename = '{}/{}/log/{}'.format(self.output_dir, self.args.mode,
                                            self.args.outfilen)
            if os.path.isfile(filename):
                os.remove(filename)
            logging.basicConfig(filename=filename, level=level,
                                format=format_)

    def init_biling(self):
        self.outfile = open(self.out_dict_filen, mode='w')
        self.train_needed = 5000 if self.args.mode == 'collect' else 20000
        self.test_needed= 1000 if self.args.mode == 'collect' else 20000

    def main(self):
        if self.args.mode == 'analogy':
            self.analogy_main()
        else:
            assert self.args.mode in ['collect', 'score']
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
        # TODO? self.tg_model.syn0.astype('float32', casting='same_kind',
        # copy=False)
        self.tg_index = bidict(enumerate(self.tg_model.index2word))
        self.read_seed()
        self.get_training_data()
        self.get_trans_model()
        self.trans_model.fit(np.array(self.sr_train), np.array(self.tg_train))
        if self.args.mode == 'collect':
            self.collect_main()
        elif self.args.mode == 'score':
            self.score_main()

    def load_embed(self, filen, write_vocab=False, type_='mx_and_bidict'):
        if re.search('gensim', filen):
            return Word2Vec.load(filen)
        elif re.search('polyglot-..\.pkl$', filen):
            model = Word2Vec()
            words_t, model.syn0 = pickle.load(open(filen, mode='rb'))
            logging.info(
                'Embedding with shape {} loaded'.format(model.syn0.shape))
            model.index2word = list(words_t)
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

    def read_seed(self):
        """
        It is assumed that the first translation is the best.  E.g. in the
        case of wikt2dict, the first translation is present in more editions
        of Wiktionary.
        """
        logging.info('Reading seed dictionary from {}'.format(
            self.args.seed_filen))
        # TODO do we need fallback?
        columns = [1,3] if 'wikt' in self.args.seed_filen else range(2)
        if self.args.reverse:
            columns.reverse()
        self.seed_dict = {}
        with codecs.open(self.args.seed_filen, encoding='utf-8') as file_:
            for line in file_.readlines():
                separator = '\t' if '\t' in line else ' '
                cells = line.strip().split(separator)
                sr_word, tg_word = [cells[index] for index in columns]
                if sr_word in self.seed_dict:
                    continue
                self.seed_dict[sr_word] = tg_word
        logging.info('{} seed pairs e.g. {}'.format(
            len(self.seed_dict), 
            self.seed_dict.items()[:5]))
        if len(self.seed_dict.keys()) < self.train_needed:
                logging.error('too few training pairs')
                raise Exception('too few training pairs')

    def get_training_data(self):
        train_dat_fn = '{}/train_dat/{}__{}__{}.pkl'.format(
            self.output_dir,
            self.strip_embed_filen(self.args.sr_langm_filen),
            self.strip_embed_filen(self.args.tg_langm_filen),
            self.args.seed_filen.split('/')[-1])
        if os.path.isfile(train_dat_fn):
            logging.info('loading training data from {}'.format(train_dat_fn))
            (self.sr_train, self.sr_freq_not_seed, self.sr_position,
             self.tg_train, self.ootg) = cPickle.load(open(train_dat_fn,
                                                           mode='rb'))
            return
        self.sr_train = []
        self.sr_freq_not_seed = []
        self.tg_train = []
        self.ootg = []
        train_collected = 0
        for i, (sr_word, sr_vec) in enumerate(izip(self.sr_model.index2word,
                                                   self.sr_model.syn0)):
            if train_collected < self.train_needed:
                if sr_word in self.seed_dict:
                    tg_word = self.seed_dict[sr_word]
                    if tg_word in self.tg_index.itervalues():
                        if not train_collected % 1000:
                            logging.debug(
                                '{} training items collected'.format(
                                    train_collected))
                        train_collected += 1
                        self.sr_train.append(sr_vec)
                        self.tg_train.append(
                            self.tg_model.syn0[self.tg_index[:tg_word]])
                    else:
                        self.ootg.append(tg_word)
                else:
                    self.sr_freq_not_seed.append((sr_word, sr_vec))
            else:
                self.sr_position = i
                logging.info('At seed item {}'.format(self.sr_position))
                break
        else:
            logging.error('Too few training pairs')
            self.sr_position = None
        logging.debug('out of target embed: {}'.format(
                '; '.join(word.encode('utf8') for word in self.ootg[:20])))
        if not self.sr_position:
            raise Exception('Too few training pairs')
        logging.info('Pickling training data to {}'.format(train_dat_fn))
        cPickle.dump((self.sr_train, self.sr_freq_not_seed, self.sr_position,
                      self.tg_train, self.ootg), open(train_dat_fn, mode='wb'))

    def get_trans_model(self):
        """
        http://stackoverflow.com/questions/19650115/which-scikit-learn-tools-\
                can-handle-multivariate-output
        """
        self.trans_model = LinearRegression()
        # TODO parametrize trans_model
        # TODO crossvalidation
        logging.info('Fitting translation model {}...'.format(
            self.trans_model))

    def collect_main(self):
        """
        First look for translations with gold data to see precision, then
        compute translations of frequent words without seed translation.
        """
        self.neighbour_k = 10
        self.populate_nearpy_engine()
        logging.info('Collecting translations...')
        self.collected = 0
        self.has_seed = 0
        self.score_at_5 = 0
        self.score_at_1 = 0
        for sr_word, sr_vec in izip(
                self.sr_model.index2word[self.sr_position:],
                self.sr_model.syn0[self.sr_position:]):
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
                'Translating frequent words without seed translation...')
            for sr_word, sr_vec in self.sr_freq_not_seed:
                self.test_item(sr_word, sr_vec)
        else:
            logging.info('Frequent words without seed translation skipped.')

    def populate_nearpy_engine(self):
        """
        Populates the nearpy engine. Note that the instanciation of the PCA
        hash means a PCA of the target embedding and consumes much memory.
        """
        logging.info('Creating nearpy engine...')
        hashes = [PCABinaryProjections(
            'ne1v', 1, self.tg_model.syn0.T[:1000,:])]
        logging.info(hashes)
        dim = self.tg_model.layer1_size
        self.engine = Engine(dim, lshashes=hashes, vector_filters=[],
                             distance=[])
        for ind in xrange(self.tg_model.syn0.shape[0]):
            if not ind % 100000:
                logging.debug(
                    '{} target words added to nearpy engine'.format(ind))
            self.engine.store_vector(self.tg_model.syn0[ind,:], ind)

    def test_item(self, sr_word, sr_vec, prec_at=9):
        if not self.collected % 100:
            logging.debug('{} translations collected, {} have reference\
                          translation'.format( self.collected, self.has_seed))
        self.collected += 1
        guessed_vec = self.trans_model.predict(sr_vec.reshape((1,-1)))
        # TODO? .astype('float32')
        if self.args.exact_neighbour:
            near_vecs = self.tg_model.syn0
        else:
            near_vecs, near_inds = izip(
                *self.engine.neighbours(guessed_vec.reshape(-1)))
        #if not self.collected % 100: logging.debug(
        #'{} approximate neighbours'.format(len(near_inds)))
        distances = cdist(near_vecs, guessed_vec, 'cosine').reshape(-1)
        # ^ 1/3 of time is spent here
        inds_among_near = np.argsort(distances)
        tg_indices_ranked = [near_inds[i] for i in inds_among_near]
        gold = self.eval_item_with_gold(
            sr_word, tg_indices_ranked[:self.neighbour_k])
        best_dist = distances[inds_among_near[0]]
        self.outfile.write(
            '{sr_w}\t{gold_tg_w}\t{gold_rank}\t{dist:.4}\t{tg_ws}\n'.format(
                sr_w=sr_word.encode('utf-8'),
                gold_tg_w=gold['word'].encode('utf-8'),
                gold_rank=gold['rank'],
                dist=best_dist,
                tg_ws=' '.join(
                    self.tg_index[ind].encode('utf-8')
                    for ind in tg_indices_ranked[:prec_at])))

    def eval_item_with_gold(self, sr_word, tg_indices_ranked):
        """
        Looks up the gold target word and books precision.
        """
        gold = {
            'word': '', 'rank': '>{}'.format(self.neighbour_k)}
        if sr_word in self.seed_dict:
            gold['word'] = self.seed_dict[sr_word]
            self.has_seed += 1
            if gold['word'] in self.tg_index.itervalues():
                index = self.tg_index[:gold['word']]
                if index in tg_indices_ranked:
                    gold['rank'] = tg_indices_ranked.index(index)
                    if gold['rank'] < 5:
                        self.score_at_5 += 1
                        if gold['rank'] == 0:
                            self.score_at_1 += 1
        return gold

    def score_main(self):
        raise NotImplementedError


if __name__=='__main__':
    output_dir = '/mnt/store/home/makrai/project/efnilex'
    #cProfile.run('
    LinearTranslator(output_dir).main()#')
