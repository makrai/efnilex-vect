import os.path
import pickle
import logging
import argparse
from collections import defaultdict
import re

from bidict import bidict
from nltk.corpus import stopwords
import numpy as np
from sklearn.linear_model import LinearRegression
from gensim.models import Word2Vec
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.filters import NearestFilter

class Vector2Dict:
    """
    Collects translational word pairs from neural language models (embedding)
    for the source and the target language.  The translation model is a linear
    mapping between the two vector spaces following Mikolov et al (2013
    Exploiting...).
    """
    def init_logging(self, log_to_err):
        level = logging.DEBUG
        format_ = "%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s"
        if log_to_err:
            logging.basicConfig(level=level, format=format_) 
        else:
            filename = self.output_dir + 'log/' + self.params.outfilen
            try:
                os.remove(filename)
            except:
                pass
            logging.basicConfig(filename=filename, level=level,
                                format=format_) 

    def init_collecting(self, train_needed):
        out_dict_filen = self.output_dir + self.params.outfilen
        if os.path.isfile(out_dict_filen): 
            raise Exception(
                'file for collected translation pairs exists {}'.format(
                    out_dict_filen))
        self.init_logging(self.params.log_to_err)
        self.outfile = open(out_dict_filen, mode='w')
        self.train_needed = train_needed
        if not self.train_needed:
            if self.params.test_mode == 'collect':
                self.train_needed = 5000  
            else:
                # This branch is for test mode 'score'
                self.train_needed = 20000
        self.test_indices = []

    def strip_embed_filen(self, filen):
        filen = filen.split('/')[-1]
        for ext in ['w2v', 'gensim', 'gz', 'bin', 'pkl']:
            filen = filen.split('.'+ext)[0]
        return filen

    def __init__(self, params, train_needed=None):
        # TODO config file
        self.vecnil_dir = '/mnt/store/home/makrai/project/efnilex/vector/'
        self.params = params
        if self.params.test_mode == 'accuracy':
            self.output_dir = self.vecnil_dir + 'test/'
        elif self.params.test_mode == 'collect':
            self.output_dir = self.vecnil_dir + 'dict/'
        elif self.params.test_mode == 'score':
            self.output_dir = self.vecnil_dir + 'score/'
        if not self.params.outfilen:
            if self.params.test_mode == 'accuracy':
                self.params.outfilen = self.strip_embed_filen(
                    self.params.source_embedding)
            else:
                self.params.outfilen = '{}__{}__{}_{}f_c{}'.format(
                    self.strip_embed_filen(self.params.source_embedding),
                    self.strip_embed_filen(self.params.target_embedding),
                    self.params.seed_name.split('/')[-1], 
                    int(self.params.forced_stem),
                    self.params.restrict_embed)
        self.embed_filens = {
            'sr': self.params.source_embedding, 
            'tg': self.params.target_embedding}
        if self.params.test_mode == 'accuracy':
            self.init_logging(self.params.log_to_err)
            self.accuracy_main()
        elif self.params.test_mode == 'vocab':
            self.init_logging(True)
            self.load_embed(self.embed_filens['sr'], write_vocab=True)
        else: 
            # This branch is for test_modes 'collect' and 'score'
            self.init_collecting(train_needed)
            self.biling_main()

    def read_hunspell_vocab(self):
        """
        self.hunspell_vocab can be used to restrict source and target
        vocabulary to words accepted by hunspell
        To create the input for this function, run
            load_embed(..., self.write_vocab=True) (a function in this class)
            hunspell -s [-d /home/ndavid/Projects/Jofogas/dat/hu_HU] \
                    embed.vocab > embed.hunspell (in bash)
        """
        logging.info('forced stemming: {}'.format(self.params.forced_stem))
        self.hunspell_vocab = defaultdict(set) 
        # There will be two keys: hunspell_vocab = {'sr': ..., 'tg': ...}
        for half, embed_filen in self.embed_filens.iteritems():
            if half[0] in self.params.restrict_embed:
                for corp_name in ['webcorp', 'slwac2.0', 'Lithuanian']:
                    if corp_name in embed_filen:
                        hv_filen = self.vecnil_dir+'hunspell/'+corp_name
                        break
                else:
                    hv_filen = embed_filen+'.hunspell'
                logging.info(
                    'reading filtered vocabulary for {} from {}'.format(
                        half, hv_filen))
                if self.params.forced_stem:
                    self.read_hunspell_vocab_forced_stem(half, hv_filen)
                else:
                    with open(hv_filen) as vocab_file:
                        for line in vocab_file:
                            if ' ' not in line:
                                continue
                            word, _ = line.strip().decode('utf8').split(' ')
                            if word in self.embeds[half]:
                                #      ^ TODO
                                self.hunspell_vocab[half].add(word)

    def read_hunspell_vocab_forced_stem(self, half, filen):
        """
        The term "forced stemming" if coined on the analogy of strong
        stemming: for each word form, if there is a stem different from the
        "inflected" one, choose that one.
        """
        inflexed = ''
        words = set()
        with open(filen) as vocab_file:
            for line in vocab_file:
                if line.strip() == '':
                    if inflexed and len(words) > 1:
                        words.discard(inflexed)
                    for word in words:
                        self.hunspell_vocab[half].add(word)
                    words = set()
                else:
                    line = line.strip().decode('utf8') 
                    if ' ' in line:
                        inflexed, word = line.split(' ')
                        # word is the stem
                        if word in self.embeds[half]:
                            #      ^ TODO
                            words.add(word)
                    else:
                        # string refused by hunspell
                        pass

    def get_seed_dict_filen(self, fallback):
        # TODO config file
        if self.params.seed_name=='wikt2dict':
            return os.path.expanduser(
                "~")+'/repo/wikt2dict/dat/{}_{}'.format(
                    self.params.pair,
                    'triang' if fallback else 'direct')
        elif self.params.seed_name=='opus':
            return os.path.expanduser(
                '~')+'/project/efnilex/data/opus/{}{}.dic'.format(
                    self.params.pair.replace('_', '-'),
                    '_big' if fallback else '_small')
        elif self.params.seed_name=='eniko':
            return os.path.expanduser(
                "~")+'/project/efnilex/data/dict/eniko/{}/{}.gpp'.format(
                    self.params.pair, self.params.pair)
        elif self.params.seed_name=='vonyo':
            return '/home/zseder/Proj/NPAlign/Data/Dicts/vonyo_en_hu.txt'
        else:
            return self.params.seed_name

    def read_seed_dict(self, fallback=False):
        columns = [1,3] if 'wikt2dict' in self.params.seed_name else range(2)
        if self.params.reverse:
            columns.reverse()
        filen = self.get_seed_dict_filen(fallback)
        if not fallback:
            self.seed_dict = {}
        if os.path.isfile(filen):
            logging.info('reading seed dictionary from '+filen)
            with open(filen) as file_:
                for line in file_.readlines():
                    separator = '\t' if '\t' in line else ' '
                    cells = line.decode('utf8').strip().split(separator)
                    sr_word, tg_word = [cells[index] for index in columns]
                    if ' ' in sr_word:
                        # TODO
                        pass
                    if sr_word in self.seed_dict:
                        # It is assumed that the first translation is the
                        # best. E.g. in the case of wikt2dict, the first
                        # translation is present in more editions of
                        # Wiktionary.
                        continue
                    self.seed_dict[sr_word] = tg_word
            logging.info('{} seed pairs'.format(len(self.seed_dict)))
        else:
            logging.warning('seed does not exist: '+filen)
        #logging.debug(self.seed_dict.items()[:20])
        seed_vocab = set(self.seed_dict.keys())
        # TODO .intersection( set(self.sr_words))  
        if len(seed_vocab) < self.train_needed:
            if fallback:
                logging.error('too few training pairs')
                raise Exception('too few training pairs')
            else:
                logging.info('fallbacking to broader seed')
                self.read_seed_dict(fallback=True)

    def load_embed(self, filen, write_vocab=False, type_='mx_and_bidict'):
        if re.search('polyglot-..\.pkl$', filen):
            words, vecs = pickle.load(open(filen, mode='rb'))
        else:
            if re.search('webcorp/polyglot', filen):
                from polyglot2.polyglot2 import Polyglot
                logging.info('loading embedding from {}'.format(filen))
                embed0 = Polyglot.load_word2vec_format(filen)
                words = embed0.index2word
                vecs = embed0.vectors
            elif re.search('gensim', filen):
                embed0 = Word2Vec.load(filen)
                words = embed0.index2word
                vecs = embed0.syn0
            elif 'hunvec' in filen:
                mx = nnlm.model.get_params()[0].get_value()
                # TODO
            else:
                # This branch is for target embeddings in the format of the original C code 
                embed0 = Word2Vec.load_word2vec_format(filen, 
                                                       binary='bin' in filen)
                words = embed0.index2word
                vecs = embed0.syn0
        logging.info(
            'shape of embedding: {}'.format(vecs.shape))
        if write_vocab:
            self.write_vocab(filen, words)
        if type_ == 'model':
            return embed0
        elif type_ == 'mx_and_bidict':
            return vecs.astype(
                'float32', casting='same_kind', copy=False), bidict(
                    enumerate(words))

    def write_vocab(self, efilen, words):
            vfilen = efilen  +'.vocab'
            with open(vfilen, mode='w') as vocab_file:
                for word in words:
                    vocab_file.write(word.encode('utf8')+'\n')
            logging.info(
                'vocab written to {}'.format(vfilen))

    def append_training_item(self, sr_word, sr_vec=None, log_ooseed=False,
                             ooseed_file=None):
        if sr_word.lower() in stopwords.words('english'):
            self.oov['stopword'] += 1
            return 0
        tg_word = self.seed_dict[sr_word]
        if tg_word in self.tg_vocab:
            if sr_vec is None:
                # This branch is for mode 'score'
                sr_vec = self.embeds['sr'][sr_word]
            self.sr_train.append(sr_vec)
            self.tg_train.append(self.tg_vecs[self.tg_index[:tg_word]])
        else:
            self.ootg.append(tg_word)
            self.oov['tg (filtered) embed'] += 1

    def read_word_and_vec(self, line):
        cells = line.decode('utf8').strip().split(' ')
        word = cells[0]
        vec = np.array([float(coord) for coord in
                        cells[1:]]).astype('float32')
        return word, vec

    def get_training_data(self):
        logging.info('collecting training data')
        train_collected = 0
        self.tg_vocab = set(self.tg_index.itervalues())
        for line in self.sr_embed_f:
            sr_word, sr_vec = self.read_word_and_vec(line)
            if train_collected < self.train_needed:
                if sr_word in self.seed_dict:
                    train_collected += 1
                    if not train_collected % 1000:
                        logging.debug(
                            '{} training items collected'.format(train_collected))
                    self.append_training_item(sr_word, sr_vec=sr_vec)
                else:
                    self.sr_freq_not_seed.append((sr_word, sr_vec))
            else:
                break
        logging.debug(
            'out of target embed: {}'.format(
                '; '.join(word.encode('utf8') for word in self.ootg[:20])))
        if train_collected < self.train_needed:
            raise Exception(
                'Too few training pairs ({})'.format(train_collected))

    def get_nearpy_engine(self, top_n=10):
        rbps = []
        for _ in xrange(1):
            # TODO 
            #   less or more projections
            #   other types of projections
            rbps.append(RandomBinaryProjections('rbp', 10))
        dim = self.tg_vecs.shape[1]
        self.engine = Engine(dim, lshashes=rbps, vector_filters=[NearestFilter(top_n)])
        for ind, vec in enumerate(self.tg_vecs):
            if not ind % 100000:
                logging.info(
                    '{} target words added to nearpy engine'.format(ind))
            self.engine.store_vector(vec, ind)

    def restrict_embed(self, sr=False, tg=False):
        # TODO At the moment, this function is not called.
        if sr:
            logging.info(
                'restricting source embedding from {:,} items '.format(
                    len(self.sr_words)))
            self.sr_words, self.sr_vecs = zip(
                *[(word, vec) 
                  for word, vec in zip(self.sr_words, self.sr_vecs) 
                  if word in self.hunspell_vocab['sr']])
            logging.info('to {:,} items'.format(len(self.sr_words)))
        if tg:
            logging.info(
                'restricting target embedding from {:,} items '.format(
                    len(self.tg_words)))
            self.tg_words, self.tg_vecs = zip(
                *[(word, vec) 
                  for word, vec in zip(self.tg_words, self.tg_vecs) 
                  if word in self.hunspell_vocab['tg']])
            logging.info('to {:,} items'.format(len(self.tg_words)))

    def prec_msg(self):
        if self.has_seed:
            return 'on {} words, prec@1: {:.2%}\tprec@5: {:.2%}'.format(
                self.has_seed, *[score/float(self.has_seed) for score in [
                    self.score_at_1, self.score_at_5]])
        else:
            logging.error('no gold data')

    def eval_item_with_gold(self, sr_word, tg_rank_row):
        """
        Looks up the gold target word, computes its similarity rank to the
        computed target vector, and books precision.
        """
        self.has_seed += 1
        gold_tg_word = self.seed_dict[sr_word]
        if gold_tg_word in self.tg_index.itervalues():
            gold_rank = np.where(
                tg_rank_row == self.tg_index[:gold_tg_word])[0][0]
            if gold_rank < 5:
                self.score_at_5 += 1
                if gold_rank == 0:
                    self.score_at_1 += 1
        else:
            gold_rank = ''
        if self.has_seed == 1000:
            logging.info(self.prec_msg())
        self.finished = bool(
            self.has_seed==self.train_needed 
            if self.params.test_mode=='score' 
            # This TODO if for not only measuring precision but collecting as
            # many translations possible. If sr embedding is properly chosen,
            # no breaking is needed. elif self.collected >= 100000)
            else self.has_seed > 1000)
        return gold_tg_word, gold_rank


    def test_item(self, sr_word, sr_vec, prec_at=9):
        self.collected += 1
        if not self.collected % 100:
            logging.debug(
                '{} translations collected, {} have reference translation'.format(
                    self.collected, self.has_seed))
        guessed_vec = self.model.predict(
            sr_vec.reshape((1,-1))).astype('float32').reshape((-1))
        guessed_norm = np.linalg.norm(guessed_vec)
        if sr_word in self.seed_dict:
            sim_row = guessed_vec.dot(self.tg_vecs)
            tg_rank_row = np.argsort(-sim_row)
            gold_tg_word, gold_rank = self.eval_item_with_gold(
                sr_word, tg_rank_row)
        else:
            _, tg_rank_row, sim_row = zip(*self.engine.neighbours(guessed_vec))
            gold_tg_word = ''
            gold_rank = ''
            self.finished = False
        self.outfile.write('\t'.join([
            sr_word, gold_tg_word, str(gold_rank),
            '{0:.4}'.format(sim_row[0]/guessed_norm)] + 
            [self.tg_index.get(ind, 'OVERWRITTEN') for ind in tg_rank_row[:prec_at]]).encode(
                    'utf8')+'\n')

    def collect_translations(self):
        self.get_nearpy_engine()
        tg_norms = np.apply_along_axis(
            np.linalg.norm, 1, self.tg_vecs).reshape(-1,1)
        self.tg_vecs /= tg_norms
        self.tg_vecs = self.tg_vecs.T
        self.collected = 0
        self.has_seed = 0
        self.score_at_5 = 0
        self.score_at_1 = 0
        if self.params.translate_oov:
            for sr_word, sr_vec in self.sr_freq_not_seed:
                self.test_item(sr_word, sr_vec)
        logging.info('Frequent words without seed translation {}'.format(
                         'translated' if self.params.translate_oov else 'skipped'))
        for line in self.sr_embed_f:
            sr_word, sr_vec = self.read_word_and_vec(line)
            self.test_item(sr_word, sr_vec)
            if self.finished:
                break
        logging.info(self.prec_msg())

    def score_translations(self):
        # TODO rewrite using the bidict
        sr_words, sr_vecs, tg_words, tg_vecs = zip(*[ 
            (sr_word, self.embeds['sr'][sr_word], tg_word, self.embeds[
                'tg'][tg_word])
            for sr_word, tg_word in self.seed_dict.iteritems()
            if sr_word in self.embeds['sr'] and tg_word in self.embeds[
                'tg']])
        guessed_vecs = self.model.predict(np.array(sr_vecs))
        guessed_vecs /= np.apply_along_axis(np.linalg.norm, 1,
                                            guessed_vecs).reshape((-1,1))
        tg_vecs /= np.apply_along_axis(
            np.linalg.norm, 1, tg_vecs).reshape((-1,1))
        sim_mx = np.sum(guessed_vecs * tg_vecs, axis=1)
        for sr_word, tg_word, sim in zip(sr_words, tg_words, sim_mx):
            self.outfile.write('\t'.join([sr_word, tg_word,
                                           str(sim)]).encode('utf8')+'\n')

    def biling_main(self):
        """
        This is the main function on the two bilingual tasks, 'collect' and
        'score'.
        """
        self.tg_vecs, self.tg_index = self.load_embed( 
            self.embed_filens['tg'], type_='mx_and_bidict')
        self.read_seed_dict()
        self.model = LinearRegression() # TODO parametrize (penalty, ...)
        # TODO self.model = ElasticNet(), Lasso, Ridge, SVR
        # http://stackoverflow.com/questions/19650115/which-scikit-learn-tools-can-handle-multivariate-output
        # TODO crossvalidation
        logging.info(str(self.model))
        self.sr_embed_f = open(self.embed_filens['sr'])
        self.sr_embed_f.readline() # The header is skipped.
        self.sr_train = []
        self.tg_train = []        
        self.sr_freq_not_seed = []
        self.oov = defaultdict(int)
        self.oosr = []
        self.ootg = []
        self.get_training_data()
        logging.info('fitting model')
        self.model.fit(np.array(self.sr_train), np.array(self.tg_train))
        logging.info('testing')
        if self.params.test_mode == 'collect':
            self.collect_translations()
        elif self.params.test_mode == 'score':
            self.score_translations()

    def accuracy_main(self):
        model = self.load_embed(self.embed_filens['sr'], type_='model')
        lower = False
        # TODO
        for label in ['+l', '1l']:
            if label in self.embed_filens['sr']:
                lower = True
        return model.accuracy( 
            os.path.expanduser('~') +
            '/project/efnilex/vector/test/hu/questions-words.txt',
            lower=lower)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--test-mode", dest='test_mode', 
        choices=['collect', 'score', 'accuracy', 'vocab'], default='collect')
    parser.add_argument(
        '-l', '--log-to-screen', action='store_true', dest='log_to_err')
    parser.add_argument(
        "-s", "--source-embedding", dest='source_embedding', type=str)
    parser.add_argument(
        "-t", "--target-embedding", dest='target_embedding', type=str)
    parser.add_argument(
        "-d", "--seed-dict", type=str, 
        default="wikt2dict",
        dest='seed_name')
    parser.add_argument(
        '-p', '--pair', default='en_hu', 
        help='language pair (in alphabetical order)')
    parser.add_argument(
        "-r", "--reverse", 
        action='store_true',
        help="use if the seed dict contains pair in reverse order")
    parser.add_argument(
        '-o', '--output', dest='outfilen')
    parser.add_argument(
        '-c', '--restrict-embed', choices = ['n', 's', 't', 'st'],
        dest='restrict_embed', default='n')
    parser.add_argument(
        "-f", "--forced-stem", dest='forced_stem', action='store_true',
        help='consider only "forced" stems')
    parser.add_argument(
        "-v", "--translate_oov", action='store_true', 
        help='Translate frequent words without a seed dictionary translation')
    return parser.parse_args()


if __name__=='__main__':
    Vector2Dict(parse_args())#', 'tottime')
