import os.path
import pickle
from itertools import izip
import logging
from collections import defaultdict
import re
import argparse

from nltk.corpus import stopwords
import numpy as np
from sklearn.linear_model import LinearRegression  #, ElasticNet, Lasso, Ridge
#from sklearn.svm import SVR
from gensim.models import Word2Vec

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

    def init_collecting(self, train_size):
        out_dict_filen = self.output_dir + self.params.outfilen
        if os.path.isfile(out_dict_filen): 
            raise Exception(
                'file for collected translation pairs exists {}'.format(
                    out_dict_filen))
        self.init_logging(self.params.log_to_err)
        self.outfile = open(out_dict_filen, mode='w')
        self.train_on_full = False
        self.train_size = train_size
        if not self.train_size:
            if self.params.test_mode == 'collect':
                self.train_size = 5000  
            else: 
                self.train_size = 20000
        self.test_indices = []

    def __init__(self, params, train_size=None):
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
                self.params.outfilen = self.params.source_embedding.split('/')[-1]
            else:
                self.params.outfilen = '{}__{}__{}_{}f_c{}'.format(
                    self.params.source_embedding.split('/')[-1],
                    self.params.target_embedding.split('/')[-1],
                    self.params.seed_name.split('/')[-1], 
                    int(self.params.forced_stem),
                    self.params.restrict_embed)
        if self.params.test_mode == 'collect':
            self.init_collecting(train_size)
        elif self.params.test_mode == 'accuracy':
            self.init_logging(self.params.log_to_err)
        elif self.params.test_mode == 'vocab':
            self.init_logging(True)

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
        self.hunspell_vocab = defaultdict(set) # {'sr': ..., 'tg': ...}
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
                            words.add(word)
                    else:
                        # string refused by hunspell
                        pass

    def get_seed_dict_filen(self, fallback):
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
        #separator = ' ' if 'opus' in self.params.seed_name else '\t'
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
                    #if ' ' in sr_word:
                    #    continue
                    if sr_word in self.seed_dict:
                        # In the case of wikt2dict, the first translation is
                        # present in more editions of Wiktionary.
                        continue
                    self.seed_dict[sr_word] = tg_word
            logging.info('{} seed pairs'.format(len(self.seed_dict)))
        else:
            logging.warning('seed does not exist: '+filen)
        if len(set(self.seed_dict.keys()).intersection(
            set(self.embeds['sr'].keys()))) < self.train_size:
            if fallback:
                raise Exception('too few training pairs')
            else:
                logging.info('fallbacking to broader seed')
                self.read_seed_dict(fallback=True)

    def load_embed(self, filen, write_vocab=False, just_model=False):
        if re.search('polyglot-..\.pkl$', filen):
            words, vecs = pickle.load(open(filen, mode='rb'))
        else:
            if re.search('webcorp/polyglot', filen):
                from polyglot2.polyglot2 import Polyglot
                logging.info('loading embedding from {}'.format(filen))
                embed0 = Polyglot.load_word2vec_format(filen)
                words = embed0.index2word
                vecs = embed0.vectors
            elif 'bin' in filen: 
                logging.debug('bin')
                embed0 = Word2Vec.load_word2vec_format(filen, binary=True)
                words = embed0.index2word
                vecs = embed0.syn0
            elif re.search('gensim', filen):
                embed0 = Word2Vec.load(filen)
                words = embed0.index2word
                vecs = embed0.syn0
            elif 'hunvec' in filen:
                mx = nnlm.model.get_params()[0].get_value()
                # TODO
            else:
                # mikolov, orig, glove, hpca, senna
                embed0 = Word2Vec.load_word2vec_format(filen)
                words = embed0.index2word
                vecs = embed0.syn0
        logging.info(
            'shape of embedding: {}'.format(vecs.shape))
        if write_vocab:
            self.write_vocab(filen, words)
        if just_model:
            return embed0
        else:
            embed = dict(izip(words, vecs))
            return words, vecs, embed

    def write_vocab(self, efilen, words):
            vfilen = efilen  +'.vocab'
            with open(vfilen, mode='w') as vocab_file:
                for word in words:
                    vocab_file.write(word.encode('utf8')+'\n')
            logging.info(
                'vocab written to {}'.format(vfilen))

    def append_training_item(self, index, log_ooseed, ooseed_file):
        """
        appends the index-th source word to the training sample
        """
        sr_word = self.sr_words[index]
        if sr_word.lower() in stopwords.words('english'):
            self.oov['stopword'] += 1
            return 0
        if sr_word not in self.seed_dict:
            # mostly punctuation and inflected forms
            # and Martin, Roger,...
            if log_ooseed:
                ooseed_file.write(sr_word.encode('utf8')+'\n')
            self.test_indices.append(index)
            self.oov['seed dict'] += 1
            return 0
        tg_word = self.seed_dict[sr_word]
        if tg_word in self.embeds['tg']:
            self.sr_train.append(self.embeds['sr'][sr_word])
            self.tg_train.append(self.embeds['tg'][tg_word])
            self.train_collected += 1
        else:
            self.ootg.append(tg_word)
            self.oov['tg (filtered) embed'] += 1

    def load_model(self, trans_model_filen):
        logging.info('loading translation mx from {}'.format(
            trans_model_filen))
        self.test_indices, self.model = pickle.load(open(
            trans_model_filen, mode='rb'))

    def train(self, log_ooseed=False, ooseed_filen=None,
              trans_model_filen=None):
        logging.info('train on full: {}'.format(self.train_on_full))
        logging.info('training translation model')
        self.sr_train = []
        self.tg_train = []
        self.train_collected = 0
        self.oov = defaultdict(int)
        ooseed_file = open( ooseed_filen, mode='w') if log_ooseed else None
        self.oosr = []
        self.ootg = []
        index = 0 # 4
        while self.train_collected < self.train_size or self.train_on_full: 
            self.append_training_item(index, log_ooseed, ooseed_file)
            index += 1
        self.test_indices += range(index, self.sr_vocab_size) 
        logging.info(
            '{} of {} words in sr embedding added to the training sample\noov: {}'.format(
                self.train_collected, index, self.oov))
        logging.info('filtered out of source embedding:\t'+'; '.join(
            [word.encode('utf8') for word in self.oosr[:20]]))
        logging.info('out of target embedding:\t'+'; '.join(
            [word.encode('utf8') for word in self.ootg[:20]]))
        logging.info('fitting model')
        self.model.fit(np.array(self.sr_train), np.array(self.tg_train))
        if trans_model_filen:
            pickle.dump((self.test_indices, self.model), open(
                trans_model_filen, mode='wb'))

    def restrict_embed(self, sr=False, tg=False):
        if sr:
            logging.info(
                'restricting source embedding from {:,} items '.format(
                    len(self.sr_words)))
            self.sr_words, self.sr_vecs = zip(
                *[(word, vec) 
                  for word, vec in zip(self.sr_words, self.sr_vecs) 
                  if word in self.hunspell_vocab['sr']])
        self.sr_vocab_size = len(self.sr_words)
        if sr:
            logging.info('to {:,} items'.format(self.sr_vocab_size))
        if tg:
            logging.info(
                'restricting target embedding from {:,} items '.format(
                    len(self.tg_words)))
            self.tg_words, self.tg_vecs = zip(
                *[(word, vec) 
                  for word, vec in zip(self.tg_words, self.tg_vecs) 
                  if word in self.hunspell_vocab['tg']])
        self.tg_vocab_size = len(self.tg_words)
        if tg:
            logging.info('to {:,} items'.format(self.tg_vocab_size))

    def get_part_argsorted(self, start, end, prec_at):
        """
        called by test_part()
        """
        sim_mx = self.guessed_vecs[start:end,:].dot(self.tg_vecs)
        # TODO scipy.linalg.blas (3/5 time is spent with this line)
        logging.debug('similarities computed')
        rankmx = np.argsort(-sim_mx, axis=1)
        logging.debug('target words ranked')
        return zip(self.sr_words_test[start:end], sim_mx,
                   self.guessed_norm_mx[start:end], rankmx)

    def prec_msg(self):
        if self.has_seed:
            return 'on {} words, prec@1: {:.2%}\tprec@5: {:.2%}'.format(
                self.has_seed, *[score/float(self.has_seed) for score in [
                    self.score_at_1, self.score_at_5]])
        else:
            logging.error('no gold data')

    def gold_tg_word_with_rank(self, sr_word, tg_rank_row):
        """
        Looks up the gold target word, computes its similarity rank to the
        computed target vector, and books precision.
        """
        self.collected += 1
        if sr_word in self.seed_dict:
            self.has_seed += 1
            gold_tg_word = self.seed_dict[sr_word]
            gold_index_l = np.where(self.tg_words == gold_tg_word)[0]
            if gold_index_l:
                gold_rank = np.where(tg_rank_row == gold_index_l[0])[0][0]
                if gold_rank < 5:
                    self.score_at_5 += 1
                    if gold_rank == 0:
                        self.score_at_1 += 1
            else:
                gold_rank = ''
            if self.has_seed == 1000:
                logging.info(self.prec_msg())
            self.finished = bool(
                self.has_seed==self.train_size 
                if self.params.test_mode=='score' 
                #else self.collected >= 100000)
                else self.has_seed > 1000)
        else:
            gold_tg_word = ''
            gold_rank = ''
            self.finished = False
        return gold_tg_word, gold_rank

    def test_part(self, start, end, prec_at):
        """
        Tests a part of the test data. Test data is split to parts so that
        similarity matrices fit into memory.
        """
        test_tuples = self.get_part_argsorted(start, end, prec_at)
        logging.info('test tuples built')
        for sr_word, sim_row, guessed_norm_row, tg_rank_row in test_tuples:
            gold_tg_word, gold_rank = self.gold_tg_word_with_rank(
                sr_word, tg_rank_row)
            self.outfile.write('\t'.join([
                sr_word, gold_tg_word, str(gold_rank),
                '{0:.4}'.format(sim_row[tg_rank_row[0]]/guessed_norm_row)] + 
                self.tg_words[
                    np.array(tg_rank_row[:prec_at])].tolist()).encode(
                        'utf8')+'\n')
            if self.finished:
                break

    def get_part_size(self, test_size):
        """
        Used by collect_translations
        """
        # TODO
        print resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        part_size = 1600000000000
        for denom_term in [self.sr_vecs_test.shape[1], self.tg_vecs.shape[0],
                           self.tg_vecs.shape[1]]:
            part_size /= denom_term
        logging.info('testing in {} parts'.format(
            (test_size / part_size) + 1))
        return part_size

    def collect_translations(self, prec_at=9):
        """
        Guessed target vectors for the whole test sample are computed in one
        step, other testing computations are done in parts of length 
        part_size.
        """
        test_size = len(self.sr_words_test)
        logging.info('test sample size {}'.format(test_size))
        logging.info('computing target vector estimates')
        self.guessed_vecs = self.model.predict(self.sr_vecs_test)
        self.guessed_norm_mx = np.apply_along_axis(
            np.linalg.norm, 1, self.guessed_vecs)
        self.tg_words = np.array(self.tg_words)
        tg_norms = np.apply_along_axis(
            np.linalg.norm, 1, self.tg_vecs).reshape(-1,1)
        self.tg_vecs = (self.tg_vecs/tg_norms).T
        self.collected = 0
        self.has_seed = 0
        self.score_at_1 = 0
        self.score_at_5 = 0
        part_size = self.get_part_size(test_size)
        for ind in xrange(0, test_size, part_size):
            end = min(ind + part_size, test_size)
            self.test_part(ind, end, prec_at=prec_at)
            if self.finished:
                break
            logging.info(
                '{} items (with {} gold translations) collected'.format(
                    #float(ind)/test_size,
                    self.collected,
                    self.has_seed))

    def score_translations(self):
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

    def test(self, measure='cos'):
        logging.info('starting testing')
        if self.train_on_full:
            self.test_indices = range(self.sr_vocab_size)
        self.sr_words_test = np.array(self.sr_words)[self.test_indices]
        self.sr_vecs_test = np.array(self.sr_vecs)[self.test_indices]
        del self.sr_vecs
        if self.params.test_mode == 'collect':
            del self.embeds
            self.collect_translations()
        elif self.params.test_mode == 'score':
            self.score_translations()

    def biling_main(self):
        """
        This is the main function on the two bilingual tasks, 'collect' and
        'score'.
        """
        self.embeds = {}
        self.sr_words, self.sr_vecs, self.embeds['sr'] = self.load_embed(
            self.embed_filens['sr'])
        self.tg_words, self.tg_vecs, self.embeds['tg'] = self.load_embed(
            self.embed_filens['tg'])
        self.read_hunspell_vocab()
        self.restrict_embed(
            sr='s' in self.params.restrict_embed,
            tg='t' in self.params.restrict_embed)
        self.read_seed_dict()
        self.model = LinearRegression() # TODO parametrize (penalty, ...)
        # TODO self.model = ElasticNet(), Lasso, Ridge, SVR
        # http://stackoverflow.com/questions/19650115/which-scikit-learn-tools-can-handle-multivariate-output
        # TODO crossvalidation
        logging.info(str(self.model))
        self.train()
        self.test()

    def main(self):
        self.embed_filens = {
            'sr': self.params.source_embedding, 
            'tg': self.params.target_embedding}
        if self.params.test_mode == 'accuracy':
            model = self.load_embed(self.embed_filens['sr'],
                                    just_model=True)
            lower = False
            # TODO
            for label in ['+l', '1l', 'stem']:
                if label in self.embed_filens['sr']:
                    lower = True
            return model.accuracy( 
                os.path.expanduser('~') +
                '/project/efnilex/vector/test/hu/questions-words.txt',
                lower=lower)
        elif self.params.test_mode == 'vocab':
            self.load_embed(self.embed_filens['sr'], write_vocab=True)
        else:
            self.biling_main()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--test-mode", dest='test_mode', 
        choices=['collect', 'score', 'accuracy', 'vocab'], default='collect')
    parser.add_argument(
        '-l', '--log-to-screen', action='store_true', dest='log_to_err')
    parser.add_argument(
        "-d", "--seed-dict", type=str, 
        #choices=["wikt2dict", "eniko", "vonyo", "opus"],
        default="wikt2dict",
        dest='seed_name')
    parser.add_argument(
        '-p', '--pair', default='en_hu', 
        help='language pair (in alphabetical order)')
    parser.add_argument(
        "-s", "--source-embedding", dest='source_embedding', type=str)
    parser.add_argument(
        "-t", "--target-embedding", dest='target_embedding', type=str)
    parser.add_argument(
        "-r", "--reverse", 
        action='store_true',
        help="use if the seed dict contains pair in reverse order")
    parser.add_argument(
        '-o', '--output', dest='outfilen')
    parser.add_argument(
        "-f", "--forced-stem", dest='forced_stem', action='store_true',
        help='consider only "forced" stems')
    parser.add_argument(
        '-c', '--restrict-embed', choices = ['n', 's', 't', 'st'],
        dest='restrict_embed', default='n')
    return parser.parse_args()


if __name__=='__main__':
    Vector2Dict(parse_args()).main()
