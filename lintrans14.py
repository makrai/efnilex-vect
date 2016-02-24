import argparse
import codecs
import cPickle
#import cProfile
from collections import defaultdict
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
    Exploiting similarities...), trained and tested on a seed dictionary.
    """
    def __init__(self, args, output_dir=None):
        self.args = args
        self.output_dir = output_dir if output_dir else self.args.output_dir
        if not os.path.isdir(output_dir):
            os.mkdir(self.args.output_dir)
        self.mode_dir = "{}/{}/".format(
            self.output_dir,
            "analogy" if self.args.analogy else "collect")
        if not os.path.isdir(self.mode_dir):
            os.mkdir(self.mode_dir)
        self.get_outfn_rel()
        self.config_logger(self.args.log_to_err)
        if not self.args.analogy:
            self.init_collect()

    def get_outfn_rel(self):
        """"
        Gets the relative path of the output file that is named according to
        the pattern

            source__target__seed__opts

        where
            source describes the source language model
            target describes the target language model
            seed describes the seed dictionary
            opts describes the value of command line options v and n-proj
        """
        if not self.args.outfn_rel:
            if self.args.analogy:
                self.args.outfn_rel = self.strip_embed_filen(
                    self.args.source_fn)
            else:
                self.args.outfn_rel = "{}__{}__{}__{}v-{}p".format(
                    self.strip_embed_filen(self.args.source_fn),
                    self.strip_embed_filen(self.args.target_fn),
                    self.args.seed_fn.split("/")[-1],
                    int(self.args.trans_freq_oov),
                    self.args.n_proj)
        self.outfn_abs = self.mode_dir + self.args.outfn_rel
        if os.path.isfile(self.outfn_abs):
            raise Exception(
                "file for collected translation pairs exists {}".format(
                    self.outfn_abs))

    def config_logger(self, log_to_err):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        if log_to_err:
            handler = logging.StreamHandler()
        else:
            log_fn = "{}/{}/log/{}".format(
                self.output_dir,
                "analogy" if self.args.analogy else "collect",
                self.args.outfn_rel)
            if os.path.isfile(log_fn):
                os.remove(log_fn)
            handler = logging.FileHandler(log_fn, encoding='utf8')
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(module)s (%(lineno)s) %(levelname)s %(message)s"))
        logger.addHandler(handler)

    def init_collect(self):
        self.train_needed = 20000 if self.args.analogy else 5000
        self.test_needed= 20000 if self.args.analogy else 1000
        self.sr_model = self.load_embed(self.args.source_fn,
                                        vocab=self.args.sr_vocab)
        self.tg_model = self.load_embed(self.args.target_fn,
                                       vocab=self.args.tg_vocab)
        # TODO? self.tg_model.syn0.astype("float32", casting="same_kind",
        # copy=False)
        self.tg_index = bidict(enumerate(self.tg_model.index2word))
        self.tg_vocab = set(self.tg_index.itervalues())
        self.read_seed()
        self.get_training_data()
        self.get_trans_model()
        logging.info("Fitting translation model {}...".format(self.trans_model))
        self.trans_model.fit(np.array(self.sr_train), np.array(self.tg_train))

    def strip_embed_filen(self, old_filen):
        path, new_filen = os.path.split(old_filen)
        new_filen, ext = os.path.splitext(new_filen)
        if ext[1:] in ["w2v", "w2vbin", "gensim", "gz", "bin", "pkl", "txt"]:
            return self.strip_embed_filen(new_filen)
        else:
            return old_filen

    def load_embed(self, filen, vocab=None, write_vocab=False, type_="mx_and_bidict"):
        pref, ext = os.path.splitext(filen)
        if ext == '.gensim':
            try:
                return Word2Vec.load(filen)
            except AttributeError:
                if False: # TODO vocab:
                    m = Word2Vec.load_word2vec_format('{}.w2vbin'.format(pref),
                                                      fvocab=vocab, binary=True)
                    m.save('{}.gensim')
                    return m
                else:
                    msg = 'To load thies embedding, vocabulary file (with\
                            frequuencies) has to be specifies'
                    logging.exception(msg)
                    raise Exception(msg)
        elif ext.startswith('.w2v'):
            return Word2Vec.load_word2vec_format(
                filen, binary=ext.endswith('bin'))# TODO, fvocab=vocab)
        elif re.search("polyglot-..\.pkl$", filen):
            model = Word2Vec()
            words_t, model.syn0 = pickle.load(open(filen, mode="rb"))
            logging.info(
                "Embedding with shape {} loaded".format(model.syn0.shape))
            model.index2word = list(words_t)
            return model
        elif re.search("webcorp/polyglot", filen):
            from polyglot2.polyglot2 import Polyglot
            logging.info("loading lang_mod from {}".format(filen))
            return Polyglot.load_word2vec_format(filen)
        elif "hunvec" in filen:
            #mx = nnlm.model.get_params()[0].get_value()
            raise NotImplementedError
        else:
            raise Exception('extension {} unknown'.format(ext))

    def read_seed(self):
        """
        It could be assumed that the first translation in the file is the
        best one (the target word is more frequent or, in the case of a
        triangulated translation, the triangle is supported by more pivots).
        """
        logging.info("Reading seed dictionary from {}".format(
            self.args.seed_fn))
        _, ext = os.path.splitext(self.args.seed_fn)
        assert ext[1:] in ['ssv', '5col']
        columns = [1,3] if ext == '.5col' else range(2)
        if self.args.reverse:
            columns.reverse()
        self.seed_dict = defaultdict(list)
        with codecs.open(self.args.seed_fn, encoding="utf-8") as file_:
            for line in file_.readlines():
                separator = "\t" if "\t" in line else " "
                cells = line.strip().split(separator)
                sr_word, tg_word = [cells[index] for index in columns]
                if self.args.ambig or sr_word not in self.seed_dict:
                    self.seed_dict[sr_word].append(tg_word)
        logging.info("{} seed pairs e.g. {}".format(
            len(self.seed_dict),
            self.seed_dict.items()[:4]))
        if len(self.seed_dict.keys()) < self.train_needed:
                logging.error("too few training pairs")
                raise Exception("too few training pairs")

    def get_training_data(self):
        train_dat_fn = "{}/train_dat/{}__{}__{}.pkl".format(
            self.output_dir,
            self.strip_embed_filen(self.args.source_fn),
            self.strip_embed_filen(self.args.target_fn),
            self.args.seed_fn.split("/")[-1])
        if os.path.isfile(train_dat_fn):
            logging.info("loading training data from {}".format(train_dat_fn))
            (self.sr_train, self.sr_freq_not_seed, self.sr_position,
             self.tg_train, self.ootg) = cPickle.load(open(train_dat_fn,
                                                           mode="rb"))
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
                    tg_word = self.seed_dict[sr_word][0]
                    if tg_word in self.tg_vocab:
                        if not train_collected % 1000:
                            logging.debug(
                                "{} training items collected".format(
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
                logging.info(
                    "Training finished seed item #{}".format(
                        self.sr_position - 1))
                break
        else:
            logging.error("Too few training pairs")
            self.sr_position = None
        logging.debug("out of target embed: {}".format(
                "; ".join(word.encode("utf8") for word in self.ootg[:20])))
        if not self.sr_position:
            raise Exception("Too few training pairs")
        logging.info("Pickling training data to {}".format(train_dat_fn))
        cPickle.dump((self.sr_train, self.sr_freq_not_seed, self.sr_position,
                      self.tg_train, self.ootg), open(train_dat_fn, mode="wb"))

    def get_trans_model(self):
        """
        http://stackoverflow.com/questions/19650115/which-scikit-learn-tools-\
                can-handle-multivariate-output
        """
        self.trans_model = LinearRegression(n_jobs=6, fit_intercept=False) 
        # parameters of LinearRegression:
            # normalize doesn't seem to cause any difference
            # for saving memory, copy_X=True could be tried (with some risk)

    def main(self):
        if self.args.analogy:
            self.analogy_main()
        else:
            self.collect_main()

    def analogy_main(self):
        model = self.load_embed(self.args.source_fn, type_="model")
        return model.accuracy(
            "/home/makrai/project/efnilex/2014/vect/analogy/hu/questions-words.txt",
            lower=False)

    def collect_main(self):
        """
        First look for translations with gold data to see precision, then
        compute translations of frequent words without seed translation.
        """
        self.neighbour_k = 10
        self.prec_thresholds = [1, 5, 10]
        self.get_nearpy_engine()
        logging.info("Collecting translations...")
        self.collected = 0
        self.has_seed = 0
        self.score_at = defaultdict(int)
        self.without_neighbour = None
        with open(self.outfn_abs, mode="w") as self.outfile:
            for sr_word, sr_vec in izip(self.sr_model.index2word[self.sr_position:],
                                        self.sr_model.syn0[self.sr_position:]):
                self.test_item(sr_word, sr_vec)
                if self.has_seed == self.test_needed:
                    # TODO If the goal is not only measuring precision but
                    # collecting as many translations as possible, and sr lang_mod
                    # is properly chosen, no breaking is needed.
                    break
            else:
                logging.error(
                    "Only {} test pairs with gold data. {} needed.".format(
                        self.has_seed, self.test_needed))
            logging.info("on {} words, prec@1: {:.2%}\tprec@5: {:.2%}".format(
                self.has_seed,
                *[float(self.score_at[k])/self.has_seed
                  for k in self.prec_thresholds]))
            if self.args.trans_freq_oov:
                logging.info(
                    "Translating frequent words without seed translation...")
                for sr_word, sr_vec in self.sr_freq_not_seed:
                    self.test_item(sr_word, sr_vec)
            else:
                logging.info("Frequent words without seed translation skipped.")

    def get_nearpy_engine(self):
        """
        Populates the nearpy engine. Note that the instanciation of the PCA
        hash means a PCA of 1000 target vectors and may consume much memory.
        """
        logging.info("Creating nearpy engine...")
        hashes = [PCABinaryProjections(
            "ne1v", self.args.n_proj, self.tg_model.syn0[:1000,:].T)]
        logging.info(hashes)
        dim = self.tg_model.layer1_size
        self.engine = Engine(dim, lshashes=hashes, vector_filters=[],
                             distance=[])
        for ind in xrange(self.tg_model.syn0.shape[0]):
            if not ind % 200000:
                logging.debug(
                    "{} target words added to nearpy engine".format(ind))
            self.engine.store_vector(self.tg_model.syn0[ind,:], ind)

    def test_item(self, sr_word, sr_vec, prec_at=9):
        if True:#not self.collected % 100:
            msg = "{} translations collected, {} have reference translation"
            if self.without_neighbour:
                msg += ', except for {} ones, e.g. {}'.format(
                    len(self.without_neighbour),
                    ', '.join(self.without_neighbour[:9]).encode('utf8'))
            logging.debug(msg.format(self.collected, self.has_seed))
            self.without_neighbour = []
        self.collected += 1
        guessed_vec = self.trans_model.predict(sr_vec.reshape((1,-1)))
        # TODO? .astype("float32")
        if self.args.n_proj:
            try:
                near_vecs, near_inds = izip(
                    *self.engine.neighbours(guessed_vec.reshape(-1)))
                logging.debug(type(near_inds))
                #if not self.collected % 100: logging.debug(
                #"{} approximate neighbours".format(len(near_inds)))
            except ValueError:
                self.without_neighbour.append(sr_word)
                return
        else:
            near_vecs, near_inds = self.tg_model.syn0, tuple(
                range( self.tg_model.syn0.shape[0]))
        distances = cdist(near_vecs, guessed_vec, "cosine").reshape(-1)
        # ^ 1/3 of time is spent here
        inds_among_near = np.argsort(distances)
        tg_indices_ranked = [near_inds[i] for i in inds_among_near]
        gold = self.eval_item_with_gold(
            sr_word, tg_indices_ranked[:self.neighbour_k])
        best_dist = distances[inds_among_near[0]]
        self.outfile.write(
            "{sr_w}\t{gold_tg_w}\t{gold_rank}\t{dist:.4}\t{tg_ws}\n".format(
                sr_w=sr_word.encode("utf-8"),
                gold_tg_w=gold["word"].encode("utf-8"),
                gold_rank=gold["rank"],
                dist=best_dist,
                tg_ws=" ".join(
                    self.tg_index[ind].encode("utf-8")
                    for ind in tg_indices_ranked[:prec_at])))

    def eval_item_with_gold(self, sr_word, tg_indices_ranked):
        """
        Looks up the gold target words and books precision.
        """
        gold = {
            "word": "", "rank": ">{}".format(self.neighbour_k)}
        if sr_word in self.seed_dict:
            self.has_seed += 1
            gold_words_embedded = filter(lambda w: w in self.tg_vocab,
                                         self.seed_dict[sr_word])
            indices_gold_embedded = [self.tg_index[:w] 
                                     for w in gold_words_embedded]
            indices_gold_found = filter(lambda i: i in tg_indices_ranked,
                                        indices_gold_embedded)
            ranks_of_gold = set(tg_indices_ranked.index(i) 
                                for i in indices_gold_found)
            if not ranks_of_gold:
                return gold
            gold["rank"] = min(ranks_of_gold)
            gold["word"] = self.tg_index[tg_indices_ranked[gold["rank"]]]
            for k in self.prec_thresholds:
                if gold["rank"] < k:
                    self.score_at[k] += 1
        return gold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("seed_fn")
    parser.add_argument("source_fn")
    parser.add_argument("target_fn")
    parser.add_argument("--sr_vocab")
    parser.add_argument("--tg_vocab")
    parser.add_argument(
        "-r", "--reverse",
        action="store_true",
        help="use if the seed dict contains pair in reverse order")
    parser.add_argument(
        "-v", "--translate-oov", dest="trans_freq_oov",
        action="store_true",
        help="Not translate frequent words that are not covered by the\
        seed")
    parser.add_argument(
        "-b", "--non-ambig", dest="ambig",
        help="don't let words have more gold translations",
        action="store_false")
    parser.add_argument(
        "--n-proj", dest="n_proj", type=int, default=1,
        help="number of PCABinaryProjections in nearpy engine")
    parser.add_argument(
        "-a", "--analogy", action="store_true")
    parser.add_argument(
        "--output-directory", dest=output_dir)
    parser.add_argument(
        "-o", "--output-file-name", dest="outfn_rel",
        help="prefix of names of output and log files without path")
    parser.add_argument(
        "-l", "--log-to-stderr", action="store_true", dest="log_to_err")
    parser.add_argument(
        "-c", "--restrict-embed", choices = ["n", "s", "t", "st"],
        dest="restrict_embed", default="n",
        help="which of the source and the target embeddings to restrict:\
        *n*one, *s*ource, *t*arget, or *st* both")
    parser.add_argument(
        "-f", "--forced-stem", dest="forced_stem", action="store_true",
        help='consider only "forced" stems')
    return parser.parse_args()


if __name__=="__main__":
    output_dir = "/mnt/store/home/makrai/project/efnilex"
    #cProfile.run("
    args = parse_args()
    LinearTranslator(args, output_dir).main()
