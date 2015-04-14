import logging
import sys
import pickle
import os.path

from gensim.models.word2vec import Word2Vec


class EmbedInterProcessor():
    def __init__(self):
        self.spaces = [
            '\xe2\x80\x80', '\xC2\xA0', '\xe2\x80\x81', '\xe2\x80\x82',
            '\xe2\x80\x83', '\xe2\x80\x84', '\xe2\x80\x85', '\xe2\x80\x86',
            '\xe2\x80\x87', '\xe2\x80\x88', '\xe2\x80\x89', '\xe2\x80\x8a',
            '\xe2\x80\x8b', '\xe2\x80\x8c', '\xe2\x80\x8d', '\xe2\x80\x8e',
            '\xe2\x80\x8f', '\xe2\x80\xa8', '\xe2\x80\xa9', '\xe2\x80\xaa',
            '\xe2\x80\xab', '\xe2\x80\xac', '\xe2\x80\xad', '\xe2\x80\xae',
            '\xe2\x80\xaf', '\xe2\x81\x9f', '\xe2\x81\xa0', '\xe2\x81\xa1',
            '\xe2\x81\xa2', '\xe2\x81\xa3', '\xe2\x81\xa4', '\xe2\x81\xa5',
            '\xe2\x81\xa6', '\xe2\x81\xa7', '\xe2\x81\xa8', '\xe2\x81\xa9']

    def normalize(self, word):
        word = word.strip()
        for space in self.spaces:
            word.replace(space, '_')
        return word
            
    def main(self):
        for in_filen in sys.argv[1:]:
            logging.info('Processing {}'.format(in_filen))
            file_pref, ext = os.path.splitext(in_filen)
            if ext == '.pkl':
                # this branch is for embeddings from
                # https://sites.google.com/site/rmyeid/projects/polyglot
                with open(file_pref+'.w2v', mode='w') as out_file:
                    with open(in_filen, mode='rb') as in_file:
                        words, vecs = pickle.load(in_file)
                    out_file.write('{} {}\n'.format(*vecs.shape))
                    for word, vec in zip(words, vecs):
                        out_file.write('{}  {}\n'.format(
                            word.encode('utf8'), 
                            ' '.join(str(coord) for coord in vec.tolist())))
            elif ext == '.w2v':
                m = Word2Vec.load_word2vec_format(in_filen)
                m.save(file_pref+'.gensim')
            elif ext == '.txt':
                with open(file_pref+'.tmp', mode='w') as out_file:
                    # Output file need adding a header and replacing any kind of 
                    with open(in_filen) as in_file:
                        vocab_size = 0
                        dim = None
                        for i, line in enumerate(in_file):
                            try:
                                fields = line.strip().decode('utf-8').split(' ')
                            except UnicodeDecodeError as e:
                                logging.warn('in line {}, {}'.format(i, e))
                                continue
                            if not dim:
                                dim = len(fields) - 1
                                logging.info(
                                    'processing {}-dimensional model'.format(dim))
                            if len(fields) != dim + 1:
                                logging.info(
                                    'line with white space skipped (#{})'.format(
                                        i))
                                continue
                            fields[0] = self.normalize(fields[0])
                            if not fields[0]:
                                logging.info('empty word in line {}, skipped'.format(
                                    i))
                                continue
                            vocab_size += 1
                            out_file.write(' '.join(fields).encode('utf-8')+'\n')
                logging.info(
                    'Now you need to add the header\n{} {}'.format(vocab_size, dim))
            elif ext == '.bin':
                m = Word2Vec.load_word2vec_format(in_filen, binary=True)
                m.save(file_pref+'.gensim')


if __name__ == '__main__':
    format_ = "%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s"
    logging.basicConfig(format=format_, level=logging.DEBUG) 
    EmbedInterProcessor().main()
